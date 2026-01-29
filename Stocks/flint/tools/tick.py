#!/usr/bin/env python3
"""
Ticker Ingestion Client - Kit (TIC-K)

Production-grade financial data pipeline for OHLCV and fundamentals data.
No-cost solution using yfinance, robin_stocks, and Alpha Vantage

USAGE:
    python tick.py                                        # Full universe
    python tick.py --test 10                              # First 10 tickers
    python tick.py --tickers AAPL MSFT                    # Specific tickers
    python tick.py --no-fundamentals                      # OHLCV only
    python tick.py --tickers AAPL MSFT --no-fundamentals  # Combine options
    python tick.py --test 5000 --no-fundamentals          # Batch processing
"""

import os
import sys
import time
import logging
import argparse
import functools
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Callable
import pandas as pd
import duckdb
import yfinance as yf
import requests

# Optional: robin_stocks
try:
    import robin_stocks.robinhood as r
    ROBINHOOD_AVAILABLE = True
except ImportError:
    ROBINHOOD_AVAILABLE = False

# =============================================================================
# CONFIG & UTILS
# =============================================================================

class Config:
    # Paths
    BASE_DIR = Path("data_drops")
    DB_PATH = BASE_DIR / "pipeline_state.duckdb"
    LOG_PATH = Path("logs/pipeline.log")
    UNIVERSE_PATH = Path("universe_common.parquet")
    
    # Date Range
    START_DATE = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    # Rate Limits & Settings
    TARGET_TPM = 18  # Tickers per minute
    PER_REQUEST_SLEEP = 0.4  # Smooths bursts between individual requests
    BATCH_SIZE = 50
    MAX_RETRIES = 4
    BACKOFF_BASE = 2.0
    
    # Data Sources
    PRICE_PRIMARY = "yfinance"
    PRICE_FALLBACK = "robinhood"
    ENABLE_FUNDAMENTALS = True
    USE_ROBINHOOD_LOGIN = False
    
    # Alpha Vantage
    AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
    AV_URL = "https://www.alphavantage.co/query"
    AV_SLEEP = 15  # Space out calls conservatively
    AV_CAP = 25  # Free tier actual limit
    AV_STATEMENTS = ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"]
    
    # Robinhood
    ROBIN_USER = os.getenv("ROBIN_USER", "")
    ROBIN_PASS = os.getenv("ROBIN_PASS", "")

    @classmethod
    def batch_sleep(cls):
        """Calculate sleep between batches to maintain target TPM."""
        per_batch_time = cls.BATCH_SIZE * cls.PER_REQUEST_SLEEP
        target_batch_time = (cls.BATCH_SIZE / cls.TARGET_TPM) * 60
        return max(0, target_batch_time - per_batch_time)


def setup_logging():
    """Configure logging to file and console."""
    Config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_PATH),
            logging.StreamHandler(sys.stdout)
        ]
    )


def retry_with_backoff(retries=3, backoff_base=2.0):
    """Decorator to retry function with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    if attempt < retries - 1:
                        backoff = backoff_base ** attempt
                        logging.warning(f"{func.__name__} attempt {attempt + 1} failed: {str(e)[:100]}. Backoff {backoff}s")
                        time.sleep(backoff)
            raise last_err
        return wrapper
    return decorator


def atomic_write(df: pd.DataFrame, path: Path):
    """Write parquet atomically using temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


# =============================================================================
# DATABASE
# =============================================================================

class StateManager:
    """Manages pipeline state using DuckDB."""
    
    def __init__(self):
        Config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(Config.DB_PATH))
        self._init_schema()

    def _init_schema(self):
        """Create state tracking tables."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_runs (
                ticker VARCHAR PRIMARY KEY,
                status VARCHAR,
                source VARCHAR,
                last_ts TIMESTAMP,
                rows BIGINT,
                start_dt DATE,
                end_dt DATE,
                error VARCHAR
            );
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS fund_runs (
                ticker VARCHAR,
                stmt VARCHAR,
                status VARCHAR,
                last_ts TIMESTAMP,
                rows BIGINT,
                p_min VARCHAR,
                p_max VARCHAR,
                error VARCHAR,
                PRIMARY KEY (ticker, stmt)
            );
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS gaps (
                ticker VARCHAR,
                start DATE,
                end DATE,
                ts TIMESTAMP
            );
        """)

    def seed(self, tickers: List[str]):
        """Add new tickers to tracking table."""
        self.con.execute("""
            INSERT INTO ohlcv_runs (ticker, status)
            SELECT t, 'pending'
            FROM (SELECT UNNEST(?) AS t)
            WHERE t NOT IN (SELECT ticker FROM ohlcv_runs)
        """, [tickers])

    def update_ohlcv(self, ticker: str, status: str, source: Optional[str] = None,
                     rows: Optional[int] = None, start: Optional[str] = None,
                     end: Optional[str] = None, error: Optional[str] = None):
        """Update OHLCV run status."""
        self.con.execute("""
            UPDATE ohlcv_runs
            SET status = ?,
                source = COALESCE(?, source),
                last_ts = CURRENT_TIMESTAMP,
                rows = COALESCE(?, rows),
                start_dt = COALESCE(?, start_dt),
                end_dt = COALESCE(?, end_dt),
                error = ?
            WHERE ticker = ?
        """, [status, source, rows, start, end, error, ticker])

    def update_fund(self, ticker: str, stmt: str, status: str,
                    rows: Optional[int] = None, p_min: Optional[str] = None,
                    p_max: Optional[str] = None, error: Optional[str] = None):
        """Update fundamentals run status."""
        self.con.execute("""
            INSERT INTO fund_runs VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
            ON CONFLICT (ticker, stmt) DO UPDATE SET
                status = excluded.status,
                last_ts = CURRENT_TIMESTAMP,
                rows = COALESCE(excluded.rows, fund_runs.rows),
                p_min = COALESCE(excluded.p_min, fund_runs.p_min),
                p_max = COALESCE(excluded.p_max, fund_runs.p_max),
                error = excluded.error
        """, [ticker, stmt, status, rows, p_min, p_max, error])

    def get_pending(self) -> List[str]:
        """Get pending OHLCV tickers."""
        return [r[0] for r in self.con.execute(
            "SELECT ticker FROM ohlcv_runs WHERE status IN ('pending', 'failed') ORDER BY ticker"
        ).fetchall()]

    def get_success(self) -> List[str]:
        """Get successfully completed OHLCV tickers."""
        return [r[0] for r in self.con.execute(
            "SELECT ticker FROM ohlcv_runs WHERE status = 'success'"
        ).fetchall()]

    def fund_exists(self, ticker: str, stmt: str) -> bool:
        """Check if fundamentals already retrieved successfully."""
        result = self.con.execute(
            "SELECT 1 FROM fund_runs WHERE ticker = ? AND stmt = ? AND status = 'success'",
            [ticker, stmt]
        ).fetchone()
        return result is not None

    def log_gap(self, ticker: str, start: str, end: str):
        """Log detected gap in data."""
        self.con.execute("INSERT INTO gaps VALUES (?, ?, ?, CURRENT_TIMESTAMP)", [ticker, start, end])

    def close(self):
        """Close database connection."""
        self.con.close()


# =============================================================================
# FETCHERS
# =============================================================================

class Fetcher:
    """Handles data retrieval from various sources."""
    
    @staticmethod
    def _clean_ohlcv(df: pd.DataFrame, ticker: str):
        """Clean and validate OHLCV data, return (df, gaps)."""
        if df.empty:
            raise ValueError("Empty data returned")
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Find and normalize date column
        date_col = next(
            (c for c in df.columns if str(c).lower() in ['date', 'index', 'begins_at']),
            None
        )
        if not date_col:
            raise ValueError(f"No date column found in {list(df.columns)}")
        
        df = df.rename(columns={date_col: 'Date'})
        df['Date'] = pd.to_datetime(df['Date'])
        df['ticker'] = ticker
        
        # Deduplicate and sort
        df = df.sort_values('Date').drop_duplicates('Date').reset_index(drop=True)
        
        # Detect gaps (> 5 days between consecutive dates)
        gaps = []
        if len(df) > 1:
            deltas = df['Date'].diff().dt.days
            gap_indices = deltas[deltas > 5].index
            for i in gap_indices:
                start = df.loc[i-1, 'Date'].strftime('%Y-%m-%d')
                end = df.loc[i, 'Date'].strftime('%Y-%m-%d')
                gaps.append((start, end))
        
        return df, gaps

    @staticmethod
    def get_yfinance(ticker: str):
        """Fetch OHLCV from Yahoo Finance."""
        df = yf.download(
            ticker,
            start=Config.START_DATE,
            end=Config.END_DATE,
            interval="1d",
            progress=False,
            auto_adjust=False
        )
        return Fetcher._clean_ohlcv(df, ticker)

    @staticmethod
    def get_robinhood(ticker: str):
        """Fetch OHLCV from Robinhood."""
        if not ROBINHOOD_AVAILABLE:
            raise ImportError("robin_stocks not installed")
        
        hist = r.stocks.get_stock_historicals(ticker, interval="day", span="5year")
        if not hist:
            raise ValueError("No data from Robinhood")
        
        df = pd.DataFrame(hist).rename(columns={
            "begins_at": "Date",
            "open_price": "Open",
            "close_price": "Close",
            "high_price": "High",
            "low_price": "Low",
            "volume": "Volume"
        })
        
        return Fetcher._clean_ohlcv(df, ticker)

    @staticmethod
    def get_fundamentals(ticker: str, stmt: str):
        """Fetch fundamentals from Alpha Vantage."""
        resp = requests.get(
            Config.AV_URL,
            params={"function": stmt, "symbol": ticker, "apikey": Config.AV_KEY},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Check for API limits or errors
        if "Note" in data or "Information" in data:
            msg = data.get("Note") or data.get("Information")
            raise RuntimeError(f"AlphaVantage error: {msg}")
        
        # Parse reports
        records = []
        for report_type in ['annualReports', 'quarterlyReports']:
            for item in data.get(report_type, []):
                item.update({
                    'ticker': ticker,
                    'period_type': 'annual' if 'annual' in report_type else 'quarterly'
                })
                records.append(item)
        
        if not records:
            raise ValueError("No fundamentals data in response")
        
        return pd.DataFrame(records)


# =============================================================================
# PIPELINE
# =============================================================================

class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, tickers: Optional[List[str]] = None, limit: Optional[int] = None):
        self.db = StateManager()
        self.limit = limit
        self.rh_active = False
        
        # Determine which tickers to process
        if tickers:
            self.targets = tickers
            logging.info(f"Using {len(tickers)} tickers from command line")
        else:
            if not Config.UNIVERSE_PATH.exists():
                logging.error(f"Universe file not found: {Config.UNIVERSE_PATH}")
                sys.exit(1)
            
            universe = pd.read_parquet(Config.UNIVERSE_PATH)
            self.targets = universe["ticker"].dropna().unique().tolist()
            logging.info(f"Loaded {len(self.targets)} tickers from universe")
        
        # Apply test limit if specified
        if self.limit:
            self.targets = self.targets[:self.limit]
            logging.info(f"TEST MODE: Limited to first {self.limit} tickers")
        
        # Seed database
        self.db.seed(self.targets)
        
        # Login to Robinhood if configured
        if Config.USE_ROBINHOOD_LOGIN and ROBINHOOD_AVAILABLE:
            try:
                r.login(Config.ROBIN_USER, Config.ROBIN_PASS)
                self.rh_active = True
                logging.info("Logged into Robinhood")
            except Exception as e:
                logging.warning(f"Robinhood login failed: {e}")

    @retry_with_backoff(retries=Config.MAX_RETRIES, backoff_base=Config.BACKOFF_BASE)
    def _fetch_ohlcv_safe(self, ticker: str):
        """Fetch OHLCV with fallback logic."""
        try:
            df, gaps = Fetcher.get_yfinance(ticker)
            return "yfinance", df, gaps
        except Exception as e:
            # Try fallback if configured
            if Config.PRICE_FALLBACK == "robinhood" and self.rh_active:
                logging.info(f"{ticker}: Falling back to Robinhood")
                df, gaps = Fetcher.get_robinhood(ticker)
                return "robinhood", df, gaps
            raise e

    def process_ohlcv(self, ticker: str) -> bool:
        """Process single ticker for OHLCV data."""
        time.sleep(Config.PER_REQUEST_SLEEP)  # Rate limiting
        self.db.update_ohlcv(ticker=ticker, status="running")
        
        try:
            source, df, gaps = self._fetch_ohlcv_safe(ticker)
            
            # Log gaps
            for gap_start, gap_end in gaps:
                self.db.log_gap(ticker, gap_start, gap_end)
                logging.warning(f"{ticker}: Gap detected {gap_start} to {gap_end}")
            
            # Save to parquet
            output_path = Config.BASE_DIR / "ohlcv_daily" / f"{ticker}.parquet"
            atomic_write(df, output_path)
            
            # Mark success
            self.db.update_ohlcv(
                ticker=ticker,
                status="success",
                source=source,
                rows=len(df),
                start=df.Date.min().strftime('%Y-%m-%d'),
                end=df.Date.max().strftime('%Y-%m-%d'),
                error=None
            )
            
            logging.info(f"{ticker}: Success ({source}, {len(df)} rows)")
            return True
            
        except Exception as e:
            self.db.update_ohlcv(ticker=ticker, status="failed", error=str(e))
            logging.error(f"{ticker}: Failed - {str(e)[:100]}")
            return False

    def process_fund(self, ticker: str, stmt: str) -> bool:
        """Process fundamentals for one ticker/statement."""
        # Check if already done (BEFORE retry logic)
        if self.db.fund_exists(ticker, stmt):
            return False
        
        try:
            return self._fetch_fund_with_retry(ticker, stmt)
        except Exception as e:
            self.db.update_fund(ticker=ticker, stmt=stmt, status="failed", error=str(e))
            logging.error(f"{ticker} {stmt}: Failed - {str(e)[:100]}")
            return False

    @retry_with_backoff(retries=Config.MAX_RETRIES, backoff_base=Config.BACKOFF_BASE)
    def _fetch_fund_with_retry(self, ticker: str, stmt: str) -> bool:
        """Fetch fundamentals with retry logic."""
        time.sleep(Config.AV_SLEEP)  # Rate limiting
        
        df = Fetcher.get_fundamentals(ticker, stmt)
        
        # Save to parquet
        output_path = Config.BASE_DIR / "fundamentals" / stmt.lower() / f"{ticker}.parquet"
        atomic_write(df, output_path)
        
        # Extract period range
        p_min = df['fiscalDateEnding'].min() if 'fiscalDateEnding' in df.columns else None
        p_max = df['fiscalDateEnding'].max() if 'fiscalDateEnding' in df.columns else None
        
        # Mark success
        self.db.update_fund(
            ticker=ticker,
            stmt=stmt,
            status="success",
            rows=len(df),
            p_min=p_min,
            p_max=p_max,
            error=None
        )
        
        logging.info(f"{ticker} {stmt}: Success ({len(df)} rows)")
        return True

    def _run_batch(self, items: List, process_func: Callable, batch_sleep: float = 0):
        """Generic batch processor with rate limiting."""
        total = len(items)
        success_count = 0
        
        for i, item in enumerate(items, 1):
            if process_func(item):
                success_count += 1
            
            # Batch-level sleep
            if batch_sleep and i % Config.BATCH_SIZE == 0 and i < total:
                logging.info(f"Batch {i // Config.BATCH_SIZE} complete: {success_count}/{i} successful. Sleeping {batch_sleep:.1f}s...")
                time.sleep(batch_sleep)
        
        return success_count

    def run(self):
        """Run complete pipeline."""
        setup_logging()
        logging.info("=" * 80)
        logging.info(f"Pipeline Starting")
        logging.info(f"Tickers to process: {len(self.targets)}")
        logging.info("=" * 80)
        
        try:
            # 1. OHLCV Backfill
            pending = self.db.get_pending()
            if pending:
                logging.info(f"Starting OHLCV backfill: {len(pending)} tickers")
                success = self._run_batch(pending, self.process_ohlcv, Config.batch_sleep())
                logging.info(f"OHLCV backfill complete: {success}/{len(pending)} successful")
            else:
                logging.info("No pending OHLCV tickers")
            
            # 2. Fundamentals Backfill
            if Config.ENABLE_FUNDAMENTALS and Config.AV_KEY:
                success_tickers = self.db.get_success()
                logging.info(f"Starting fundamentals backfill: {len(success_tickers)} tickers")
                logging.warning(f"Alpha Vantage FREE TIER: {Config.AV_CAP} calls/day")
                logging.warning(f"With {len(Config.AV_STATEMENTS)} statements/ticker, ~{Config.AV_CAP // len(Config.AV_STATEMENTS)} tickers/day max")
                
                calls_made = 0
                tickers_completed = 0
                
                for ticker in success_tickers:
                    if calls_made >= Config.AV_CAP:
                        logging.warning("=" * 80)
                        logging.warning(f"ALPHA VANTAGE DAILY LIMIT REACHED: {calls_made}/{Config.AV_CAP}")
                        logging.warning(f"Completed {tickers_completed} tickers successfully")
                        logging.warning(f"Run again tomorrow to continue")
                        logging.warning("=" * 80)
                        break
                    
                    ticker_success = True
                    for stmt in Config.AV_STATEMENTS:
                        if self.process_fund(ticker, stmt):
                            calls_made += 1
                        else:
                            ticker_success = False
                    
                    if ticker_success:
                        tickers_completed += 1
                        logging.info(f"✓ {ticker}: All statements complete ({tickers_completed} tickers done)")
                
                logging.info(f"Fundamentals backfill complete: {calls_made} API calls, {tickers_completed} tickers")
            elif not Config.ENABLE_FUNDAMENTALS:
                logging.info("Fundamentals disabled, skipping")
            else:
                logging.warning("Alpha Vantage API key not set, skipping fundamentals")
            
            logging.info("=" * 80)
            logging.info("Pipeline completed successfully")
            logging.info("=" * 80)
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            self.db.close()


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Financial data pipeline for OHLCV and fundamentals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full universe backfill
  python tick.py
  
  # Test with first 10 tickers
  python tick.py --test 10
  
  # Specific tickers only
  python tick.py --tickers AAPL MSFT GOOGL
  
  # OHLCV only (no fundamentals)
  python tick.py --no-fundamentals
  
  # Batch processing (5000 tickers)
  python tick.py --test 5000 --no-fundamentals
        """
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific ticker symbols to process'
    )
    
    parser.add_argument(
        '--test',
        type=int,
        metavar='N',
        help='Test mode: process only first N tickers'
    )
    
    parser.add_argument(
        '--no-fundamentals',
        action='store_true',
        help='Skip fundamentals retrieval (OHLCV only)'
    )
    
    parser.add_argument(
        '--universe',
        default=str(Config.UNIVERSE_PATH),
        help='Path to universe parquet file'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Validate arguments
    if args.tickers and args.test:
        print("Error: Cannot use both --tickers and --test. Choose one.")
        sys.exit(1)
    
    # Update config based on arguments
    if args.no_fundamentals:
        Config.ENABLE_FUNDAMENTALS = False
    
    Config.UNIVERSE_PATH = Path(args.universe)
    
    # Set Robinhood login flag
    Config.USE_ROBINHOOD_LOGIN = bool(Config.ROBIN_USER and Config.ROBIN_PASS)
    
    # Convert tickers to uppercase if provided
    ticker_override = [t.upper() for t in args.tickers] if args.tickers else None
    
    # Create and run pipeline
    Pipeline(tickers=ticker_override, limit=args.test).run()