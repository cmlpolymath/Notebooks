#!/usr/bin/env python3
"""
Production-grade financial data pipeline for OHLCV and fundamentals data.
No-cost solution using yfinance, robin_stocks, and Alpha Vantage.

USAGE:
    # Use full universe from parquet
    python tick.py
    
    # Test with subset (first N tickers)
    python tick.py --test 10
    
    # Specific tickers only
    python tick.py --tickers AAPL MSFT GOOGL TSLA
    
    # Skip fundamentals
    python tick.py --no-fundamentals
    
    # Combine options
    python tick.py --tickers AAPL MSFT --no-fundamentals
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import sys
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import pandas as pd
import duckdb
import yfinance as yf
import requests

# Optional: robin_stocks (install: pip install robin-stocks)
try:
    import robin_stocks.robinhood as r
    ROBINHOOD_AVAILABLE = True
except ImportError:
    ROBINHOOD_AVAILABLE = False
    print("Warning: robin_stocks not available. Install with: uv add robin-stocks")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    UNIVERSE_PARQUET = "data_drops/universe_common.parquet"
    OUT_OHLCV_DIR = "data_drops/ohlcv_daily"
    OUT_FUND_DIR = "data_drops/fundamentals"
    DB_PATH = "data_drops/pipeline_state.duckdb"
    LOG_PATH = "logs/pipeline.log"

    # Try a few likely locations (choose what you want)
    candidates = [
        Path.cwd() / ".env",                      # repo root if running from there
        Path("/workspaces") / ".env",             # less common
        Path.home() / ".config/myapp/.env",       # private location
    ]

    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)
            break
    
    # Date range
    START_DATE = (datetime.now() - timedelta(days=365*10)).strftime("%Y-%m-%d")
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    # Rate limiting (yfinance scraping safe defaults)
    TARGET_TPM = 18  # tickers per minute
    PER_REQUEST_SLEEP_SEC = 0.40
    BATCH_SIZE = 50
    MAX_RETRIES = 3
    BACKOFF_BASE_SEC = 2.0
    
    # Data sources
    PRICE_PRIMARY = "yfinance"
    PRICE_FALLBACK = "robinhood"
    ENABLE_FUNDAMENTALS = True
    
    # Alpha Vantage
    ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
    AV_BASE_URL = "https://www.alphavantage.co/query"
    AV_CALL_SLEEP_SEC = 12.5  # ~5 calls/min
    AV_DAILY_CAP = 500
    AV_STATEMENTS = ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"]
    
    # Robinhood credentials
    ROBIN_USER = os.getenv("ROBIN_USER", "")
    ROBIN_PASS = os.getenv("ROBIN_PASS", "")
    USE_ROBINHOOD_LOGIN = False  # Set to True if using Robinhood
    
    @classmethod
    def calc_batch_sleep(cls) -> float:
        """Calculate sleep time between batches to maintain target TPM."""
        per_batch_time = cls.BATCH_SIZE * cls.PER_REQUEST_SLEEP_SEC
        target_batch_time = (cls.BATCH_SIZE / cls.TARGET_TPM) * 60
        return max(0, target_batch_time - per_batch_time)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_path: str):
    """Configure logging to file and console."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )

# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

class StateManager:
    """Manages pipeline state using DuckDB."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Ensure directory exists before connecting
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.con = duckdb.connect(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Create state tracking tables."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_runs (
                ticker VARCHAR PRIMARY KEY,
                status VARCHAR,
                source_used VARCHAR,
                last_attempt_ts TIMESTAMP,
                rows BIGINT,
                start_dt DATE,
                end_dt DATE,
                error VARCHAR
            );
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals_runs (
                ticker VARCHAR,
                statement_type VARCHAR,
                status VARCHAR,
                last_attempt_ts TIMESTAMP,
                rows BIGINT,
                period_min VARCHAR,
                period_max VARCHAR,
                error VARCHAR,
                PRIMARY KEY (ticker, statement_type)
            );
        """)
        
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS gaps (
                ticker VARCHAR,
                gap_start DATE,
                gap_end DATE,
                created_ts TIMESTAMP
            );
        """)
    
    def seed_tickers(self, tickers: List[str]):
        """Add new tickers to ohlcv_runs table."""
        self.con.execute("""
            INSERT INTO ohlcv_runs (ticker, status)
            SELECT t.ticker, 'pending'
            FROM (SELECT UNNEST(?) AS ticker) t
            LEFT JOIN ohlcv_runs r ON r.ticker = t.ticker
            WHERE r.ticker IS NULL;
        """, [tickers])
    
    def mark_ohlcv(self, ticker: str, status: str, source_used: Optional[str] = None,
                   rows: Optional[int] = None, start_dt: Optional[str] = None,
                   end_dt: Optional[str] = None, error: Optional[str] = None):
        """Update OHLCV run status."""
        self.con.execute("""
            UPDATE ohlcv_runs
            SET status = ?,
                source_used = COALESCE(?, source_used),
                last_attempt_ts = CURRENT_TIMESTAMP,
                rows = COALESCE(?, rows),
                start_dt = COALESCE(?, start_dt),
                end_dt = COALESCE(?, end_dt),
                error = ?
            WHERE ticker = ?;
        """, [status, source_used, rows, start_dt, end_dt, error, ticker])
    
    def mark_fundamentals(self, ticker: str, statement_type: str, status: str,
                         rows: Optional[int] = None, period_min: Optional[str] = None,
                         period_max: Optional[str] = None, error: Optional[str] = None):
        """Update fundamentals run status."""
        self.con.execute("""
            INSERT INTO fundamentals_runs 
            (ticker, statement_type, status, last_attempt_ts, rows, period_min, period_max, error)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
            ON CONFLICT (ticker, statement_type) DO UPDATE SET
                status = excluded.status,
                last_attempt_ts = excluded.last_attempt_ts,
                rows = COALESCE(excluded.rows, fundamentals_runs.rows),
                period_min = COALESCE(excluded.period_min, fundamentals_runs.period_min),
                period_max = COALESCE(excluded.period_max, fundamentals_runs.period_max),
                error = excluded.error;
        """, [ticker, statement_type, status, rows, period_min, period_max, error])
    
    def get_pending_ohlcv(self) -> List[str]:
        """Get list of pending OHLCV tickers."""
        result = self.con.execute("""
            SELECT ticker FROM ohlcv_runs
            WHERE status IN ('pending', 'failed')
            ORDER BY ticker;
        """).fetchall()
        return [row[0] for row in result]
    
    def get_successful_tickers(self) -> List[str]:
        """Get tickers with successful OHLCV downloads."""
        result = self.con.execute("""
            SELECT ticker FROM ohlcv_runs WHERE status = 'success'
        """).fetchall()
        return [row[0] for row in result]
    
    def check_fundamentals_exists(self, ticker: str, statement_type: str) -> bool:
        """Check if fundamentals already successfully retrieved."""
        result = self.con.execute("""
            SELECT 1 FROM fundamentals_runs
            WHERE ticker = ? AND statement_type = ? AND status = 'success'
        """, [ticker, statement_type]).fetchone()
        return result is not None
    
    def log_gap(self, ticker: str, gap_start: str, gap_end: str):
        """Log a detected gap in data."""
        self.con.execute("""
            INSERT INTO gaps VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, [ticker, gap_start, gap_end])
    
    def close(self):
        """Close database connection."""
        self.con.close()

# =============================================================================
# DATA FETCHERS
# =============================================================================

class DataFetcher:
    """Handles data retrieval from various sources."""
    
    @staticmethod
    def fetch_ohlcv_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            df = yf.download(ticker, start=start_date, end=end_date, 
                           interval="1d", progress=False, auto_adjust=False)
            
            if df is None or df.empty:
                raise ValueError("Empty OHLCV from yfinance")
            
            # Flatten MultiIndex columns if present (happens with single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Reset index to convert DatetimeIndex to column
            df = df.reset_index()
            
            # Ensure the date column is named 'Date'
            if 'Date' not in df.columns:
                # Find the date column (usually first column or named 'index')
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]) or col.lower() in ['date', 'index']:
                        df = df.rename(columns={col: 'Date'})
                        break
            
            # Add ticker column
            df["ticker"] = ticker
            
            # Ensure Date is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Verify we have essential columns
            essential_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing = [col for col in essential_cols if col not in df.columns]
            if missing:
                logging.warning(f"{ticker}: Missing columns {missing}. Available: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logging.error(f"yfinance fetch failed for {ticker}: {e}")
            raise
    
    @staticmethod
    def fetch_ohlcv_robinhood(ticker: str) -> pd.DataFrame:
        """Fetch OHLCV data from Robinhood (requires login)."""
        if not ROBINHOOD_AVAILABLE:
            raise RuntimeError("robin_stocks not installed")
        
        try:
            # Fetch last 5 years
            hist = r.stocks.get_stock_historicals(
                ticker, interval="day", span="5year", bounds="regular"
            )
            
            if not hist:
                raise ValueError("Empty OHLCV from Robinhood")
            
            df = pd.DataFrame(hist)
            
            # Normalize column names
            df = df.rename(columns={
                "begins_at": "Date",
                "open_price": "Open",
                "high_price": "High",
                "low_price": "Low",
                "close_price": "Close",
                "volume": "Volume"
            })
            
            df["Date"] = pd.to_datetime(df["Date"])
            df["ticker"] = ticker
            
            # Convert price columns to float
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            if "Volume" in df.columns:
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
            
            return df
        except Exception as e:
            logging.error(f"Robinhood fetch failed for {ticker}: {e}")
            raise
    
    @staticmethod
    def fetch_fundamentals_alphavantage(ticker: str, statement_type: str, 
                                       api_key: str) -> pd.DataFrame:
        """Fetch fundamentals from Alpha Vantage."""
        try:
            params = {
                "function": statement_type,
                "symbol": ticker,
                "apikey": api_key
            }
            
            response = requests.get(Config.AV_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            
            # Check for throttling or error messages
            if "Note" in payload or "Information" in payload:
                msg = payload.get("Note") or payload.get("Information")
                raise RuntimeError(f"AlphaVantage throttled/error: {msg}")
            
            # Parse annual and quarterly reports
            annual = payload.get("annualReports", [])
            quarterly = payload.get("quarterlyReports", [])
            
            if not annual and not quarterly:
                raise ValueError("No fundamentals data in response")
            
            records = []
            for report in annual:
                report["period_type"] = "annual"
                report["ticker"] = ticker
                records.append(report)
            
            for report in quarterly:
                report["period_type"] = "quarterly"
                report["ticker"] = ticker
                records.append(report)
            
            df = pd.DataFrame(records)
            return df
            
        except Exception as e:
            logging.error(f"AlphaVantage fetch failed for {ticker} {statement_type}: {e}")
            raise

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dirs(*paths):
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def atomic_write_parquet(df: pd.DataFrame, path: str):
    """Write parquet atomically using temp file."""
    ensure_dirs(os.path.dirname(path))
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)

def validate_and_clean_ohlcv(df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Validate and clean OHLCV data, return cleaned df and list of gaps."""
    
    # Ensure Date column exists
    if "Date" not in df.columns:
        # Try to find date column
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if not date_cols:
            raise ValueError(f"No Date column found. Available columns: {list(df.columns)}")
        df = df.rename(columns={date_cols[0]: 'Date'})
    
    # Ensure Date is datetime type
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Sort and deduplicate
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    
    # Detect gaps (simple version - consecutive dates)
    gaps = []
    
    for i in range(1, len(df)):
        prev_date = df.loc[i-1, "Date"]
        curr_date = df.loc[i, "Date"]
        gap_days = (curr_date - prev_date).days
        
        # If gap > 5 days (accounting for weekends), log it
        if gap_days > 5:
            gaps.append((prev_date.strftime("%Y-%m-%d"), curr_date.strftime("%Y-%m-%d")))
    
    return df, gaps

def chunk_list(lst: List, size: int):
    """Split list into chunks."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def should_use_fallback(attempt: int, error_msg: str) -> bool:
    """Determine if fallback source should be used."""
    return attempt >= 2

# =============================================================================
# MAIN PIPELINE
# =============================================================================

class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: Config, ticker_override: Optional[List[str]] = None,
                 test_limit: Optional[int] = None):
        self.config = config
        self.state = StateManager(config.DB_PATH)
        self.fetcher = DataFetcher()
        self.robinhood_logged_in = False
        self.ticker_override = ticker_override
        self.test_limit = test_limit
    
    def initialize(self):
        """Initialize pipeline: directories, logging, universe."""
        ensure_dirs(
            self.config.OUT_OHLCV_DIR,
            self.config.OUT_FUND_DIR,
            os.path.join(self.config.OUT_FUND_DIR, "income_statement"),
            os.path.join(self.config.OUT_FUND_DIR, "balance_sheet"),
            os.path.join(self.config.OUT_FUND_DIR, "cash_flow"),
            "logs",
            "data"
        )
        
        setup_logging(self.config.LOG_PATH)
        logging.info("=" * 80)
        logging.info("Pipeline Starting")
        logging.info("=" * 80)
        
        # Determine tickers to process
        if self.ticker_override:
            # Use command-line provided tickers
            tickers = self.ticker_override
            logging.info(f"Using {len(tickers)} tickers from command line: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        else:
            # Load from universe file
            if not os.path.exists(self.config.UNIVERSE_PARQUET):
                logging.error(f"Universe file not found: {self.config.UNIVERSE_PARQUET}")
                sys.exit(1)
            
            universe = pd.read_parquet(self.config.UNIVERSE_PARQUET)
            tickers = universe["ticker"].dropna().unique().tolist()
            
            # Apply test limit if specified
            if self.test_limit:
                tickers = tickers[:self.test_limit]
                logging.info(f"TEST MODE: Limited to first {self.test_limit} tickers")
            
            logging.info(f"Loaded {len(tickers)} tickers from universe")
        
        self.state.seed_tickers(tickers)
        
        # Login to Robinhood if configured
        if self.config.USE_ROBINHOOD_LOGIN and ROBINHOOD_AVAILABLE:
            try:
                r.login(username=self.config.ROBIN_USER, password=self.config.ROBIN_PASS)
                self.robinhood_logged_in = True
                logging.info("Logged into Robinhood")
            except Exception as e:
                logging.warning(f"Robinhood login failed: {e}")
    
    def process_ohlcv_ticker(self, ticker: str) -> bool:
        """Process single ticker for OHLCV data. Returns True on success."""
        self.state.mark_ohlcv(ticker, "running", error=None)
        
        attempt = 0
        last_error = None
        source_used = None
        
        while attempt < self.config.MAX_RETRIES:
            try:
                time.sleep(self.config.PER_REQUEST_SLEEP_SEC)
                
                # Try primary source
                if self.config.PRICE_PRIMARY == "yfinance":
                    source_used = "yfinance"
                    df = self.fetcher.fetch_ohlcv_yfinance(
                        ticker, self.config.START_DATE, self.config.END_DATE
                    )
                else:
                    source_used = "robinhood"
                    df = self.fetcher.fetch_ohlcv_robinhood(ticker)
                
                # Validate and clean
                df_clean, gaps = validate_and_clean_ohlcv(df, ticker)
                
                # Log gaps
                for gap_start, gap_end in gaps:
                    self.state.log_gap(ticker, gap_start, gap_end)
                    logging.warning(f"{ticker}: Gap detected {gap_start} to {gap_end}")
                
                # Save to parquet
                out_path = os.path.join(self.config.OUT_OHLCV_DIR, f"{ticker}.parquet")
                atomic_write_parquet(df_clean, out_path)
                
                # Mark success
                self.state.mark_ohlcv(
                    ticker, "success",
                    source_used=source_used,
                    rows=len(df_clean),
                    start_dt=df_clean["Date"].min().strftime("%Y-%m-%d"),
                    end_dt=df_clean["Date"].max().strftime("%Y-%m-%d"),
                    error=None
                )
                
                logging.info(f"{ticker}: Success ({source_used}, {len(df_clean)} rows)")
                return True
                
            except Exception as e:
                attempt += 1
                last_error = str(e)
                
                # Try fallback if configured
                if (self.config.PRICE_PRIMARY == "yfinance" and 
                    self.config.PRICE_FALLBACK == "robinhood" and 
                    should_use_fallback(attempt, last_error) and
                    self.robinhood_logged_in):
                    
                    try:
                        logging.info(f"{ticker}: Trying fallback (Robinhood)")
                        source_used = "robinhood"
                        df = self.fetcher.fetch_ohlcv_robinhood(ticker)
                        df_clean, gaps = validate_and_clean_ohlcv(df, ticker)
                        
                        out_path = os.path.join(self.config.OUT_OHLCV_DIR, f"{ticker}.parquet")
                        atomic_write_parquet(df_clean, out_path)
                        
                        self.state.mark_ohlcv(
                            ticker, "success",
                            source_used=source_used,
                            rows=len(df_clean),
                            start_dt=df_clean["Date"].min().strftime("%Y-%m-%d"),
                            end_dt=df_clean["Date"].max().strftime("%Y-%m-%d"),
                            error=None
                        )
                        
                        logging.info(f"{ticker}: Success via fallback ({len(df_clean)} rows)")
                        return True
                        
                    except Exception as e2:
                        last_error = f"Primary: {e}; Fallback: {e2}"
                
                # Exponential backoff
                if attempt < self.config.MAX_RETRIES:
                    backoff = self.config.BACKOFF_BASE_SEC ** attempt
                    logging.warning(f"{ticker}: Attempt {attempt} failed: {last_error[:100]}. Backoff {backoff}s")
                    time.sleep(backoff)
        
        # All retries exhausted
        self.state.mark_ohlcv(ticker, "failed", source_used=source_used, error=last_error)
        logging.error(f"{ticker}: Failed after {self.config.MAX_RETRIES} attempts")
        return False
    
    def run_ohlcv_backfill(self):
        """Run OHLCV backfill for all pending tickers."""
        logging.info("Starting OHLCV backfill")
        
        pending = self.state.get_pending_ohlcv()
        logging.info(f"Found {len(pending)} pending tickers")
        
        batch_sleep = self.config.calc_batch_sleep()
        
        for batch_num, batch in enumerate(chunk_list(pending, self.config.BATCH_SIZE), 1):
            logging.info(f"Processing batch {batch_num} ({len(batch)} tickers)")
            
            success_count = 0
            for ticker in batch:
                if self.process_ohlcv_ticker(ticker):
                    success_count += 1
            
            logging.info(f"Batch {batch_num} complete: {success_count}/{len(batch)} successful")
            
            if batch_num * self.config.BATCH_SIZE < len(pending):
                logging.info(f"Batch sleep {batch_sleep:.1f}s (target {self.config.TARGET_TPM} TPM)")
                time.sleep(batch_sleep)
        
        logging.info("OHLCV backfill complete")
    
    def process_fundamentals_ticker(self, ticker: str, statement_type: str,
                                   calls_made: int) -> Tuple[bool, int]:
        """Process fundamentals for one ticker/statement. Returns (success, calls_made)."""
        
        if calls_made >= self.config.AV_DAILY_CAP:
            logging.warning("Alpha Vantage daily cap reached")
            return False, calls_made
        
        if self.state.check_fundamentals_exists(ticker, statement_type):
            return True, calls_made
        
        self.state.mark_fundamentals(ticker, statement_type, "running", error=None)
        
        attempt = 0
        last_error = None
        
        while attempt < self.config.MAX_RETRIES:
            try:
                time.sleep(self.config.AV_CALL_SLEEP_SEC)
                
                df = self.fetcher.fetch_fundamentals_alphavantage(
                    ticker, statement_type, self.config.ALPHAVANTAGE_API_KEY
                )
                
                # Determine output directory
                stmt_dir = statement_type.lower()
                out_dir = os.path.join(self.config.OUT_FUND_DIR, stmt_dir)
                ensure_dirs(out_dir)
                
                out_path = os.path.join(out_dir, f"{ticker}.parquet")
                atomic_write_parquet(df, out_path)
                
                # Extract period range
                period_min = None
                period_max = None
                if "fiscalDateEnding" in df.columns:
                    period_min = str(df["fiscalDateEnding"].min())
                    period_max = str(df["fiscalDateEnding"].max())
                
                self.state.mark_fundamentals(
                    ticker, statement_type, "success",
                    rows=len(df),
                    period_min=period_min,
                    period_max=period_max,
                    error=None
                )
                
                logging.info(f"{ticker} {statement_type}: Success ({len(df)} rows)")
                return True, calls_made + 1
                
            except Exception as e:
                attempt += 1
                last_error = str(e)
                
                if attempt < self.config.MAX_RETRIES:
                    backoff = self.config.BACKOFF_BASE_SEC ** attempt
                    logging.warning(f"{ticker} {statement_type}: Attempt {attempt} failed. Backoff {backoff}s")
                    time.sleep(backoff)
        
        self.state.mark_fundamentals(ticker, statement_type, "failed", error=last_error)
        logging.error(f"{ticker} {statement_type}: Failed after {self.config.MAX_RETRIES} attempts")
        return False, calls_made
    
    def run_fundamentals_backfill(self):
        """Run fundamentals backfill for successful OHLCV tickers."""
        if not self.config.ENABLE_FUNDAMENTALS:
            logging.info("Fundamentals disabled, skipping")
            return
        
        if not self.config.ALPHAVANTAGE_API_KEY:
            logging.warning("Alpha Vantage API key not set, skipping fundamentals")
            return
        
        logging.info("Starting fundamentals backfill")
        
        tickers = self.state.get_successful_tickers()
        logging.info(f"Processing fundamentals for {len(tickers)} tickers")
        
        calls_made = 0
        
        for ticker in tickers:
            for statement_type in self.config.AV_STATEMENTS:
                success, calls_made = self.process_fundamentals_ticker(
                    ticker, statement_type, calls_made
                )
                
                if calls_made >= self.config.AV_DAILY_CAP:
                    logging.warning("Reached Alpha Vantage daily cap, stopping fundamentals")
                    return
        
        logging.info(f"Fundamentals backfill complete ({calls_made} API calls)")
    
    def run(self):
        """Run complete pipeline."""
        try:
            self.initialize()
            self.run_ohlcv_backfill()
            self.run_fundamentals_backfill()
            logging.info("=" * 80)
            logging.info("Pipeline completed successfully")
            logging.info("=" * 80)
        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            self.state.close()

# =============================================================================
# ENTRY POINT
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
  
  # Test with first 10 tickers from universe
  python tick.py --test 10
  
  # Process specific tickers only
  python tick.py --tickers AAPL MSFT GOOGL TSLA
  
  # Specific tickers without fundamentals
  python tick.py --tickers AAPL MSFT --no-fundamentals
  
  # Test mode without fundamentals (faster)
  python tick.py --test 5 --no-fundamentals
        """
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Specific ticker symbols to process (e.g., AAPL MSFT GOOGL)'
    )
    
    parser.add_argument(
        '--test',
        type=int,
        metavar='N',
        help='Test mode: process only first N tickers from universe file'
    )
    
    parser.add_argument(
        '--no-fundamentals',
        action='store_true',
        help='Skip fundamentals retrieval (OHLCV only)'
    )
    
    parser.add_argument(
        '--universe',
        default=Config.UNIVERSE_PARQUET,
        help=f'Path to universe parquet file (default: {Config.UNIVERSE_PARQUET})'
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
    
    if args.universe:
        Config.UNIVERSE_PARQUET = args.universe
    
    # Convert tickers to uppercase if provided
    ticker_override = [t.upper() for t in args.tickers] if args.tickers else None
    
    # Create and run pipeline
    pipeline = Pipeline(Config, ticker_override=ticker_override, test_limit=args.test)
    pipeline.run()