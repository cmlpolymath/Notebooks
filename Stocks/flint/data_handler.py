# FILE: data_handler.py

from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import yfinance as yf
import re
import asyncio
import structlog

from economics import fetch_macro_indicators
from config import settings

# Instantiate the logger. This will automatically use the config from settings.
# Using __name__ gives us the module context in the logs.
logger = structlog.get_logger(__name__)

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens multi-index columns into a single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def sanitize_ticker_for_filename(ticker: str) -> str:
    """Replaces special characters for safe filenames."""
    return re.sub(r'[^A-Z0-9_]', '_', ticker.upper())

def get_ticker_info(ticker: str) -> dict:
    """Fetches the .info dictionary for a given ticker."""
    try:
        tkr = yf.Ticker(ticker)
        return tkr.info
    except Exception as e:
        # Use the logger to report errors with context.
        logger.error("ticker_info_failed", ticker=ticker, error=str(e))
        return {}

def get_stock_data(ticker: str, start_date: str, end_date: str, force_redownload: bool = False) -> pd.DataFrame | None:
    """Fetch stock data intelligently with structured logging."""
    Path('data').mkdir(exist_ok=True)
    safe_ticker = sanitize_ticker_for_filename(ticker)
    parquet_path = Path(f'data/{safe_ticker}.parquet')
    
    log = logger.bind(ticker=ticker, path=str(parquet_path))

    if force_redownload and parquet_path.exists():
        log.warning("forcing_redownload", reason="User flag")
        parquet_path.unlink()

    df_existing = None
    fetch_start_date = start_date

    if parquet_path.exists():
        # This is a noisy, repetitive message, so we demote it to DEBUG.
        log.debug("cache_hit")
        df_existing = pd.read_parquet(parquet_path)
        last_stored_date = df_existing['Date'].max().date()
        
        if last_stored_date >= (datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=1)):
             # Also a noisy message, demoted to DEBUG.
             log.debug("cache_up_to_date")
             return df_existing
        
        fetch_start_date = (last_stored_date + timedelta(days=1)).strftime('%Y-%m-%d')
        # This is an important action, so it stays at INFO level.
        log.info("updating_cache", start=fetch_start_date, end=end_date)
    else:
        # This is also an important, one-time action.
        log.info("cache_miss", reason="No cache file found")

    df_new = yf.download(ticker, start=fetch_start_date, end=end_date, progress=False, auto_adjust=True)

    if df_new.empty:
        # This is good to know but not critical, so DEBUG is appropriate.
        log.debug("no_new_data_found", period_start=fetch_start_date, period_end=end_date)
        return df_existing
    
    df_new = clean_column_names(df_new)
    df_new.reset_index(inplace=True)

    if df_existing is not None:
        if not df_existing.columns.equals(df_new.columns):
            log.warning("schema_mismatch", reason="Cached and new data have different columns. Re-downloading.")
            return get_stock_data(ticker, start_date, end_date, force_redownload=True)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.drop_duplicates(subset=['Date'], keep='last', inplace=True)
    df_combined.sort_values('Date', inplace=True)
    df_combined.to_parquet(parquet_path, index=False)
    
    return df_combined

def get_macro_data(force_redownload: bool = False) -> pd.DataFrame | None:
    """
    Fetch macroeconomic data from FRED, with intelligent caching.
    1. Checks for a cached Parquet file.
    2. If the cache is recent (from today), it's used.
    3. Otherwise, it fetches fresh data from the FRED API and updates the cache.
    """
    Path('data').mkdir(exist_ok=True)
    parquet_path = Path('data/macro_indicators_daily.parquet')
    log = logger.bind(path=str(parquet_path))

    if not force_redownload and parquet_path.exists():
        last_mod_time = date.fromtimestamp(parquet_path.stat().st_mtime)
        if last_mod_time >= date.today():
            log.debug("macro_cache_hit")
            return pd.read_parquet(parquet_path)
        else:
            log.info("macro_cache_stale", reason="Data is from a previous day")
    else:
        log.info("macro_cache_miss", reason="No cache file found")

    try:
        log.info("fetching_fred_data")
        df_macro = asyncio.run(fetch_macro_indicators(settings.features.fred_indicators))
        df_macro.to_parquet(parquet_path)
        log.info("macro_cache_updated")
        return df_macro
    except Exception as e:
        log.error("macro_data_failed", error=str(e))
        if parquet_path.exists():
            log.warning("using_stale_cache", reason="FRED API fetch failed")
            return pd.read_parquet(parquet_path)
        return None