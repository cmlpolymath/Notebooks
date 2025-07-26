# data_handler.py
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import yfinance as yf
import re
import asyncio
from economics import fetch_macro_indicators
from config import settings

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens multi-index columns (e.g., from yfinance) into a single level.
    Example: ('Close', '') -> 'Close'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# Sanitize ticker for safe filenames
def sanitize_ticker_for_filename(ticker: str) -> str:
    """Replaces special characters in a ticker with underscores for safe filenames."""
    return re.sub(r'[^A-Z0-9_]', '_', ticker.upper())

# ADDED: Function to get fundamental info for a ticker
def get_ticker_info(ticker: str) -> dict:
    """
    Fetches the .info dictionary for a given ticker.
    A simple wrapper around yfinance.Ticker().info.
    """
    try:
        tkr = yf.Ticker(ticker)
        # .info can be slow; this is a basic example. Caching could be added.
        return tkr.info
    except Exception as e:
        print(f"Could not fetch info for {ticker}: {e}")
        return {}

def get_stock_data(ticker: str, start_date: str, end_date: str, force_redownload: bool = False) -> pd.DataFrame | None:
    """
    Fetch stock data intelligently for any ticker (e.g., 'AAPL', 'SPY').
    1. Reads existing data from Parquet.
    2. Fetches only the missing data since the last entry.
    3. Appends and saves back to Parquet.
    """
    Path('data').mkdir(exist_ok=True)

    # Use sanitized ticker for the filename
    safe_ticker = sanitize_ticker_for_filename(ticker)
    parquet_path = Path(f'data/{safe_ticker}.parquet')
    
    if force_redownload and parquet_path.exists():
        print(f"Forcing redownload for {ticker}. Deleting cache at '{parquet_path}'.")
        parquet_path.unlink()

    df_existing = None
    fetch_start_date = start_date

    if parquet_path.exists():
        print(f"Found cached data for {ticker} at '{parquet_path}'")
        df_existing = pd.read_parquet(parquet_path)
        last_stored_date = df_existing['Date'].max().date()
        
        if last_stored_date >= (datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=1)):
             print(f"Data for {ticker} is up-to-date. Loading from cache.")
             return df_existing
        
        fetch_start_date = (last_stored_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Updating data for {ticker} from {fetch_start_date} to {end_date}...")
    else:
        print(f"No cache found. Downloading full history for {ticker}...")

    # Fetch new data (either full or delta)
    df_new = yf.download(
        ticker,
        start=fetch_start_date,
        end=end_date,
        progress=False,
        auto_adjust=True
    )

    if df_new.empty:
        print(f"No new data found for {ticker} for the period.")
        return df_existing
    
    # CRITICAL FIX: Clean columns and reset index to ensure a standard format.
    df_new = clean_column_names(df_new)
    df_new.reset_index(inplace=True)

    # Combine existing and new data
    if df_existing is not None:
        # Ensure schema consistency before concatenating
        if not df_existing.columns.equals(df_new.columns):
            print("Warning: Schema mismatch between cached and new data. Re-downloading full history.")
            # Fallback to full download if schema changes
            return get_stock_data(ticker, start_date, end_date, force_redownload=True)
        
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Final validation and save
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

    if not force_redownload and parquet_path.exists():
        last_mod_time = date.fromtimestamp(parquet_path.stat().st_mtime)
        if last_mod_time >= date.today():
            print(f"Found recent macro data cache. Loading from '{parquet_path}'.")
            return pd.read_parquet(parquet_path)
        else:
            print("Macro data cache is outdated. Fetching fresh data...")
    else:
        print("No macro data cache found. Fetching from FRED API...")

    try:
        # Pass the dictionary to the function
        print("Running async FRED fetcher...")
        df_macro = asyncio.run(fetch_macro_indicators(settings.features.fred_indicators))
        
        # Save to cache for future runs
        df_macro.to_parquet(parquet_path)
        print(f"Successfully fetched and cached macro data to '{parquet_path}'.")
        return df_macro
    except Exception as e:
        print(f"!!! FAILED to fetch macroeconomic data: {e}")
        # If fetching fails but an old cache exists, use it as a fallback
        if parquet_path.exists():
            print(f"Using stale cache from '{parquet_path}' as a fallback.")
            return pd.read_parquet(parquet_path)
        return None