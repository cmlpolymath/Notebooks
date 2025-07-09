# data_handler.py
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens multi-index columns (e.g., from yfinance) into a single level.
    Example: ('Close', '') -> 'Close'
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_stock_data(ticker: str, start_date: str, end_date: str, force_redownload: bool = False) -> pd.DataFrame | None:
    """
    Fetch stock data intelligently for any ticker (e.g., 'AAPL', 'SPY').
    1. Reads existing data from Parquet.
    2. Fetches only the missing data since the last entry.
    3. Appends and saves back to Parquet.
    """
    Path('data').mkdir(exist_ok=True)
    parquet_path = Path(f'data/{ticker}.parquet')
    
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