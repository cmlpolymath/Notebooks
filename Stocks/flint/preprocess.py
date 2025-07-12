# preprocess.py
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse

# Import project modules
import config
import data_handler
from feature_engineering import FeatureCalculator
import models

def prepare_and_cache_data(ticker: str, start_date: str, end_date: str, force_reprocess: bool = False):
    """
    Orchestrates the entire data preprocessing pipeline.
    1. Checks for a cached version of the processed data.
    2. If not found or forced, it generates the data:
       - Loads raw ticker, market index, sector, and macro data.
       - Calculates all features.
       - Defines the target variable.
       - Splits into training and testing sets.
    3. Saves the processed data to a cache file for future runs.
    4. Returns a dictionary containing all processed data objects.
    """
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the same sanitized filename as in data_handler
    safe_ticker = data_handler.sanitize_ticker_for_filename(ticker)
    cache_file = processed_dir / f"{safe_ticker}_data.pt"

    if not force_reprocess and cache_file.exists():
        print(f"Loading cached processed data for {ticker} from '{cache_file}'")
        return torch.load(cache_file, weights_only=False)

    print(f"Processing and caching data for {ticker}...")

    # Step 1: Load raw data
    df_raw = data_handler.get_stock_data(ticker=ticker, start_date=start_date, end_date=end_date)
    if df_raw is None or len(df_raw) < 350:
        raise ValueError(f"Insufficient raw data for {ticker} (need ~350 days).")
    
    market_df = data_handler.get_stock_data(ticker=config.MARKET_INDEX_TICKER, start_date=start_date, end_date=end_date)
    if market_df is None:
        raise ValueError(f"Could not load market index data for {config.MARKET_INDEX_TICKER}.")

    # --- MODIFIED: Load Sector Data only for non-crypto assets ---
    sector_df = None
    # Simple heuristic to detect crypto tickers
    is_crypto = '-USD' in ticker.upper() or '-USDT' in ticker.upper()

    if not is_crypto:
        try:
            info = data_handler.get_ticker_info(ticker)
            sector = info.get('sector')
            if sector and sector in config.SECTOR_ETF_MAP:
                sector_ticker = config.SECTOR_ETF_MAP[sector]
                print(f"Stock in '{sector}' sector. Fetching data for ETF: {sector_ticker}")
                sector_df = data_handler.get_stock_data(ticker=sector_ticker, start_date=start_date, end_date=end_date)
            else:
                print(f"Warning: Could not determine sector for stock {ticker}. Sector: {sector}")
        except Exception as e:
            print(f"Warning: Failed to load sector data for {ticker}. Error: {e}")
    else:
        print("Crypto asset detected. Skipping sector-specific feature engineering.")

    # Load Macro Data (applies to both stocks and crypto)
    macro_dfs = {}
    print("Fetching macroeconomic data...")
    for name, macro_ticker in config.MACRO_TICKERS.items():
        macro_dfs[name] = data_handler.get_stock_data(ticker=macro_ticker, start_date=start_date, end_date=end_date)

    # Step 2: Calculate all features
    feature_calculator = FeatureCalculator(df_raw.copy())
    df_features = feature_calculator.add_all_features(
        market_df=market_df.copy() if market_df is not None else None,
        sector_df=sector_df.copy() if sector_df is not None else None,
        macro_dfs=macro_dfs
    )
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.dropna(inplace=True)
    
    # Step 3: Define target and finalize DataFrame
    df_model = df_features.copy()
    future_price = df_model['Close'].shift(-5)
    future_ma = df_model['Close'].rolling(20).mean().shift(-5)
    df_model['UpNext'] = (future_price > future_ma).astype(int)
    df_model.dropna(inplace=True)

    # Step 4: Split data
    train_size = int(len(df_model) * config.TRAIN_SPLIT_RATIO)
    train_df = df_model.iloc[:train_size]
    test_df = df_model.iloc[train_size:]

    # Ensure only available columns are used
    feature_cols = [col for col in config.FEATURE_COLS if col in df_model.columns]
    X_train, y_train = train_df[feature_cols], train_df['UpNext']
    X_test, y_test_orig = test_df[feature_cols], test_df['UpNext']

    # Step 5: Prepare sequences for Transformer
    X_seq_train, y_seq_train = models.prepare_sequences(X_train.values, y_train.values, config.SEQUENCE_WINDOW_SIZE)
    X_seq_test, y_seq_test = models.prepare_sequences(X_test.values, y_test_orig.values, config.SEQUENCE_WINDOW_SIZE)
    
    # Step 6: Package and cache
    data_package = {
        'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test_orig': y_test_orig,
        'feature_cols': feature_cols, 'X_seq_train': X_seq_train, 'y_seq_train': y_seq_train,
        'X_seq_test': X_seq_test, 'y_seq_test': y_seq_test, 'df_features': df_features, 'test_df': test_df
    }
    
    torch.save(data_package, cache_file)
    print(f"Successfully cached processed data to '{cache_file}'")
    
    return data_package

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess and cache stock/crypto data.")
    parser.add_argument('tickers', metavar='TICKER', type=str, nargs='+',
                        help='A space-separated list of tickers to preprocess.')
    args = parser.parse_args()
    
    for ticker in args.tickers:
        try:
            print(f"\n--- Preprocessing {ticker} ---")
            prepare_and_cache_data(
                ticker=ticker, start_date=config.START_DATE, end_date=config.END_DATE, force_reprocess=True
            )
        except Exception as e:
            print(f"!!! FAILED to preprocess {ticker}: {e}")
            continue