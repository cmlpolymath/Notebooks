# preprocess.py
from pathlib import Path
from datetime import date
import torch
import numpy as np
import argparse
import structlog

# Import project modules
from config import settings
import data_handler
from data_store import B2Data
from feature_engineering import FeatureCalculator
import models

logger = structlog.get_logger(__name__)

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
    data_store = B2Data()
    safe_ticker = data_handler.sanitize_ticker_for_filename(ticker)
    log = logger.bind(ticker=ticker)

    # --- FIX: Call the method on the INSTANCE (data_store) and pass the CORRECT variable (safe_ticker) ---
    if not force_reprocess and data_store.is_cache_valid(safe_ticker):
        log.info("cache_hit_valid", storage_format="structured_directory")
        return data_store.load_data_package(safe_ticker)
    
    if force_reprocess:
        log.warning("cache_ignored", reason="--force-reprocess flag used")
    else:
        log.info("cache_miss_or_stale", reason="No valid cache from today found")

    # --- Full data pipeline continues here ---
    log.info("data_pipeline_start", reason="Generating new data package")
    
    df_macro_fred = data_handler.get_macro_data(force_redownload=force_reprocess)
    macro_dfs_yf = {}
    for name, macro_ticker in settings.features.macro_tickers_yf.items():
        macro_dfs_yf[name] = data_handler.get_stock_data(
            ticker=macro_ticker, start_date=start_date, end_date=end_date, force_redownload=force_reprocess
        )

    df_raw = data_handler.get_stock_data(ticker=ticker, start_date=start_date, end_date=end_date)
    if df_raw is None or len(df_raw) < 350:
        raise ValueError(f"Insufficient raw data for {ticker} (need ~350 days).")
    
    market_df = data_handler.get_stock_data(ticker=settings.data.market_index_ticker, start_date=start_date, end_date=end_date)
    if market_df is None:
        raise ValueError(f"Could not load market index data for {settings.data.market_index_ticker}.")

    sector_df = None
    is_crypto = '-USD' in ticker.upper() or '-USDT' in ticker.upper()
    if not is_crypto:
        try:
            info = data_handler.get_ticker_info(ticker)
            sector = info.get('sector')
            if sector and sector in settings.features.sector_etf_map:
                sector_ticker = settings.features.sector_etf_map[sector]
                sector_df = data_handler.get_stock_data(ticker=sector_ticker, start_date=start_date, end_date=end_date)
        except Exception as e:
            log.warning("sector_data_load_failed", error=str(e))

    feature_calculator = FeatureCalculator(df_raw.copy())
    df_features = feature_calculator.add_all_features(
        market_df=market_df.copy() if market_df is not None else None,
        sector_df=sector_df.copy() if sector_df is not None else None,
        macro_dfs_yf=macro_dfs_yf,
        df_macro_fred=df_macro_fred
    )
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.dropna(inplace=True)

    min_required_rows = settings.models.sequence_window_size + 100
    if len(df_features) < min_required_rows:
        raise ValueError(
            f"Insufficient data for {ticker} after feature engineering. "
            f"Need at least {min_required_rows} rows, but only {len(df_features)} remain. "
            "This is often due to long lookback periods in features (e.g., 252 days for YoY inflation)."
        )    
    df_model = df_features.copy()
    future_price = df_model['Close'].shift(-5)
    future_ma = df_model['Close'].rolling(20).mean().shift(-5)
    df_model['UpNext'] = (future_price > future_ma).astype(int)
    df_model.dropna(inplace=True)

    train_size = int(len(df_model) * settings.models.train_split_ratio)
    train_df = df_model.iloc[:train_size]
    test_df = df_model.iloc[train_size:]

    feature_cols = [col for col in settings.features.get_all_feature_names() if col in df_model.columns]
    X_train, y_train = train_df[feature_cols], train_df['UpNext']
    X_test, y_test_orig = test_df[feature_cols], test_df['UpNext']

    X_seq_train, y_seq_train = models.prepare_sequences(X_train.values, y_train.values, settings.models.sequence_window_size)
    X_seq_test, y_seq_test = models.prepare_sequences(X_test.values, y_test_orig.values, settings.models.sequence_window_size)
    
    data_package = {
        'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test_orig': y_test_orig,
        'feature_cols': feature_cols, 'X_seq_train': X_seq_train, 'y_seq_train': y_seq_train,
        'X_seq_test': X_seq_test, 'y_seq_test': y_seq_test, 'df_features': df_features, 'test_df': test_df
    }
    
    data_store.save_data_package(data_package, safe_ticker)
    
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
                ticker=ticker, start_date=settings.data.start_date, end_date=settings.data.end_date, force_reprocess=True
            )
        except Exception as e:
            print(f"!!! FAILED to preprocess {ticker}: {e}")
            continue