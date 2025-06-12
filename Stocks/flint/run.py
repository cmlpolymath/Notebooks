# run.py
import initialize
import argparse
import re
import time
import numpy as np
import pandas as pd
# Import project modules
import config
import data_handler
import audit_logger
from feature_engineering import FeatureCalculator
import models
import predictors

def validate_ticker(ticker):
    """Sanitize and validate a stock ticker format."""
    ticker = ticker.strip().upper()
    if re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$', ticker):
        return ticker
    raise argparse.ArgumentTypeError(f"'{ticker}' is not a valid ticker format.")

def run_single_ticker_analysis(ticker: str):
    """Orchestrates a simplified analysis using XGBoost and standard Monte Carlo."""
    print(f"\n{'='*20} Processing Ticker: {ticker} {'='*20}")
    
    # 1. Load Data
    df_raw = data_handler.get_stock_data(
        ticker=ticker,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    if df_raw is None or len(df_raw) < 252:
        print(f"Insufficient data for {ticker}. Skipping.")
        return
    
    # 2. Feature Engineering
    feature_calculator = FeatureCalculator(df_raw.copy())
    df_features = feature_calculator.add_all_features()
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_features.dropna(inplace=True)
    
    # 3. Define Target and Split Data
    df_features['UpNext'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
    df_features.dropna(inplace=True)

    train_size = int(len(df_features) * config.TRAIN_SPLIT_RATIO)
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:]

    feature_cols = [col for col in config.FEATURE_COLS if col in df_features.columns]
    X_train, y_train = train_df[feature_cols], train_df['UpNext']
    X_test, y_test = test_df[feature_cols], test_df['UpNext']
    
    # 4. Train XGBoost Model
    xgb_model = models.train_xgboost(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    # 5. Generate Monte-Carlo Signal
    print("Generating final Monte-Carlo signal...")
    mc_final_signal = {}
    try:
        mc_filter = predictors.MonteCarloTrendFilter(**config.MC_FILTER_PARAMS)
        mc_filter.fit(df_features) 
        mc_final_signal = mc_filter.predict()
        print(f"MC Signal: Trend Strength={mc_final_signal.get('trend_strength', 0):.3f}")
    except Exception as e:
        print(f"Could not generate Monte-Carlo signal: {e}")

    # 6. Generate Metrics & Importance
    shap_values = models.get_shap_importance(xgb_model, X_test)
    
    predictions = (xgb_proba > 0.5).astype(int)
    accuracy = np.mean(predictions == y_test.values)
    kelly_fraction = 2 * xgb_proba[-1] - 1 if len(xgb_proba) > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "kelly_fraction": kelly_fraction,
        "mc_trend_strength": mc_final_signal.get('trend_strength'),
        "mc_up_prob": mc_final_signal.get('up_prob'),
        "mc_down_prob": mc_final_signal.get('down_prob'),
        "mc_ci": mc_final_signal.get('ci', [None, None]),
        "mc_simulated_slopes": mc_final_signal.get('simulated_slopes', [])         
    }
    
    # 7. Log Results for Auditing
    audit_logger.log_analysis_result(
        ticker=ticker,
        model_name="XGBoost_v1", # Simplified model name
        run_config=config.XGB_PARAMS,
        predictions={"probabilities": xgb_proba.tolist()},
        metrics=metrics,
        shap_importance={
            'features': X_test.columns.tolist(),
            'values': shap_values.tolist()
        }
    )
    print(f"Analysis for {ticker} complete. Results logged.")

def main():
    parser = argparse.ArgumentParser(description="Run the stock analysis pipeline.")
    parser.add_argument('tickers', metavar='TICKER', type=validate_ticker, nargs='+',
                        help='A space-separated list of stock tickers to analyze.')
    args = parser.parse_args()
    
    audit_logger.setup_audit_db()

    for ticker in args.tickers:
        try:
            run_single_ticker_analysis(ticker)
            time.sleep(1) 
        except Exception as e:
            print(f"!!! FAILED to process {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()