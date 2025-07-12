# run.py
import initialize
import time
import argparse
import re
import numpy as np
import pandas as pd
import torch

# Import project modules
import config
import audit_logger
import models
import predictors
from preprocess import prepare_and_cache_data

def validate_ticker(ticker):
    """Sanitize and validate a stock or crypto ticker format."""
    ticker = ticker.strip().upper()
    # MODIFIED: Updated regex to allow hyphens and be more flexible for crypto.
    if re.match(r'^[A-Z0-9\.\-]{1,12}$', ticker):
        return ticker
    raise argparse.ArgumentTypeError(f"'{ticker}' is not a valid ticker format.")

# In run.py

def main():
    parser = argparse.ArgumentParser(description="Run the stock/crypto analysis pipeline.")
    parser.add_argument('tickers', metavar='TICKER', type=validate_ticker, nargs='+',
                        help='A space-separated list of tickers to analyze.')
    parser.add_argument(
        '--model', 
        type=str, 
        default='ensemble', 
        choices=['ensemble', 'xgb', 'rf'],
        help="Specify the model to run: 'ensemble' (XGB+Transformer), 'xgb' (XGBoost only), or 'rf' (Random Forest only)."
    )
    args = parser.parse_args()
    
    audit_logger.setup_audit_db()

    for ticker in args.tickers:
        print(f"\n{'='*25} Processing Ticker: {ticker} | Model: {args.model.upper()} {'='*25}")
        try:
            # Step 1: Load Data
            data_package = prepare_and_cache_data(ticker, config.START_DATE, config.END_DATE)
            X_train, y_train = data_package['X_train'], data_package['y_train']
            X_test, y_test_orig = data_package['X_test'], data_package['y_test_orig']
            feature_cols, df_features = data_package['feature_cols'], data_package['df_features']

            # Step 2: Run Monte Carlo simulation (run once, regardless of model)
            print("\n--- Generating Monte-Carlo Signal ---")
            mc_final_signal = {}
            try:
                mc_filter = predictors.MonteCarloTrendFilter(**config.ADVANCED_MC_PARAMS)
                mc_filter.fit(df_features) 
                mc_final_signal = mc_filter.predict(horizon=21)
                print(f"MC Signal: Trend Strength={mc_final_signal.get('trend_strength', 0):.3f}")
            except Exception as e:
                print(f"Could not generate Monte-Carlo signal: {e}")

            # --- Step 3: MODEL SELECTION & TRAINING ---
            final_proba = np.array([]) #"None" aggrevated the linter
            model_name_log = ""
            shap_values = np.zeros(len(feature_cols)) # Default SHAP values
            
            if args.model == 'rf':
                from sklearn.ensemble import RandomForestClassifier
                print("\n--- Training Random Forest model ---")
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf_model.fit(X_train, y_train)
                final_proba = rf_model.predict_proba(X_test)[:, 1]
                y_test_final = y_test_orig.values
                model_name_log = "RandomForest_v1"
                # Note: Using default zero SHAP for RF

            elif args.model == 'xgb':
                print("\n--- Training XGBoost model ---")
                xgb_model = models.train_xgboost(X_train, y_train)
                final_proba = xgb_model.predict_proba(X_test)[:, 1]
                y_test_final = y_test_orig.values
                shap_values = models.get_shap_importance(xgb_model, X_test)
                model_name_log = "XGBoost_v1"

            elif args.model == 'ensemble':
                print("\n--- Training XGBoost component ---")
                xgb_model = models.train_xgboost(X_train, y_train)
                xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
                shap_values = models.get_shap_importance(xgb_model, X_test)

                print("\n--- Identifying Top Features for Transformer ---")
                shap_series = pd.Series(shap_values, index=feature_cols)
                top_12_features = shap_series.nlargest(12).index.tolist()
                print(f"Selected Top 12 Features: {top_12_features}")
                X_train_reduced, X_test_reduced = X_train[top_12_features], X_test[top_12_features]

                print("\n--- Training Transformer component ---")
                trans_model, device, scaler, y_seq_test = models.train_transformer(
                    X_train_reduced.values, y_train.values, X_test_reduced.values, y_test_orig.values
                )
                trans_proba = models.predict_transformer(
                    trans_model, device, scaler, X_test_reduced.values
                )[:, 1]
                
                print("\n--- Creating Ensemble Prediction ---")
                min_len = min(len(xgb_proba), len(trans_proba))
                final_proba = (xgb_proba[:min_len] + trans_proba[:min_len]) / 2
                y_test_final = y_seq_test
                model_name_log = "Ensemble_v2_CryptoAware"

            # --- Step 4: Common Post-Processing & Logging ---
            predictions = (final_proba > 0.5).astype(int)
            accuracy = np.mean(predictions == y_test_final) if len(y_test_final) > 0 else 0.0
            kelly_fraction = 2 * final_proba[-1] - 1 if len(final_proba) > 0 else 0.0

            # Actionable Forecast Logic (as implemented previously)
            verdict = "HOLD / NEUTRAL"
            entry_price, stop_loss, take_profit = None, None, None
            confidence_threshold = 0.10

            if len(final_proba) > 0:
                last_close = df_features['Close'].iloc[-1]
                last_atr = df_features['ATR14'].iloc[-1]
                
                if predictions[-1] == 1 and kelly_fraction > confidence_threshold:
                    verdict = "BUY"
                    entry_price, stop_loss, take_profit = last_close, last_close - (1.5 * last_atr), last_close + (3.0 * last_atr)
                elif predictions[-1] == 0 and kelly_fraction < -confidence_threshold:
                    verdict = "SELL (SHORT)"
                    entry_price, stop_loss, take_profit = last_close, last_close + (1.5 * last_atr), last_close - (3.0 * last_atr)

            print(f"\n--- Actionable Forecast for {ticker} ---")
            print(f"Last Close Price: {last_close:.2f}")
            print(f"Model Verdict: {verdict} (Kelly: {kelly_fraction:.2f})")
            if verdict != "HOLD / NEUTRAL":
                print(f"   -> Suggested Entry: ~{entry_price:.2f}")
                print(f"   -> Suggested Stop-Loss: ~{stop_loss:.2f}")
                print(f"   -> Suggested Take-Profit: ~{take_profit:.2f}")

            # Logging
            metrics = {
                "accuracy": accuracy, "kelly_fraction": kelly_fraction, 'verdict': verdict,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                **mc_final_signal
            }
            
            run_config = {"model_type": args.model, "XGBoost": config.XGB_PARAMS, "Transformer": {**config.TRANSFORMER_PARAMS, **config.TRANSFORMER_TRAINING}}

            audit_logger.log_analysis_result(
                ticker=ticker, model_name=model_name_log, run_config=run_config,
                predictions={"probabilities": final_proba.tolist()}, metrics=metrics,
                shap_importance={'features': feature_cols, 'values': shap_values.tolist()}
            )
            print(f"Analysis for {ticker} using {model_name_log} complete. Results logged.")
            print(f"Final Accuracy: {accuracy:.2%}")

        except Exception as e:
            print(f"!!! FAILED to process {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")