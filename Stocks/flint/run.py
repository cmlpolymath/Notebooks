# run.py
import os
from rich import print
from pathlib import Path
import json
import initialize
import time
import argparse
import re
import numpy as np
import pandas as pd
import torch

# Import project modules
from config import settings
import audit_logger
import models
import predictors
from preprocess import prepare_and_cache_data

settings.environment = 'production'  # Set the environment to production

def validate_ticker(ticker):
    """Sanitize and validate a stock or crypto ticker format."""
    ticker = ticker.strip().upper()
    # MODIFIED: Updated regex to allow hyphens and be more flexible for crypto.
    if re.match(r'^[A-Z0-9\.\-]{1,12}$', ticker):
        return ticker
    raise argparse.ArgumentTypeError(f"'{ticker}' is not a valid ticker format.")

def main():
    parser = argparse.ArgumentParser(description="Run the stock/crypto analysis pipeline.")
    parser.add_argument('tickers', metavar='TICKER', type=validate_ticker, nargs='+',
                        help='A space-separated list of tickers to analyze.')
    parser.add_argument(
        '--model', '-m',
        metavar='MODEL',
        type=str, 
        default='ensemble', 
        choices=['ensemble', 'xgb', 'rf'],
        help="Model to run: 'ensemble' (XGB+Transformer), 'xgb' (XGBoost only), or 'rf' (Random Forest only)."
    )
    args = parser.parse_args()
    
    audit_logger.setup_audit_db()


    for ticker in args.tickers:
        print(f"\n{'='*25} Processing Ticker: {ticker} | Model: {args.model.upper()} {'='*25}")
        # --- UNIFIED PARAMETER LOADING ---
        tuned_params = {}
        params_path = Path("results") / "tuning" / f"best_params_{ticker}.json"
        if params_path.exists():
            print(f"\n--- Found tuned parameter file for {ticker}: {params_path} ---")
            with params_path.open("r") as f:
                tuned_params = json.load(f)
        
        # Extract specific params for each model type, defaulting to empty dict if not found
        tuned_xgb_params = tuned_params.get("xgboost")
        tuned_transformer_params = tuned_params.get("transformer")

        try:
            # Step 1: Load Data
            data_package = prepare_and_cache_data(ticker, settings.data.start_date, settings.data.end_date)
            X_train, y_train = data_package['X_train'], data_package['y_train']
            X_test, y_test_orig = data_package['X_test'], data_package['y_test_orig']
            feature_cols, df_features = data_package['feature_cols'], data_package['df_features']

            # Step 2: Run Monte Carlo simulation
            print("\n--- Generating Monte-Carlo Signal ---")
            mc_final_signal = {}
            try:
                mc_params = settings.models.monte_carlo.model_dump(exclude_unset=True)
                mc_filter = predictors.MonteCarloTrendFilter(**mc_params)
                mc_filter.fit(df_features) 
                mc_final_signal = mc_filter.predict(horizon=21)
                print(f"MC Signal: Trend Strength= {mc_final_signal.get('trend_strength', 0):+.3f}")
            except Exception as e:
                print(f"Could not generate Monte-Carlo signal: {e}")

            # --- Step 3: MODEL SELECTION & TRAINING ---
            final_proba = np.array([])
            model_name_log = ""
            shap_values = np.zeros(len(feature_cols))
            run_config = {}

            if args.model == 'rf':
                rf_model, scaler, selector, run_config = models.train_enhanced_random_forest(
                    X_train, y_train
                )
                
                # Fix 1: Maintain DataFrame structure after scaling
                X_test_scaled = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
                
                # Fix 2: Maintain DataFrame structure after feature selection
                # Get the selected feature names from the selector
                selected_feature_mask = selector.get_support()
                selected_feature_names = X_test.columns[selected_feature_mask].tolist()
                
                X_test_selected = pd.DataFrame(
                    selector.transform(X_test_scaled),
                    columns=selected_feature_names,
                    index=X_test_scaled.index
                )

                final_proba = rf_model.predict_proba(X_test_selected)[:, 1]
                y_test_final = y_test_orig.values
                model_name_log = run_config["model_type"]       
                         
            elif args.model == 'xgb':
                print("\n--- Training XGBoost model ---")
                if tuned_xgb_params:
                    print("Using tuned XGBoost parameters.")
                    xgb_params_to_use = tuned_xgb_params
                else:
                    print("No tuned XGBoost parameters found, using defaults from config.")
                    xgb_params_to_use = settings.models.xgboost.model_dump(exclude_unset=True)
                
                xgb_model = models.train_xgboost(X_train, y_train, params=xgb_params_to_use)
                final_proba = xgb_model.predict_proba(X_test)[:, 1]
                y_test_final = y_test_orig.values
                shap_values = models.get_shap_importance(xgb_model, X_test)
                model_name_log = "XGBoost_v2_Tuned" if tuned_xgb_params else "XGBoost_v2_Default"
                run_config = {
                    "model_type": "xgb",
                    "xgboost_params": xgb_params_to_use
                }

            elif args.model == 'ensemble':
                print("\n--- Training XGBoost component ---")
                if tuned_xgb_params:
                    print("Using tuned XGBoost parameters.")
                    xgb_params_to_use = tuned_xgb_params
                else:
                    print("No tuned XGBoost parameters found, using defaults from config.")
                    xgb_params_to_use = settings.models.xgboost.model_dump(exclude_unset=True)
                
                xgb_model = models.train_xgboost(X_train, y_train, params=xgb_params_to_use)
                xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
                shap_values = models.get_shap_importance(xgb_model, X_test)

                # --- Transformer Component ---
                print("\n--- Identifying Top Features for Transformer ---")
                shap_series = pd.Series(shap_values, index=feature_cols)
                top_12_features = shap_series.nlargest(12).index.tolist()
                print(f"Selected Top 12 Features: {top_12_features}")
                X_train_reduced, X_test_reduced = X_train[top_12_features], X_test[top_12_features]

                print("\n--- Training Transformer component ---")
                
                if tuned_transformer_params:
                    print("Using tuned Transformer parameters.")
                    # Start with default training params and override/add the tuned ones
                    trans_training_params = settings.models.transformer_training.model_dump(exclude_unset=True)
                    trans_training_params.update(tuned_transformer_params)
                else:
                    print("No tuned Transformer parameters found, using defaults from config.")
                    trans_training_params = settings.models.transformer_training.model_dump(exclude_unset=True)

                trans_model, device, scaler, y_seq_test = models.train_transformer(
                    X_train_reduced.values, y_train.values, X_test_reduced.values, y_test_orig.values,
                    training_params=trans_training_params
                )
                trans_proba = models.predict_transformer(
                    trans_model, device, scaler, X_test_reduced.values
                )[:, 1]
                
                # --- Ensemble Prediction ---
                print("\n--- Creating Ensemble Prediction ---")
                min_len = min(len(xgb_proba), len(trans_proba))
                final_proba = (xgb_proba[:min_len] + trans_proba[:min_len]) / 2
                y_test_final = y_seq_test
                model_name_log = "Ensemble_v3_Tuned" if tuned_xgb_params or tuned_transformer_params else "Ensemble_v3_Default"
                run_config = {
                    "model_type": "ensemble",
                    "xgboost_params": xgb_params_to_use,
                    "transformer_arch_params": settings.models.transformer_arch.model_dump(exclude_unset=True),
                    "transformer_training_params": trans_training_params
                }

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

            # get the actual pd.Index (the dates) for the test predictions
            num_predictions = len(final_proba)
            test_index_for_log = pd.DatetimeIndex(data_package['test_df']['Date'].tail(num_predictions))

            # Logging
            metrics = {
                "accuracy": accuracy, "kelly_fraction": kelly_fraction, 'verdict': verdict,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                **mc_final_signal
            }

            audit_logger.log_analysis_result(
                ticker=ticker, 
                model_name=model_name_log, 
                run_config=run_config,
                predictions={"probabilities": final_proba.tolist()}, 
                test_index=test_index_for_log,
                metrics=metrics,
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