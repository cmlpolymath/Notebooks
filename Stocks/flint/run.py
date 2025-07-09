# run.py
import initialize
import time
import argparse
import re
import time
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
    """Sanitize and validate a stock ticker format."""
    ticker = ticker.strip().upper()
    if re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$', ticker):
        return ticker
    raise argparse.ArgumentTypeError(f"'{ticker}' is not a valid ticker format.")

def main():
    parser = argparse.ArgumentParser(description="Run the Ensemble stock analysis pipeline.")
    parser.add_argument('tickers', metavar='TICKER', type=validate_ticker, nargs='+',
                        help='A space-separated list of stock tickers to analyze.')
    args = parser.parse_args()
    
    audit_logger.setup_audit_db()

    for ticker in args.tickers:
        print(f"\n{'='*25} Processing Ticker: {ticker} {'='*25}")
        try:
            # Step 1: Load pre-processed and cached data
            # This function will either load from cache or create it if it doesn't exist.
            data_package = prepare_and_cache_data(ticker, config.START_DATE, config.END_DATE)
            
            # Unpack the data
            X_train = data_package['X_train']
            y_train = data_package['y_train']
            X_test = data_package['X_test']
            y_test_orig = data_package['y_test_orig']
            feature_cols = data_package['feature_cols']
            df_features = data_package['df_features']
            
            # Step 2: Run Monte Carlo simulation ONCE per ticker
            print("\n--- Generating Monte-Carlo Signal (Once per Ticker) ---")
            mc_final_signal = {}
            try:
                mc_filter = predictors.MonteCarloTrendFilter(**config.ADVANCED_MC_PARAMS)
                mc_filter.fit(df_features) 
                mc_final_signal = mc_filter.predict(horizon=21)
                print(f"MC Signal: Trend Strength={mc_final_signal.get('trend_strength', 0):.3f}")
            except Exception as e:
                print(f"Could not generate Monte-Carlo signal: {e}")
                import traceback
                traceback.print_exc()

            # Step 3: Train XGBoost and get its probabilities and SHAP values
            print("\n--- Training XGBoost component ---")
            xgb_model = models.train_xgboost(X_train, y_train)
            xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
            shap_values = models.get_shap_importance(xgb_model, X_test)
            print("XGBoost training complete.")

            # --- MODIFIED: Feature Reduction based on SHAP ---
            print("\n--- Identifying Top Features for Transformer ---")
            shap_series = pd.Series(shap_values, index=feature_cols)
            top_12_features = shap_series.nlargest(12).index.tolist()
            print(f"Selected Top 12 Features: {top_12_features}")
            
            X_train_reduced = X_train[top_12_features]
            X_test_reduced = X_test[top_12_features]

            # Step 4: Train Transformer on the REDUCED feature set
            print("\n--- Training Transformer component on reduced feature set ---")
            if len(X_train_reduced) < config.SEQUENCE_WINDOW_SIZE + 50: # Check against training data length
                print("Skipping Transformer: Insufficient data. Ensemble will not be created.")
                continue
            
            trans_model, device, scaler, y_seq_test = models.train_transformer(
                X_train_reduced.values, y_train.values, X_test_reduced.values, y_test_orig.values
            )
            trans_proba = models.predict_transformer(
                trans_model, device, scaler, X_test_reduced.values
            )[:, 1]
            print("Transformer training complete.")

            # Step 5: Create the Ensemble prediction
            print("\n--- Creating Ensemble Prediction ---")
            min_len = min(len(xgb_proba), len(trans_proba))
            ensemble_proba = (xgb_proba[:min_len] + trans_proba[:min_len]) / 2
            
            # Align the test labels to the shortest prediction length
            # The transformer's test labels (y_seq_test) are the correct length
            y_test_final = y_seq_test

            # Step 6: Calculate final metrics and log the ENSEMBLE result
            predictions = (ensemble_proba > 0.5).astype(int)
            accuracy = np.mean(predictions == y_test_final) if len(y_test_final) > 0 else 0.0
            kelly_fraction = 2 * ensemble_proba[-1] - 1 if len(ensemble_proba) > 0 else 0.0
            
            metrics = {
                "accuracy": accuracy,
                "kelly_fraction": kelly_fraction,
                **mc_final_signal
            }
            
            # The final model name is "Ensemble_v2" to reflect changes
            model_name = "Ensemble_v2"
            run_config = {
                "XGBoost": config.XGB_PARAMS, 
                "Transformer": {**config.TRANSFORMER_PARAMS, **config.TRANSFORMER_TRAINING},
                "Feature_Reduction": {"method": "SHAP", "n_features": 12}
            }

            audit_logger.log_analysis_result(
                ticker=ticker, 
                model_name=model_name, 
                run_config=run_config,
                predictions={"probabilities": ensemble_proba.tolist()}, 
                metrics=metrics,
                # Use the SHAP values from the interpretable XGBoost model for the ensemble
                shap_importance={'features': feature_cols, 'values': shap_values.tolist()}
            )
            print(f"Analysis for {ticker} using {model_name} complete. Results logged.")
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