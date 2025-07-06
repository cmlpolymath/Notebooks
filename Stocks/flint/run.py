# run.py
import initialize
import argparse
import re
import time
import numpy as np
import pandas as pd
import torch

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

def main():
    parser = argparse.ArgumentParser(description="Run the Ensemble stock analysis pipeline.")
    parser.add_argument('tickers', metavar='TICKER', type=validate_ticker, nargs='+',
                        help='A space-separated list of stock tickers to analyze.')
    args = parser.parse_args()
    
    audit_logger.setup_audit_db()

    for ticker in args.tickers:
        print(f"\n{'='*25} Processing Ticker: {ticker} {'='*25}")
        try:
            # Step 1: Load and prepare data ONCE per ticker
            df_raw = data_handler.get_stock_data(ticker=ticker, start_date=config.START_DATE, end_date=config.END_DATE)
            if df_raw is None or len(df_raw) < 350:
                print(f"Insufficient data for {ticker} (need ~350 days). Skipping.")
                continue
            
            feature_calculator = FeatureCalculator(df_raw.copy())
            df_features = feature_calculator.add_all_features()
            df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_features.dropna(inplace=True)

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

            # Step 3: Prepare data for models
            df_model = df_features.copy()
            
            # --- MODIFIED: NEW, MORE ROBUST TARGET DEFINITION ---
            # The target is 1 if the price 5 days from now is above the 20-day MA at that future time.
            # This is a much more stable signal than next-day prediction.
            future_price = df_model['Close'].shift(-5)
            future_ma = df_model['Close'].rolling(20).mean().shift(-5)
            df_model['UpNext'] = (future_price > future_ma).astype(int)
            
            # This correctly drops rows at the end where the future target is unknown
            df_model.dropna(inplace=True)

            train_size = int(len(df_model) * config.TRAIN_SPLIT_RATIO)
            train_df = df_model.iloc[:train_size]
            test_df = df_model.iloc[train_size:]

            feature_cols = [col for col in config.FEATURE_COLS if col in df_model.columns]
            X_train, y_train = train_df[feature_cols], train_df['UpNext']
            X_test, y_test_orig = test_df[feature_cols], test_df['UpNext']

            # Step 4: Train XGBoost and get its probabilities and SHAP values
            print("\n--- Training XGBoost component ---")
            xgb_model = models.train_xgboost(X_train, y_train)
            xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
            shap_values = models.get_shap_importance(xgb_model, X_test)
            print("XGBoost training complete.")

            # Step 5: Train Transformer and get its probabilities
            print("\n--- Training Transformer component ---")
            if len(df_model) < config.SEQUENCE_WINDOW_SIZE + 100:
                print("Skipping Transformer: Insufficient data. Ensemble will not be created.")
                continue
            
            trans_model, device, scaler, y_seq_test = models.train_transformer(X_train.values, y_train.values, X_test.values, y_test_orig.values)
            trans_proba = models.predict_transformer(trans_model, device, scaler, X_test.values)[:, 1]
            print("Transformer training complete.")

            # Step 6: Create the Ensemble prediction
            print("\n--- Creating Ensemble Prediction ---")
            min_len = min(len(xgb_proba), len(trans_proba))
            ensemble_proba = (xgb_proba[:min_len] + trans_proba[:min_len]) / 2
            
            # Align the test labels to the shortest prediction length
            # The transformer's test labels (y_seq_test) are the correct length
            y_test_final = y_seq_test

            # Step 7: Calculate final metrics and log the ENSEMBLE result
            predictions = (ensemble_proba > 0.5).astype(int)
            accuracy = np.mean(predictions == y_test_final) if len(y_test_final) > 0 else 0.0
            kelly_fraction = 2 * ensemble_proba[-1] - 1 if len(ensemble_proba) > 0 else 0.0
            
            metrics = {
                "accuracy": accuracy,
                "kelly_fraction": kelly_fraction,
                **mc_final_signal
            }
            
            # The final model name is "Ensemble_v1"
            model_name = "Ensemble_v1"
            run_config = {"XGBoost": config.XGB_PARAMS, "Transformer": {**config.TRANSFORMER_PARAMS, **config.TRANSFORMER_TRAINING}}

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
            print(f"Final Accuracy: {accuracy:.2%}") # ADDED: Print final accuracy

        except Exception as e:
            print(f"!!! FAILED to process {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    main()