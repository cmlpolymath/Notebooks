# validate.py
import initialize
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier # ADDED for 'rf' option
import torch

# Import project modules
import config
import models
from preprocess import prepare_and_cache_data
from feature_engineering import FeatureCalculator # ADDED to re-run feature engineering on filtered data

def run_walk_forward_validation(ticker: str, years: int, model_type: str):
    """
    Performs a robust walk-forward validation of a specified modeling strategy.

    Args:
        ticker (str): The stock ticker to validate.
        years (int): The number of recent years of data to use for the validation.
        model_type (str): The model to validate ('rf', 'xgb', or 'tf').
    """
    print(f"\n{'='*25} Starting Walk-Forward Validation {'='*25}")
    print(f"Ticker: {ticker} | Years of Data: {years} | Model Type: {model_type.upper()}")

    # 1. Load and Filter Data based on the 'years' parameter
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * years)
        
        print(f"Loading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        df_raw = data_handler.get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        market_df = data_handler.get_stock_data(config.MARKET_INDEX_TICKER, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if df_raw is None or market_df is None or len(df_raw) < 252: # Need at least a year of data
            raise ValueError("Insufficient data for the specified period.")

        # Re-run feature engineering on the filtered data
        feature_calculator = FeatureCalculator(df_raw.copy())
        df_features = feature_calculator.add_all_features(market_df=market_df.copy())
        
        # Define the target variable
        future_price = df_features['Close'].shift(-5)
        future_ma = df_features['Close'].rolling(20).mean().shift(-5)
        df_model = df_features.copy()
        df_model['UpNext'] = (future_price > future_ma).astype(int)
        df_model.dropna(inplace=True)

    except Exception as e:
        print(f"Could not prepare data for validation: {e}")
        return

    # 2. Define Walk-Forward Parameters
    n = len(df_model)
    initial_train_size = int(n * 0.50) # Start with 50% of data for initial training
    validation_size = int(n * 0.15)    # Use 15% for the validation set in each fold
    step_size = int(n * 0.05)          # Step forward by 5% of the data each time
    
    # Adjust validation size for non-transformer models which don't need it for early stopping
    if model_type in ['rf', 'xgb']:
        initial_train_size += validation_size # Fold validation set into training
        validation_size = 0

    if initial_train_size + step_size > n:
        print("Not enough data for even one walk-forward fold. Aborting.")
        return

    all_preds = []
    all_true = []
    fold_num = 0
    
    feature_cols = [col for col in config.FEATURE_COLS if col in df_model.columns]

    # 3. Loop through the data, creating folds
    start_idx = 0
    while start_idx + initial_train_size + validation_size + step_size <= n:
        fold_num += 1
        print(f"\n--- Processing Fold {fold_num} ---")

        # Define indices for this fold
        train_end_idx = start_idx + initial_train_size
        val_end_idx = train_end_idx + validation_size
        test_end_idx = val_end_idx + step_size

        train_df = df_model.iloc[start_idx:train_end_idx]
        test_df = df_model.iloc[val_end_idx:test_end_idx]
        
        X_train, y_train = train_df[feature_cols], train_df['UpNext']
        X_test, y_test = test_df[feature_cols], test_df['UpNext']
        
        # --- Model-Specific Training and Prediction ---
        
        if model_type == 'rf':
            print("Training Random Forest model...")
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            predictions = rf_model.predict(X_test)
            y_test_aligned = y_test.values
        
        elif model_type == 'xgb':
            print("Training XGBoost model...")
            xgb_model = models.train_xgboost(X_train, y_train)
            predictions = xgb_model.predict(X_test)
            y_test_aligned = y_test.values

        elif model_type == 'tf':
            print("Training Transformer Ensemble model...")
            val_df = df_model.iloc[train_end_idx:val_end_idx]
            X_val, y_val = val_df[feature_cols], val_df['UpNext']

            xgb_model = models.train_xgboost(X_train, y_train)
            xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

            shap_values = models.get_shap_importance(xgb_model, X_val)
            shap_series = pd.Series(shap_values, index=feature_cols)
            top_12_features = shap_series.nlargest(12).index.tolist()
            
            X_train_reduced = X_train[top_12_features]
            X_val_reduced = X_val[top_12_features]
            X_test_reduced = X_test[top_12_features]

            trans_model, device, scaler, _ = models.train_transformer(
                X_train_reduced.values, y_train.values, X_val_reduced.values, y_val.values
            )
            trans_proba = models.predict_transformer(
                trans_model, device, scaler, X_test_reduced.values
            )[:, 1]

            num_trans_preds = len(trans_proba)
            xgb_proba_aligned = xgb_proba[-num_trans_preds:]
            y_test_aligned = y_test.values[-num_trans_preds:]

            ensemble_proba = (xgb_proba_aligned + trans_proba) / 2
            predictions = (ensemble_proba > 0.5).astype(int)

        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose 'rf', 'xgb', or 'tf'.")

        # --- Store results for this fold ---
        all_preds.extend(predictions.tolist())
        all_true.extend(y_test_aligned.tolist())

        # Move to the next fold
        start_idx += step_size

    # 4. Report Final, Aggregated Metrics
    if not all_true:
        print("\nValidation could not be completed. No results to report.")
        return
        
    final_accuracy = accuracy_score(all_true, all_preds)
    print(f"\n{'='*30}")
    print("Walk-Forward Validation Complete")
    print(f"{'='*30}")
    print(f"Ticker: {ticker}")
    print(f"Model: {model_type.upper()}")
    print(f"Data Period: Last {years} year(s)")
    print(f"Total Folds: {fold_num}")
    print(f"Total Out-of-Sample Predictions: {len(all_true)}")
    print(f"Overall Accuracy: {final_accuracy:.2%}")
    print(f"{'-'*30}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation on a specified model and data period.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('ticker', metavar='TICKER', type=str, 
                        help='The stock ticker to validate (e.g., RYCEY).')
    parser.add_argument('years', metavar='YEARS', type=int, 
                        help='The number of recent years of data to use (e.g., 5).')
    parser.add_argument('model_type', metavar='MODEL', type=str, choices=['rf', 'xgb', 'tf'],
                        help="The type of model to validate:\n"
                             "  'rf'  - Random Forest\n"
                             "  'xgb' - XGBoost\n"
                             "  'tf'  - Transformer Ensemble")
    
    args = parser.parse_args()

    # Import data_handler here to avoid circular dependency issues if it were at the top
    # and to ensure it's imported after initialize.py has run.
    import data_handler
    run_walk_forward_validation(args.ticker, args.years, args.model_type)