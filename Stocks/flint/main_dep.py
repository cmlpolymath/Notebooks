# main.py
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch

# Import our new modules
import config
import data_handler
from feature_engineering import FeatureCalculator
import models
import visualization
# The function to generate the MC feature
from predictors import add_mc_trend_feature # Let's assume you put the function above in a helper file

def run_analysis():
    """Main orchestration function."""
    # 1. Load Data
    df_raw = data_handler.get_stock_data(config.TICKER, config.START_DATE, config.END_DATE)
    df_raw.index = pd.to_datetime(df['Date'])

    # 2. Feature Engineering
    # This is the standard feature set
    feature_calculator = FeatureCalculator(df_raw)
    df_features = feature_calculator.add_all_features()
    
    # Add the advanced Monte Carlo feature
    df_features = add_mc_trend_feature(df_features)
    df_features.dropna(inplace=True)

    # 3. Define Target and Split Data
    df_features['UpNext'] = (df_features['Close'].shift(-1) > df_features['Close']).astype(int)
    df_features.dropna(inplace=True)

    train_size = int(len(df_features) * config.TRAIN_SPLIT_RATIO)
    train_df = df_features.iloc[:train_size]
    test_df = df_features.iloc[train_size:]

    X_train = train_df[config.FEATURE_COLS]
    y_train = train_df['UpNext']
    X_test = test_df[config.FEATURE_COLS]
    y_test = test_df['UpNext']

    # 4. Train Models
    # XGBoost
    xgb_model = models.train_xgboost(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Transformer
    X_seq, y_seq = models.prepare_sequences(
        df_features[config.FEATURE_COLS].values, 
        df_features['UpNext'].values, 
        config.TRANSFORMER_SEQUENCE_WINDOW
    )
    # Align sequence data with train/test split
    seq_train_size = len(train_df) - config.TRANSFORMER_SEQUENCE_WINDOW
    X_seq_train, y_seq_train = X_seq[:seq_train_size], y_seq[:seq_train_size]
    X_seq_test, y_seq_test = X_seq[seq_train_size:], y_seq[seq_train_size:]
    
    transformer_model = models.train_transformer(X_seq_train, y_seq_train)
    
    # 5. Evaluate and Ensemble
    transformer_model.eval()
    with torch.no_grad():
        test_outputs = transformer_model(torch.tensor(X_seq_test))
        trans_proba = F.softmax(test_outputs, dim=1).numpy()[:, 1]

    min_len = min(len(xgb_proba), len(trans_proba))
    ensemble_proba = (xgb_proba[:min_len] + trans_proba[:min_len]) / 2

    # 6. Generate Verdict
    latest_p = ensemble_proba[-1]
    kelly_fraction = 2 * latest_p - 1
    verdict = visualization.get_fuzzy_verdict(kelly_fraction)
    print(f"\n--- Final Verdict for {config.TICKER} ---")
    print(f"Latest Ensemble 'Up' Probability: {latest_p:.2%}")
    print(f"Kelly Fraction: {kelly_fraction:.2f}")
    print(f"Verdict: {verdict}")

    # 7. Visualization
    # SHAP Importance
    shap_importance = models.get_shap_importance(xgb_model, X_test)
    feature_importance = pd.Series(shap_importance, index=X_test.columns).sort_values(ascending=False)
    
    # Generate Plots
    fig1 = visualization.plot_price_with_signals(df_features, test_df.index, ensemble_proba > 0.5)
    fig2 = visualization.plot_feature_importance(feature_importance)
    fig1.show()
    fig2.show()


if __name__ == '__main__':
    # You will need to create the helper file and visualization.py
    # For now, let's just demonstrate the structure
    # For visualization.py, move your plotting and verdict logic there.
    # For the helper file, just copy the add_mc_trend_feature function.
    print("This is a structural demonstration. Run the complete refactored code.")
    # To run for real:
    # run_analysis()