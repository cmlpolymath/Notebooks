# config.py
from datetime import datetime, timedelta

# --- Data Configuration ---
START_DATE = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')
END_DATE = datetime.today().strftime('%Y-%m-%d')

# --- Feature Configuration ---
# List of feature columns to be used by the models
FEATURE_COLS = [
    'RSI14', 'MACD_line', 'MACD_signal', 'MACD_hist', 'ATR14',
    'BB_mid', 'BB_upper', 'BB_lower', 'OBV', '%K', '%D',
    'MFI14', 'CCI20', 'Williams_%R', 'ROC10', 'GARCH_vol',
    'Dominant_Period', 'Return1', 'Close', 'Volume',
]

# --- Model Configuration ---
# Train/Test Split
TRAIN_SPLIT_RATIO = 0.8

# XGBoost Parameters
XGB_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# Monte Carlo Predictor (Simplified Parameters)
MC_FILTER_PARAMS = {
    'lookback_days': 252,  # Use 1 year of data to calculate drift and volatility
    'n_simulations': 1000, # Number of simulation paths
    'horizon_days': 21      # Predict 1 month into the future
}