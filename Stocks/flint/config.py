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
SEQUENCE_WINDOW_SIZE = 60 # The number of past time steps the Transformer looks at for each prediction.

# XGBoost Parameters
XGB_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# Transformer Model Hyperparameters
TRANSFORMER_PARAMS = {
    'd_model': 128,
    'nhead': 8,
    'num_layers': 4,
    'num_classes': 2 # Binary classification (Up/Down)
}

# Transformer Training Configuration
TRANSFORMER_TRAINING = {
    'lr': 0.0001,
    'batch_size': 64,
    'epochs': 50
}

# Advanced Monte Carlo Predictor Parameters
# These now match the constructor of MonteCarloTrendFilter
ADVANCED_MC_PARAMS = {
    'gbm_lookback': 63,
    'jump_window': 21,
    'n_optimize': 50,
    'base_n_sims': 1000,
    'n_jobs': -1
}