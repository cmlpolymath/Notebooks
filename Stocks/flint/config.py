# config.py
from datetime import datetime, timedelta

# --- Data Configuration ---
START_DATE = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')
END_DATE = datetime.today().strftime('%Y-%m-%d')
MARKET_INDEX_TICKER = 'SPY' # Added for market context

# Mapping of yfinance sectors to representative ETFs
SECTOR_ETF_MAP = {
    'Technology': 'XLK',
    'Communication Services': 'XLC',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Industrials': 'XLI',
    'Consumer Discretionary': 'XLY',
    'Consumer Cyclical': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Basic Materials': 'XLB'
}

# Tickers for macroeconomic indicators from yfinance
MACRO_TICKERS = {
    'VIX': '^VIX',             # Volatility Index
    '10Y_Treasury': '^TNX',    # 10-Year Treasury Yield
    'Crude_Oil': 'CL=F',       # Crude Oil Futures
    'Gold': 'GC=F'             # Gold Futures
}

FRED_INDICATORS = {
    'CPI':             'CPIAUCSL',
    'TreasurySpread':  'T10Y2Y',
    'Unemployment':    'UNRATE',
    'GDP':             'GDP',
    'FedFunds':        'FEDFUNDS',
    'Mortgage30Yr':    'MORTGAGE30US',
    'IndustrialProd':  'INDPRO',
    'RetailSales':     'RSXFS',
    'HousingStarts':   'HOUST',
    'NASDAQ':          'NASDAQCOM',
    'ConsumerSenti':   'UMCSENT',
    'ProdPriceIdx':    'PPIACO',
}

fred_engineered_features = []
for name in FRED_INDICATORS.keys():
    fred_engineered_features.extend([
        f"{name}_DaysSinceUpdate",
        f"{name}_InEventWindow"
    ])

relational_macro_features = [
    'Corr_Stock_FedFunds_60D',
    'Corr_Stock_CPI_60D',
    'TreasurySpread_RealVol_21D',
    'Real_FedFunds',
    'Stock_vs_GDP_Ratio',
    'CPI_ROC_3M'
]

# --- Feature Configuration ---
# List of feature columns to be used by the models
# ADDED: SPY_RSI14 and SPY_Return1 for market context
# ADDED: Sector and Macroeconomic features
FEATURE_COLS = [
    'RSI14', 'MACD_line', 'MACD_signal', 'MACD_hist', 'ATR14',
    'BB_mid', 'BB_upper', 'BB_lower', 'OBV', '%K', '%D',
    'MFI14', 'CCI20', 'Williams_%R', 'ROC10', 'GARCH_vol',
    'Dominant_Period', 'Return1', 'Close', 'Volume',
    'Kal_filter', 'Realized_Vol', 'Hurst_Exponent',
    'Fractal_Dimension', 'Wavelet_Ratio', 'Efficiency_Ratio',
    'VWAP_zscore',
    'SPY_RSI14', 'SPY_Return1', # Market context features
    'SECTOR_RSI14', 'SECTOR_Return1', # Sector features
    'VIX', '10Y_Treasury', 'Crude_Oil', 'Gold', # Macro features
    *FRED_INDICATORS.keys(), # Unpack all the FRED indicator names here
    *fred_engineered_features,
    *relational_macro_features
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
    'batch_size': 256,
    'epochs': 50,
    'patience': 5 # ADDED: Early stopping patience
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