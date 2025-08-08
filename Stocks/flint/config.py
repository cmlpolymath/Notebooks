# config.py
"""
Expert-level configuration for high-performance trading system.
Optimized for speed, interpretability, and dynamic modification.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from functools import lru_cache
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text  
import os
import logging
import json
import sys
import structlog

# ==============================================================================
# ENUMS FOR TYPE SAFETY AND PERFORMANCE
# ==============================================================================

class ModelType(str, Enum):
    RF = "randomforest"
    XGBOOST = "xgboost"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    LSTM = "lstm"

class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"

class DataFrequency(str, Enum):
    DAILY = "1d"
    HOURLY = "1h"
    MINUTE = "1m"
    WEEKLY = "1wk"

class FeatureGroup(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    VOLATILITY = "volatility"

# ==============================================================================
# PERFORMANCE OPTIMIZED BASE CLASSES
# ==============================================================================

class MyConfig(BaseModel):
    name: str
    values: list[int]
    
    @lru_cache(maxsize=128)
    def get_cached_config(self) -> Dict[str, Any]:
        return self.model_dump()

    
class OptimizedBaseModel(BaseModel):
    """Base model with performance optimizations."""
    
    model_config = {
        'validate_assignment': True,
        'use_enum_values': True,
        'frozen': False,  # Allow modification for interpretability
        'extra': 'forbid',  # Catch typos early
    }
        
    def update_field(self, field: str, value: Any) -> None:
        """Safely update a field with validation."""
        setattr(self, field, value)

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

class DataConfig(OptimizedBaseModel):
    """Data fetching and preprocessing configuration."""
    
    # Date range with smart defaults
    start_date: str = Field(
        default_factory=lambda: (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d'),
        description="Historical data start date"
    )
    end_date: str = Field(
        default_factory=lambda: datetime.now().strftime('%Y-%m-%d'),
        description="Historical data end date"
    )
    
    # Market data
    market_index_ticker: str = Field(default='SPY', description="Primary market benchmark")
    data_frequency: DataFrequency = Field(default=DataFrequency.DAILY, description="Data frequency")
    
    # Performance optimizations
    max_workers: int = Field(default=8, ge=1, le=32, description="Parallel data fetching threads")
    cache_expiry_hours: int = Field(default=24, ge=1, description="Data cache expiry in hours")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="API retry attempts")
    
    # Data quality
    min_trading_days: int = Field(default=252, ge=50, description="Minimum trading days required")
    max_missing_pct: float = Field(default=0.05, ge=0, le=1, description="Maximum missing data percentage")
    epsilon: int = Field(default=1e-9, description="Small value to avoid division by zero")

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

# From: config.py

class RandomForestParams(OptimizedBaseModel):
    """Enhanced Random Forest parameters for financial data with time-aware validation and stacking."""
    
    # Existing parameters
    scaler_quantile_range: Tuple[int, int] = Field(default=(5, 95), description="Quantile range for RobustScaler")
    adasyn_sampling_strategy: str = Field(default='minority', description="ADASYN sampling strategy")
    adasyn_neighbors: int = Field(default=5, ge=1, description="Number of neighbors for ADASYN")
    feature_selection_max_features: int = Field(default=15, ge=5, le=50, description="Max features to select")
    feature_selection_threshold: str = Field(default='1.25*median', description="Threshold for feature selection from model")
    hyperparam_tuning_iterations: int = Field(default=10, ge=1, description="Number of iterations for RandomizedSearchCV")  # Reduced from 25
    hyperparam_tuning_cv_folds: int = Field(default=3, ge=2, description="Number of cross-validation folds")
    base_n_estimators: int = Field(default=200, ge=100)  # Reduced from 500
    base_max_depth: int = Field(default=10, ge=5)  # Reduced from 15
    base_min_samples_leaf: int = Field(default=5, ge=1)
    base_max_features: float = Field(default=0.5, gt=0, le=1.0)
    param_distributions: Dict[str, List] = Field(default_factory=lambda: {
        'n_estimators': [100, 200, 300],  # Reduced from [300, 500, 700]
        'max_depth': [8, 12, 16],  # Reduced from [10, 15, 20, None]
        'min_samples_split': [5, 10],  # Reduced options
        'min_samples_leaf': [3, 5],  # Reduced options
        'max_features': ['sqrt', 0.5],  # Reduced options
        'max_samples': [0.7]  # Single option
    })
    
    # New time-aware validation parameters
    tscv_splits: int = Field(default=5, ge=3, le=10, description="Number of splits for TimeSeriesSplit")
    tscv_gap: int = Field(default=5, ge=0, description="Gap between train/test in TimeSeriesSplit to prevent leakage")
    
    # Stacking parameters
    enable_stacking: bool = Field(default=True, description="Whether to use StackingClassifier")
    lgb_max_depth: int = Field(default=6, ge=3, le=15, description="Max depth for LightGBM base estimator")
    lgb_learning_rate: float = Field(default=0.1, gt=0, le=1.0, description="Learning rate for LightGBM")
    lgb_n_estimators: int = Field(default=100, ge=50, le=500, description="Number of estimators for LightGBM")
    lgb_num_leaves: int = Field(default=31, ge=10, le=100, description="Number of leaves for LightGBM")
    stack_final_estimator_C: float = Field(default=1.0, gt=0, description="Regularization parameter for stacking LogisticRegression")

class XGBoostParams(OptimizedBaseModel):
    """Optimized XGBoost parameters for financial data."""
    
    # Core parameters
    n_estimators: int = Field(default=100, ge=50, le=5000)
    max_depth: int = Field(default=6, ge=3, le=15)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    
    # Performance optimization
    tree_method: str = Field(default='hist', description="Fast histogram-based algorithm")
    device: DeviceType = Field(default=DeviceType.AUTO)
    n_jobs: int = Field(default=-1, description="Use all CPU cores")
    
    # Regularization for financial noise
    reg_alpha: float = Field(default=0.1, ge=0, description="L1 regularization")
    reg_lambda: float = Field(default=1.0, ge=0, description="L2 regularization")
    gamma: float = Field(default=0.1, ge=0, description="Minimum split loss")
    min_child_weight: float = Field(default=3, ge=0, description="Minimum child weight")
    
    # Advanced parameters
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    scale_pos_weight: float = Field(default=1.0, gt=0, description="Handle class imbalance")
    
    # Evaluation
    eval_metric: str = Field(default='logloss')
    early_stopping_rounds: int = Field(default=50, ge=10)
    
    # Financial-specific
    objective: str = Field(default='binary:logistic', description="Classification objective")
    random_state: int = Field(default=42)

class TransformerArchParams(OptimizedBaseModel):
    """Optimized Transformer architecture for time series."""
    
    # Core architecture
    d_model: int = Field(default=256, description="Model dimension")
    nhead: int = Field(default=8, description="Number of attention heads")
    num_layers: int = Field(default=6, ge=2, le=12)
    dim_feedforward: int = Field(default=1024, description="FFN dimension")
    
    # Regularization
    dropout: float = Field(default=0.1, ge=0, le=0.5)
    layer_norm_eps: float = Field(default=1e-5)
    
    # Output
    num_classes: int = Field(default=2, ge=2)
    
    # Performance optimizations
    batch_first: bool = Field(default=True, description="Batch dimension first")
    norm_first: bool = Field(default=True, description="Pre-layer normalization")
    
    @field_validator('d_model')
    @classmethod
    def validate_d_model(cls, v, info):
        if info.data and 'nhead' in info.data and v % info.data['nhead'] != 0:
            raise ValueError("d_model must be divisible by nhead")
        return v

class TransformerTrainingParams(OptimizedBaseModel):
    """Optimized training parameters."""
    
    # Learning parameters
    lr: float = Field(default=1e-4, gt=0, le=1)
    weight_decay: float = Field(default=1e-5, ge=0)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.999, gt=0, lt=1)
    
    # Training schedule
    batch_size: int = Field(default=128, ge=8, le=1024)
    epochs: int = Field(default=100, ge=10, le=1000)
    patience: int = Field(default=15, ge=5)
    
    # Performance optimization
    accumulation_steps: int = Field(default=4, ge=1)
    mixed_precision: bool = Field(default=True, description="Use automatic mixed precision")
    compile_model: bool = Field(default=True, description="Use torch.compile for speed")
    
    # Advanced optimization
    warmup_steps: int = Field(default=1000, ge=0)
    scheduler: str = Field(default='cosine', description="Learning rate scheduler")
    gradient_clip_norm: float = Field(default=1.0, gt=0)

class MonteCarloParams(OptimizedBaseModel):
    """Monte Carlo simulation parameters."""
    
    # Lookback periods
    gbm_lookback: int = Field(default=252, ge=21, description="GBM estimation period")
    jump_window: int = Field(default=21, ge=5, description="Jump detection window")
    
    # Simulation parameters
    base_n_sims: int = Field(default=10000, ge=1000, description="Base simulations")
    adaptive_sims: bool = Field(default=True, description="Adaptive simulation count")
    confidence_levels: List[float] = Field(default=[0.05, 0.95], description="VaR confidence levels")
    
    # Performance
    parallel_sims: bool = Field(default=True)
    chunk_size: int = Field(default=1000, ge=100)
    
    # Model parameters
    include_jumps: bool = Field(default=True, description="Include jump diffusion")
    vol_clustering: bool = Field(default=True, description="GARCH-based volatility")

class ModelConfig(OptimizedBaseModel):
    """Unified model configuration."""
    
    # Training split
    train_split_ratio: float = Field(default=0.7, gt=0, lt=1)
    validation_split_ratio: float = Field(default=0.2, gt=0, lt=1)
    test_split_ratio: float = Field(default=0.1, gt=0, lt=1)
    
    # Time series specific
    sequence_window_size: int = Field(default=60, ge=10, le=500)
    prediction_horizon: int = Field(default=1, ge=1, le=30)
    
    # Walk-forward parameters
    walk_forward_periods: int = Field(default=12, ge=1, description="Retraining frequency")
    min_train_size: int = Field(default=1000, ge=252, description="Minimum training samples")
    
    # Model instances
    random_forest: RandomForestParams = Field(default_factory=RandomForestParams)
    xgboost: XGBoostParams = Field(default_factory=XGBoostParams)
    transformer_arch: TransformerArchParams = Field(default_factory=TransformerArchParams)
    transformer_training: TransformerTrainingParams = Field(default_factory=TransformerTrainingParams)
    monte_carlo: MonteCarloParams = Field(default_factory=MonteCarloParams)
    
    # Ensemble configuration
    ensemble_weights: Dict[str, float] = Field(default={
        'xgboost': 0.4,
        'transformer': 0.4,
        'monte_carlo': 0.2
    })
    
    @model_validator(mode='after')
    def validate_splits(self):
        train = self.train_split_ratio
        val = self.validation_split_ratio
        test = self.test_split_ratio
        if abs(train + val + test - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        return self

# ==============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ==============================================================================

class FeatureConfig(OptimizedBaseModel):
    """Advanced feature engineering configuration."""
    
    # ETF sector mapping (optimized for fast lookup)
    sector_etf_map: Dict[str, str] = Field(default={
        'Technology': 'XLK', 'Communication Services': 'XLC', 'Healthcare': 'XLV',
        'Financials': 'XLF', 'Industrials': 'XLI', 'Consumer Discretionary': 'XLY',
        'Consumer Cyclical': 'XLY', 'Consumer Staples': 'XLP', 'Energy': 'XLE',
        'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB',
        'Semiconductor': 'SMH', 'Biotech': 'XBI', 'Regional Banks': 'KRE'
    })
    
    # Yahoo Finance tickers for macro data
    macro_tickers_yf: Dict[str, str] = Field(default={
        'VIX': '^VIX', 'VIX9D': '^VIX9D', 'VVIX': '^VVIX',
        '10Y_Treasury': '^TNX', '2Y_Treasury': '^IRX', '30Y_Treasury': '^TYX',
        'DXY': 'DX-Y.NYB', 'Crude_Oil': 'CL=F', 'Natural_Gas': 'NG=F',
        'Gold': 'GC=F', 'Silver': 'SI=F', 'Copper': 'HG=F',
        'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD'
    })
    
    # FRED economic indicators
    fred_indicators: Dict[str, str] = Field(default={
        'CPI': 'CPIAUCSL', 'Core_CPI': 'CPILFESL', 'PCE': 'PCE',
        'TreasurySpread': 'T10Y2Y', 'TED_Spread': 'TEDRATE',
        'Unemployment': 'UNRATE', 'Initial_Claims': 'ICSA',
        'GDP': 'GDP', 'GDP_Growth': 'A191RL1Q225SBEA',
        'FedFunds': 'FEDFUNDS', 'EFFR': 'EFFR',
        'Mortgage30Yr': 'MORTGAGE30US', 'Mortgage15Yr': 'MORTGAGE15US',
        'IndustrialProd': 'INDPRO', 'Capacity_Util': 'TCU',
        'RetailSales': 'RSXFS', 'HousingStarts': 'HOUST',
        'Building_Permits': 'PERMIT', 'Existing_Home_Sales': 'EXHOSLUSM495S',
        'ConsumerSenti': 'UMCSENT', 'ConsumerConf': 'CSCICP03USM665S',
        'ProdPriceIdx': 'PPIACO', 'Import_Price': 'IR',
        'Money_Supply': 'M2SL', 'Bank_Credit': 'TOTLL'
    })
    
    # Feature groups for organized computation
    feature_groups: Dict[FeatureGroup, List[str]] = Field(default={
        FeatureGroup.TECHNICAL: [
            'RSI14', 'RSI30', 'MACD_line', 'MACD_signal', 'MACD_hist',
            'ATR14', 'ATR30', 'BB_mid', 'BB_upper', 'BB_lower', 'BB_width',
            'OBV', '%K', '%D', 'MFI14', 'CCI20', 'Williams_%R',
            'ROC10', 'ROC20', 'Aroon_up', 'Aroon_down', 'ADX',
            'PSAR', 'CMO', 'TRIX'
        ],
        FeatureGroup.VOLATILITY: [
            'GARCH_vol', 'RealVol_5D', 'RealVol_21D', 'RealVol_63D',
            'Vol_Ratio', 'Vol_Surface', 'EWMA_vol', 'Parkinson_vol',
            'Garman_Klass_vol', 'Yang_Zhang_vol'
        ],
        FeatureGroup.FUNDAMENTAL: [
            'PE_Ratio', 'PB_Ratio', 'PS_Ratio', 'EV_EBITDA',
            'ROE', 'ROA', 'Debt_Equity', 'Current_Ratio',
            'Quick_Ratio', 'Operating_Margin'
        ]
    })
    
    # Advanced feature parameters
    lookback_periods: List[int] = Field(default=[5, 10, 21, 63, 126, 252])
    correlation_windows: List[int] = Field(default=[21, 63, 126])
    quantile_levels: List[float] = Field(default=[0.05, 0.25, 0.5, 0.75, 0.95])
    
    # Performance optimization flags
    use_numba: bool = Field(default=True, description="Use Numba JIT compilation")
    parallel_features: bool = Field(default=True, description="Parallel feature computation")
    feature_caching: bool = Field(default=True, description="Cache computed features")
    
    def get_all_feature_names(self) -> List[str]:
        """Get complete feature list with caching."""
        features = []
        
        # Base technical features
        features.extend(self.feature_groups[FeatureGroup.TECHNICAL])
        features.extend(self.feature_groups[FeatureGroup.VOLATILITY])
        
        # Market context
        features.extend([f'SPY_{feat}' for feat in ['RSI14', 'Return1', 'Volume_Ratio']])
        
        # Sector context
        features.extend([f'SECTOR_{feat}' for feat in ['RSI14', 'Return1', 'Relative_Strength']])
        
        # Macro features
        features.extend(list(self.macro_tickers_yf.keys()))
        features.extend(list(self.fred_indicators.keys()))
        
        # Engineered macro features
        fred_engineered = [
            f"{name}_{suffix}" for name in self.fred_indicators.keys()
            for suffix in ["DaysSinceUpdate", "InEventWindow", "ZScore", "Percentile"]
        ]
        features.extend(fred_engineered)
        
        # Relational features
        relational_features = [
            'Corr_Stock_FedFunds_60D', 'Corr_Stock_CPI_60D', 'Corr_Stock_VIX_21D',
            'TreasurySpread_RealVol_21D', 'Real_FedFunds', 'Stock_vs_GDP_Ratio',
            'CPI_ROC_3M', 'VIX_Term_Structure', 'Yield_Curve_Slope',
            'Credit_Spread', 'Liquidity_Index'
        ]
        features.extend(relational_features)

        # Market Regime Features
        regime_features = [
            'is_bull_regime',
            'is_bear_regime',
            'regime_diff',
            'regime_ratio'
        ]
        features.extend(regime_features)

        # Interaction Features
        interaction_features = [
            'Price_x_Volume',
            'RSI_div_ATR',
            'Bull_x_RSI',
            'Bear_x_ATR',
            'Vol_x_Dev',
            'Regime_Mom_Ratio'
        ]
        features.extend(interaction_features)
        
        # Time-based features
        time_features = [
            'DayOfWeek', 'MonthOfYear', 'QuarterOfYear', 'IsMonthEnd',
            'IsQuarterEnd', 'DaysToEarnings', 'DaysToFOMC', 'IsHoliday'
        ]
        features.extend(time_features)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(features))

# ==============================================================================
# SYSTEM CONFIGURATION
# ==============================================================================

class SystemConfig(OptimizedBaseModel):
    """System-wide configuration."""
    
    # Logging
    log_level: str = Field(default='INFO', description="Logging level")
    log_format: str = Field(
        default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        description="Log format"
    )
    
    # Paths
    data_dir: Path = Field(default=Path('data'), description="Data directory")
    model_dir: Path = Field(default=Path('models'), description="Model directory")
    log_dir: Path = Field(default=Path('logs'), description="Log directory")
    cache_dir: Path = Field(default=Path('.cache'), description="Cache directory")
    
    # Performance monitoring
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    memory_limit_gb: float = Field(default=16.0, gt=0, description="Memory usage limit")
    
    # API settings
    max_api_calls_per_minute: int = Field(default=100, ge=1, description="API rate limit")
    api_timeout_seconds: int = Field(default=30, ge=1, description="API timeout")
    
    # Resource management
    max_cpu_usage: float = Field(default=0.8, gt=0, le=1, description="Max CPU utilization")
    gc_threshold: int = Field(default=1000, ge=100, description="Garbage collection threshold")

    # Rich controls
    enable_progress_bars: bool = Field(default=True, description="Enable rich progress bars for long tasks.")

# ==============================================================================
# MAIN CONFIGURATION CLASS
# ==============================================================================

class Settings(OptimizedBaseModel):
    """Main configuration object - single source of truth."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    
    # Environment-specific overrides
    environment: str = Field(default='development', description="Environment: dev/staging/prod")
    debug_mode: bool = Field(default=True, description="Debug mode flag")
    
    # Hot-reloading for interpretability
    auto_reload_config: bool = Field(default=True, description="Auto-reload config on file change")
    config_file_path: Optional[str] = Field(default=None, description="Config file path for reloading")
    
    # Logging configuration
    suppress_third_party_logs: bool = Field(default=True, description="Suppress noisy third-party debug logs")
    app_log_level: str = Field('INFO', description="Log level for our application's code.")
    third_party_log_level: str = Field('WARNING', description="Default log level for all third-party libraries.")
    
    model_config = {
        'env_prefix': 'TRADING_',
        'case_sensitive': False,
    }

    def model_post_init(self, __context):
        """Called after the model is initialized. Applies overrides and configures logging."""
        env_overrides = self.get_environment_overrides()
        self.apply_overrides(env_overrides)
        # The configure_logging() is now called *after* overrides are applied.
    
    def configure_logging(self) -> None:
        """
        Configures application-wide logging using structlog correctly piped into rich.
        This version resolves the garbled output by ensuring only RichHandler
        is responsible for console rendering in development.
        """
        app_level = getattr(logging, self.app_log_level.upper(), logging.INFO)
        third_party_level = getattr(logging, self.third_party_log_level.upper(), logging.WARNING)

        # This is the shared processing pipeline for ALL logs.
        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # This processor must come last to format the log record for the handler.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ]

        # --- THE ENVIRONMENT-AWARE SWITCH ---
        if os.environ.get("LOG_FORMAT", "console").lower() == "json":
            # --- PRODUCTION: JSON LOGS ---
            # The final processor is a machine-readable JSON renderer.
            log_renderer = structlog.processors.JSONRenderer()
            handler = logging.StreamHandler(sys.stdout)
        else:
            # --- DEVELOPMENT: RICH CONSOLE LOGS ---
            # RichHandler will handle all rendering, including colors and formatting.
            handler = RichHandler(rich_tracebacks=True, tracebacks_suppress=[])
            
            # --- THE FIX IS HERE ---
            # We use a renderer that ONLY formats the key-value pairs into a string
            # WITHOUT adding any color codes. RichHandler will add the color.
            log_renderer = structlog.dev.ConsoleRenderer(colors=False, sort_keys=False)

        # Configure structlog to use the shared processors.
        structlog.configure(
            processors=shared_processors,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # This formatter connects the structlog pipeline to the standard logging handler.
        formatter = structlog.stdlib.ProcessorFormatter(
            # The `processor` is the final step that turns the log dictionary into a string.
            processor=log_renderer,
            # `foreign_pre_chain` is for logs from other libraries (e.g., yfinance).
            foreign_pre_chain=[
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
            ],
        )

        handler.setFormatter(formatter)
        
        # Configure the root logger to use ONLY our configured handler.
        root_logger = logging.getLogger()
        root_logger.handlers = [handler]
        root_logger.setLevel(third_party_level)

        # Set the level for our application's specific logger.
        app_logger = logging.getLogger("flint")
        app_logger.setLevel(app_level)
        app_logger.propagate = False # Prevent duplicate messages.

        log = structlog.get_logger("flint.config")
        log.info(
            "logging_configured", 
            renderer=log_renderer.__class__.__name__,
            handler=handler.__class__.__name__,
            app_level=self.app_log_level
        )

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides and re-configure logging."""
        for key, value in overrides.items():
            if '.' in key:
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)
        
        # Re-configure logging AFTER applying overrides to ensure the correct levels are used.
        # self.configure_logging()
        
    def reload_from_file(self, config_path: str) -> None:
        """Reload configuration from file for dynamic updates."""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                new_config = json.load(f)
            
            # Update fields dynamically
            for key, value in new_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            # Reconfigure logging after reload
            self.configure_logging()
                    
    def save_to_file(self, config_path: str) -> None:
        """Save current configuration to file."""
        with open(config_path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
            
    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides."""
        overrides = {}
        if self.environment == 'production':
            overrides.update({
                'debug_mode': False,
                'app_log_level': 'INFO',
                'third_party_log_level': 'WARNING', # Keep third-party logs quiet
                'models.xgboost.n_estimators': 1000,
            })
        elif self.environment == 'development':
            overrides.update({
                'debug_mode': True,
                'app_log_level': 'DEBUG', # Verbose logging for our app
                'third_party_log_level': 'INFO', # Slightly more verbose for third-party
            })
        return overrides
    
class RichLogRenderer:
    """
    A structlog processor that intelligently prepares log records for rendering
    by the rich library.
    
    It leaves rich-renderable objects (like Tables) untouched so that
    RichHandler can render them, and converts all other values to strings.
    """
    def __call__(self, logger, method_name, event_dict):
        # The __call__ method makes instances of this class behave like functions.
        for key, value in event_dict.items():
            # If the value is a rich object, let RichHandler deal with it.
            if isinstance(value, (Table, Text)):
                continue
            # Otherwise, ensure it's a clean string for printing.
            event_dict[key] = str(value)
        return event_dict



# ==============================================================================
# GLOBAL SETTINGS INSTANCE
# ==============================================================================

# Initialize with environment-specific settings
settings = Settings(environment='development')


# Logging is now automatically configured in Settings.__post_init__
# No need for manual logging.basicConfig() anymore!

# Create necessary directories
for directory in [settings.system.data_dir, settings.system.model_dir, 
                  settings.system.log_dir, settings.system.cache_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# UTILITY FUNCTIONS FOR DYNAMIC CONFIGURATION
# ==============================================================================

def get_device() -> str:
    """Automatically detect best available device."""
    if settings.models.xgboost.device == DeviceType.AUTO:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
    return "cpu"

def update_config_field(field_path: str, value: Any) -> None:
    """Update configuration field dynamically with validation."""
    try:
        keys = field_path.split('.')
        obj = settings
        
        for key in keys[:-1]:
            obj = getattr(obj, key)
        
        obj.update_field(keys[-1], value)
        logging.info(f"Updated {field_path} = {value}")
        
    except Exception as e:
        logging.error(f"Failed to update {field_path}: {e}")
        raise

def get_optimized_batch_size() -> int:
    """Calculate optimal batch size based on available memory."""
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Heuristic: use larger batches with more memory
        if available_memory_gb > 32:
            return min(512, settings.models.transformer_training.batch_size * 2)
        elif available_memory_gb > 16:
            return settings.models.transformer_training.batch_size
        else:
            return max(32, settings.models.transformer_training.batch_size // 2)
            
    except ImportError:
        return settings.models.transformer_training.batch_size

# Update batch size based on system capabilities
settings.models.transformer_training.batch_size = get_optimized_batch_size()
settings.models.xgboost.device = DeviceType(get_device())

# Export key objects for easy importing
__all__ = [
    'settings', 'Settings', 'ModelType', 'DeviceType', 'DataFrequency',
    'FeatureGroup', 'update_config_field', 'get_device'
]