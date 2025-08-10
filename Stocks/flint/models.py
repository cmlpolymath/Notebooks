# models.py
import copy
import math
import os
import structlog
import time
import threading
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from config import settings

logger = structlog.get_logger(__name__)

class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # The original paper adds PE to the input.
        # We need to permute x to [seq_len, batch_size, d_model] for broadcasting with pe
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2) # Permute back to [batch_size, seq_len, d_model]
        return self.dropout(x)


# 1. Transformer Model Definition (MODIFIED)
class StockTransformer(nn.Module):
    def __init__(self, input_features, d_model, nhead, num_layers, num_classes, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.feature_embed = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout) # ADDED
        
        # ADDED dropout to the encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model
        
        # ADDED LayerNorm for stability before the final classification
        self.pre_classifier_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.feature_embed(x) * math.sqrt(self.d_model) # Scale embedding
        x = self.pos_encoder(x) # ADDED positional encoding
        x = self.transformer_encoder(x)
        
        # Take the output of the last token in the sequence for classification
        out = x[:, -1, :]
        out = self.pre_classifier_norm(out) # ADDED LayerNorm
        out = self.fc_out(out)
        return out

# 2. Model Training and Prediction Functions
def train_enhanced_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Trains a robust Random Forest classifier using a sophisticated pipeline with time-aware validation 
    and model stacking. Expects pre-engineered features including market regimes and interaction features.
    """
    
    
    cfg = settings.models.random_forest
    logger.info("\n--- Training Enhanced Random Forest with Time-Aware Validation & Stacking ---")
    
    # Establish a single source of truth for n_jobs based on the correct config flag.
    if settings.system.enable_profiling:
        logger.warning("profiling_mode_active", reason="All model training will be single-threaded for stability.")
        n_jobs_safe = 1
    else:
        n_jobs_safe = -1

    # 1. Validate expected pre-engineered features are present
    expected_features = ['is_bull_regime', 'is_bear_regime', 'Price_x_Volume', 'RSI_div_ATR']
    available_features = [feat for feat in expected_features if feat in X_train.columns]
    missing_features = [feat for feat in expected_features if feat not in X_train.columns]
    
    logger.info(f"Pre-engineered features found: {available_features}")
    if missing_features:
        logger.warn(f"Missing expected features: {missing_features}")
    
    # 2. Check class imbalance
    initial_dist = Counter(y_train)
    logger.info(f"Initial class distribution: {initial_dist}")
    
    # 3. Time-series aware scaling
    logger.info("Applying RobustScaler...")
    scaler = RobustScaler(quantile_range=cfg.scaler_quantile_range)
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 4. ROBUST ADAPTIVE ADASYN for class imbalance (FIX #1)
    logger.info("Applying robust adaptive ADASYN for minority class synthesis...")
    
    # Check minority class size to determine if ADASYN is viable
    class_counts = Counter(y_train)
    minority_class_count = min(class_counts.values())
    majority_class_count = max(class_counts.values())
    
    # Calculate imbalance ratio to determine if resampling is needed
    imbalance_ratio = majority_class_count / minority_class_count
    logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    # Skip ADASYN if classes are already balanced (ratio < 1.5)
    if imbalance_ratio < 1.5:
        logger.info(f"Classes are already balanced (ratio: {imbalance_ratio:.2f}). Skipping ADASYN.")
        X_resampled, y_resampled = X_train_scaled, y_train
        resampled_dist = initial_dist
        sampling_applied = "None_Balanced"
    # Apply ADASYN with error handling
    elif minority_class_count > cfg.adasyn_neighbors:
        try:
            ada = ADASYN(
                sampling_strategy=cfg.adasyn_sampling_strategy,
                n_neighbors=cfg.adasyn_neighbors,
                random_state=42
            )
            X_resampled, y_resampled = ada.fit_resample(X_train_scaled, y_train)
            resampled_dist = Counter(y_resampled)
            logger.info("Standard ADASYN applied successfully")
            sampling_applied = "ADASYN_Standard"
        except ValueError as e:
            logger.info(f"Standard ADASYN failed: {e}")
            logger.info("Falling back to no resampling...")
            X_resampled, y_resampled = X_train_scaled, y_train
            resampled_dist = initial_dist
            sampling_applied = "None_ADASYN_Failed"
    elif minority_class_count > 1:
        # Adaptive n_neighbors: use minority_class_count - 1
        adaptive_neighbors = minority_class_count - 1
        logger.info(f"Minority class size ({minority_class_count}) requires adaptive n_neighbors: {adaptive_neighbors}")
        
        try:
            ada = ADASYN(
                sampling_strategy=cfg.adasyn_sampling_strategy,
                n_neighbors=adaptive_neighbors,
                random_state=42
            )
            X_resampled, y_resampled = ada.fit_resample(X_train_scaled, y_train)
            resampled_dist = Counter(y_resampled)
            logger.info("Adaptive ADASYN applied successfully")
            sampling_applied = "ADASYN_Adaptive"
        except ValueError as e:
            logger.info(f"Adaptive ADASYN failed: {e}")
            logger.info("Falling back to no resampling...")
            X_resampled, y_resampled = X_train_scaled, y_train
            resampled_dist = initial_dist
            sampling_applied = "None_ADASYN_Failed"
    else:
        # Skip ADASYN if minority class is too small (≤1 sample)
        logger.warn(f"Minority class too small ({minority_class_count} samples). Skipping ADASYN resampling.")
        X_resampled, y_resampled = X_train_scaled, y_train
        resampled_dist = initial_dist
        sampling_applied = "None_TooSmall"
    
    logger.info(f"Final class distribution: {resampled_dist}")
    logger.info(f"Dataset size after resampling: {X_resampled.shape[0]} samples, {X_resampled.shape[1]} features")
    
    # Convert back to DataFrame with original feature names to avoid warnings
    X_resampled_df = pd.DataFrame(X_resampled, columns=X_train.columns)
    
    # 5. Model-based feature selection with proper DataFrame handling
    logger.info("Selecting most predictive features...")
    selector = SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs_safe),
        max_features=cfg.feature_selection_max_features,
        threshold=cfg.feature_selection_threshold
    )
    selector.fit(X_resampled_df, y_resampled)
    selected_feature_names = X_resampled_df.columns[selector.get_support()].tolist()
    X_selected_df = X_resampled_df[selected_feature_names]
    logger.info("feature_selection_complete", count=len(selected_feature_names))

    
    logger.info(f"Selected {X_selected_df.shape[1]} features: {selected_feature_names}")
    
    # 6. Time-aware cross-validation setup
    logger.info(f"Setting up TimeSeriesSplit with {cfg.tscv_splits} splits and {cfg.tscv_gap} day gap...")
    tscv = TimeSeriesSplit(n_splits=cfg.tscv_splits, gap=cfg.tscv_gap)
    
    # Print expected computation load
    total_fits = cfg.hyperparam_tuning_iterations * cfg.tscv_splits
    logger.info(f"Expected total model fits: {total_fits}")
    logger.info(f"Estimated training time: {total_fits * 2:.2f}-{total_fits * 8:.2f} seconds")    
    
    # 7. ULTRA-ROBUST HYPERPARAMETER TUNING with time-aware validation (FIX #2)
    rf_base = RandomForestClassifier(
        n_estimators=cfg.base_n_estimators,
        max_depth=cfg.base_max_depth,
        min_samples_leaf=cfg.base_min_samples_leaf,
        max_features=cfg.base_max_features,
        n_jobs=n_jobs_safe,
        random_state=42,
        oob_score=True
    )
    
    logger.info("Starting ultra-robust hyperparameter optimization...")
    logger.info(f"Search space: {cfg.hyperparam_tuning_iterations} candidates × {cfg.tscv_splits} folds = {total_fits} total fits")

    # Check if we have both classes in the dataset - if not, use simpler scoring
    unique_classes = len(set(y_resampled))
    
    if unique_classes < 2:
        logger.warn("Only one class present in dataset. Using accuracy scoring only.")
        scoring_strategy = 'accuracy'
        refit_strategy = 'accuracy'
    else:
        # Custom scorer that handles single-class folds gracefully
        from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
        
        def safe_roc_auc_score(y_true, y_pred_proba, **kwargs):
            """ROC AUC that returns 0.5 (random chance) for single-class cases"""
            try:
                if len(set(y_true)) < 2:
                    return 0.5  # Random chance baseline for single-class folds
                else:
                    return roc_auc_score(y_true, y_pred_proba)
            except Exception:
                return 0.5  # Fallback to random chance on any error
        
        # Create safe scorers - let's use a simpler approach
        scoring_strategy = {
            'roc_auc': make_scorer(safe_roc_auc_score, needs_proba=True),
            'accuracy': 'accuracy'  # Use built-in accuracy scorer
        }
        refit_strategy = 'roc_auc'

    rf_tuner = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=cfg.param_distributions,
        n_iter=cfg.hyperparam_tuning_iterations,
        cv=tscv,
        scoring=scoring_strategy,
        refit=refit_strategy,
        n_jobs=n_jobs_safe,
        verbose=0,
        random_state=42,
        error_score=0.5
    )


    # Config-driven, profiler-safe progress feedback
    import time
    
    start_time = time.time()
    total_fits = cfg.hyperparam_tuning_iterations * cfg.tscv_splits
    
    # Check if profiling is enabled in your settings
    if hasattr(settings.system, 'enable_profiling') and settings.system.enable_profiling:
        # PROFILING MODE - Zero threading, completely safe
        print(f"🔍 Profiling mode active - using thread-safe feedback")
        print(f"⏳ Starting hyperparameter optimization: {total_fits} total fits")
        
        # Simple, blocking execution - no threads at all
        rf_tuner.fit(X_selected_df, y_resampled)
        
    else:
        # NORMAL MODE - Enhanced user experience with threading
        import sys
        import threading
        from itertools import cycle
        
        training_complete = threading.Event()
        
        print(f"🚀 Starting hyperparameter optimization: {total_fits} total fits")
        
        def console_progress():
            """Thread-safe progress spinner for normal execution"""
            spinner = cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'])
            
            while not training_complete.is_set():
                elapsed = time.time() - start_time
                spinner_char = next(spinner)
                
                # Simple elapsed time display
                sys.stdout.write(f"\r{spinner_char} Training in progress... {elapsed:.1f}s elapsed")
                sys.stdout.flush()
                time.sleep(0.3)  # Update ~3 times per second
        
        # Start progress thread only in normal mode
        progress_thread = threading.Thread(target=console_progress, daemon=True)
        progress_thread.start()
        
        try:
            # Run the training
            rf_tuner.fit(X_selected_df, y_resampled)
        finally:
            # Always clean up the spinner, even if training fails
            training_complete.set()
            time.sleep(0.5)  # Let spinner finish gracefully
            
            # Clear the progress line
            sys.stdout.write("\r" + " " * 60 + "\r")
            sys.stdout.flush()
    
    # Get results (same for both modes)
    best_rf = rf_tuner.best_estimator_
    elapsed_time = time.time() - start_time

    logger.info(f"Best Parameters: {rf_tuner.best_params_}")
    logger.info(f"Best Random Forest CV Score: {rf_tuner.best_score_:.4f}")
    logger.info(f"Random Forest OOB Score: {best_rf.oob_score_:.4f}")

    logger.info(f"Hyperparameter search completed in {elapsed_time:.2f} seconds")
    # 8. Model Stacking (simplified to avoid redundant validation)
    final_model = best_rf  # Default to RF if stacking disabled
    stacking_cv_score = None
    
    if cfg.enable_stacking:
        logger.info("stacking_start")
        
        # Use the safe n_jobs setting here
        lgb_base = lgb.LGBMClassifier(
            max_depth=cfg.lgb_max_depth,
            learning_rate=cfg.lgb_learning_rate,
            n_estimators=cfg.lgb_n_estimators,
            num_leaves=cfg.lgb_num_leaves,
            random_state=42,
            n_jobs=n_jobs_safe,
            verbose=-1,
            force_col_wise=True
        )

        # Use regular KFold for stacking instead of TimeSeriesSplit
        # TimeSeriesSplit doesn't work with StackingClassifier's cross_val_predict
        from sklearn.model_selection import KFold
        rf_for_stacking = RandomForestClassifier(**best_rf.get_params())

        stacking_cv = KFold(n_splits=3, shuffle=False)
        
        stacking_clf = StackingClassifier(
            estimators=[
                ('rf', rf_for_stacking),
                ('lgb', lgb_base)
            ],
            final_estimator=LogisticRegression(
                C=cfg.stack_final_estimator_C,
                random_state=42,
                max_iter=1000
            ),
            cv=stacking_cv,
            n_jobs=n_jobs_safe # Use the safe setting for the stacker itself
        )
        
        logger.info("Training StackingClassifier...")
        stacking_clf.fit(X_selected_df, y_resampled)
        final_model = stacking_clf
        
        stacking_cv_score = rf_tuner.best_score_
        logger.info("stacking_complete")
    
    # 9. Prepare enhanced metadata for logging
    run_config = {
        "model_type": "EnhancedRandomForest_TimeAware_Stacking" if cfg.enable_stacking else "EnhancedRandomForest_TimeAware",
        "stacking_enabled": cfg.enable_stacking,
        "best_rf_params": rf_tuner.best_params_,
        "best_rf_cv_score": rf_tuner.best_score_,
        "stacking_cv_score": stacking_cv_score,
        "features_selected_count": X_selected_df.shape[1],
        "selected_feature_names": selected_feature_names,
        "available_engineered_features": available_features,
        "missing_engineered_features": missing_features,
        "rf_oob_score": best_rf.oob_score_,
        "class_distribution_initial": dict(initial_dist),
        "class_distribution_resampled": dict(resampled_dist),
        "time_series_cv_splits": cfg.tscv_splits,
        "time_series_cv_gap": cfg.tscv_gap,
        "sampling_strategy": sampling_applied,
        "config_params": cfg.model_dump()
    }
    
    return final_model, scaler, selector, run_config

def train_xgboost(X_train, y_train, params: dict | None = None):
    """
    Trains an XGBoost model using either provided parameters or defaults from config.
    """
    logger.info("Training XGBoost model...")
    # If no specific params are passed, use the defaults from the config file
    if params is None:
        params = settings.models.xgboost.model_dump(exclude_unset=True)
        
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    xgb_model.get_booster().set_param({'nthread' : os.cpu_count() or 1, "device":"cpu"})  # Use all available CPU threads
    return xgb_model

def get_shap_importance(xgb_model, X_test):
    logger.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        # For binary classification, shap_values is a list of two arrays (for class 0 and 1)
        # We are interested in the importances for the "Up" class (class 1)
        shap_values = shap_values[1]
    return np.mean(np.abs(shap_values), axis=0)

def prepare_sequences(X, y, window_size: int):
    """
    Vectorized function to create overlapping sequences for time-series models.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y)

    unique_labels = np.unique(y_arr)
    if np.issubdtype(y_arr.dtype, np.integer) and set(unique_labels).issubset({0, 1}):
        y_arr = y_arr.astype(np.uint8)
    else:
        y_arr = y_arr.astype(np.int64)

    if X_arr.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features); got ndim={X_arr.ndim}")
    n_samples, n_features = X_arr.shape
    if n_samples < window_size:
        raise ValueError(
            f"Need at least window_size samples to create one sequence: "
            f"n_samples={n_samples}, window_size={window_size}"
        )
    if y_arr.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of rows")

    windows = sliding_window_view(X_arr, window_shape=(window_size, n_features))
    windows = np.squeeze(windows, axis=1)

    y_seq = y_arr[window_size - 1:]

    X_seq = np.ascontiguousarray(windows)
    y_seq = np.ascontiguousarray(y_seq)

    return X_seq, y_seq

def train_transformer(X_train, y_train, X_test, y_test, training_params: dict | None = None):
    log = logger.bind(model_name="Transformer")
    log.info("training_start")
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    seq_window = settings.models.sequence_window_size
    X_seq_train, y_seq_train = prepare_sequences(X_train_scaled, y_train, seq_window)
    X_seq_test, y_seq_test = prepare_sequences(X_test_scaled, y_test, seq_window)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device_setup", device=str(device))

    model_params = settings.models.transformer_arch.model_dump(exclude_unset=True)
    
    if training_params is None:
        train_cfg = settings.models.transformer_training.model_dump(exclude_unset=True)
    else:
        train_cfg = training_params
    
    # --- GRADIENT ACCUMULATION ---
    # Get the number of accumulation steps from the config. Default to 1 if not specified.
    accumulation_steps = train_cfg.get('accumulation_steps', 1)
    
    # Adjust batch size if tuned params are too large.
    # This is a safeguard against OOM even with accumulation.
    batch_size = min(train_cfg.get('batch_size', 256), 128) # Cap batch size at a safe limit
    if accumulation_steps > 1:
        logger.info(f"Using gradient accumulation with {accumulation_steps} steps.")
        logger.info(f"Physical batch size: {batch_size}, Effective batch size: {batch_size * accumulation_steps}")

    model = StockTransformer(
        input_features=X_seq_train.shape[2],
        d_model=model_params.get('d_model', 64),
        nhead=model_params.get('nhead', 4),
        num_layers=model_params.get('num_layers', 4),
        num_classes=model_params.get('num_classes', 2),  # Binary classification
        dropout=model_params.get('dropout', 0.1) 
    ).to(device)
        
    # Check the global settings object to see if we are in profiling mode.
    if not settings.system.enable_profiling:
        try:
            model = torch.compile(model, backend="aot_eager")
            log.info("torch_compile_enabled", backend="aot_eager")
        except Exception as e:
            log.warning("torch_compile_failed", error=str(e))
    else:
        log.warning("enable_profiling_active", reason="torch.compile has been disabled.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get('lr', 1e-4), weight_decay=train_cfg.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    train_dataset = TensorDataset(torch.tensor(X_seq_train, dtype=torch.float32), torch.tensor(y_seq_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_seq_test, dtype=torch.float32), torch.tensor(y_seq_test, dtype=torch.long))
    
    if settings.system.enable_profiling:
        num_workers = 0
        log.warning("enable_profiling_active", reason="DataLoader multi-processing has been disabled.")
    else:
        num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)    
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, pin_memory=True, num_workers=num_workers)

    scaler_amp = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = train_cfg.get('patience', 5)
    best_model_wts = copy.deepcopy(model.state_dict())
    
    epochs = train_cfg.get('epochs', 50)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        # --- GRADIENT ACCUMULATION ---
        # The optimizer is cleared only once at the beginning of the accumulation cycle.
        optimizer.zero_grad(set_to_none=True)

        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # --- GRADIENT ACCUMULATION ---
                # Normalize the loss by the number of accumulation steps.
                # This ensures the magnitude of the final accumulated gradient is correct.
                loss = loss / accumulation_steps
            
            if torch.isnan(loss):
                logger.info(f"Epoch {epoch}: Loss is NaN. Stopping training.")
                model.load_state_dict(best_model_wts)
                return model, device, scaler, y_seq_test

            # Accumulate the scaled loss
            scaler_amp.scale(loss).backward()
            
            total_loss += loss.item() * accumulation_steps # Un-scale for logging

            # --- GRADIENT ACCUMULATION ---
            # Perform the optimizer step only after accumulating gradients for `accumulation_steps` batches.
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update the model weights
                scaler_amp.step(optimizer)
                scaler_amp.update()
                
                # Clear the gradients for the next accumulation cycle
                optimizer.zero_grad(set_to_none=True)
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch}/{epochs} -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(f"   -> New best validation loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break

    logger.info("Loading best model weights...")
    model.load_state_dict(best_model_wts)
        
    return model, device, scaler, y_seq_test

def predict_transformer(model, device, scaler, X_test):
    """Generates predictions from a trained transformer model."""
    model.eval()
    
    seq_window = settings.models.sequence_window_size
    X_test_scaled = scaler.transform(X_test)
    X_seq_test, _ = prepare_sequences(X_test_scaled, np.zeros(len(X_test_scaled)), seq_window)

    test_dataset = TensorDataset(torch.tensor(X_seq_test, dtype=torch.float32))
    # Use a fixed, reasonable batch size for prediction to avoid OOM
    pred_batch_size = settings.models.transformer_training.batch_size * 2
    test_loader = DataLoader(test_dataset, batch_size=pred_batch_size)


    all_probabilities = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = model(X_batch)
            probabilities = torch.softmax(logits, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())
            
    return np.concatenate(all_probabilities, axis=0)