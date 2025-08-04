# models.py
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import RobustScaler
from collections import Counter
import shap
from config import settings
from sklearn.preprocessing import StandardScaler
import copy
import math


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
def train_enhanced_random_forest(X_train, y_train) -> tuple:
    """
    Trains a robust Random Forest classifier using a sophisticated pipeline.

    This pipeline includes:
    1.  Robust scaling to handle outliers.
    2.  ADASYN over-sampling to address class imbalance.
    3.  Domain-specific feature engineering.
    4.  Model-based feature selection.
    5.  Hyperparameter tuning with RandomizedSearchCV.

    All parameters are controlled via the central config file.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.

    Returns:
        A tuple containing:
        - The final trained model object.
        - The fitted scaler object.
        - The fitted feature selector object.
        - A dictionary of metadata for logging.
    """
    print("\n--- Training Enhanced Random Forest with ADASYN Pipeline ---")

    # Load configuration from the central settings object
    cfg = settings.models.random_forest

    # 1. Check class imbalance
    initial_dist = Counter(y_train)
    print(f"Initial class distribution: {initial_dist}")

    # 2. Time-series aware scaling
    print("Applying RobustScaler...")
    scaler = RobustScaler(quantile_range=cfg.scaler_quantile_range)
    X_train_scaled = scaler.fit_transform(X_train)

    # 3. ADASYN for class imbalance
    print("Applying ADASYN for minority class synthesis...")
    ada = ADASYN(
        sampling_strategy=cfg.adasyn_sampling_strategy,
        n_neighbors=cfg.adasyn_neighbors,
        random_state=42
    )
    X_resampled, y_resampled = ada.fit_resample(X_train_scaled, y_train)
    resampled_dist = Counter(y_resampled)
    print(f"Resampled class distribution: {resampled_dist}")

    # 4. Domain-specific feature engineering
    print("Creating financial interaction features...")
    feature_names = X_train.columns.tolist()
    
    # This internal function is safer as it operates on indices
    def _add_financial_interactions(X, names):
        X_out = X.copy()
        new_names = []
        # Price-volume interactions
        if 'Volume' in names and 'Close' in names:
            vol_idx = names.index('Volume')
            close_idx = names.index('Close')
            price_vol = X[:, close_idx] * X[:, vol_idx]
            X_out = np.hstack((X_out, price_vol.reshape(-1, 1)))
            new_names.append('Price_x_Volume')

        # Momentum-volatility interaction
        if 'RSI14' in names and 'ATR14' in names:
            rsi_idx = names.index('RSI14')
            atr_idx = names.index('ATR14')
            mom_vol = X[:, rsi_idx] / (X[:, atr_idx] + 1e-8)
            X_out = np.hstack((X_out, mom_vol.reshape(-1, 1)))
            new_names.append('RSI_div_ATR')
        
        return X_out, new_names

    X_resampled, new_feature_names = _add_financial_interactions(X_resampled, feature_names)
    final_feature_names = feature_names + new_feature_names

    # 5. Model-based feature selection
    print("Selecting most predictive features...")
    selector = SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        max_features=cfg.feature_selection_max_features,
        threshold=cfg.feature_selection_threshold
    )
    X_selected = selector.fit_transform(X_resampled, y_resampled)
    
    # Get the names of the selected features
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [final_feature_names[i] for i in selected_indices]
    print(f"Selected {X_selected.shape[1]} features: {selected_feature_names}")

    # 6. Hyperparameter tuning with RandomizedSearchCV
    print("Tuning hyperparameters with RandomizedSearchCV...")
    rf_base = RandomForestClassifier(
        n_estimators=cfg.base_n_estimators,
        max_depth=cfg.base_max_depth,
        min_samples_leaf=cfg.base_min_samples_leaf,
        max_features=cfg.base_max_features,
        n_jobs=-1,
        random_state=42,
        oob_score=True # Enable OOB score for model evaluation
    )

    rf_tuner = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=cfg.param_distributions,
        n_iter=cfg.hyperparam_tuning_iterations,
        cv=cfg.hyperparam_tuning_cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    rf_tuner.fit(X_selected, y_resampled)
    best_rf = rf_tuner.best_estimator_
    
    print(f"Best Parameters found: {rf_tuner.best_params_}")
    print(f"Best OOB Score: {best_rf.oob_score_:.4f}")

    # 7. Prepare metadata for logging
    run_config = {
        "model_type": "RandomForest_ADASYN_Finance",
        "best_params": rf_tuner.best_params_,
        "features_selected_count": X_selected.shape[1],
        "selected_feature_names": selected_feature_names,
        "oob_score": best_rf.oob_score_,
        "class_distribution_initial": dict(initial_dist),
        "class_distribution_resampled": dict(resampled_dist),
        "sampling_strategy": "ADASYN",
        "config_params": cfg.model_dump() # Log the config used
    }

    # The final model is already trained by RandomizedSearchCV on the full data
    # when refit=True (the default). No need to call .fit() again.
    return best_rf, scaler, selector, _add_financial_interactions, run_config

def train_xgboost(X_train, y_train, params: dict | None = None):
    """
    Trains an XGBoost model using either provided parameters or defaults from config.
    """
    print("Training XGBoost model...")
    # If no specific params are passed, use the defaults from the config file
    if params is None:
        params = settings.models.xgboost.model_dump(exclude_unset=True)
        
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(X_train, y_train)
    xgb_model.get_booster().set_param({'nthread' : os.cpu_count() or 1, "device":"cpu"})  # Use all available CPU threads
    return xgb_model

def get_shap_importance(xgb_model, X_test):
    print("Calculating SHAP values...")
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
    print("Training Transformer model...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    seq_window = settings.models.sequence_window_size
    X_seq_train, y_seq_train = prepare_sequences(X_train_scaled, y_train, seq_window)
    X_seq_test, y_seq_test = prepare_sequences(X_test_scaled, y_test, seq_window)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        print(f"Using gradient accumulation with {accumulation_steps} steps.")
        print(f"Physical batch size: {batch_size}, Effective batch size: {batch_size * accumulation_steps}")

    model = StockTransformer(
        input_features=X_seq_train.shape[2],
        d_model=model_params.get('d_model', 64),
        nhead=model_params.get('nhead', 4),
        num_layers=model_params.get('num_layers', 4),
        num_classes=model_params.get('num_classes', 2),  # Binary classification
        dropout=model_params.get('dropout', 0.1) 
    ).to(device)
        
    try:
        model = torch.compile(model, backend="aot_eager")
        print("Model compiled successfully with 'aot_eager' backend.")
    except Exception as e:
        print(f"Could not compile model, will run in eager mode. Error: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.get('lr', 1e-4), weight_decay=train_cfg.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    train_dataset = TensorDataset(torch.tensor(X_seq_train, dtype=torch.float32), torch.tensor(y_seq_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_seq_test, dtype=torch.float32), torch.tensor(y_seq_test, dtype=torch.long))
    
    # Use the potentially smaller, safer batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=min(4, os.cpu_count() or 1))    
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, pin_memory=True, num_workers=min(0, os.cpu_count() or 1))

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
                print(f"Epoch {epoch}: Loss is NaN. Stopping training.")
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
        
        print(f"Epoch {epoch}/{epochs} -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"   -> New best validation loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break

    print("Loading best model weights...")
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