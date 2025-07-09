# models.py
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
import shap
import config
from sklearn.preprocessing import StandardScaler
import copy
import math # ADDED for PositionalEncoding

# --- ADDED: Positional Encoding Class ---
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
def train_xgboost(X_train, y_train):
    print("Training XGBoost model...")
    xgb_model = XGBClassifier(**config.XGB_PARAMS)
    xgb_model.fit(X_train, y_train)
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

def train_transformer(X_train, y_train, X_test, y_test):
    print("Training Transformer model...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_seq_train, y_seq_train = prepare_sequences(X_train_scaled, y_train, config.SEQUENCE_WINDOW_SIZE)
    X_seq_test, y_seq_test = prepare_sequences(X_test_scaled, y_test, config.SEQUENCE_WINDOW_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    params = config.TRANSFORMER_PARAMS
    train_cfg = config.TRANSFORMER_TRAINING
    
    model = StockTransformer(
        input_features=X_seq_train.shape[2],
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers'],
        num_classes=params['num_classes'],
        dropout=0.1
    )
    model.to(device)
    
    try:
        model = torch.compile(model, backend="aot_eager")
        print("Model compiled successfully with 'aot_eager' backend.")
    except Exception as e:
        print(f"Could not compile model, will run in eager mode. Error: {e}")


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    
    # --- THE FIX IS HERE ---
    # The 'verbose' argument was removed in newer PyTorch versions.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    train_dataset = TensorDataset(torch.tensor(X_seq_train, dtype=torch.float32), torch.tensor(y_seq_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_seq_test, dtype=torch.float32), torch.tensor(y_seq_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, pin_memory=True, num_workers=min(4, os.cpu_count() or 1))    
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'] * 2, pin_memory=True, num_workers=min(4, os.cpu_count() or 1))

    scaler_amp = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = train_cfg['patience']
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, train_cfg['epochs'] + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
            
            if torch.isnan(loss):
                print(f"Epoch {epoch}: Loss is NaN. Stopping training.")
                model.load_state_dict(best_model_wts)
                return model, device, scaler, y_seq_test

            scaler_amp.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
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
        
        print(f"Epoch {epoch}/{train_cfg['epochs']} -> Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
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
    
    X_test_scaled = scaler.transform(X_test)
    X_seq_test, _ = prepare_sequences(X_test_scaled, np.zeros(len(X_test_scaled)), config.SEQUENCE_WINDOW_SIZE)

    test_dataset = TensorDataset(torch.tensor(X_seq_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=config.TRANSFORMER_TRAINING['batch_size'] * 2)

    all_probabilities = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch[0].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                logits = model(X_batch)
            probabilities = torch.softmax(logits, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())
            
    return np.concatenate(all_probabilities, axis=0)