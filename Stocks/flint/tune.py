# tune.py
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
import structlog

import numpy as np
import optuna
import torch
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

# Import custom models
from config import settings
from models import StockTransformer, prepare_sequences
from preprocess import prepare_and_cache_data

settings.environment = 'development'  # Set the environment to production

# Suppress the specific experimental warning AFTER all imports
warnings.filterwarnings(
    "ignore",
    category=ExperimentalWarning,
    message=".*Argument ``multivariate``.*"
)

STORAGE_PATH = Path("results/tuning.db").resolve()
STORAGE_URL = f"sqlite:///{STORAGE_PATH}"

# Configure logging
logger = structlog.get_logger(__name__)

def _ensure_dirs():
    Path("results/tuning").mkdir(parents=True, exist_ok=True)

def _save_params(ticker: str, model_type: str, params: dict):
    """Saves the best parameters to a single JSON file."""
    path = Path("results") / "tuning" / f"best_params_{ticker}.json"
    if path.exists():
        with path.open("r") as f:
            data = json.load(f)
    else:
        data = {}
    data[model_type] = params
    with path.open("w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Updated '{model_type}' params in {path}")

# --- TUNE XGBOOST  ---
def tune_xgboost(ticker: str, n_trials: int = 100, early_stop: int = 50):
    _ensure_dirs()
    sampler = TPESampler(multivariate=True)
    pruner  = HyperbandPruner(min_resource=1, max_resource=early_stop, reduction_factor=3)
    study = optuna.create_study(
        storage=STORAGE_URL, sampler=sampler, pruner=pruner,
        direction="minimize", study_name=f"xgb_tuning_{ticker}", load_if_exists=True
    )

    pkg = prepare_and_cache_data(ticker, settings.data.start_date, settings.data.end_date)
    X, y = pkg["X_train"], pkg["y_train"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"XGBoost Train/Val sizes: {len(X_train)}/{len(X_val)}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
            "device": "cuda", "random_state": 42, "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = XGBClassifier(**params, callbacks=[EarlyStopping(rounds=early_stop, save_best=True)])
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model.best_score

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best XGBoost validation logloss: {study.best_value:.6f}")
    _save_params(ticker, "xgboost", study.best_params)

# --- TUNE_TRANSFORMER ---
def tune_transformer(ticker: str, n_trials: int = 50, max_epochs: int = 10):
    _ensure_dirs()
    sampler = TPESampler(multivariate=True)
    pruner = HyperbandPruner(min_resource=1, max_resource=max_epochs, reduction_factor=3)
    study = optuna.create_study(
        storage=STORAGE_URL, sampler=sampler, pruner=pruner,
        direction="minimize", study_name=f"transformer_tuning_{ticker}", load_if_exists=True
    )
    logger.info(f"Loaded {len(study.trials)} existing trials for Transformer, running {n_trials} more")

    # --- DATA LOADING AND PREPARATION ---
    # 1. Load the single, unified data package
    pkg = prepare_and_cache_data(ticker, settings.data.start_date, settings.data.end_date)
    X_train_full, y_train_full = pkg["X_train"], pkg["y_train"]
    
    # 2. Create a validation set from the training data
    X_train_df, X_val_df, y_train_s, y_val_s = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    # 3. Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)

    # 4. Create time-series sequences
    X_seq_train, y_seq_train = prepare_sequences(X_train_scaled, y_train_s.values, settings.models.sequence_window_size)
    X_seq_val, y_seq_val = prepare_sequences(X_val_scaled, y_val_s.values, settings.models.sequence_window_size)
    
    # 5. Create PyTorch Datasets and DataLoaders
    train_ds = TensorDataset(torch.tensor(X_seq_train, dtype=torch.float32), torch.tensor(y_seq_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_seq_val, dtype=torch.float32), torch.tensor(y_seq_val, dtype=torch.long))
    
    logger.info(f"Transformer Train/Val sequence counts: {len(train_ds)}/{len(val_ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        # Define the hyperparameter search space for YOUR StockTransformer
        params = {
            'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
            'nhead': trial.suggest_categorical('nhead', [4, 8]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
        
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'] * 2)

        model = StockTransformer(
            input_features=X_seq_train.shape[2],
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            num_classes=2 # Binary classification
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')

        # Training loop for a given trial
        for epoch in range(max_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation loop
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
            
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            # Report intermediate results to Optuna for pruning
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_loss

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best Transformer validation loss: {study.best_value:.6f}")
    _save_params(ticker, "transformer", study.best_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for project models")
    parser.add_argument("ticker", help="Stock ticker for data")
    parser.add_argument("--model", "-m", choices=["xgb", "tf"], required=True)
    parser.add_argument("--trials", "-t", type=int, default=50)
    
    # XGBoost-specific
    parser.add_argument("--early-stop", "-es", type=int, default=50)
    
    # Transformer-specific
    parser.add_argument("--max-epochs", "-me", type=int, default=10)

    args = parser.parse_args()
    start = time.perf_counter()

    if args.model == "xgb":
        tune_xgboost(ticker=args.ticker, n_trials=args.trials, early_stop=args.early_stop)
    else:
        tune_transformer(ticker=args.ticker, n_trials=args.trials, max_epochs=args.max_epochs)

    end = time.perf_counter()
    logger.info(f"Total execution time: {end - start:.3f} seconds")