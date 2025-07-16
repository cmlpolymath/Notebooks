# tune.py

import argparse
import json
import logging
import sys

import optuna
from optuna.integration import XGBoostPruningCallback
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import config
from preprocess import prepare_and_cache_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def tune_xgboost(ticker: str, n_trials: int = 100, early_stop: int = 50):
    logging.info(f"XGBoost tuning for '{ticker}' ({n_trials} trials)")

    # 1. Load & split data
    data_pkg = prepare_and_cache_data(ticker, config.START_DATE, config.END_DATE)
    X_full, y_full = data_pkg["X_train"], data_pkg["y_train"]
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2,
        random_state=42, stratify=y_full
    )
    logging.info(f"Train/Validation sizes: {len(X_train)}/{len(X_val)}")

    def objective(trial: optuna.Trial) -> float:
        # 2. Set up pruning & early stopping callbacks
        pruning_cb = XGBoostPruningCallback(trial, "validation_0-logloss")
        early_stop_cb = EarlyStopping(rounds=early_stop, save_best=True)

        # 3. Hyperparameter search space
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": "cuda",
            "random_state": 42,
            "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        # 4. Instantiate with both callbacks in constructor
        model = XGBClassifier(
            **params,
            early_stopping_rounds=early_stop,
            callbacks=[pruning_cb, early_stop_cb]
        )

        # 5. Train with ONLY eval_set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # 6. Extract validation loss safely
        try:
            return model.best_score
        except AttributeError:
            return model.evals_result()['validation_0']['logloss'][-1]

    # 7. Run the Optuna study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=f"xgb_tuning_{ticker}"
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 8. Save best params
    logging.info(f"Best logloss: {study.best_value:.6f}")
    with open("best_xgb_params.json", "w") as fout:
        json.dump(study.best_params, fout, indent=4)
    logging.info("Saved best hyperparameters to best_xgb_params.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for XGBoost via Optuna"
    )
    parser.add_argument("ticker", help="Stock ticker for data")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of Optuna trials")
    parser.add_argument("--early-stop", type=int, default=50,
                        help="Rounds for early stopping")
    args = parser.parse_args()
    tune_xgboost(args.ticker, args.trials, args.early_stop)