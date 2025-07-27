# audit_logger.py
import duckdb
import json
from datetime import datetime
import git
from pathlib import Path
import numpy as np
import pandas as pd
import optuna

DB_PATH = Path('results/audit_log.duckdb')

def get_git_hash():
    """Gets the current git commit hash for versioning."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        return "not_a_git_repo"

def setup_audit_db():
    """Initializes the DuckDB database and the results table if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Use a 'with' statement for robust connection handling
    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                run_id UBIGINT,
                ticker VARCHAR,
                execution_timestamp TIMESTAMP,
                model_name VARCHAR,
                git_hash VARCHAR,
                run_config JSON,
                predictions JSON,
                metrics JSON,
                shap_importance JSON,
                PRIMARY KEY (run_id, ticker)
            );
        """)
        # Explicitly commit the table creation
        con.commit()
    print(f"Audit database ready at '{DB_PATH}'")

def numpy_safe_json_serializer(obj):
    """
    Custom JSON serializer for objects that may contain numpy types.
    This function is passed to the `default` argument of json.dumps().
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert arrays to lists
    if isinstance(obj, np.integer):
        return int(obj)      # Convert numpy integers to python int
    if isinstance(obj, np.floating):
        # Handle NaN separately before converting to float
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)     # Convert numpy bools to python bool
    
    # If the type is not recognized, let the default encoder raise the TypeError
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def log_analysis_result(ticker: str,
                        model_name: str,
                        run_config: dict,
                        predictions: dict,
                        test_index: pd.Index,
                        metrics: dict,
                        shap_importance: dict):
    """Logs the results of a single analysis run to the database."""
    
    # --- THE DEFINITIVE FIX ---
    # 1. Use a 'with' statement to ensure the connection is always closed.
    # 2. Use an explicit 'con.commit()' to force the write to disk.
    # This approach is version-agnostic and does not rely on special config keys.
    with duckdb.connect(str(DB_PATH)) as con:
        run_id = int(datetime.now().timestamp() * 1e9)
        execution_timestamp = datetime.now()
        git_hash = get_git_hash()
        
        predictions_payload = {
            "probabilities": predictions.get("probabilities", []),
            "dates": test_index.strftime('%Y-%m-%d').tolist()
        }
        
        # Serialize dicts to JSON strings for storage
        config_json = json.dumps(run_config, default=numpy_safe_json_serializer)
        predictions_json = json.dumps(predictions_payload, default=numpy_safe_json_serializer)
        metrics_json = json.dumps(metrics, default=numpy_safe_json_serializer)
        shap_json = json.dumps(shap_importance, default=numpy_safe_json_serializer)
        
        con.execute(
            """
            INSERT OR REPLACE INTO analysis_results 
            (run_id, ticker, execution_timestamp, model_name, git_hash, run_config, predictions, metrics, shap_importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [run_id, ticker, execution_timestamp, model_name, git_hash, config_json, predictions_json, metrics_json, shap_json]
        )
        
        con.commit()

def setup_tuning_table():
    """Creates Optuna tuning table in DuckDB if not exists"""
    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS tuning (
            study_id VARCHAR PRIMARY KEY,
            study_name VARCHAR,
            ticker VARCHAR,
            git_hash VARCHAR,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            best_params JSON,
            best_value DOUBLE,
            trials JSON,
            pareto_front JSON
        );
        """)
        con.commit()

def log_tuning_study(study: optuna.Study, study_name: str, ticker: str):
    """Logs Optuna study to DuckDB tuning table"""
    git_hash = get_git_hash()
    start_time = study.trials[0].datetime_start
    end_time = study.trials[-1].datetime_complete
    
    # Serialize trials data
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "number": trial.number,
            "params": trial.params,
            "value": trial.value,
            "duration": (trial.datetime_complete - trial.datetime_start).total_seconds(),
            "state": trial.state.name
        })
    
    # For multi-objective studies
    pareto_front = []
    if study._is_multi_objective():
        pareto_front = [{"values": t.values, "params": t.params} for t in study.best_trials]
    
    # Prepare data
    data = {
        "study_id": str(study._study_id),
        "study_name": study_name,
        "ticker": ticker,
        "git_hash": git_hash,
        "start_time": start_time,
        "end_time": end_time,
        "best_params": json.dumps(study.best_params),
        "best_value": study.best_value,
        "trials": json.dumps(trials_data),
        "pareto_front": json.dumps(pareto_front)
    }
    
    # Insert into DuckDB
    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("""
        INSERT OR REPLACE INTO tuning 
        (study_id, study_name, ticker, git_hash, start_time, end_time, best_params, best_value, trials, pareto_front)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, list(data.values()))
        con.commit()