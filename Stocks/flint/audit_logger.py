# audit_logger.py
import duckdb
import json
from datetime import datetime
import git
from pathlib import Path
import numpy as np

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

def log_analysis_result(ticker: str, model_name: str, run_config: dict, predictions: dict, metrics: dict, shap_importance: dict):
    """Logs the results of a single analysis run to the database."""
    
    # --- THE DEFINITIVE FIX ---
    # 1. Use a 'with' statement to ensure the connection is always closed.
    # 2. Use an explicit 'con.commit()' to force the write to disk.
    # This approach is version-agnostic and does not rely on special config keys.
    with duckdb.connect(str(DB_PATH)) as con:
        run_id = int(datetime.now().timestamp() * 1e9)
        execution_timestamp = datetime.now()
        git_hash = get_git_hash()
        
        # Serialize dicts to JSON strings for storage
        config_json = json.dumps(run_config, default=numpy_safe_json_serializer)
        predictions_json = json.dumps(predictions, default=numpy_safe_json_serializer)
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
        
        # Explicitly force the transaction to be written to the database file.
        con.commit()
        # The 'with' statement will handle closing the connection after the commit.