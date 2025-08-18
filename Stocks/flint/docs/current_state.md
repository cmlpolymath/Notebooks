---

### **1. Project Overview**

Project Flint is a sophisticated, modular Python pipeline for algorithmic trading analysis that has been upgraded with a professional-grade `structlog` and `rich` logging system for enhanced observability. The system ingests and caches market data, engineers a comprehensive feature set including TA-Lib candlestick patterns, and trains a suite of advanced, profiler-aware machine learning models. All operations are centrally configured and results are logged to a DuckDB database, providing a robust foundation for the primary strategic goal of migrating the current CLI to a modern web UI and API.

### **2. Script Inventory**

| Script | Responsibility | Key Functions |
|---|---|---|
| `config.py` | Single source of truth for all parameters and settings. | - `Settings` class<br>- `configure_logging()` |
| `run.py` | Main entry point for orchestrating a complete analysis run. | - `main()`<br>- `run_analysis()` |
| `data_handler.py` | Fetches and caches raw market data from yfinance. | - `get_stock_data()`<br>- `get_macro_data()` |
| `economics.py` | Specialized handler for fetching macroeconomic data from FRED. | - `fetch_macro_indicators()` |
| `preprocess.py` | Assembles raw data and orchestrates feature engineering. | - `prepare_and_cache_data()` |
| `data_store.py` | Manages storage of processed data artifacts (`.parquet`, `.blosc2`). | - `B2Data.save_data_package()`<br>- `B2Data.load_data_package()` |
| `feature_engineering.py` | Contains all feature creation logic. | - `FeatureCalculator.add_all_features()` |
| `aggressor.py` | Provides high-performance, Numba-optimized feature calculations. | - `AggressorFeatures.add_hurst()`<br>- `AggressorFeatures.add_fractal_dimension()` |
| `models.py` | Defines and trains all machine learning models. | - `train_enhanced_random_forest()`<br>- `train_transformer()`<br>- `train_xgboost()` |
| `predictors.py` | Contains the Monte Carlo simulation engine. | - `MonteCarloTrendFilter.fit()`<br>- `MonteCarloTrendFilter.predict()` |
| `tune.py` | Standalone script for hyperparameter tuning with Optuna. | - `tune_xgboost()`<br>- `tune_transformer()` |
| `validate.py` | Standalone script for robust, walk-forward validation. | - `run_walk_forward_validation()` |
| `audit_logger.py` | Handles writing all analysis results to the DuckDB database. | - `log_analysis_result()` |
| `visualization.py` | (Non-functional) Intended to visualize results. | N/A |

### **3. Model & API Specifications**

#### **Models Used**

-   **Model name/type:** `Enhanced Random Forest` (with ADASYN and Stacking)
    -   **Location:** `models.py`, `train_enhanced_random_forest()`
    -   **Input:** `X_train` (pandas DataFrame), `y_train` (pandas Series)
    -   **Output:** Trained `StackingClassifier` object, `scaler`, `selector`, `run_config`
    -   **Training data source:** `data_store.py` (`B2Data` class)

-   **Model name/type:** `XGBoost`
    -   **Location:** `models.py`, `train_xgboost()`
    -   **Input:** `X_train` (pandas DataFrame), `y_train` (pandas Series)
    -   **Output:** Trained `XGBClassifier` object
    -   **Training data source:** `data_store.py` (`B2Data` class)

-   **Model name/type:** `Stock Transformer`
    -   **Location:** `models.py`, `train_transformer()`
    -   **Input:** `X_train` (NumPy array), `y_train` (NumPy array)
    -   **Output:** Trained PyTorch `StockTransformer` object, `device`, `scaler`, `y_seq_test`
    -   **Training data source:** `data_store.py` (`B2Data` class)

-   **Model name/type:** `Monte Carlo Trend Filter`
    -   **Location:** `predictors.py`, `MonteCarloTrendFilter.fit()`
    -   **Input:** `df_features` (pandas DataFrame)
    -   **Output:** Fitted `MonteCarloTrendFilter` object
    -   **Training data source:** `data_store.py` (`B2Data` class)

#### **API Interactions**

-   **Service:** `yfinance` (Yahoo Finance)
    -   **Endpoint/operations used:** `yf.download()`, `yf.Ticker().info`, `yf.Ticker().financials`, etc.
    -   **Authentication method:** N/A (Public API)

-   **Service:** `FRED` (Federal Reserve Economic Data)
    -   **Endpoint/operations used:** `api.stlouisfed.org/fred/series/observations`
    -   **Authentication method:** API Key (managed via `.env` file)

-   **Service:** Local Filesystem (Parquet, Blosc2, JSON)
    -   **Endpoint/operations used:** `pandas.read_parquet()`, `blosc2.unpack_array()`, `json.load()`
    -   **Authentication method:** N/A

### **4. Pending Features**

1.  **[HIGH PRIORITY] UI/API Implementation:**
    -   **Description:** Replace the current command-line interface with a modern, interactive web UI (likely using **NiceGUI**) and a programmatic REST API (using **FastAPI**). This is the primary strategic goal to make the tool usable and accessible.
    -   **Blockers:** Requires dedicated development time for UI design and API endpoint structuring.
    -   **Estimated complexity:** [L]

2.  **[HIGH PRIORITY] Feature Calculation Optimization:**
    -   **Description:** Refactor the `FeatureCalculator` to support **incremental, "stateful" updates**. This will prevent the recalculation of the entire 10-year feature set each day and dramatically speed up daily runs.
    -   **Blockers:** Requires careful handling of path-dependent features like EMA and OBV.
    -   **Estimated complexity:** [M]

3.  **[MEDIUM] CLI Enhancement with Typer:**
    -   **Description:** Migrate all `argparse`-based CLIs (`run.py`, `tune.py`, etc.) to use **Typer**. This will simplify the code, provide automatic help generation, and allow for a more professional, unified command structure.
    -   **Blockers:** None, this is a direct refactoring task.
    -   **Estimated complexity:** [S]

4.  **[MEDIUM] Visualization Module (`visualization.py`):**
    -   **Description:** Build a functional dashboard to visualize the results stored in the `audit_log.duckdb` database. This was previously attempted with Dash but should now be built using the chosen UI framework (e.g., NiceGUI).
    -   **Blockers:** Depends on the decision from the UI/API implementation.
    -   **Estimated complexity:** [M]

5.  **[MEDIUM] Data Storage Migration for `tune.py`:**
    -   **Description:** The `tune.py` script currently uses a separate SQLite database (`tuning.db`) for Optuna storage. This should be migrated to use the central **DuckDB** database (`audit_log.duckdb`) to unify all persistent storage into a single, manageable artifact.
    -   **Blockers:** Requires updating the Optuna storage connection string and ensuring the schema is compatible.
    -   **Estimated complexity:** [S]

6.  **[LOW] Advanced Fundamental Feature Integration:**
    -   **Description:** Expand the feature set to include more sophisticated fundamental metrics beyond the basic `yfinance.info` snapshot. This involves parsing historical financial statements (`tkr.financials`, `tkr.balance_sheet`) to create time-series features like **Return on Invested Capital (ROIC)**, historical margin stability, and debt-to-equity ratios.
    -   **Blockers:** Requires careful handling of point-in-time data to avoid look-ahead bias (i.e., merging quarterly financial data onto a daily price series correctly).
    -   **Estimated complexity:** [M]

7.  **[LOW] Monte Carlo Model Caching:**
    -   **Description:** The `MonteCarloTrendFilter` currently re-trains its internal `JumpNetworkLSTM` neural network on every single run. This model should be **cached to disk** after the first training for a given ticker, and subsequent runs should load the pre-trained weights, reducing the simulation setup time from several seconds to milliseconds.
    -   **Blockers:** Requires implementing `torch.save` and `torch.load` logic within the `predictors.py` module.
    -   **Estimated complexity:** [S]