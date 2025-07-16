Excellent questions. It's smart to be skeptical of raw feature importance and to think critically about *why* certain features might be influential. Let's break this down.

### 1. Why Macro Features (Oil, Rates) Are Influential & How to Improve Them

It might seem counterintuitive for crude oil or treasury yields to impact a tech stock, but there are several indirect, yet powerful, economic mechanisms at play.

**Why it's happening:**

*   **Interest Rates (10-Year Treasury):** This is the most direct link. Tech stocks, especially growth-oriented ones, are valued based on their *future* earnings. The 10-year treasury yield is the "risk-free rate" used in valuation models (like a Discounted Cash Flow model) to discount those future earnings back to today's value.
    *   **Higher Rates -> Higher Discount Rate -> Lower Present Value.** When yields go up, the present value of a tech company's future profits goes down, making the stock less attractive. This is a market-wide effect that hits growth sectors hardest. Your model is correctly identifying that rising rates are a headwind for tech stocks.
*   **Crude Oil (Energy Prices):** This is a second-order effect, but still significant.
    *   **Inflation Indicator:** Rising oil prices are a major driver of inflation. High inflation often forces the central bank (the Fed) to raise interest rates to cool the economy down. So, oil prices act as a leading indicator for future interest rate hikes, which, as we just discussed, hurts tech stocks.
    *   **Consumer Spending:** High oil prices (e.g., at the gas pump) reduce consumers' discretionary income. This can lead to lower spending on non-essential goods and services, including tech products and subscriptions, eventually impacting company revenues.
    *   **Input Costs:** While less direct for pure software, many tech companies have significant operational costs (e.g., data centers) that are sensitive to energy prices.

**Proposed Feature Engineering (The Solution):**

The raw values of oil and treasury yields are useful, but we can make them much more powerful by transforming them into features that represent **relationships and momentum**, which is what the market truly cares about.

*   **Create Spreads and Ratios:** Instead of just the raw value, calculate the *relationship* between the asset and the macro indicator.
    *   **`Stock_vs_Treasury_Ratio`**: `df['Close'] / df['10Y_Treasury']`. This normalizes the stock price against the prevailing interest rate environment. A falling ratio could be a bearish signal.
    *   **`Stock_vs_Oil_Ratio`**: `df['Close'] / df['Crude_Oil']`.
*   **Calculate Rolling Correlations:** This is a very powerful technique. How has the correlation between the stock's returns and the macro indicator's returns changed over time?
    *   **`Corr_Stock_Treasury_30D`**: `df['Return1'].rolling(30).corr(df['10Y_Treasury_Return1'])`. A rising positive correlation might mean the stock is becoming more sensitive to interest rate moves.
*   **Momentum of Macro Indicators:** The model should know if rates are rising or falling quickly.
    *   **`10Y_Treasury_ROC10`**: `df['10Y_Treasury'].pct_change(10)`. The 10-day Rate of Change for the treasury yield.
*   **Volatility of Macro Indicators:** Is the oil market stable or chaotic?
    *   **`Crude_Oil_RealVol`**: Calculate the 21-day realized volatility of crude oil returns. High volatility in macro assets often signals market uncertainty, which is typically bad for stocks.

By engineering these relational and momentum-based features, you are giving the model much richer, more contextual information than just the raw price levels.

### 2. How to Improve Model Accuracy

Your current ensemble approach is solid. Here are the next logical steps to boost performance:

1.  **Feature Selection & Engineering (As Above):** This is the #1 lever you can pull. Adding the relational macro features, sector correlation features, and more advanced technical indicators (from your validation list) will provide more signal and less noise.
2.  **Hyperparameter Tuning:** You are currently using fixed parameters for XGBoost and the Transformer. This is a low-hanging fruit. Use a systematic approach like `Optuna` or `scikit-learn's GridSearchCV` to find the optimal set of hyperparameters (`n_estimators`, `learning_rate`, `max_depth` for XGBoost; `d_model`, `nhead`, `num_layers` for the Transformer) for each specific stock.
3.  **Refine the Target Variable:** Your current target (`future_price > future_ma`) is good. You could experiment with alternatives to see if they are more predictable:
    *   **Volatility-Adjusted Target:** `future_price > df['Close'] + 0.5 * df['ATR14']`. This defines "up" as a move that is significant relative to recent volatility.
    *   **Outperformance Target:** `(future_price / df['Close'] - 1) > (future_spy_price / df['SPY_Close'] - 1)`. This targets stocks that are set to *outperform the market*, which can be more robust in a bear market.
4.  **Ensemble Weighting:** Instead of a simple 50/50 average, use the models' confidence. For example, you could weight the ensemble based on the validation accuracy of each model from the walk-forward folds. If XGBoost is consistently more accurate, give it a 60% weight and the Transformer 40%.
5.  **Use a Meta-Model (Stacking):** This is an advanced ensemble technique.
    *   Train XGBoost and the Transformer as "Level 1" models.
    *   Use their predictions (the probabilities) as *new features* for a simple "Level 2" meta-model (e.g., a Logistic Regression).
    *   This meta-model learns the optimal way to combine the predictions from the base models, often outperforming a simple average.

### 3. Displaying Forecasted Price and Enter/Exit Levels

This is a great idea to make the output more actionable.

**In `run.py`:**

After you calculate the final predictions, you can get the last row of the test data to determine the entry/exit prices.

```python
# In run.py, inside the main loop, after calculating metrics

# ... existing code ...
y_test_final = y_seq_test
predictions = (ensemble_proba > 0.5).astype(int)
# ... existing code ...

# --- NEW: Calculate and Display Forecast ---
if len(ensemble_proba) > 0:
    last_signal = predictions[-1]
    last_close = df_features['Close'].iloc[-1]
    last_atr = df_features['ATR14'].iloc[-1]

    # Define a simple entry/exit logic
    if last_signal == 1: # Buy signal
        verdict = "BUY"
        entry_price = last_close
        stop_loss = last_close - (1.5 * last_atr)
        take_profit = last_close + (3.0 * last_atr)
    else: # Sell/Hold signal
        verdict = "SELL/HOLD"
        entry_price = 'N/A'
        stop_loss = 'N/A'
        take_profit = 'N/A'

    print(f"\n--- Actionable Forecast for {ticker} ---")
    print(f"Last Close Price: {last_close:.2f}")
    print(f"Model Verdict: {verdict}")
    if last_signal == 1:
        print(f"   -> Suggested Entry: ~{entry_price:.2f}")
        print(f"   -> Suggested Stop-Loss: ~{stop_loss:.2f} (based on 1.5 * ATR)")
        print(f"   -> Suggested Take-Profit: ~{take_profit:.2f} (based on 3.0 * ATR)")
else:
    print("\n--- No forecast generated (insufficient data) ---")

# You can also add these to the `metrics` dict to be logged
metrics['verdict'] = verdict
metrics['entry_price'] = entry_price if isinstance(entry_price, float) else None
# ... etc.
```

**In `visualization.py`:**

You would modify `build_dashboard_content` to create a new card or update an existing one with this information, which you'd load from the `metrics` JSON in the database.

### 4. Validation of New `FeatureCalculator` Methods

Here is a quick audit of the new methods you provided.

*   **`_add_sma_ema`**: **Works.** Standard implementation.
*   **`_add_rsi`**: **Works, but the original is better.** Your new version uses `.rolling().mean()` on the gains and losses. The standard and more responsive RSI calculation uses an Exponentially Weighted Moving Average (`.ewm()`), which is what was in the original `feature_engineering.py`. I recommend keeping the original EWM-based implementation.
*   **`_add_macd`**: **Works.** Correct implementation.
*   **`_add_bollinger`**: **Works.** Correct implementation.
*   **`_add_atr`**: **Works.** Correct implementation.
*   **`_add_obv`**: **Works.** Correct implementation.
*   **`_add_stochastic`**: **Works.** Correct implementation.
*   **`_add_mfi`**: **Works.** Correct implementation.
*   **`_add_cci`**: **Works, but risky.** The denominator `(0.015 * md)` can be zero if the mean deviation is zero (e.g., in flat markets), which would cause a division-by-zero error. The original code added `+ 1e-9` to prevent this. **Fix:** `self.df['CCI20'] = (tp - ma) / (0.015 * md + 1e-9)`
*   **`_add_williams_r`**: **Works.** Correct implementation.
*   **`_add_roc`**: **Works.** Correct implementation.
*   **`_add_realized_vol`**: **Works.** Good feature.
*   **`_add_fourier_period`**: **Doesn't Work as a Rolling Feature.** This calculates a *single* dominant period for the *entire series* and assigns that same value to every row. This is not useful for a time-series model. The original `_add_dominant_period` correctly used a `sliding_window_view` to calculate this on a rolling basis. The original implementation is superior.
*   **`_add_garch_vol`**: **Doesn't Work as a Rolling Feature.** Similar to the Fourier issue, this fits a GARCH model on the *entire series* and then assigns the *single* final forecasted volatility value to every row in the new column. The original implementation correctly iterated through the returns to build a time series of GARCH volatility.
*   **`_add_hurst`**: **Doesn't Work as a Rolling Feature.** Same issue. Calculates one Hurst exponent for the entire series. To be useful, this needs to be calculated on a rolling window.
*   **`_add_fractal_dimension`**: **Works.** This is correctly implemented as a rolling feature. Good job.
*   **`_add_kalman`**: **Works.** Correct implementation.
*   **`_add_wavelet_ratio`**: **Works.** Correctly implemented as a rolling feature.
*   **`_add_volume_weighted_rsi`**: **Doesn't Work.** This is the exact same calculation as `_add_mfi`. The Money Flow Index (MFI) *is* a volume-weighted RSI. This feature is redundant.
*   **`_add_efficiency_ratio`**: **Works.** Correctly implemented as a rolling feature. Also known as the Fractal Efficiency Ratio. Good feature.
*   **`_add_vwap_zscore`**: **Works.** Correctly implemented as a rolling feature. Excellent feature.
*   **`_add_dtw_dist`**: **Too Slow for Production.** Dynamic Time Warping (DTW) is computationally very expensive (O(N^2)). Running this in a nested loop over a 252-day lookback will be extremely slow and likely make your script unusable for a 10-year dataset. While the logic is interesting, I would not recommend using this feature without significant optimization (e.g., using a faster library like `fastdtw`).
*   **`_add_wasserstein`**: **Too Slow for Production.** Same issue as DTW. This is a great idea from a mathematical perspective, but the nested loop implementation will be prohibitively slow.
*   **`_add_arima_resid_z`**: **Doesn't Work as a Rolling Feature.** This fits one ARIMA model on the entire series and calculates the residuals. A model needs to see how the "surprise" (residual) changes over time. This should be implemented on a rolling or expanding window basis to be a valid time-series feature.

**Summary of Validation:** The rolling features (`Fractal_Dim`, `Wavelet_Ratio`, `Eff_Ratio`, `VWAP_Z`) are excellent additions. The features that calculate a single value for the whole series (`Fourier`, `GARCH`, `Hurst`, `ARIMA`) need to be re-implemented on a rolling basis to be useful. The distance metrics (`DTW`, `Wasserstein`) are too slow for practical use in their current form.

---
This is a substantial and very impressive update to `predictors.py`. You've introduced several advanced concepts that significantly elevate the sophistication of the Monte Carlo simulation.

Let's break it down.

### Will it work with the current project?

**Yes, it will work.** The public-facing API of the `MonteCarloTrendFilter` class remains the same:

1.  It is initialized with the same parameters from `config.py`.
2.  The `fit(data)` method still accepts a DataFrame with a 'Close' column.
3.  The `predict(horizon)` method still returns a dictionary with the same keys (`up_prob`, `trend_strength`, `simulated_slopes`, etc.).

Because the "contract" with the rest of the application (`run.py`) is unchanged, you can drop this new file in, and the project will run without any code changes in other files.

---

### What I Think of the Additions (They are excellent)

This is a very thoughtful evolution of the predictor. You've moved from a simpler model to one that captures more complex market dynamics.

1.  **`JumpNetworkLSTM` (The Star of the Show):**
    *   **What it is:** You replaced the original `MultiheadAttention` network with an `LSTM` (Long Short-Term Memory) network.
    *   **Why it's a great idea:** LSTMs are specifically designed to learn from sequential data like time series. They maintain a "memory" of past events, which is perfect for detecting patterns in price returns and volatility that might precede a jump. While attention is great for seeing how all points in a window relate to each other, LSTM is arguably better at capturing the *temporal flow* and evolving state of the market leading up to the most recent data point. This is a more classic and often very effective approach for this kind of problem.

2.  **Vectorized `simulate_paths` Function:**
    *   **What it is:** The previous version used a `for` loop inside a `joblib.Parallel` call to run simulations one by one (on different CPU cores). This new version creates a single, large PyTorch tensor for all simulations (`(num_paths, steps)`) and performs all calculations (Brownian motion, jumps, price updates) in a fully vectorized way on the GPU (or CPU).
    *   **Why it's a massive improvement:** This is a huge performance win. Vectorized GPU operations are orders of magnitude faster than parallelized CPU loops for this kind of numerical task. This will make the Monte Carlo simulation run much, much faster, especially for a high number of simulations (`n_sims`).

3.  **Vectorized Slope Calculation:**
    *   **What it is:** You replaced the call to `scipy.stats.linregress` (which runs on one path at a time) with a vectorized PyTorch implementation of the linear regression slope formula.
    *   **Why it's a great idea:** This complements the vectorized path simulation. By keeping the slope calculation on the GPU and vectorized, you avoid the bottleneck of moving data back to the CPU and looping through it. This maintains the high performance of the simulation from start to finish.

4.  **Volatility Regime-Switching:**
    *   **What it is:** The model now uses a `GaussianMixture` model to identify two distinct volatility regimes (e.g., "low vol" and "high vol") from the historical data. When it runs the simulation, it allocates a portion of the paths to each regime based on the probability that the *current* market volatility belongs to one or the other.
    *   **Why it's a great idea:** This is a very sophisticated and realistic addition. Markets are not static; they switch between periods of calm and periods of chaos. By simulating paths with different volatility parameters based on the current market state, your forecast will be much more adaptive and realistic than a model that assumes a single, constant volatility.

---

### Missing or Changed Elements from the Prior Implementation

You've captured all the core ideas of the previous implementation and improved upon them. Here's a summary of the key changes:

| Feature | Previous Implementation | New Implementation | Analysis of Change |
| :--- | :--- | :--- | :--- |
| **Jump Detection** | `MultiheadAttention` network (`JumpNetwork`). | `LSTM` network (`JumpNetworkLSTM`). | **Major Improvement.** LSTM is arguably a better fit for learning from the temporal flow of price/volatility sequences to predict jumps. |
| **Simulation** | `joblib.Parallel` loop, one path per CPU core. | Fully vectorized PyTorch tensor operations on GPU/CPU. | **Massive Performance Upgrade.** This is the most significant change for speed. The new version will be much faster. |
| **Slope Calculation** | `scipy.stats.linregress` in a loop. | Vectorized linear regression formula in PyTorch. | **Performance Upgrade.** Keeps the entire simulation pipeline fast and avoids CPU bottlenecks. |
| **Volatility** | A single `sigma` was used for all simulations. | `GaussianMixture` model identifies two volatility regimes (`sigma_low`, `sigma_high`) and allocates simulations proportionally. | **Major Conceptual Improvement.** Makes the model far more realistic and adaptive to changing market conditions. |
| **Jump Intensity** | `lam` was calculated as `jp * 252`. | `lam` is still calculated as `jp * 252`. | **Unchanged.** The logic for annualizing the jump probability remains the same. |
| **Policy Network** | `PolicyNetwork` class existed. | `PolicyNetwork` class still exists. | **Unchanged.** The mechanism for dynamically selecting the number of simulations is the same. |
| **GPR for Drift** | `GaussianProcessRegressor` was used. | `LinRegressor` found to be optimal  | **Improved** GPR suggested that trend was a linear model |

**Conclusion:**

This new `predictors.py` is a superior version in every meaningful way. It is:

*   **More Sophisticated:** The LSTM and volatility-regime models capture more complex market dynamics.
*   **Faster:** The full vectorization on the GPU will drastically reduce the time it takes to run the Monte Carlo simulation.
*   **More Realistic:** The regime-switching volatility makes the forecasts more adaptive to the market's current state.

---
### How to Construct More Features in the Future (The Pattern)

You can now easily add more features by following this simple, repeatable pattern. Let's say you want to create a new feature: **"6-Month Realized Volatility of Oil Prices"**.

1.  **Identify Required Data:**
    *   You need the price of Crude Oil. This comes from `yfinance` and is already merged into your DataFrame as the `Crude_Oil` column.

2.  **Choose the Right Location:**
    *   Does the feature depend *only* on the external data source?
        *   If yes (like `Days_Since_Update`), engineer it in `economics.py` or in the `_add_yf_macro_features` method.
    *   Does the feature depend on an *interaction* between the stock and external data?
        *   If yes (like our oil volatility feature), engineer it in the `_add_relational_macro_features` method. This is the correct place for our new feature.

3.  **Implement the Logic in `_add_relational_macro_features`:**
    *   Open `feature_engineering.py`.
    *   Go to the `_add_relational_macro_features` method.
    *   Add the new logic, wrapped in a `check_cols` guard to prevent errors if the data is missing.

    ```python
    # Inside _add_relational_macro_features in feature_engineering.py

    # ... (existing 6 features) ...

    # 7. 6-Month Realized Volatility of Oil Prices
    if check_cols('Crude_Oil'):
        oil_returns = self.df['Crude_Oil'].pct_change().fillna(0)
        # 126 trading days is approximately 6 months
        self.df['Oil_RealVol_6M'] = oil_returns.rolling(126).std() * np.sqrt(252)
    ```

4.  **Add the New Feature Name to `config.py`:**
    *   Open `config.py`.
    *   Find the `relational_macro_features` list.
    *   Add the new column name to the list.

    ```python
    # In config.py
    relational_macro_features = [
        'Corr_Stock_FedFunds_60D',
        # ...
        'CPI_ROC_3M',
        'Oil_RealVol_6M' # Add the new feature here
    ]
    ```

Of course. Here is a comprehensive summary of the project in Markdown format, detailing its evolution, features, architecture, and future steps.

---

# Algorithmic Trading & Forecasting Engine: Project Overview

## 1. Project Evolution: From V1 to V2

### Summary

This project has evolved from a foundational machine learning pipeline (V1) into a sophisticated, multi-faceted forecasting engine (V2). The initial version focused on using standard technical indicators to train multiple models. The current version has expanded dramatically to incorporate a rich set of macroeconomic data, sector analysis, and advanced simulation techniques. It now features a modular architecture with multiple, selectable predictive models, robust data caching, and a stable, reproducible forecasting process suitable for rigorous backtesting and analysis.

Of course. My apologies for the mischaracterizations. You are absolutely right to correct the record—the initial version was far more sophisticated than my summary portrayed. The goal is to accurately capture the *evolution* of the features, not to downplay the starting point.

Here is the revised and more accurate V1 vs. V2 Feature Comparison table, reflecting the facts you've provided.

---

### V1 vs. V2 Feature Comparison

| Feature | V1 (Advanced Start)| V2 (Current State) |
| :--- | :--- | :--- |
| **Data Sources** | Stock, Market (SPY), Sector ETFs, and yfinance Macro (VIX, Oil, etc.). | **Expanded with FRED Macroeconomics:** Adds a new, robust data layer for CPI, GDP, Fed Funds Rate, etc., fetched via an asynchronous `economics.py` module. |
| **Feature Set** | A strong set of technical indicators and basic macro features. | **Significantly Enhanced with Relational & Event-Based Features:** Now includes over 60 features, creating powerful interactions like *Real Fed Funds Rate*, *rolling correlations* between stock returns and macro data, and event-based indicators like *Days Since CPI Update*. |
| **Predictive Models** | A single model at runtime (e.g., `XGBoost` + `Transformer`). | **Multiple selectable models:** `rf` (Random Forest), `xgb` (XGBoost), and a complex `ensemble` (XGBoost + Transformer), selectable via a command-line argument. |
| **Forecasting** | A dual-component forecast with an ML prediction and a Monte Carlo simulation. | **Refined Dual-Component Forecasting:** The ML component is now selectable and can be weighted. The Monte Carlo component is now a stable, independent pillar of the forecast. |
| **Monte Carlo** | An advanced Jump-Diffusion model, but prone to instability between runs. | **Stabilized & Hardened Monte Carlo:** Now fully reproducible using comprehensive **seeding**, quasi-random **Sobol sequences**, and **antithetic variates**. The drift calculation is simplified to a more robust `linregress` model. |
| **Data Pipeline** | An intelligent caching system for stock and yfinance data. | **More Robust & Granular Caching:** The caching system is now more resilient and handles the new, independent FRED macro data cache, ensuring all data sources are checked for freshness on every run. |
| **Validation** | A robust Walk-Forward Validation script was already in place. | **Validation Integrated with All Data Sources:** The `validate.py` script has been updated to correctly pull and process the new FRED macro data, ensuring rigorous testing of the full feature set. |
| **User Interface** | An interactive dashboard (`visualization.py`) displayed results. | **Enhanced Dashboard with Deeper Insights:** The dashboard now adapts to the asset type (stocks vs. crypto) and presents a more actionable forecast with clear entry, stop-loss, and take-profit levels derived from the model's output. |
| **Code Architecture** | A modular and extensible architecture. | **Refined Modular Architecture:** The architecture has been further improved by encapsulating all FRED data fetching and initial feature creation within the dedicated `economics.py` module, making the system even cleaner and easier to maintain. |

## 2. Key Project Features

This algorithmic analysis engine represents a state-of-the-art framework for generating quantitative financial forecasts. It leverages a powerful synthesis of machine learning, stochastic modeling, and comprehensive data integration to produce actionable, multi-horizon insights.

-   **Hybrid Forecasting Model:** The system's core strength lies in its dual-component approach. It combines the pattern-recognition power of **gradient-boosted trees (XGBoost) and Transformers** for short-term predictions with a sophisticated **Jump-Diffusion Monte Carlo simulation** that models long-term trend dynamics, providing a complete picture of an asset's potential trajectory.

-   **Rich, Multi-Source Feature Engineering:** The engine moves beyond simple price data by integrating a vast array of predictive features:
    -   **Technical Indicators:** A full suite of classic and advanced indicators, from RSI and MACD to GARCH Volatility and Dominant Cycle Periods.
    -   **Macroeconomic Intelligence:** Seamlessly pulls and processes data from both `yfinance` (VIX, Oil) and the **FRED database** (CPI, GDP, Fed Funds Rate), creating powerful relational features like the *Real Fed Funds Rate* and *rolling correlations* between stock returns and inflation.
    -   **Sector & Market Context:** Automatically identifies an asset's sector, incorporating the performance of benchmark ETFs (e.g., SPY, XLK) to gauge relative strength and market sentiment.

-   **Advanced Monte Carlo Simulation:** The `predictors.py` module is a professional-grade simulation engine. It is stabilized for **full reproducibility** using quasi-random **Sobol sequences** and **antithetic variates** for variance reduction. It dynamically models **volatility regime-switching** and incorporates a neural network (LSTM) to predict market jumps, capturing the "fat-tailed" nature of real-world financial returns.

-   **Robust Validation & Auditing:** The project is built for rigor and transparency.
    -   A dedicated **Walk-Forward Validation** script (`validate.py`) provides an honest measure of out-of-sample performance, simulating real-world trading conditions.
    -   Every analysis run is meticulously logged in a **DuckDB audit database**, capturing the model used, its configuration, feature importance, and the final prediction for complete traceability.

-   **Flexible & User-Friendly Interface:**
    -   The command-line interface allows users to analyze any stock or cryptocurrency ticker and **select the desired predictive model (`rf`, `xgb`, `ensemble`)** for the run.
    -   The interactive **Dash-based dashboard** provides a comprehensive visual audit of any run, displaying price charts with signals, SHAP feature importances, Monte Carlo distributions, and key fundamental data.

## 3. Architecture and Module Interaction

### Module Summaries

This project is composed of several specialized modules that work together in a sequential pipeline.

-   **`config.py` (The Brain):** This is the central configuration file. It defines all key parameters, including date ranges, feature lists, model hyperparameters, and macroeconomic indicator codes. It governs the behavior of the entire application.

-   **`run.py` (The Conductor):** This is the main entry point of the application. It parses user commands (ticker, model choice), orchestrates the pipeline by calling other modules in the correct order, and prints the final actionable forecast.

-   **`economics.py` (The Economist):** A specialized, asynchronous data-fetching module. Its sole purpose is to connect to the FRED API, download a pre-defined list of economic indicators, and perform initial processing and feature engineering (e.g., `DaysSinceUpdate`) before saving the data.

-   **`data_handler.py` (The Librarian):** This module manages all data persistence. It contains functions to intelligently fetch and cache data for stocks, ETFs, and the macroeconomic data from `economics.py`. It ensures that data is fresh and avoids redundant API calls.

-   **`preprocess.py` (The Assembler):** This module acts as the primary data orchestrator. It calls `data_handler` to gather all the necessary raw data (stock, market, sector, macro). It then passes this collection of data to the `FeatureCalculator`.

-   **`feature_engineering.py` (The Scientist):** This module receives the raw data and transforms it into a feature-rich dataset for the models. It calculates all technical indicators and creates powerful relational features by combining the stock data with the macroeconomic data.

-   **`models.py` (The Learners):** This module defines the machine learning models (XGBoost, Transformer) and contains the logic for training them, preparing data sequences, and generating predictions.

-   **`predictors.py` (The Physicist):** This module contains the advanced Monte Carlo simulation. It uses principles of stochastic calculus (Geometric Brownian Motion, Jump-Diffusion) to generate thousands of possible future price paths and forecast the underlying trend.

-   **`validate.py` (The Skeptic):** A standalone script for rigorously testing model performance. It uses a walk-forward methodology to provide an honest assessment of a model's predictive accuracy over time.

-   **`visualization.py` (The Artist):** The front-end of the project. It queries the audit database to fetch the results of a completed run and presents them in an interactive web-based dashboard.

-   **`audit_logger.py` (The Scribe):** A simple utility that handles writing all results from a run into the `audit_log.duckdb` database, ensuring every prediction is recorded.

### `predictors.py` Version Comparison

The `predictors.py` module underwent the most significant evolution to improve its stability and realism.

| Feature | Previous Version | Latest Version | Gain / Loss |
| :--- | :--- | :--- | :--- |
| **Randomness** | Standard `torch.randn`, results varied between runs. | **Sobol sequences** and **antithetic variates**. | **Gain:** Massive improvement in stability and faster convergence. |
| **Reproducibility** | Not guaranteed. `Trend Strength` could flip signs. | **Fully seeded** with deterministic settings. | **Gain:** Guaranteed reproducibility. The same input will always produce the same output. |
| **Jump Detection** | Present, using a Multi-Head Attention network. | Present, using a more suitable **LSTM network**. | **Gain:** LSTM is better suited for capturing temporal patterns in time series. |
| **Jump Simulation** | Present in the simulation loop. | **Fully restored** and integrated with the new random sampling. | **Feature Preserved.** |
| **Drift Calculation** | `GaussianProcessRegressor` (slow, prone to warnings). | Simple and fast `scipy.stats.linregress`. | **Gain:** Faster, more robust, and directly implements the solution the GPR was pointing to. |
| **Volatility Model** | `GaussianMixture` for regime-switching. | `GaussianMixture` for regime-switching. | **Feature Preserved.** |

## 4. Outstanding Needs & Future Work

The project has a strong foundation, but several high-impact improvements are ready to be implemented to further enhance its performance and capabilities.

-   **Hyperparameter Tuning:** The XGBoost model is the current top performer but uses default parameters. Implementing a systematic tuning process using a library like **`Optuna`** is the highest-priority next step to maximize its accuracy.

-   **Advanced Ensemble Construction (Meta-Modeling):** The current weighted-average ensemble is an improvement but is still static. The next evolution is to build a **stacked ensemble**, where a simple meta-model (e.g., `LogisticRegression`) is trained to learn the optimal way to combine the outputs of the XGBoost and Transformer models, likely yielding higher accuracy and robustness.

-   **Dynamic Feature Selection:** The Transformer model is currently underperforming, likely due to being overwhelmed by the large feature set. While we manually select the top 12 features from SHAP, a more dynamic approach could be implemented where a feature selection process (e.g., Recursive Feature Elimination) is run inside the pipeline to choose the optimal feature subset for each model independently.

-   **Refined Target Variable:** Experiment with alternative definitions for the `UpNext` target variable. Instead of a simple price-over-moving-average, one could test:
    -   **Volatility-Adjusted Target:** e.g., `future_price > current_price + 0.5 * ATR`.
    -   **Market Outperformance Target:** e.g., `stock_return > spy_return`.

-   **Time-Weighted SHAP Importance:** Currently, SHAP values are averaged over the entire test set. A more advanced analysis would be to weigh recent SHAP values more heavily to understand how feature importances are changing over time.

---

### Implementation Plan: "Lite" Model for New Assets

*   **Objective:** Create a "Lite" mode that can run analysis on assets with limited historical data (e.g., less than 1.5 years), which currently fail due to long lookback periods in feature engineering.

*   **Trigger Condition:**
    *   In `preprocess.py`, after loading the raw stock data (`df_raw`), check its length: `if len(df_raw) < 350:`.
    *   If the condition is met, set a boolean flag, `is_lite_mode = True`. This flag will control subsequent logic.

*   **Dynamic Feature Set:**
    *   In `config.py`, create a new list named `FEATURE_COLS_LITE`.
    *   This list should contain only features with short lookback windows (e.g., 30 days or less).
    *   **Include:** `RSI14`, `MACD_line`, `MACD_signal`, `MACD_hist`, `ATR14`, `Return1`, `Close`, `Volume`.
    *   **Exclude:** All features with long lookbacks, especially those requiring 60, 126, or 252 days (e.g., `Real_FedFunds`, `CPI_ROC_3M`, `Corr_Stock_CPI_60D`, `Dominant_Period`).

*   **Conditional Logic in `preprocess.py`:**
    *   When calling `feature_calculator.add_all_features`, the features will still be calculated as before.
    *   However, after the feature-rich DataFrame is created, use the `is_lite_mode` flag to select the final columns for the model.
    *   If `is_lite_mode` is `True`, redefine the `feature_cols` variable: `feature_cols = [col for col in config.FEATURE_COLS_LITE if col in df_model.columns]`.
    *   If `is_lite_mode` is `False`, use the standard `FEATURE_COLS` as is currently done.

*   **Model Selection Override:**
    *   In `run.py`, if `is_lite_mode` is detected (it can be passed back from `prepare_and_cache_data` in the `data_package`), the model selection should be overridden.
    *   The Transformer model should be disabled in "Lite" mode because its `SEQUENCE_WINDOW_SIZE` of 60 is too large for the limited data.
    *   Force the model to be `xgb` or `rf`, as they can operate on smaller datasets without a sequence requirement.
    *   Log the model name clearly, for example: `model_name_log = "XGBoost_Lite_v1"`.

*   **Final Output:**
    *   The final actionable forecast should include a clear warning message indicating that the prediction was generated using "Lite" mode with a limited feature set and should be interpreted with extra caution.
    *   Example: `print("WARNING: Running in Lite Mode due to limited data history. Forecast is less reliable.")`

---
* Our machine learning models require a complete, uninterrupted dataset for every single day to function, but NaN (missing) values are inevitably introduced from two sources. First, when we merge external data that updates at different frequencies (like daily stock prices with monthly CPI) or when a data source's cache is stale, gaps appear at the end of our timeline; a forward-fill (ffill) is essential to carry the last known valid observation forward to fill these recent gaps. Second, many of our most powerful features have a "long-lookback," meaning they require a long window of historical data to be calculated, such as a 252-day Year-over-Year inflation rate. This process creates NaNs at the beginning of our dataset where no prior history exists to look back on. To solve this, we use a back-fill (bfill) to propagate the first validly calculated value backward into these initial gaps. Therefore, we use both ffill and bfill as a robust, two-pronged strategy to fill all NaNs, which is critical to prevent the final dropna() command from incorrectly deleting valuable recent data and ensuring the model receives a complete data matrix for its analysis.
---