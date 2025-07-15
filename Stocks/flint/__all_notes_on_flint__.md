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

That's it. By following these four steps, you can systematically add any number of new, complex, relational features to your model. This pattern keeps your code organized, robust, and easy to extend.