# predictors.py
import numpy as np
import pandas as pd
from tqdm import tqdm
import config

class MonteCarloTrendFilter:
    """
    A standard Monte Carlo simulator using Geometric Brownian Motion (GBM).

    This is a robust, textbook implementation that simulates future price paths
    based on historical volatility and drift.
    """
    def __init__(self, lookback_days, n_simulations, horizon_days):
        self.lookback_days = lookback_days
        self.n_simulations = n_simulations
        self.horizon_days = horizon_days
        self._is_fitted = False

    def fit(self, data: pd.DataFrame):
        """
        Calculates historical drift and volatility from the data.
        """
        if len(data) < self.lookback_days:
            raise ValueError(f"Data length ({len(data)}) is less than lookback period ({self.lookback_days}).")
        
        # Use the last `lookback_days` of data to calculate parameters
        relevant_data = data.iloc[-self.lookback_days:]
        log_returns = np.log(relevant_data['Close'] / relevant_data['Close'].shift(1))
        
        # Mu: Annualized drift (average daily return)
        self.mu = log_returns.mean() * 252
        # Sigma: Annualized volatility (standard deviation of daily returns)
        self.sigma = log_returns.std() * np.sqrt(252)
        # S0: The last known closing price
        self.s0 = relevant_data['Close'].iloc[-1]
        
        self._is_fitted = True
        return self

    def predict(self):
        """
        Runs the GBM simulation to predict future trend.
        """
        if not self._is_fitted:
            raise RuntimeError("The `fit` method must be called before `predict`.")

        dt = 1 / 252  # Time step (1 trading day)
        
        # Generate random shocks
        # Shape: (horizon_days, n_simulations)
        random_shocks = np.random.standard_normal(size=(self.horizon_days, self.n_simulations))
        
        # Create an array to hold the simulated price paths
        price_paths = np.zeros_like(random_shocks)
        price_paths[0] = self.s0
        
        # Simulate paths day by day
        for t in range(1, self.horizon_days):
            # Standard GBM formula
            price_paths[t] = price_paths[t-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * random_shocks[t]
            )
        
        # Analyze the results
        final_prices = price_paths[-1]
        up_prob = np.mean(final_prices > self.s0)
        down_prob = np.mean(final_prices < self.s0)
        
        # Trend strength: A value from -1 (strong down) to +1 (strong up)
        # This is the difference between the probability of an up move vs. a down move.
        trend_strength = up_prob - down_prob

        return {
            'up_prob': up_prob,
            'down_prob': down_prob,
            'trend_strength': trend_strength,
            'ci': np.percentile(final_prices, [5, 95]).tolist(),
            'simulated_slopes': final_prices - self.s0 # Simplified to final price change
        }