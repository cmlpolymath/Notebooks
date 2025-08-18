# aggressors.py
"""
A module for computationally intensive or "aggressive" feature calculations,
optimized using Numba for high performance.
"""

import numpy as np
import pandas as pd
from numba import jit

# --- Numba-Optimized Helper Functions ---
# These are kept outside the class as they are pure, low-level functions.
# The @jit decorator works best on standalone functions.

# Numba-optimized helper functions
@jit(nopython=True)
def _hurst_numba(ts, min_lag=2, max_lag=50):
    """Numba-optimized Hurst calculation"""
    n = len(ts)
    if n < max_lag + 1:
        return np.nan
        
    lags = np.arange(min_lag, min(max_lag, n//2))
    n_lags = len(lags)
    
    if n_lags < 2:
        return np.nan
        
    log_lags = np.log(lags.astype(np.float64))
    log_tau = np.empty(n_lags)
    
    for i in range(n_lags):
        lag = lags[i]
        diffs = ts[lag:] - ts[:-lag]
        tau = np.sqrt(np.var(diffs))
        if tau <= 0:
            return np.nan
        log_tau[i] = np.log(tau)
    
    # Manual linear regression for numba compatibility
    n_points = len(log_lags)
    sum_x = np.sum(log_lags)
    sum_y = np.sum(log_tau)
    sum_xy = np.sum(log_lags * log_tau)
    sum_x2 = np.sum(log_lags * log_lags)
    
    slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x * sum_x)
    return slope * 2.0

@jit(nopython=True)
def _vectorized_katz_fd(close_values, window_size):
    """Fully vectorized Katz fractal dimension calculation"""
    n = len(close_values)
    fd_values = np.full(n, np.nan)
    
    for i in range(window_size - 1, n):
        start_idx = i - window_size + 1
        ts = close_values[start_idx:i+1]
        
        n_points = len(ts) - 1
        if n_points <= 0:
            continue
            
        # Calculate differences
        diffs = np.diff(ts)
        
        # Path length
        L = np.sum(np.sqrt(1 + diffs**2))
        
        # Diameter  
        d = np.max(np.abs(ts - ts[0]))
        
        if d <= 0 or L <= 0:
            continue

        fd_values[i] = np.log(n_points) / (np.log(n_points) + np.log(d/L))
    
    return fd_values

@jit(nopython=True)
def _vectorized_hurst(close_values, window_size, min_lag=2, max_lag=50):
    """Fully vectorized Hurst calculation"""
    n = len(close_values)
    hurst_values = np.full(n, np.nan)
    
    for i in range(window_size, n):
        start_idx = max(0, i - window_size + 1)
        ts = close_values[start_idx:i+1]
        hurst_values[i] = _hurst_numba(ts, min_lag, max_lag)
    
    return hurst_values

class AggressorFeatures:
    """
    A class to encapsulate and apply high-performance, Numba-optimized
    feature calculations to a DataFrame.
    """
    def __init__(self):
        self._warmed_up = False

    def _warmup(self):
        """
        Pre-compiles the Numba functions to avoid a delay on the first
        real data call.
        """
        if not self._warmed_up:
            # print("Warming up Numba-optimized functions in AggressorFeatures...")
            dummy_data = np.random.randn(200)
            _hurst_numba(dummy_data)
            _vectorized_katz_fd(dummy_data, 14)
            _vectorized_hurst(dummy_data, 100)
            self._warmed_up = True
            # print("Warm-up complete.")

    def add_hurst(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        """Calculates the Hurst Exponent and adds it to the DataFrame."""
        self._warmup() # Ensure functions are compiled
        close_values = df['Close'].to_numpy()
        hurst_values = _vectorized_hurst(close_values, window)
        df['Hurst'] = hurst_values
        return df

    def add_fractal_dimension(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculates the Katz Fractal Dimension and adds it to the DataFrame."""
        self._warmup()
        close_values = df['Close'].to_numpy()
        fd_values = _vectorized_katz_fd(close_values, window)
        df['Fractal_Dim'] = fd_values
        return df