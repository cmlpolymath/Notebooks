# feature_engineering.py
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks

class FeatureCalculator:
    def __init__(self, df):
        # Work on a copy to avoid SettingWithCopyWarning
        self.df = df.copy()

        # Force the underlying numpy arrays of numeric columns to be writable.
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].values.flags.writeable = True

    def add_all_features(self, market_df: pd.DataFrame | None = None):
        """Orchestrator method to add all features."""
        self._add_rsi()
        self._add_macd()
        self._add_atr()
        self._add_bollinger_bands()
        self._add_obv()
        self._add_stochastic_oscillator()
        self._add_mfi()
        self._add_cci()
        self._add_williams_r()
        self._add_roc()
        self._add_garch_vol()
        self._add_returns()
        self._add_dominant_period()
        self._add_kalman_filter()
        
        # ADDED: Incorporate market context features if provided
        if market_df is not None:
            self._add_spy_rsi(market_df)
            self._add_spy_return(market_df)
        
        self.df.dropna(inplace=True)
        return self.df

    # Each indicator is its own private method, keeping logic separate
    def _add_rsi(self, period=14):
        delta = self.df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        self.df['RSI14'] = 100 - (100 / (1 + rs))

    def _add_macd(self, span1=12, span2=26, signal_span=9):
        ema1 = self.df['Close'].ewm(span=span1, adjust=False).mean()
        ema2 = self.df['Close'].ewm(span=span2, adjust=False).mean()
        self.df['MACD_line'] = ema1 - ema2
        self.df['MACD_signal'] = self.df['MACD_line'].ewm(span=signal_span, adjust=False).mean()
        self.df['MACD_hist'] = self.df['MACD_line'] - self.df['MACD_signal']

    def _add_atr(self, period=14):
        high_low = self.df['High'] - self.df['Low']
        high_prev_close = (self.df['High'] - self.df['Close'].shift(1)).abs()
        low_prev_close  = (self.df['Low'] - self.df['Close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        self.df['ATR14'] = true_range.rolling(window=period).mean()

    def _add_bollinger_bands(self, period=20, std_dev=2):
        rolling_mean = self.df['Close'].rolling(window=period).mean()
        rolling_std  = self.df['Close'].rolling(window=period).std()
        self.df['BB_mid']   = rolling_mean
        self.df['BB_upper'] = rolling_mean + std_dev * rolling_std
        self.df['BB_lower'] = rolling_mean - std_dev * rolling_std
        
    def _add_obv(self):
        direction = np.sign(self.df['Close'].diff().fillna(0))
        self.df['OBV'] = (direction * self.df['Volume']).cumsum()

    def _add_stochastic_oscillator(self, period=14, smooth_k=3):
        lowest_low  = self.df['Low'].rolling(period).min()
        highest_high = self.df['High'].rolling(period).max()
        self.df['%K'] = (self.df['Close'] - lowest_low) / (highest_high - lowest_low + 1e-9) * 100
        self.df['%D'] = self.df['%K'].rolling(smooth_k).mean()

    def _add_mfi(self, period=14):
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3.0
        mf = typical_price * self.df['Volume']
        tp_diff = typical_price.diff()
        pos_mf = mf.where(tp_diff > 0, 0.0).rolling(period).sum()
        neg_mf = mf.where(tp_diff < 0, 0.0).rolling(period).sum()
        self.df['MFI14'] = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-9))

    def _add_cci(self, period=20):
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3.0
        TP20 = typical_price.rolling(period).mean()
        MD20 = (typical_price - TP20).abs().rolling(period).mean()
        self.df['CCI20'] = (typical_price - TP20) / (0.015 * MD20 + 1e-9)

    def _add_williams_r(self, period=14):
        lowest_low  = self.df['Low'].rolling(period).min()
        highest_high = self.df['High'].rolling(period).max()
        self.df['Williams_%R'] = (highest_high - self.df['Close']) / (highest_high - lowest_low + 1e-9) * -100
        
    def _add_roc(self, period=10):
        self.df['ROC10'] = self.df['Close'].pct_change(periods=period) * 100

    def _add_garch_vol(self):
        returns = self.df['Close'].pct_change().fillna(0)
        var0 = returns.var()
        alpha, beta = 0.1, 0.85
        omega = var0 * max(0, (1 - alpha - beta))
        garch_vars = [var0]
        for r in returns.iloc[1:]:
            new_var = omega + alpha * (r**2) + beta * garch_vars[-1]
            garch_vars.append(new_var)
        # Cap volatility at a reasonable maximum (e.g., 200% annualized) to prevent overflows
        garch_vol_series = pd.Series(np.sqrt(garch_vars), index=self.df.index)
        self.df['GARCH_vol'] = garch_vol_series.clip(upper=2.0 / np.sqrt(252))
        
    def _add_returns(self):
        self.df['Return1'] = self.df['Close'].pct_change().fillna(0) * 100
        
    def _add_dominant_period(self, window_size: int = 60, min_period: int = 2):
        """
        Vectorized calculation of the dominant period using real FFT
        on rolling windows of the 'Close' price series.
        """
        n = len(self.df)
        if n < window_size:
            # Not enough data to compute any window
            self.df['Dominant_Period'] = np.nan
            return

        close = self.df['Close'].to_numpy(dtype=float)

        # 1. Create rolling windows safely
        windows = sliding_window_view(close, window_shape=window_size)
        # windows.shape == (n - window_size + 1, window_size)

        # 2. Detrend & apply Hamming window
        hamming = np.hamming(window_size)  # (window_size,)
        # subtract row means, then multiply
        windows = windows - windows.mean(axis=1, keepdims=True)
        windows *= hamming

        # 3. Compute real FFT & positive frequencies
        fft_vals = np.fft.rfft(windows, axis=1)
        freqs = np.fft.rfftfreq(window_size, d=1)

        # 4. Power spectrum & noise threshold
        power = np.abs(fft_vals) ** 2  # shape: (num_windows, window_size//2+1)
        noise = np.median(power, axis=1, keepdims=True)  # (num_windows, 1)
        # mask out anything below 2× noise floor
        masked_power = np.where(power > 2 * noise, power, 0.0)

        # 5. Peak detection: index of max power in each window
        peak_idx = np.argmax(masked_power, axis=1)  # (num_windows,)

        # 6. Convert to periods, guard div-by-zero
        peak_freqs = freqs[peak_idx]  # (num_windows,)
        periods = np.empty_like(peak_freqs)
        np.divide(1.0, peak_freqs, out=periods, where=peak_freqs != 0)

        # 7. Invalidate out-of-range periods
        valid = (periods >= min_period) & (periods <= window_size)
        periods[~valid] = np.nan  # mark invalid peaks

        # 8. Pad to match original index length
        padded = np.full(n, np.nan)
        padded[window_size - 1 :] = periods

        # 9. Interpolate & smooth
        series = pd.Series(padded, index=self.df.index)
        interp = series.interpolate(method='linear', limit_area='inside')
        smooth = interp.rolling(window=3, center=True, min_periods=1).mean()

        self.df['Dominant_Period'] = smooth

    def _align_and_join(self, feature_series: pd.Series):
        """Helper to safely join a feature series based on the 'Date' column."""
        # Ensure both dataframes have a 'Date' index for joining
        if 'Date' not in self.df.columns or 'Date' not in feature_series.index.names:
             self.df.set_index('Date', inplace=True)
             if 'Date' not in self.df.index.name: # Handle MultiIndex case
                 self.df.index.names = ['Date' if name is None else name for name in self.df.index.names]

        # Join and reset index to keep 'Date' as a column
        self.df = self.df.join(feature_series, on='Date')


    def _add_spy_rsi(self, market_df: pd.DataFrame, period=14):
        """Calculates RSI for the market index and merges it."""
        market_df_c = market_df.copy()
        # Ensure 'Date' is a datetime column for merging
        market_df_c['Date'] = pd.to_datetime(market_df_c['Date'])
        
        # Set index for calculation, but don't modify the original
        market_df_calc = market_df_c.set_index('Date')

        delta = market_df_calc['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        spy_rsi = 100 - (100 / (1 + rs))
        spy_rsi.name = 'SPY_RSI14'
        
        # Align and merge using the 'Date' column
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.merge(spy_rsi, on='Date', how='left')

    # --- MODIFIED: _add_spy_return ---
    def _add_spy_return(self, market_df: pd.DataFrame):
        """Calculates daily return for the market index and merges it."""
        market_df_c = market_df.copy()
        # Ensure 'Date' is a datetime column for merging
        market_df_c['Date'] = pd.to_datetime(market_df_c['Date'])
        
        # Set index for calculation, but don't modify the original
        market_df_calc = market_df_c.set_index('Date')

        spy_return = market_df_calc['Close'].pct_change().fillna(0) * 100
        spy_return.name = 'SPY_Return1'
        
        # Align and merge using the 'Date' column
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.merge(spy_return, on='Date', how='left')

    # def _add_fractal_dimension(x, kmax=10):
    #     """Compute Higuchi Fractal Dimension for 1D array x."""
    #     N = len(x)
    #     Lk = []
    #     # Loop over k (segment length)
    #     for k in range(1, kmax + 1):
    #         Lm = []
    #         for m in range(k):  # starting offset m = 0,...,k-1
    #             length = 0.0
    #             n_max = int(np.floor((N - m - 1) / k))
    #             for i in range(1, n_max + 1):
    #                 # Accumulate absolute increments |x[m + i*k] - x[m + (i-1)*k]|
    #                 length += abs(x[m + i*k] - x[m + (i-1)*k])
    #             if n_max > 0:
    #                 # Scale length by factor (N-1)/(n_max * k) as per Higuchi’s formula
    #                 length = length * (N - 1) / (n_max * k)
    #                 Lm.append(length)
    #         if Lm:
    #             Lk.append(np.mean(Lm))
    #     # If we have lengths for different k, perform linear fit in log-log to estimate dimension
    #     Lk = np.array(Lk)
    #     if Lk.size < 2:
    #         return np.nan  # not enough points
    #     logk = np.log(np.arange(1, len(Lk) + 1))
    #     logLk = np.log(Lk)
    #     # Slope of log(Lk) vs log(k). Fractal dimension D ≈ -slope
    #     slope, _ = np.polyfit(logk, logLk, 1)
    #     # return -slope

    #     # Compute rolling Higuchi FD on Close price (e.g., 252-day window)
    #     close_prices = self.df['Close'].values
    #     window = 252
    #     fractal_dim_vals = [np.nan] * (window - 1)  # no value for initial window-1 days
    #     for t in range(window - 1, len(close_prices)):
    #         segment = close_prices[t-window+1 : t+1]
    #         fd = _add_fractal_dimension(segment, kmax=10)
    #         fractal_dim_vals.append(fd)

    #     # Store as pandas Series in metrics
    #     self.df['fractal_dim'] = pd.Series(fractal_dim_vals, index=self.df.index)

    def _add_kalman_filter(self, Q=1e-5, R=1e-2):
        """
        Applies a 1D Kalman filter to the 'Close' price to estimate the
        true underlying value, creating a smoothed price series.
        """
        def _kalman_filter_internal(series, Q, R):
            n = len(series)
            filtered = np.zeros(n)
            P = np.zeros(n)
            
            # Initial estimates
            filtered[0] = series[0]
            P[0] = 1.0  # initial estimate covariance
            
            for t in range(1, n):
                # Time update (predict)
                x_pred = filtered[t-1]
                P_pred = P[t-1] + Q
                
                # Measurement update (correct)
                K = P_pred / (P_pred + R)
                filtered[t] = x_pred + K * (series[t] - x_pred)
                P[t] = (1 - K) * P_pred
            return filtered

        # Ensure we are working with a numpy array
        close_prices = self.df['Close'].values
        
        # Apply the filter
        filtered_close = _kalman_filter_internal(close_prices, Q, R)
        
        # Assign the result back to the DataFrame
        self.df['Kalman_Close'] = pd.Series(filtered_close, index=self.df.index)
