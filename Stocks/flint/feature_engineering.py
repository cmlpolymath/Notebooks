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

    def add_all_features(
        self,
        market_df: pd.DataFrame | None = None,
        sector_df: pd.DataFrame | None = None,
        macro_dfs_yf: dict[str, pd.DataFrame] | None = None,
        df_macro_fred: pd.DataFrame | None = None
    ):
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
        self._add_realized_vol()
        self._add_hurst()
        self._add_fractal_dimension()
        # self._add_wavelet_ratio()
        self._add_efficiency_ratio()
        self._add_vwap_zscore()
        
        # Incorporate market context features if provided
        if market_df is not None:
            self._add_spy_rsi(market_df)
            self._add_spy_return(market_df)

        # ADDED: Incorporate sector context features if provided
        if sector_df is not None:
            self._add_sector_rsi(sector_df)
            self._add_sector_return(sector_df)
        
        # Incorporate yfinance macroeconomic features if provided
        if macro_dfs_yf:
            self._add_yf_macro_features(macro_dfs_yf)
        
        if df_macro_fred is not None:
            self._add_fred_macro_features(df_macro_fred)

        # This is called AFTER the merge, so we can create interaction features.
        self._add_relational_macro_features()

        # --- Final Cleanup ---
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

    def _add_spy_rsi(self, market_df: pd.DataFrame, period=14):
        """Calculates RSI for the market index and merges it."""
        market_df_c = market_df.copy()
        market_df_c['Date'] = pd.to_datetime(market_df_c['Date'])
        market_df_calc = market_df_c.set_index('Date')

        delta = market_df_calc['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        spy_rsi = 100 - (100 / (1 + rs))
        spy_rsi.name = 'SPY_RSI14'
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.merge(spy_rsi, on='Date', how='left')

    def _add_spy_return(self, market_df: pd.DataFrame):
        """Calculates daily return for the market index and merges it."""
        market_df_c = market_df.copy()
        market_df_c['Date'] = pd.to_datetime(market_df_c['Date'])
        market_df_calc = market_df_c.set_index('Date')

        spy_return = market_df_calc['Close'].pct_change().fillna(0) * 100
        spy_return.name = 'SPY_Return1'
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.merge(spy_return, on='Date', how='left')

    # --- ADDED: Sector Feature Methods ---
    def _add_sector_rsi(self, sector_df: pd.DataFrame, period=14):
        """Calculates RSI for the sector ETF and merges it."""
        sector_df_c = sector_df.copy()
        sector_df_c['Date'] = pd.to_datetime(sector_df_c['Date'])
        sector_df_calc = sector_df_c.set_index('Date')

        delta = sector_df_calc['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        sector_rsi = 100 - (100 / (1 + rs))
        sector_rsi.name = 'SECTOR_RSI14'
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.merge(sector_rsi, on='Date', how='left')

    def _add_sector_return(self, sector_df: pd.DataFrame):
        """Calculates daily return for the sector ETF and merges it."""
        sector_df_c = sector_df.copy()
        sector_df_c['Date'] = pd.to_datetime(sector_df_c['Date'])
        sector_df_calc = sector_df_c.set_index('Date')

        sector_return = sector_df_calc['Close'].pct_change().fillna(0) * 100
        sector_return.name = 'SECTOR_Return1'
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.merge(sector_return, on='Date', how='left')

    # --- ADDED: Macroeconomic Feature Method ---
    def _add_yf_macro_features(self, macro_dfs_yf: dict[str, pd.DataFrame]):
        """Merges multiple yfinance indicators into the main dataframe."""
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        for name, df_macro in macro_dfs_yf.items():
            if df_macro is None or df_macro.empty:
                print(f"Warning: Macro indicator '{name}' data is empty. Skipping.")
                self.df[name] = np.nan # Add empty column to maintain schema
                continue
            
            macro_series = df_macro.copy()
            macro_series['Date'] = pd.to_datetime(macro_series['Date'])
            
            if 'Close' not in macro_series.columns:
                 print(f"Warning: 'Close' column not found for macro indicator '{name}'. Skipping.")
                 continue

            macro_series = macro_series[['Date', 'Close']].rename(columns={'Close': name})
            self.df = pd.merge(self.df, macro_series, on='Date', how='left')
            
        # Forward-fill the macro data to handle non-trading days (e.g., weekends, holidays for VIX)
        macro_cols = [name for name in macro_dfs_yf.keys() if name in self.df.columns]
        self.df[macro_cols] = self.df[macro_cols].ffill()

    def _add_fred_macro_features(self, df_macro_fred: pd.DataFrame):
        """
        Merges the pre-processed, daily-indexed FRED data into the main dataframe.
        """
        if 'Date' not in self.df.columns:
             self.df.reset_index(inplace=True)

        self.df['Date'] = pd.to_datetime(self.df['Date'])
        df_macro_fred = df_macro_fred.set_index(pd.to_datetime(df_macro_fred.index))
        
        # Merge the entire pre-engineered DataFrame.
        # All columns from df_macro_fred will be added.
        self.df = pd.merge(self.df, df_macro_fred, left_on='Date', right_index=True, how='left')
        
        # Forward-fill to handle any weekend/holiday misalignments from the merge.
        # The original ffill in economics.py handles the monthly/quarterly gaps.
        self.df[df_macro_fred.columns] = self.df[df_macro_fred.columns].ffill()

    def _add_relational_macro_features(self):
        """
        Creates features based on the interaction between the stock's data
        and the already-merged macroeconomic data.
        """
        print("Engineering relational macroeconomic features...")
        
        # Helper to check if required columns exist before creating a feature
        def check_cols(*args):
            return all(col in self.df.columns for col in args)

        # 1. Rolling Correlation: Stock Returns vs. Fed Funds Rate Changes
        if check_cols('Return1', 'FedFunds'):
            # First, calculate the daily % change of the FedFunds rate
            fedfunds_change = self.df['FedFunds'].pct_change().fillna(0)
            self.df['Corr_Stock_FedFunds_60D'] = self.df['Return1'].rolling(60).corr(fedfunds_change)

        # 2. Rolling Correlation: Stock Returns vs. CPI Changes
        if check_cols('Return1', 'CPI'):
            cpi_change = self.df['CPI'].pct_change().fillna(0)
            self.df['Corr_Stock_CPI_60D'] = self.df['Return1'].rolling(60).corr(cpi_change)

        # 3. Realized Volatility of the Treasury Spread
        if check_cols('TreasurySpread'):
            spread_change = self.df['TreasurySpread'].pct_change().fillna(0)
            # Calculate rolling std dev and annualize it
            self.df['TreasurySpread_RealVol_21D'] = spread_change.rolling(21).std() * np.sqrt(252)

        # 4. Real Fed Funds Rate (Nominal Rate - Inflation)
        if check_cols('FedFunds', 'CPI'):
            # Calculate Year-over-Year CPI inflation rate
            yoy_inflation = self.df['CPI'].pct_change(252) * 100
            self.df['Real_FedFunds'] = self.df['FedFunds'] - yoy_inflation

        # 5. Stock Price vs. GDP Ratio (Valuation)
        if check_cols('Close', 'GDP'):
            # GDP is in Billions of USD. To make the ratio sensible, we don't need to scale it,
            # but the model will learn the magnitude.
            self.df['Stock_vs_GDP_Ratio'] = self.df['Close'] / self.df['GDP']

        # 6. 3-Month Rate of Change for CPI
        if check_cols('CPI'):
            # 63 trading days is approximately 3 months
            self.df['CPI_ROC_3M'] = self.df['CPI'].pct_change(63) * 100
            
        print("Relational features engineered.")

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

        close_prices = self.df['Close'].values
        filtered_close = _kalman_filter_internal(close_prices, Q, R)
        self.df['Kalman_Close'] = pd.Series(filtered_close, index=self.df.index)

    def _add_realized_vol(self, window: int = 21):
        lr = np.log(self.df['Close']/self.df['Close'].shift(1)).dropna()
        vol = lr.rolling(window).std() * np.sqrt(252)
        self.df['RealVol'] = vol.reindex(self.df.index)
    
    
    def _add_hurst(self):
        def hurst(ts):
            lags = [2,4,8,16,32,64]
            variances = [np.var(ts[lag:] - ts[:-lag]) for lag in lags]
            slope, _ = np.polyfit(np.log(lags), np.log(variances), 1)
            return slope/2
        self.df['Hurst'] = hurst(self.df['Close'].values)
    
    def _add_fractal_dimension(self, window: int = 252, kmax: int = 10):
        prices = self.df['Close'].values
        fd_vals = [np.nan]*(window-1)
        for t in range(window-1, len(prices)):
            seg = prices[t-window+1:t+1]
            # Higuchi FD
            N = len(seg)
            Lk = []
            for k in range(1, kmax+1):
                Lm = []
                for m in range(k):
                    length = 0
                    nmax = (N-m-1)//k
                    for i in range(1,nmax+1):
                        length += abs(seg[m+i*k] - seg[m+(i-1)*k])
                    if nmax>0:
                        Lm.append(length*(N-1)/(nmax*k))
                if Lm:
                    Lk.append(np.mean(Lm))
            if len(Lk)>1:
                slope,_ = np.polyfit(np.log(np.arange(1,len(Lk)+1)), np.log(Lk),1)
                fd_vals.append(-slope)
            else:
                fd_vals.append(np.nan)
        self.df['Fractal_Dim'] = pd.Series(fd_vals, index=self.df.index)
    
    # def _add_wavelet_ratio(self, window: int = 30, wavelet: str = 'haar', level: int = 3):
    #     prices = self.df['Close'].values
    #     vals = [np.nan]*(window-1)
    #     for t in range(window-1, len(prices)):
    #         seg = prices[t-window+1:t+1]
    #         coeffs = pywt.wavedec(seg, wavelet, level=level)
    #         detail = np.sum([np.sum(c**2) for c in coeffs[1:]])
    #         approx = np.sum(coeffs[0]**2)
    #         vals.append(detail/(detail+approx+1e-9))
    #     self.df['Wavelet_Ratio'] = pd.Series(vals, index=self.df.index)
    
    def _add_efficiency_ratio(self, period: int = 10):
        p = self.df['Close']
        vals = [np.nan]*period
        for t in range(period, len(p)):
            change = abs(p.iloc[t] - p.iloc[t-period])
            volatility = p.iloc[t-period+1:t+1].diff().abs().sum()
            vals.append(change/(volatility+1e-9))
        self.df['Eff_Ratio'] = pd.Series(vals, index=self.df.index)
    
    def _add_vwap_zscore(self, window: int = 20):
        p = self.df['Close'].values
        v = self.df['Volume'].values
        vals = [np.nan]*window
        for t in range(window, len(p)):
            ps = p[t-window:t]
            vs = v[t-window:t]
            vwap = np.sum(ps*vs)/np.sum(vs)
            std = np.std(ps)
            vals.append((p[t]-vwap)/(std+1e-9))
        self.df['VWAP_Z'] = pd.Series(vals, index=self.df.index)
    
    # def _add_arima_resid_z(self, order=(5,0,0)):
    #     rets = self.df['Close'].pct_change().dropna()
    #     mod = sm.tsa.ARIMA(rets, order=order).fit()
    #     resid = mod.resid; mu=resid.mean(); sd=resid.std(ddof=0)
    #     z = (resid - mu)/ (sd+1e-9)
    #     full = pd.Series(np.nan, index=self.df.index)
    #     full.iloc[1:] = z.values
    #     self.df['ARIMA_Resid_Z'] = full
