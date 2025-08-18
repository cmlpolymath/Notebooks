# feature_engineering.py
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numba import jit
from scipy.signal import find_peaks
import talib
import structlog

from config import settings, FeatureGroup
from aggressors import AggressorFeatures

logger = structlog.get_logger(__name__)

class FeatureCalculator:
    def __init__(self, df):
        self.df = df.copy()
        self.aggressors = AggressorFeatures()
        self._prepare_talib_arrays()
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].values.flags.writeable = True
    
    def _prepare_talib_arrays(self):
        """Prepares and caches NumPy arrays for TA-Lib functions for performance."""
        # Ensure the DataFrame index is sorted for correct calculations
        self.df.sort_index(inplace=True)
        self.open = self.df['Open'].values.astype(np.float64)
        self.high = self.df['High'].values.astype(np.float64)
        self.low = self.df['Low'].values.astype(np.float64)
        self.close = self.df['Close'].values.astype(np.float64)
        self.volume = self.df['Volume'].values.astype(np.float64)

    def _add_candlestick_patterns(self):
        """
        Adds candlestick pattern features to the DataFrame, reading the list
        of patterns to generate directly from the central config.
        """
        # Get the list of pattern names from the settings object.
        # This makes config.py the single source of truth.
        pattern_names = settings.features.feature_groups.get(FeatureGroup.CANDLESTICK, [])
        
        if not pattern_names:
            logger.info("No candlestick patterns found in config. Skipping.")
            return

        for pattern_name in pattern_names:
            try:
                # Get the actual function from the talib module
                pattern_function = getattr(talib, pattern_name)
                # Call the function with the prepared numpy arrays
                result = pattern_function(self.open, self.high, self.low, self.close)
                # Add the result as a new column to the DataFrame
                self.df[pattern_name] = result
            except Exception as e:
                # Use a structured log if you have a logger instance here, otherwise print
                logger.warn(f"Warning: Could not calculate pattern {pattern_name}: {e}")

    def _robust_fill(self, reason: str):
        """
        A defense-in-depth method to fill NaNs. First forward-fills, then back-fills.
        This ensures no NaNs remain, which is critical before calculations or final dropna.
        """
        # Select only numeric columns for filling
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        
        # Count NaNs before filling for diagnostics
        nans_before = self.df[numeric_cols].isnull().sum().sum()
        if nans_before > 0:
            logger.info(f"Performing robust fill ({reason})... Found {nans_before} total NaN values.")
            self.df.ffill(inplace=True)
            self.df.bfill(inplace=True)
            nans_after = self.df[numeric_cols].isnull().sum().sum()
            logger.info(f"Fill complete. {nans_after} NaN values remain.")

    def add_all_features(
        self,
        market_df: pd.DataFrame | None = None,
        sector_df: pd.DataFrame | None = None,
        macro_dfs_yf: dict[str, pd.DataFrame] | None = None,
        df_macro_fred: pd.DataFrame | None = None
    ):
        """Orchestrator method to add all features with robust NaN handling."""
        
        # --- Stage 1: Base Technical Indicators ---
        # These are calculated on the raw stock data first.
        logger.info("Calculating base technical indicators...")
        logger.info("Warming up Numba-optimized functions in AggressorFeatures...")
        self.df = self.aggressors.add_hurst(self.df, window=100)
        self.df = self.aggressors.add_fractal_dimension(self.df, window=14)
        logger.info("Warm-up complete.")
        self._add_returns() # Add returns early as many features depend on it
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
        self._add_dominant_period()
        self._add_kalman_filter()
        self._add_realized_vol()
        self._add_efficiency_ratio()
        self._add_vwap_zscore()
        self._add_candlestick_patterns()
        
        # --- Stage 2: Merge All External Data Sources ---
        logger.info("Merging external data sources...")
        if market_df is not None:
            self._add_spy_rsi(market_df)
            self._add_spy_return(market_df)
        if sector_df is not None:
            self._add_sector_rsi(sector_df)
            self._add_sector_return(sector_df)
        if macro_dfs_yf:
            self._add_yf_macro_features(macro_dfs_yf)
        if df_macro_fred is not None:
            self._add_fred_macro_features(df_macro_fred)

        # --- Stage 3: First Robust Fill ---
        # Fill all NaNs from merges and initial calculations before creating relational features.
        self._robust_fill(reason="Post-Merge")

        # --- Stage 4: Relational & Interaction Features ---
        # These features depend on the merged data from Stage 2.
        logger.info("Engineering market regime and interaction features...")
        self._add_market_regime()
        self._add_interaction_features()
        self._add_relational_macro_features()

        # --- Stage 5: Second Robust Fill ---
        # Fill any NaNs created during the relational calculations (e.g., from .corr()).
        self._robust_fill(reason="Post-Calculation")

        # --- Final Cleanup ---
        # This final dropna should now only remove rows at the very beginning of the
        # dataset where long lookback windows prevented any calculation.
        logger.info("Performing final cleanup of initial lookback period...")
        initial_len = len(self.df)
        self.df.dropna(inplace=True)
        final_len = len(self.df)
        logger.info(f"Dropped {initial_len - final_len} initial rows.")
        
        return self.df

    # --- All individual feature methods below ---

    def _add_returns(self):
        self.df['Return1'] = self.df['Close'].pct_change() * 100

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
        direction = np.sign(self.df['Close'].diff())
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
        returns = self.df['Return1'] / 100 # Use pre-calculated returns
        var0 = returns.var()
        alpha, beta = 0.1, 0.85
        omega = var0 * max(0, (1 - alpha - beta))
        garch_vars = [var0]
        for r in returns.iloc[1:]:
            new_var = omega + alpha * (r**2) + beta * garch_vars[-1]
            garch_vars.append(new_var)
        garch_vol_series = pd.Series(np.sqrt(garch_vars), index=self.df.index)
        self.df['GARCH_vol'] = garch_vol_series.clip(upper=2.0 / np.sqrt(252))
        
    def _add_dominant_period(self, window_size: int = 60, min_period: int = 2):
        n = len(self.df)
        if n < window_size:
            self.df['Dominant_Period'] = np.nan
            return
        close = self.df['Close'].to_numpy(dtype=float)
        windows = sliding_window_view(close, window_shape=window_size)
        windows = windows - windows.mean(axis=1, keepdims=True)
        windows *= np.hamming(window_size)
        fft_vals = np.fft.rfft(windows, axis=1)
        freqs = np.fft.rfftfreq(window_size, d=1)
        power = np.abs(fft_vals) ** 2
        noise = np.median(power, axis=1, keepdims=True)
        masked_power = np.where(power > 2 * noise, power, 0.0)
        peak_idx = np.argmax(masked_power, axis=1)
        peak_freqs = freqs[peak_idx]
        periods = np.divide(1.0, peak_freqs, out=np.full_like(peak_freqs, np.nan), where=peak_freqs != 0)
        valid = (periods >= min_period) & (periods <= window_size)
        periods[~valid] = np.nan
        padded = np.full(n, np.nan)
        padded[window_size - 1 :] = periods
        series = pd.Series(padded, index=self.df.index)
        interp = series.interpolate(method='linear', limit_area='inside')
        smooth = interp.rolling(window=3, center=True, min_periods=1).mean()
        self.df['Dominant_Period'] = smooth

    def _add_spy_rsi(self, market_df: pd.DataFrame, period=14):
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
        market_df_c = market_df.copy()
        market_df_c['Date'] = pd.to_datetime(market_df_c['Date'])
        market_df_calc = market_df_c.set_index('Date')
        spy_return = market_df_calc['Close'].pct_change() * 100; spy_return.name = 'SPY_Return1'
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.merge(spy_return, on='Date', how='left')

    def _add_sector_rsi(self, sector_df: pd.DataFrame, period=14):
        sector_df_c = sector_df.copy()
        sector_df_c['Date'] = pd.to_datetime(sector_df_c['Date'])
        sector_df_calc = sector_df_c.set_index('Date')
        delta = sector_df_calc['Close'].diff()
        gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        sector_rsi = 100 - (100 / (1 + rs)); sector_rsi.name = 'SECTOR_RSI14'
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.merge(sector_rsi, on='Date', how='left')

    def _add_sector_return(self, sector_df: pd.DataFrame):
        sector_df_c = sector_df.copy()
        sector_df_c['Date'] = pd.to_datetime(sector_df_c['Date'])
        sector_df_calc = sector_df_c.set_index('Date')
        sector_return = sector_df_calc['Close'].pct_change() * 100; sector_return.name = 'SECTOR_Return1'
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.merge(sector_return, on='Date', how='left')

    def _add_yf_macro_features(self, macro_dfs_yf: dict[str, pd.DataFrame]):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        for name, df_macro in macro_dfs_yf.items():
            if df_macro is None or df_macro.empty: self.df[name] = np.nan; continue
            macro_series = df_macro.copy()
            macro_series['Date'] = pd.to_datetime(macro_series['Date'])
            if 'Close' not in macro_series.columns: continue
            macro_series = macro_series[['Date', 'Close']].rename(columns={'Close': name})
            self.df = pd.merge(self.df, macro_series, on='Date', how='left')
        
    def _add_fred_macro_features(self, df_macro_fred: pd.DataFrame):
        if 'Date' not in self.df.columns: self.df.reset_index(inplace=True)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        df_macro_fred.index = pd.to_datetime(df_macro_fred.index)
        self.df = pd.merge(self.df, df_macro_fred, left_on='Date', right_index=True, how='left')

    def _add_relational_macro_features(self):
        logger.info("Engineering relational macroeconomic features...")
        def check_cols(*args): return all(col in self.df.columns for col in args)

        if check_cols('Return1', 'FedFunds'):
            self.df['Corr_Stock_FedFunds_60D'] = self.df['Return1'].rolling(60).corr(self.df['FedFunds'].pct_change())
        if check_cols('Return1', 'CPI'):
            self.df['Corr_Stock_CPI_60D'] = self.df['Return1'].rolling(60).corr(self.df['CPI'].pct_change())
        if check_cols('TreasurySpread'):
            self.df['TreasurySpread_RealVol_21D'] = self.df['TreasurySpread'].pct_change().rolling(21).std() * np.sqrt(252)
        if check_cols('FedFunds', 'CPI'):
            self.df['Real_FedFunds'] = self.df['FedFunds'] - (self.df['CPI'].pct_change(252) * 100)
        if check_cols('Close', 'GDP'):
            self.df['Stock_vs_GDP_Ratio'] = self.df['Close'] / self.df['GDP']
        if check_cols('CPI'):
            self.df['CPI_ROC_3M'] = self.df['CPI'].pct_change(63) * 100
        logger.info("Relational features engineered.")

    def _add_kalman_filter(self, Q=1e-5, R=1e-2):
        close_prices = self.df['Close'].values
        filtered = np.zeros(len(close_prices)); P = np.zeros(len(close_prices))
        filtered[0], P[0] = close_prices[0], 1.0
        for t in range(1, len(close_prices)):
            x_pred, P_pred = filtered[t-1], P[t-1] + Q
            K = P_pred / (P_pred + R)
            filtered[t] = x_pred + K * (close_prices[t] - x_pred)
            P[t] = (1 - K) * P_pred
        self.df['Kalman_Close'] = pd.Series(filtered, index=self.df.index)

    def _add_realized_vol(self, window: int = 21):
        lr = np.log(self.df['Close']/self.df['Close'].shift(1))
        self.df['RealVol'] = lr.rolling(window).std() * np.sqrt(252)
    
    def _add_efficiency_ratio(self, period: int = 10):
        change = self.df['Close'].diff(period).abs()
        volatility = self.df['Close'].diff().abs().rolling(period).sum()
        self.df['Eff_Ratio'] = change / volatility
    
    def _add_vwap_zscore(self, window: int = 20):
        p, v = self.df['Close'], self.df['Volume']
        vwap = (p * v).rolling(window).sum() / v.rolling(window).sum()
        std = p.rolling(window).std()
        self.df['VWAP_Z'] = (p - vwap) / std

    def _add_market_regime(self, short_window=20, long_window=60):
        if 'Close' not in self.df.columns:
            logger.debug("Skipping regime: no Close column")
            return

        # 1) compute returns once
        returns = self.df['Close'].pct_change().fillna(0)

        # 2) generate all regime series
        def regime_series():
            short_ma = returns.rolling(short_window, min_periods=1).mean()
            long_ma  = returns.rolling(long_window,  min_periods=1).mean()
            yield 'short_ma',      short_ma
            yield 'long_ma',       long_ma
            yield 'regime_diff',   short_ma - long_ma
            yield 'regime_ratio',  short_ma.div(long_ma + 1e-9)
            yield 'is_bull_regime',(short_ma > long_ma).astype(int)
            yield 'is_bear_regime',(short_ma <= long_ma).astype(int)

        # 3) assign directly onto self.df
        for name, series in regime_series():
            self.df[name] = series


    def _add_interaction_features(self):
        # declarative mapping: new_col -> (required_cols, generator_func)
        feature_map = {
            'Price_x_Volume':   (['Close', 'Volume'], lambda df: df['Close'] * df['Volume']),
            'RSI_div_ATR':      (['RSI14', 'ATR14'],lambda df: df['RSI14'].div(df['ATR14'] + 1e-9)),
            'Bull_x_RSI':       (['is_bull_regime', 'RSI14'], lambda df: df['is_bull_regime'] * df['RSI14']),
            'Bear_x_ATR':       (['is_bear_regime', 'ATR14'], lambda df: df['is_bear_regime']  * df['ATR14']),
            'Vol_x_Dev':        (['Volume', 'Close'],
                                lambda df: df['Volume'] * df['Close'].pct_change().rolling(10).std().fillna(0)),
            'Regime_Mom_Ratio': (['is_bull_regime', 'RSI14'],
                                lambda df: df['is_bull_regime'] * (df['RSI14'] / (df['RSI14'].rolling(5).mean() + 1e-9))
                                )
        }

        for col, (reqs, fn) in feature_map.items():
            if all(rc in self.df.columns for rc in reqs):
                self.df[col] = fn(self.df)