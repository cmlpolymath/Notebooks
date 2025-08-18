import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # optional
from typing import Dict, Union
import warnings
from datetime import datetime

class Backtester:
    """
    Sophisticated backtesting engine for Project Flint algorithmic trading strategies.
    """

    def __init__(self,
                 price_data: pd.DataFrame,
                 predictions: Union[pd.Series, pd.DataFrame],
                 atr_series: pd.Series,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 risk_per_trade: float = 0.01):
        self._validate_inputs(price_data, predictions, atr_series)

        self.prices = price_data.copy()
        self.predictions = predictions.copy()
        self.atr = atr_series.copy()

        self.initial_capital = float(initial_capital)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.risk_per_trade = float(risk_per_trade)

        self.portfolio: pd.DataFrame | None = None
        self.trade_log: list[dict] = []
        self.signals: pd.Series | None = None

        self._align_data()

    # -------------------------- Utilities & Validation --------------------------

    @staticmethod
    def _ensure_dt_index(df_or_s: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Guarantee a tz-naive, sorted, unique DatetimeIndex."""
        obj = df_or_s
        if isinstance(obj.index, pd.MultiIndex):
            raise ValueError("MultiIndex not supported; provide a single DatetimeIndex.")
        if not isinstance(obj.index, pd.DatetimeIndex):
            # Try common column names
            for candidate in ("Date", "date", "timestamp", "Timestamp"):
                if isinstance(obj, pd.DataFrame) and candidate in obj.columns:
                    obj = obj.copy()
                    obj[candidate] = pd.to_datetime(obj[candidate], errors="coerce", utc=False)
                    obj.set_index(candidate, inplace=True)
                    break
            if not isinstance(obj.index, pd.DatetimeIndex):
                raise ValueError("Input must have a DatetimeIndex or a parsable 'Date' column.")

        # Coerce, sort, drop duplicate index values
        idx = pd.to_datetime(obj.index, utc=False)
        obj = obj.copy()
        obj.index = idx
        obj = obj[~obj.index.duplicated(keep="last")].sort_index()
        return obj

    def _validate_inputs(self, price_data, predictions, atr_series):
        # Relaxed requirement: need at least High/Low/Close (Open optional)
        base_needed = ['High', 'Low', 'Close']
        missing_cols = [c for c in base_needed if c not in price_data.columns]
        if missing_cols:
            # If we at least have Close, we can degrade by copying Close into missing OHLC (still runnable)
            if 'Close' in price_data.columns:
                price_data = price_data.copy()
                if 'High' not in price_data.columns:
                    price_data['High'] = price_data['Close']
                if 'Low' not in price_data.columns:
                    price_data['Low'] = price_data['Close']
            else:
                raise ValueError(f"Price data must include at least 'Close'; missing {missing_cols}.")

        # Basic index checks will be handled in _align_data
        if not isinstance(predictions, (pd.Series, pd.DataFrame)):
            raise ValueError("predictions must be a pandas Series or DataFrame")
        if not isinstance(atr_series, pd.Series):
            raise ValueError("atr_series must be a pandas Series")

    def _align_data(self):
        """Align all series on a common, clean DatetimeIndex; coerce dtypes; fill ATR NAs."""
        self.prices = self._ensure_dt_index(self.prices)
        self.predictions = self._ensure_dt_index(self.predictions)
        self.atr = self._ensure_dt_index(self.atr)

        # If predictions is a DataFrame, use the first column consistently
        if isinstance(self.predictions, pd.DataFrame):
            if self.predictions.shape[1] == 0:
                raise ValueError("Predictions DataFrame has no columns.")
            self.predictions = self.predictions.iloc[:, 0].rename("probability")
        else:
            self.predictions = self.predictions.rename("probability")

        # Intersect dates
        common = self.prices.index.intersection(self.predictions.index).intersection(self.atr.index)
        if len(common) < 50:
            warnings.warn("Less than 50 common dates found. Results may be unreliable.")

        # Reindex to common set
        self.prices = self.prices.reindex(common).sort_index()
        self.predictions = self.predictions.reindex(common).astype(float)
        self.atr = self.atr.reindex(common).astype(float)

        # Coerce OHLC to float
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.prices.columns:
                self.prices[col] = pd.to_numeric(self.prices[col], errors="coerce").astype(float)

        # ATR safety: forward-fill then fill remaining with median; avoid zeros
        self.atr = self.atr.ffill()
        if self.atr.isna().any():
            med = float(np.nanmedian(self.atr.values))
            self.atr = self.atr.fillna(med if np.isfinite(med) and med > 0 else 1e-6)
        self.atr = self.atr.clip(lower=1e-8)

    # ------------------------------- Signals ------------------------------------

    def generate_signals(self, signal_threshold: float = 0.55) -> pd.Series:
        probs = self.predictions
        signals = pd.Series(0, index=probs.index, dtype=np.int8)
        signals[probs > signal_threshold] = 1
        sell_threshold = 1 - signal_threshold
        signals[probs < sell_threshold] = -1
        return signals

    # --------------------------- Position Sizing --------------------------------

    def calculate_position_size(self,
                                portfolio_value: float,
                                current_price: float,
                                atr: float,
                                signal: int,
                                stop_loss_atr: float = 1.5) -> int:
        risk_per_share = float(atr) * float(stop_loss_atr)
        if not np.isfinite(risk_per_share) or risk_per_share <= 0:
            return 0
        risk_amount = float(portfolio_value) * float(self.risk_per_trade)
        raw = risk_amount / risk_per_share
        shares = int(raw) * int(np.sign(signal))  # direction
        if current_price <= 0 or not np.isfinite(current_price):
            return 0
        max_shares = int((portfolio_value * 0.95) / current_price)
        return int(np.sign(shares) * min(abs(shares), max_shares))

    def calculate_slippage_price(self, price: Union[float, pd.Series, np.ndarray],
                                 signal: Union[int, float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
        """Vectorized-friendly: accepts scalars or array-like."""
        sl_adj = self.slippage * np.sign(signal)
        return price * (1 + sl_adj)

    # ------------------------------ Runner --------------------------------------

    def run(self,
            signal_threshold: float = 0.55,
            stop_loss_atr: float = 1.5,
            take_profit_atr: float = 3.0) -> Dict:
        print(f"Starting vectorized backtest with {len(self.prices)} periods...")
        self.signals = self.generate_signals(signal_threshold)
        self._run_trade_simulation_fast(stop_loss_atr, take_profit_atr)
        self._calculate_portfolio_finances_vectorized()
        metrics = self.calculate_performance_metrics()
        print(f"Backtest completed. Total trades: {len(self.trade_log)}")
        return metrics

    # ------------------------ Fast Trade Simulation -----------------------------

    def _run_trade_simulation_fast(self, stop_loss_atr: float, take_profit_atr: float):
        idx = self.prices.index
        n = len(idx)

        high = self.prices['High'].to_numpy(dtype=float)
        low = self.prices['Low'].to_numpy(dtype=float)
        close = self.prices['Close'].to_numpy(dtype=float)
        atr = self.atr.to_numpy(dtype=float)
        sig = self.signals.to_numpy(dtype=int)

        position = np.zeros(n, dtype=float)
        stop_loss = np.full(n, np.nan, dtype=float)
        take_profit = np.full(n, np.nan, dtype=float)

        self.trade_log = []

        # State machine
        for i in range(1, n):
            pos_prev, sl_prev, tp_prev = position[i-1], stop_loss[i-1], take_profit[i-1]
            position[i], stop_loss[i], take_profit[i] = pos_prev, sl_prev, tp_prev

            # Exits
            if pos_prev != 0.0 and (np.isfinite(sl_prev) or np.isfinite(tp_prev)):
                long_side = pos_prev > 0
                stop_hit = (low[i] <= sl_prev) if long_side else (high[i] >= sl_prev)
                tp_hit = (high[i] >= tp_prev) if long_side else (low[i] <= tp_prev)

                if stop_hit or tp_hit:
                    exit_price = sl_prev if stop_hit else tp_prev
                    # Log an exit (shares sign closes the position)
                    self._log_trade(idx[i], float(exit_price), -pos_prev, "Stop Loss" if stop_hit else "Take Profit")
                    position[i] = 0.0
                    stop_loss[i] = np.nan
                    take_profit[i] = np.nan
                    continue

            # Entries (flat only)
            if position[i] == 0.0 and sig[i] != 0:
                # NOTE: using initial_capital for sizing keeps it simple & stable
                size = self.calculate_position_size(self.initial_capital, close[i], atr[i], sig[i], stop_loss_atr)
                if size != 0:
                    self._log_trade(idx[i], float(close[i]), float(size), "Entry")
                    side = np.sign(size)
                    stop_loss[i] = close[i] - side * (atr[i] * stop_loss_atr)
                    take_profit[i] = close[i] + side * (atr[i] * take_profit_atr)
                    position[i] = size

        # Write back
        self.portfolio = pd.DataFrame(index=idx)
        self.portfolio['Close'] = close
        self.portfolio['Signal'] = sig.astype(np.int8)
        self.portfolio['Position'] = position
        self.portfolio['Stop_Loss'] = stop_loss
        self.portfolio['Take_Profit'] = take_profit

    def _log_trade(self, date, price, shares, reason):
        sp  = float(self.calculate_slippage_price(price, np.sign(shares)))
        val = float(shares) * sp
        com = abs(val) * self.commission
        self.trade_log.append({
            'Date': pd.to_datetime(date),
            'Type': reason,
            'Shares': float(shares),
            'Price': float(price),
            'Slippage_Price': sp,
            'Value': val,
            'Commission': com,
        })

    # ---------------- Vectorized Portfolio Accounting ---------------------------

    def _trade_df_enriched(self) -> pd.DataFrame:
        """
        Returns a sorted, enriched trade DataFrame with Slippage_Price, Value, Commission.
        Works whether self.trade_log already contains these columns or not.
        """
        if not self.trade_log:
            return pd.DataFrame(columns=['Type','Shares','Price','Slippage_Price','Value','Commission']) \
                    .astype({'Shares':'float64','Price':'float64','Slippage_Price':'float64',
                            'Value':'float64','Commission':'float64'})

        tdf = pd.DataFrame(self.trade_log).copy()

        # Ensure datetime index
        tdf['Date'] = pd.to_datetime(tdf['Date'])
        tdf.set_index('Date', inplace=True)
        tdf.sort_index(inplace=True)

        # Compute if missing (keeps compatibility if you later store these at log time)
        if 'Slippage_Price' not in tdf:
            tdf['Slippage_Price'] = self.calculate_slippage_price(tdf['Price'], np.sign(tdf['Shares']))
        if 'Value' not in tdf:
            tdf['Value'] = tdf['Shares'] * tdf['Slippage_Price']
        if 'Commission' not in tdf:
            tdf['Commission'] = tdf['Value'].abs() * self.commission

        return tdf


    def _calculate_portfolio_finances_vectorized(self):
        if self.portfolio is None:
            self.portfolio = pd.DataFrame(index=self.prices.index)

        tdf = self._trade_df_enriched()
        if tdf.empty:
            self.portfolio['Cash'] = self.initial_capital
            self.portfolio['Holdings'] = 0.0
            self.portfolio['Total'] = self.initial_capital
            self.portfolio['Returns'] = 0.0
            return

        daily_cash_change = (-tdf['Value'] - tdf['Commission']).groupby(level=0).sum()
        self.portfolio['Cash_Change'] = daily_cash_change.reindex(self.portfolio.index, fill_value=0.0)
        self.portfolio['Cash'] = self.initial_capital + self.portfolio['Cash_Change'].cumsum()

        self.portfolio['Holdings'] = self.portfolio['Position'] * self.portfolio['Close']
        self.portfolio['Total'] = self.portfolio['Cash'] + self.portfolio['Holdings']
        self.portfolio['Returns'] = (self.portfolio['Total']
                                    .pct_change()
                                    .replace([np.inf, -np.inf], np.nan)
                                    .fillna(0.0))

        # cache for metrics reuse
        self._last_trade_df = tdf


    # -------------------------- Performance Metrics -----------------------------

    def calculate_performance_metrics(self) -> Dict:
        if self.portfolio is None or self.portfolio.empty:
            raise ValueError("Must run backtest first")

        final_total = float(self.portfolio['Total'].iloc[-1])
        total_return = (final_total / self.initial_capital) - 1.0
        returns = self.portfolio['Returns'].astype(float)
        years = max(len(returns) / 252.0, 1e-9)

        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        volatility = float(returns.std(ddof=0)) * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0.0

        neg = returns[returns < 0]
        downside_dev = float(neg.std(ddof=0)) * np.sqrt(252) if len(neg) else 0.0
        sortino_ratio = (annualized_return - 0.02) / downside_dev if downside_dev > 0 else 0.0

        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative / rolling_max) - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # --- Trade statistics (enriched, consistent with accounting) ---
        tdf = getattr(self, "_last_trade_df", None)
        if tdf is None:
            tdf = self._trade_df_enriched()

        total_trades = 0
        win_rate = avg_win = avg_loss = profit_factor = total_commissions = 0.0

        if not tdf.empty:
            entry_trades = tdf[tdf['Type'] == 'Entry']
            exit_trades  = tdf[tdf['Type'].isin(['Stop Loss', 'Take Profit'])].copy()
            total_trades = int(len(exit_trades))
            total_commissions = float(tdf['Commission'].sum())

            pnl_list = []
            if total_trades > 0 and not entry_trades.empty:
                # Pair each exit with the most recent prior entry (flat-only model)
                for exit_time, exit_row in exit_trades.iterrows():
                    prior_entries = entry_trades[entry_trades.index <= exit_time]
                    if prior_entries.empty:
                        continue
                    entry_row = prior_entries.iloc[-1]

                    # Realized PnL NET of fees (matches cash aggregation)
                    pnl = (-entry_row['Value'] - entry_row['Commission']) + \
                        (-exit_row['Value']  - exit_row['Commission'])
                    pnl_list.append(float(pnl))

            if pnl_list:
                wins   = [p for p in pnl_list if p > 0]
                losses = [p for p in pnl_list if p < 0]
                win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
                avg_win  = float(np.mean(wins))   if wins   else 0.0
                avg_loss = float(np.mean(losses)) if losses else 0.0
                profit_factor = (sum(wins) / abs(sum(losses))) if losses else (np.inf if wins else 0.0)

        return {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Sortino Ratio': f"{sortino_ratio:.2f}",
            'Maximum Drawdown': f"{max_drawdown:.2%}",
            'Calmar Ratio': f"{calmar_ratio:.2f}",
            'Total Trades': total_trades,
            'Win Rate': f"{win_rate:.2%}",
            'Average Win': f"${avg_win:.2f}",
            'Average Loss': f"${avg_loss:.2f}",
            'Profit Factor': f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞",
            'Final Portfolio Value': f"${final_total:,.2f}",
            'Total Commissions': f"${total_commissions:.2f}",
        }

    def plot_equity_curve(self, figsize=(12, 8)):
        """Generate equity curve plot with drawdown."""
        if self.portfolio is None:
            raise ValueError("Must run backtest first")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Equity curve
        ax1.plot(self.portfolio.index, self.portfolio['Total'], 
                linewidth=2, label='Portfolio Value', color='navy')
        ax1.axhline(y=self.initial_capital, color='gray', 
                   linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown
        returns = self.portfolio['Returns'].dropna()
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max) - 1
        
        ax2.fill_between(drawdown.index, drawdown, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_performance_report(self):
        """Print a formatted performance report."""
        if self.portfolio is None:
            raise ValueError("Must run backtest first")
        
        metrics = self.calculate_performance_metrics()
        
        print("\n" + "="*60)
        print("PROJECT FLINT - BACKTEST PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\n📊 RETURN METRICS")
        print(f"   Total Return:        {metrics['Total Return']}")
        print(f"   Annualized Return:   {metrics['Annualized Return']}")
        print(f"   Final Value:         {metrics['Final Portfolio Value']}")
        
        print(f"\n⚖️  RISK METRICS")
        print(f"   Volatility:          {metrics['Volatility']}")
        print(f"   Sharpe Ratio:        {metrics['Sharpe Ratio']}")
        print(f"   Sortino Ratio:       {metrics['Sortino Ratio']}")
        print(f"   Maximum Drawdown:    {metrics['Maximum Drawdown']}")
        print(f"   Calmar Ratio:        {metrics['Calmar Ratio']}")
        
        print(f"\n🎯 TRADE STATISTICS")
        print(f"   Total Trades:        {metrics['Total Trades']}")
        print(f"   Win Rate:            {metrics['Win Rate']}")
        print(f"   Average Win:         {metrics['Average Win']}")
        print(f"   Average Loss:        {metrics['Average Loss']}")
        print(f"   Profit Factor:       {metrics['Profit Factor']}")
        
        print(f"\n💰 COST ANALYSIS")
        print(f"   Total Commissions:   {metrics['Total Commissions']}")
        
        print("\n" + "="*60)



    # -------------------------- Optional (unused) -------------------------------

    def _execute_trade(self, date, index, price, shares, reason):
        # Kept for compatibility; not used in the fast path
        pass

    def _update_portfolio_value(self, date, index):
        # Kept for compatibility; not used in the fast path
        pass


# ------------------------------- CLI Driver ------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a backtest for a specific ticker and model.")
    parser.add_argument('ticker', metavar='TICKER', type=str, help='The ticker symbol to backtest (e.g., "RYCEY").')
    parser.add_argument('model_name', metavar='MODEL_NAME', type=str, help='Model name for signals (e.g., "EnhancedRandomForest_Stacking").')
    args = parser.parse_args()

    # database setup
    try:
        from audit_logger import setup_backtest_db, log_backtest_result
        setup_backtest_db()
    except ImportError:
        print("Warning: audit_logger not found. Results will not be saved to the database.")
        log_backtest_result = None

    # 1) Load processed data
    try:
        from preprocess import prepare_and_cache_data
        from config import settings

        print(f"Loading processed data for {args.ticker}...")
        data_package = prepare_and_cache_data(args.ticker, settings.data.start_date, settings.data.end_date)
        df_features = data_package['df_features']

        # Respect existing DatetimeIndex, otherwise try to set from 'Date'
        if not isinstance(df_features.index, pd.DatetimeIndex):
            if 'Date' in df_features.columns:
                df_features = df_features.copy()
                df_features['Date'] = pd.to_datetime(df_features['Date'], errors="coerce")
                df_features.set_index('Date', inplace=True)
            else:
                raise KeyError("'Date' column not found and index is not DatetimeIndex. Cannot proceed with backtest.")

        df_features = df_features.sort_index()
    except Exception as e:
        print(f"Error: Could not load processed data for {args.ticker}. Please run preprocess.py first.")
        print(f"Details: {e}")
        return

    # 2) Load predictions
    try:
        import duckdb, json
        from pathlib import Path

        print(f"Loading predictions for model '{args.model_name}'...")
        db_path = Path('results/audit_log.duckdb')
        with duckdb.connect(str(db_path), read_only=True) as con:
            result = con.execute(
                "SELECT predictions FROM analysis_results "
                "WHERE ticker = ? AND model_name = ? "
                "ORDER BY execution_timestamp DESC LIMIT 1",
                [args.ticker, args.model_name]
            ).fetchone()

        if not result:
            raise ValueError(f"No predictions found in the database for {args.ticker} with model {args.model_name}.")

        predictions_data = json.loads(result[0])
        probabilities = pd.Series(
            predictions_data['probabilities'],
            index=pd.to_datetime(predictions_data['dates']),
            name="probability"
        ).sort_index()
    except Exception as e:
        print("Error: Could not load predictions from the audit database.")
        print(f"Details: {e}")
        return

    # 3) Prepare inputs
    price_data = df_features  # must contain at least Close (High/Low auto-filled if missing)

    if 'ATR14' not in df_features.columns:
        print("Error: 'ATR14' column not found in the feature data. Cannot run backtest.")
        return
    atr_series = df_features['ATR14']

    # 4) Run backtest
    print("\nInitializing Backtester...")
    backtester = Backtester(
        price_data=price_data,
        predictions=probabilities,
        atr_series=atr_series,
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )

    params = {'signal_threshold': 0.55, 'stop_loss_atr': 1.5, 'take_profit_atr': 3.0}
    results = backtester.run(**params)

    # 5) Report
    backtester.print_performance_report()
    backtester.plot_equity_curve()

    if 'log_backtest_result' in locals() and log_backtest_result:
        print("\nLogging backtest results to the database...")
        log_backtest_result(
            ticker=args.ticker,
            model_name=args.model_name,
            parameters=params,
            metrics=results
        )
        print("Results successfully logged.")

if __name__ == "__main__":
    main()