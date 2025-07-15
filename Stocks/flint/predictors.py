import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import linregress 
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Custom loss for jump size (negative log-likelihood of normal)
def jump_size_loss(y_pred, y_true):
    mu, log_sigma = y_pred.chunk(2, dim=-1)
    sigma = torch.exp(log_sigma) + 1e-6
    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    return -torch.mean(dist.log_prob(y_true))

# Alternative JumpNetwork using LSTM for sequential pattern learning
class JumpNetworkLSTM(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        seq_len, feature_dim = input_shape  # feature_dim is 1 for returns sequence
        hidden_dim = 16
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.vol_dense = nn.Linear(seq_len, 16)
        self.softplus = nn.Softplus()
        # Combined features: hidden state + vol features
        merged_dim = hidden_dim + 16
        self.jump_prob_head = nn.Sequential(nn.Linear(merged_dim, 1), nn.Sigmoid())
        self.jump_size_head = nn.Linear(merged_dim, 2)  # outputs [mu, log_sigma]
    def forward(self, price_seq, vol_seq):
        # LSTM: use final hidden state as representation of price sequence
        lstm_out, (h_n, c_n) = self.lstm(price_seq)   # h_n: (1, batch, hidden_dim)
        seq_features = h_n[-1]                       # (batch, hidden_dim)
        # Volatility sequence features
        vol_features = self.softplus(self.vol_dense(vol_seq))  # (batch, 16), ensure positive
        merged = torch.cat([seq_features, vol_features], dim=1)
        jump_prob = self.jump_prob_head(merged)
        jump_size = self.jump_size_head(merged)
        return jump_prob, jump_size

# (Optional) Original QuantumInspiredDense remains unchanged
class QuantumInspiredDense(nn.Module):
    def __init__(self, in_features, units):
        super().__init__()
        self.theta_layer = nn.Linear(in_features, units, bias=False)
        self.phi_layer = nn.Linear(in_features, units, bias=False)
    def forward(self, inputs):
        theta_prod = self.theta_layer(inputs)
        phi_prod = self.phi_layer(inputs)
        real_part = torch.cos(theta_prod)
        imag_part = torch.sin(phi_prod)
        return torch.cat([real_part, imag_part], dim=-1)

# Policy network (unchanged, but should ideally be trained or replaced with a heuristic)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus()
        )
    def forward(self, x):
        return self.net(x)

class MonteCarloTrendFilter:
    def __init__(self, gbm_lookback=63, jump_window=21, n_optimize=50, base_n_sims=1000, n_jobs=-1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gbm_lookback = gbm_lookback
        self.jump_window = jump_window
        self.n_optimize = n_optimize
        self.base_n_sims = base_n_sims
        self.n_jobs = n_jobs
        self._is_fitted = False
        self.policy_net = PolicyNetwork(input_dim=2).to(self.device)
        self.jump_model = None  # will be initialized in fit()

    def _select_simulation_count(self, vol, jump_prob):
        """Select number of simulations based on current volatility and jump probability."""
        self.policy_net.eval()
        with torch.no_grad():
            state = torch.tensor([[vol, jump_prob]], dtype=torch.float32).to(self.device)
            mult_factor = self.policy_net(state).item()
        # Ensure at least 100 simulations, scale base count by factor
        return max(100, int(self.base_n_sims * mult_factor))

    def _train_jump_model(self, X_prices, X_vol, y_prob, y_size):
        """Train the jump detection model on historical data."""
        input_shape = X_prices.shape[1:]  # (lookback, 1)
        # Initialize jump_model (using LSTM-based network for better sequence learning)
        self.jump_model = JumpNetworkLSTM(input_shape).to(self.device)
        optimizer = torch.optim.Adam(self.jump_model.parameters(), lr=1e-3)
        bce_loss = nn.BCELoss()
        # Prepare data loader
        dataset = TensorDataset(
            torch.tensor(X_prices, dtype=torch.float32),
            torch.tensor(X_vol, dtype=torch.float32),
            torch.tensor(y_prob, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_size, dtype=torch.float32).unsqueeze(1)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        # Training loop (could incorporate validation and early stopping for real use)
        self.jump_model.train()
        for epoch in range(50):  # increased epochs for potentially better learning
            for xp_batch, xv_batch, yp_batch, ys_batch in loader:
                xp_batch = xp_batch.to(self.device)
                xv_batch = xv_batch.to(self.device)
                yp_batch = yp_batch.to(self.device)
                ys_batch = ys_batch.to(self.device)
                optimizer.zero_grad()
                prob_pred, size_pred = self.jump_model(xp_batch, xv_batch)
                # Compute losses for jump probability and jump size
                loss_prob = bce_loss(prob_pred, yp_batch)
                loss_size = jump_size_loss(size_pred, ys_batch)
                total_loss = loss_prob + loss_size
                total_loss.backward()
                optimizer.step()
        return self.jump_model

    def _detect_jumps_neural(self, close_series, lookback=None):
        """Detect jump probability and size using neural network model."""
        lookback = lookback or self.jump_window
        returns = np.log(close_series / close_series.shift(1)).dropna()
        volatilities = returns.rolling(5).std().dropna()
        data = pd.concat([returns, volatilities], axis=1, keys=['returns', 'vol']).dropna()
        ret_vals = data['returns'].values  # aligned returns
        vol_vals = data['vol'].values      # aligned volatilities
        Xp, Xv, y_prob, y_size = [], [], [], []
        std_dev = returns.std()
        for i in range(lookback, len(ret_vals)):
            # Sequence of last `lookback` returns and vols
            Xp.append(ret_vals[i-lookback:i].reshape(-1, 1))
            Xv.append(vol_vals[i-lookback:i])
            # Label: jump if return at i is an outlier (>3 std dev)
            is_jump = abs(ret_vals[i]) > 3 * std_dev
            y_prob.append(1 if is_jump else 0)
            y_size.append(ret_vals[i] if is_jump else 0.0)
        Xp = np.array(Xp)
        Xv = np.array(Xv)
        # Train jump detection model on this data
        self._train_jump_model(Xp, Xv, y_prob, y_size)
        # Use the trained model to predict jump parameters for the most recent window
        self.jump_model.eval()
        with torch.no_grad():
            rp = torch.tensor(ret_vals[-lookback:].reshape(1, -1, 1), dtype=torch.float32).to(self.device)
            rv = torch.tensor(vol_vals[-lookback:].reshape(1, -1), dtype=torch.float32).to(self.device)
            prob_pred, size_pred = self.jump_model(rp, rv)
            jump_prob = prob_pred.item()
            mu = size_pred[0, 0].item()
            log_sigma = size_pred[0, 1].item()
            jump_sigma = np.exp(log_sigma)
        return jump_prob, mu, jump_sigma

    def _optimize_gbm_params(self, close_series):
        """
        Estimate drift (mu) and volatility (sigma) for GBM.
        
        The Gaussian Process Regressor consistently found that the optimal model
        for the returns series is one with a very large length_scale, which
        approximates a simple linear regression. 
        
        Therefore, we replace the GPR with a direct linear regression on the log returns
        to estimate the drift (mu). This is simpler, faster, and resolves the
        ConvergenceWarning permanently.
        """
        returns = np.log(close_series / close_series.shift(1)).dropna()
        
        # Create a time index [0, 1, 2, ...] for the x-axis of the regression
        time_index = np.arange(len(returns))
        
        # Perform linear regression of returns against time
        # The slope of this regression is the drift (mu)
        slope, intercept, r_value, p_value, std_err = linregress(time_index, returns.values)
        
        # The estimated drift (mu) is the slope of the trend in returns.
        mu_est = slope
        
        # The volatility (sigma) is the standard deviation of the returns themselves.
        sigma_est = returns.std()

        # --- STABILIZATION FIX ---
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, # Increased from 5 to 10 for more robustness
            random_state=42 # Add a random state for reproducibility
        )
        
        return mu_est, sigma_est

    def fit(self, data, warmup=252):
        """
        Calibrate the model on historical data.
        `data` is a DataFrame with at least a 'Close' column.
        """
        if len(data) < warmup:
            raise ValueError(f"Need >= {warmup} days of data for calibration.")
        print(f"[MonteCarloTrendFilter] Using device: {self.device}")
        self.close_series = data['Close']
        recent_hist = self.close_series.iloc[-warmup:]
        # 1. Estimate GBM parameters (drift mu, volatility sigma)
        self.gbm_params_ = self._optimize_gbm_params(recent_hist)
        # 2. Estimate jump parameters via neural network
        jp, jm, js = self._detect_jumps_neural(recent_hist)
        # Annualize jump intensity: jp is probability in window, scale to per year (~252 trading days)
        lam_annual = jp * 252
        self.jump_params_ = (lam_annual, jm, js)
        # 3. Fit a Gaussian Mixture to recent volatility (to capture regimes)
        recent_vols = np.log(recent_hist / recent_hist.shift(1)).dropna().rolling(5).std().dropna().values.reshape(-1, 1)
        self.vol_regimes_ = GaussianMixture(n_components=2, random_state=42).fit(recent_vols)
        self._is_fitted = True
        return self

    def predict(self, horizon=21):
        """
        Perform Monte Carlo simulation to forecast trend over given horizon (days).
        Returns a dictionary with up/down probabilities, trend strength, confidence interval, etc.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        # Setup parameters
        mu, sigma = self.gbm_params_        # drift and base volatility
        lam, jump_mu, jump_sigma = self.jump_params_  # jump intensity (annual), jump size distribution
        # Determine current volatility state (std of last jump_window returns)
        recent_returns = np.log(self.close_series.iloc[-(self.jump_window+1):] / 
                                 self.close_series.iloc[-(self.jump_window+1):-1])
        state_vol = float(np.std(recent_returns))
        # Determine number of simulations (using policy network or default)
        n_sims = self._select_simulation_count(state_vol, lam)
        # **Incorporate volatility regime**: adjust sigma or split simulations by regime
        regime_means = self.vol_regimes_.means_.flatten()
        # Identify which component is high-vol and which is low-vol
        high_idx = int(np.argmax(regime_means))
        low_idx = 1 - high_idx
        # Probability that current state_vol belongs to high or low regime
        regime_proba = self.vol_regimes_.predict_proba(np.array([[state_vol]]))[0]
        p_high = regime_proba[high_idx]
        p_low = regime_proba[low_idx]
        # Decide how many simulations per regime (proportionally)
        n_high = int(n_sims * p_high)
        n_low = n_sims - n_high
        # Fetch regime-specific volatilities (means from GMM)
        sigma_low = float(regime_means[low_idx])
        sigma_high = float(regime_means[high_idx])
        
        # Prepare to run simulations vectorized
        slopes_list = []
        with tqdm(total=n_sims, desc="[MC Simulation]") as pbar:
            
            def simulate_paths(num_paths, vol, update_pbar=True):
                if num_paths <= 0: return np.array([])
                # (The inner logic of simulate_paths remains exactly the same)
                dt = 1.0; steps = int(horizon / dt)
                dw = torch.randn((num_paths, steps), device=self.device) * np.sqrt(dt)
                jumps = torch.poisson(torch.full((num_paths, steps), lam * dt, device=self.device))
                jump_size_rvs = torch.randn((num_paths, steps), device=self.device) * jump_sigma + jump_mu
                jump_increments = jumps * jump_size_rvs
                log_increments = (mu - 0.5 * (vol**2)) * dt + vol * dw + jump_increments
                S0 = float(self.close_series.iloc[-1])
                log_paths = torch.cumsum(torch.cat([torch.zeros((num_paths, 1), device=self.device), log_increments], dim=1), dim=1)
                price_paths = S0 * torch.exp(log_paths)
                T = steps + 1; t = torch.arange(0, T, device=self.device, dtype=torch.float32)
                sum_t, sum_t2 = t.sum(), (t**2).sum()
                sum_y = price_paths.sum(dim=1)
                sum_ty = torch.matmul(price_paths, t)
                numerator, denominator = T * sum_ty - sum_t * sum_y, T * sum_t2 - (sum_t ** 2)
                slopes = (numerator / denominator).cpu().numpy()
                
                if update_pbar:
                    pbar.update(num_paths) # Update the progress bar by the number of paths simulated
                return slopes

            # Run simulations for each regime and update the progress bar
            if n_low > 0:
                slopes_list.append(simulate_paths(n_low, sigma_low))
            if n_high > 0:
                slopes_list.append(simulate_paths(n_high, sigma_high))
        
        if slopes_list:
            slopes = np.concatenate(slopes_list)
        else:
            slopes = np.array([])

        if len(slopes) == 0:
            return {
                'up_prob': 0.5,
                'down_prob': 0.5,
                'trend_strength': 0.0,
                'ci': [0.0, 0.0],
                'n_sims': 0,
                'simulated_slopes': []
                }

        # Calculate up/down probabilities and trend strength
        up_prob = float(np.mean(slopes > 0))
        down_prob = float(np.mean(slopes < 0))
        # Trend strength: use tanh of mean slope * factor for bounded output
        strength = float(np.tanh(np.mean(slopes) * 10))
        # Apply QuantumInspiredDense adjustment for tail effects (if needed)
        q_layer = QuantumInspiredDense(in_features=1, units=8).to(self.device)
        q_in = torch.tensor(slopes.reshape(-1, 1), dtype=torch.float32).to(self.device)
        q_out = q_layer(q_in)              # shape: (n_sims, 16)
        tail_factor = float(torch.mean(q_out))
        strength *= (1 + 0.1 * tail_factor)  # small adjustment based on distribution shape

        # 90% confidence interval for the trend slope distribution (5th and 95th percentiles)
        ci_lower, ci_upper = np.percentile(slopes, [5, 95]).tolist()
        return {
            'up_prob': up_prob,
            'down_prob': down_prob,
            'trend_strength': strength,
            'ci': [ci_lower, ci_upper],
            'n_sims': int(n_sims),
            'simulated_slopes': slopes.tolist()
        }
