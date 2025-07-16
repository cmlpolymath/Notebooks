# predictors.py

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from scipy.stats import linregress

# --- Helper Functions and Networks ---

def jump_size_loss(y_pred, y_true):
    """Custom loss for jump size (negative log-likelihood of normal distribution)."""
    mu, log_sigma = y_pred.chunk(2, dim=-1)
    sigma = torch.exp(log_sigma) + 1e-6
    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    return -torch.mean(dist.log_prob(y_true))

class JumpNetworkLSTM(nn.Module):
    """LSTM-based network to detect jump probability and size from return/volatility sequences."""
    def __init__(self, input_shape):
        super().__init__()
        seq_len, feature_dim = input_shape
        hidden_dim = 16
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.vol_dense = nn.Linear(seq_len, 16)
        self.softplus = nn.Softplus()
        merged_dim = hidden_dim + 16
        self.jump_prob_head = nn.Sequential(nn.Linear(merged_dim, 1), nn.Sigmoid())
        self.jump_size_head = nn.Linear(merged_dim, 2)

    def forward(self, price_seq, vol_seq):
        lstm_out, (h_n, c_n) = self.lstm(price_seq)
        seq_features = h_n[-1]
        vol_features = self.softplus(self.vol_dense(vol_seq))
        merged = torch.cat([seq_features, vol_features], dim=1)
        jump_prob = self.jump_prob_head(merged)
        jump_size = self.jump_size_head(merged)
        return jump_prob, jump_size

class QuantumInspiredDense(nn.Module):
    """Quantum-inspired layer for tail adjustments."""
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

class PolicyNetwork(nn.Module):
    """Network to determine the number of simulations based on market state."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus()
        )

    def forward(self, x):
        return self.net(x)

# --- Main Predictor Class ---

class MonteCarloTrendFilter:
    def __init__(self,
                 gbm_lookback=63,
                 jump_window=21,
                 base_n_sims=1000,
                 seed: int = 42):
        # --- Reproducibility ---
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Note: Deterministic algorithms can sometimes be slower.
        # Set to False if performance is a critical issue and slight variations are acceptable.
        try:
        #     torch.use_deterministic_algorithms(True)
        #     torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as e:
            print(f"Warning: Could not set deterministic algorithms. Results may vary slightly. Error: {e}")


        # --- Device & Parameters ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gbm_lookback = gbm_lookback
        self.jump_window = jump_window
        self.base_n_sims = base_n_sims

        # --- Model Components ---
        self.q_layer = QuantumInspiredDense(in_features=1, units=8).to(self.device)
        for p in self.q_layer.parameters():
            if p.requires_grad:
                torch.nn.init.xavier_uniform_(p)

        self.policy_net = PolicyNetwork(input_dim=2).to(self.device)
        self.sobol = SobolEngine(dimension=1, scramble=True, seed=self.seed)
        self.jump_model = None
        self.vol_regimes_ = None
        self._is_fitted = False

    def _select_simulation_count(self, vol, jump_prob):
        self.policy_net.eval()
        with torch.no_grad():
            state = torch.tensor([[vol, jump_prob]], dtype=torch.float32, device=self.device)
            factor = self.policy_net(state).item()
        return max(100, int(self.base_n_sims * factor))

    def _optimize_gbm_params(self, close_series):
        """Estimates drift (mu) and volatility (sigma) using linear regression on log returns."""
        returns = np.log(close_series / close_series.shift(1)).dropna()
        t_idx = np.arange(len(returns))
        slope, _, _, _, _ = linregress(t_idx, returns.values)
        sigma = returns.std()
        return slope, sigma

    def _train_jump_model(self, X_prices, X_vol, y_prob, y_size):
        """Trains the JumpNetworkLSTM on historical data."""
        input_shape = X_prices.shape[1:]
        self.jump_model = JumpNetworkLSTM(input_shape).to(self.device)
        optimizer = torch.optim.Adam(self.jump_model.parameters(), lr=1e-3)
        bce_loss = nn.BCELoss()
        
        dataset = TensorDataset(
            torch.tensor(X_prices, dtype=torch.float32),
            torch.tensor(X_vol, dtype=torch.float32),
            torch.tensor(y_prob, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_size, dtype=torch.float32).unsqueeze(1)
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.jump_model.train()
        for epoch in range(30): # A fixed number of epochs for simplicity
            for xp_batch, xv_batch, yp_batch, ys_batch in loader:
                xp_batch, xv_batch = xp_batch.to(self.device), xv_batch.to(self.device)
                yp_batch, ys_batch = yp_batch.to(self.device), ys_batch.to(self.device)
                
                optimizer.zero_grad()
                prob_pred, size_pred = self.jump_model(xp_batch, xv_batch)
                
                loss_prob = bce_loss(prob_pred, yp_batch)
                loss_size = jump_size_loss(size_pred, ys_batch)
                total_loss = loss_prob + loss_size
                
                total_loss.backward()
                optimizer.step()
        return self.jump_model

    def _detect_jumps_neural(self, close_series):
        """Prepares data and uses the trained LSTM to find jump parameters for the latest period."""
        lookback = self.jump_window
        returns = np.log(close_series / close_series.shift(1)).dropna()
        volatilities = returns.rolling(5).std().dropna()
        data = pd.concat([returns, volatilities], axis=1, keys=['returns', 'vol']).dropna()
        
        ret_vals, vol_vals = data['returns'].values, data['vol'].values
        Xp, Xv, y_prob, y_size = [], [], [], []
        
        std_dev = returns.std()
        for i in range(lookback, len(ret_vals)):
            Xp.append(ret_vals[i-lookback:i].reshape(-1, 1))
            Xv.append(vol_vals[i-lookback:i])
            is_jump = abs(ret_vals[i]) > 3 * std_dev
            y_prob.append(1 if is_jump else 0)
            y_size.append(ret_vals[i] if is_jump else 0.0)
        
        Xp, Xv = np.array(Xp), np.array(Xv)
        
        self._train_jump_model(Xp, Xv, y_prob, y_size)
        
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

    def fit(self, data, warmup=252):
        """Calibrates all model components on historical data."""
        if len(data) < warmup:
            raise ValueError(f"Need >= {warmup} days of data for calibration.")
        
        print(f"[MonteCarloTrendFilter] Using device: {self.device}")
        self.close_series = data['Close']
        hist = self.close_series.iloc[-warmup:]
        
        # 1. Estimate GBM parameters
        self.gbm_params_ = self._optimize_gbm_params(hist)
        
        # 2. Estimate Jump parameters using the neural network
        jp, jm, js = self._detect_jumps_neural(hist)
        self.jump_params_ = (jp * 252, jm, js) # Annualize jump intensity
        
        # 3. Fit volatility regimes
        vols = np.log(hist / hist.shift(1)).dropna().rolling(5).std().dropna().values.reshape(-1, 1)
        self.vol_regimes_ = GaussianMixture(n_components=2, random_state=self.seed).fit(vols)
        
        self._is_fitted = True
        return self

    def predict(self, horizon=21):
        """Performs a stable, reproducible Monte Carlo simulation."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        mu, _ = self.gbm_params_ # We use regime-specific sigma, so only mu is needed here
        lam, jump_mu, jump_sigma = self.jump_params_

        # Determine current volatility state
        rets = np.log(self.close_series.iloc[-(self.jump_window+1):] / self.close_series.iloc[-(self.jump_window+1):-1])
        state_vol = float(np.std(rets))

        n_sims = self._select_simulation_count(state_vol, lam)
        
        # Split simulations by GMM volatility regimes
        probs = self.vol_regimes_.predict_proba([[state_vol]])[0]
        idx_high = int(np.argmax(self.vol_regimes_.means_))
        n_low = int(n_sims * probs[1-idx_high])
        n_high = n_sims - n_low
        comp_counts = [n_low, n_high]
        sigmas = [float(self.vol_regimes_.means_[1-idx_high]), float(self.vol_regimes_.means_[idx_high])]

        slopes = []
        dt = 1.0
        steps = int(horizon / dt)
        S0 = float(self.close_series.iloc[-1])

        with tqdm(total=n_sims, desc="[MC Simulation]") as pbar:
            for count, vol_val in zip(comp_counts, sigmas):
                if count <= 0:
                    continue
                
                num_draws = (count + 1) // 2
                
                # 1. Brownian Motion (Sobol + Antithetic)
                u = self.sobol.draw(num_draws * steps).view(num_draws, steps)
                z = torch.erfinv(2 * u - 1) * np.sqrt(2)
                z = z.to(self.device)
                z_pair = torch.cat([z, -z], dim=0)[:count]
                brownian_increments = vol_val * z_pair * np.sqrt(dt)

                # 2. Jump Process
                jumps = torch.poisson(torch.full((count, steps), lam * dt, device=self.device))
                jump_sizes = torch.randn((count, steps), device=self.device) * jump_sigma + jump_mu
                jump_increments = jumps * jump_sizes

                # 3. Combine all increments
                drift = (mu - 0.5 * vol_val**2) * dt
                incr = drift + brownian_increments + jump_increments
                
                # 4. Calculate paths and slopes
                log_paths = torch.cumsum(torch.cat([torch.zeros((count, 1), device=self.device), incr], dim=1), dim=1)
                price_paths = S0 * torch.exp(log_paths)
                
                t = torch.arange(0, steps + 1, device=self.device, dtype=torch.float32)
                T = steps + 1
                denom = T * (t**2).sum() - (t.sum()**2)
                sum_y = price_paths.sum(dim=1)
                sum_ty = price_paths.matmul(t)
                slope_vals = ((T * sum_ty - t.sum() * sum_y) / denom).cpu().numpy()
                slopes.append(slope_vals)
                
                pbar.update(count)

        slopes = np.concatenate(slopes) if slopes else np.array([])

        if slopes.size == 0:
            return dict(up_prob=0.5, down_prob=0.5, trend_strength=0.0, ci=[0, 0], n_sims=0, simulated_slopes=[])

        up = float(np.mean(slopes > 0))
        down = float(np.mean(slopes < 0))
        strength = float(np.tanh(slopes.mean() * 10))

        with torch.no_grad():
            q_in = torch.tensor(slopes.reshape(-1, 1), dtype=torch.float32, device=self.device)
            tail_factor = self.q_layer(q_in).mean().item()
        strength *= (1 + 0.1 * tail_factor)

        ci = np.percentile(slopes, [5, 95]).tolist()
        return {
            'up_prob': up,
            'down_prob': down,
            'trend_strength': strength,
            'ci': ci,
            'n_sims': int(n_sims),
            'simulated_slopes': slopes.tolist()
        }