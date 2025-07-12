# predictors.py (Corrected PyTorch-only version)
import initialize
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import linregress
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from joblib import Parallel, delayed
from tqdm import tqdm
import config

# Helper function for the loss, equivalent to the original TF version
def jump_size_loss(y_pred, y_true):
    mu, log_sigma = y_pred.chunk(2, dim=-1)
    sigma = torch.exp(log_sigma) + 1e-6
    dist = torch.distributions.Normal(loc=mu, scale=sigma)
    return -torch.mean(dist.log_prob(y_true))

# PyTorch equivalent of the QuantumInspiredDense layer
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

# PyTorch equivalent of the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )
    def forward(self, x):
        return self.net(x)

# --- CORRECTED JumpNetwork ---
class JumpNetwork(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        seq_len, feature_dim = input_shape
        num_heads = 4
        
        self.projected_dim = 8
        self.input_projection = nn.Linear(feature_dim, self.projected_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=self.projected_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.projected_dim)
        
        self.vol_dense = nn.Linear(seq_len, 16)
        self.softplus = nn.Softplus()
        
        merged_size = self.projected_dim + 16
        
        self.jump_prob_head = nn.Sequential(nn.Linear(merged_size, 1), nn.Sigmoid())
        self.jump_size_head = nn.Linear(merged_size, 2)

    def forward(self, price_seq, vol_seq):
        projected_price_seq = self.input_projection(price_seq)
        
        attn_out, _ = self.attn(projected_price_seq, projected_price_seq, projected_price_seq)
        attn_out = self.norm(attn_out)
        attn_out_mean = torch.mean(attn_out, dim=1)
        
        vol_dense_out = self.softplus(self.vol_dense(vol_seq))
        
        merged = torch.cat([attn_out_mean, vol_dense_out], dim=1)
        
        jump_prob = self.jump_prob_head(merged)
        jump_size = self.jump_size_head(merged)
        return jump_prob, jump_size

# The main MonteCarloTrendFilter class, now using PyTorch internally
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
        self.jump_model = None # Will be initialized in fit()

    def _select_simulation_count(self, vol, jump_prob):
        self.policy_net.eval()
        with torch.no_grad():
            state = torch.tensor([[vol, jump_prob]], dtype=torch.float32).to(self.device)
            mult = self.policy_net(state).item()
        return max(100, int(self.base_n_sims * mult))

    def _train_jump_model(self, X_prices, X_vol, y_prob, y_size):
        input_shape = X_prices.shape[1:]
        self.jump_model = JumpNetwork(input_shape).to(self.device)
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
        for epoch in range(30): # Simplified training loop
            for xp_batch, xv_batch, yp_batch, ys_batch in loader:
                xp_batch, xv_batch, yp_batch, ys_batch = xp_batch.to(self.device), xv_batch.to(self.device), yp_batch.to(self.device), ys_batch.to(self.device)
                
                optimizer.zero_grad()
                prob_pred, size_pred = self.jump_model(xp_batch, xv_batch)
                
                loss_prob = bce_loss(prob_pred, yp_batch)
                loss_size = jump_size_loss(size_pred, ys_batch)
                total_loss = loss_prob + loss_size
                
                total_loss.backward()
                optimizer.step()
        return self.jump_model

    def _detect_jumps_neural(self, close, lookback=None):
        lookback = lookback or self.jump_window
        returns = np.log(close/close.shift(1)).dropna()
        volatilities = returns.rolling(5).std().dropna()
        aligned_data = pd.concat([returns, volatilities], axis=1, keys=['returns', 'vol']).dropna()
        
        aligned_returns = aligned_data['returns'].values
        aligned_vol = aligned_data['vol'].values
        
        Xp, Xv, y_prob, y_size = [], [], [], []
        std = returns.std()
        for i in range(lookback, len(aligned_returns)):
            Xp.append(aligned_returns[i-lookback:i].reshape(-1, 1))
            Xv.append(aligned_vol[i-lookback:i])
            is_jump = abs(aligned_returns[i]) > 3 * std
            y_prob.append(int(is_jump))
            y_size.append(aligned_returns[i] if is_jump else 0.0)
        
        Xp, Xv = np.array(Xp), np.array(Xv)
        
        jump_model = self._train_jump_model(Xp, Xv, y_prob, y_size)
        jump_model.eval()
        with torch.no_grad():
            rp = torch.tensor(returns.values[-lookback:].reshape(1, -1, 1), dtype=torch.float32).to(self.device)
            rv = torch.tensor(volatilities.values[-lookback:].reshape(1, -1), dtype=torch.float32).to(self.device)
            prob_pred, size_pred = jump_model(rp, rv)
            
            prob = prob_pred.item()
            mu = size_pred[0, 0].item()
            log_sigma = size_pred[0, 1].item()
        
        return prob, mu, np.exp(log_sigma)

    def _optimize_gbm_params(self, close):
        returns = np.log(close/close.shift(1)).dropna().values

        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e8)) + WhiteKernel(noise_level=1.0)
        
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=5
        )
        
        X = np.arange(len(returns)).reshape(-1,1)
        gpr.fit(X, returns)
        return np.mean(gpr.predict(X)), np.std(returns)

    def fit(self, data, warmup=252):
        if len(data) < warmup:
            raise ValueError(f"Need >= {warmup} days for calibration")
        print(f"MonteCarloTrendFilter using device: {self.device}")
        self.close_ = data['Close']
        hist = self.close_.iloc[-warmup:]
        self.gbm_params_ = self._optimize_gbm_params(hist)
        jp, jm, js = self._detect_jumps_neural(hist)
        self.jump_params_ = (jp * 252, jm, js)
        self.vol_regimes_ = GaussianMixture(2).fit(np.log(hist/hist.shift(1)).dropna().rolling(5).std().dropna().values.reshape(-1,1))
        self._is_fitted = True
        return self

    def predict(self, horizon=21):
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        dt = 1.0
        state_vol = np.std(np.log(self.close_[-self.jump_window:] / self.close_[-self.jump_window-1:-1]))
        n_sims = self._select_simulation_count(state_vol, self.jump_params_[0])

        slopes = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_single_simulation)(
                seed=i, S0=self.close_.iloc[-1], gbm_params=self.gbm_params_,
                jump_params=self.jump_params_, horizon=horizon, dt=dt
            ) for i in tqdm(range(n_sims))
        )
        
        slopes = [s for s in slopes if not np.isnan(s)]
        if not slopes:
            return {'up_prob': 0.5, 'down_prob': 0.5, 'trend_strength': 0.0, 'ci': [0,0], 'n_sims': n_sims, 'simulated_slopes': []}

        up_prob = np.mean(np.array(slopes) > 0)
        down_prob = np.mean(np.array(slopes) < 0)
        strength = np.tanh(np.mean(slopes) * 10)

        q_layer = QuantumInspiredDense(in_features=1, units=8).to(self.device)
        q_in = torch.tensor(np.array(slopes).reshape(-1,1), dtype=torch.float32).to(self.device)
        q_out = q_layer(q_in)
        tail_factor = torch.mean(q_out).item()
        strength *= (1 + 0.1 * tail_factor)

        return {
            'up_prob': up_prob, 'down_prob': down_prob, 'trend_strength': strength,
            'ci': np.percentile(slopes, [5, 95]).tolist(), 'n_sims': n_sims,
            'simulated_slopes': slopes
        }

def _run_single_simulation(seed, S0, gbm_params, jump_params, horizon, dt):
    np.random.seed(seed)
    mu, sigma = gbm_params
    lam, jump_mu, jump_sigma = jump_params
    steps = int(horizon / dt)
    dw = np.random.normal(scale=np.sqrt(dt), size=steps)
    jumps = np.random.poisson(lam * dt, size=steps)
    jump_sizes = np.random.normal(jump_mu, jump_sigma, size=steps) * jumps
    path = np.zeros(steps + 1)
    path[0] = S0
    max_price = S0 * 1000
    for t in range(1, steps + 1):
        next_price = path[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dw[t-1] + jump_sizes[t-1])
        path[t] = min(next_price, max_price) if np.isfinite(next_price) else max_price
    return linregress(np.arange(steps + 1), path).slope