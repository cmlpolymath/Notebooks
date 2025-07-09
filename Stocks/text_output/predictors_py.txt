import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfd
from scipy.stats import linregress, norm
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.optimize import linprog
from scipy.signal import find_peaks

class MonteCarloTrendFilter:
    """
    Advanced Monte Carlo simulator with:
    - Neural jump diffusion estimation
    - Bayesian parameter optimization
    - Regime-switching volatility
    - Ensemble trend scoring
    
    Methods:
    - fit(data, warmup=252): Calibrate model to historical OHLCV data
    - predict(horizon=5, n_sims=5000): Generate probabilistic forecasts
    - get_ensemble_signal(lookback=21): Composite trend strength score
    """
    
    def __init__(self, gbm_lookback=63, jump_window=21, n_optimize=50, n_jobs=-1):
        self.gbm_lookback = gbm_lookback
        self.jump_window = jump_window
        self.n_optimize = n_optimize
        self.n_jobs = n_jobs
        self._is_fitted = False

    def _build_jump_network(self, input_shape):
        """Dual-input neural network for jump detection"""
        price_input = Input(shape=input_shape)
        vol_input = Input(shape=(input_shape[0],))
        
        # Temporal pattern analysis
        lstm_out = LSTM(32, return_sequences=False)(price_input)
        
        # Volatility feature engineering
        vol_dense = Dense(16, activation='softplus')(vol_input)
        
        # Joint processing
        merged = concatenate([lstm_out, vol_dense])
        jump_prob = Dense(1, activation='sigmoid', name='jump_prob')(merged)
        jump_size = Dense(2, activation='linear', name='jump_size')(merged)
        
        return Model(inputs=[price_input, vol_input], 
                    outputs=[jump_prob, jump_size])

    def _train_jump_model(self, X_prices, X_vol, y_jumps):
        """Train the neural jump estimator"""
        self.jump_model = self._build_jump_network(X_prices.shape[1:])
        self.jump_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'jump_prob': 'binary_crossentropy',
                'jump_size': self._jump_size_loss
            },
            metrics={'jump_prob': 'accuracy'}
        )
        self.jump_model.fit(
            [X_prices, X_vol],
            y_jumps,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

    def _jump_size_loss(self, y_true, y_pred):
        """Negative log likelihood for jump sizes"""
        mu, log_sigma = tf.split(y_pred, 2, axis=-1)
        sigma = tf.exp(log_sigma)
        return -tfd.Normal(mu, sigma).log_prob(y_true)

    def _detect_jumps_neural(self, close, lookback=21):
        """Neural jump detection pipeline"""
        returns = np.log(close / close.shift(1)).dropna().values
        volatilities = returns.rolling(5).std().dropna().values
        
        # Create labeled dataset
        X_p, X_v, y = [], [], []
        for i in range(lookback, len(returns)):
            X_p.append(returns[i-lookback:i].reshape(-1, 1))
            X_v.append(volatilities[i-lookback:i])
            y.append([
                int(abs(returns[i]) > 3*returns.std()),
                returns[i] if abs(returns[i]) > 3*returns.std() else 0
            ])
        
        self._train_jump_model(np.array(X_p), np.array(X_v), np.array(y))
        
        # Current regime prediction
        recent_p = returns[-lookback:].reshape(1, -1, 1)
        recent_v = volatilities[-lookback:].reshape(1, -1)
        prob, (mu, log_sigma) = self.jump_model.predict([recent_p, recent_v])
        return prob[0][0], mu[0][0], np.exp(log_sigma)[0][0]

    def _optimize_gbm_params(self, close):
        """Bayesian optimization of GBM parameters"""
        returns = np.log(close / close.shift(1)).dropna()
        
        def neg_log_likelihood(params):
            mu, sigma = params
            ll = norm.logpdf(returns, loc=mu, scale=sigma)
            return -np.sum(ll)
        
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        X = np.arange(len(returns)).reshape(-1, 1)
        gpr.fit(X, returns)
        
        opt_result = linprog(
            lambda x: neg_log_likelihood(x),
            bounds=[(returns.min(), returns.max()), 
                   (0.001, returns.std()*3)]
        )
        return opt_result.x

    def _simulate_jump_diffusion(self, S0, params, T=1, dt=1/252, n_sims=1000):
        """Merton Jump Diffusion simulation"""
        mu, sigma, lam, jump_mu, jump_sigma = params
        n_steps = int(T/dt)
        
        # Brownian motion
        dw = np.random.normal(scale=np.sqrt(dt), size=(n_sims, n_steps))
        # Compound Poisson process
        jumps = np.random.poisson(lam*dt, (n_sims, n_steps))
        jump_sizes = np.random.normal(jump_mu, jump_sigma, 
                                    (n_sims, n_steps)) * jumps
        
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = S0
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5*sigma**2)*dt + sigma*dw[:, t-1] + jump_sizes[:, t-1]
            )
        return paths

    def fit(self, data, warmup=252):
        """Calibrate model to historical data"""
        if len(data) < warmup:
            raise ValueError(f"Need ≥{warmup} days for calibration")
            
        self.close_ = data['Close']
        close = self.close_.iloc[-warmup:]
        returns = np.log(close / close.shift(1)).dropna()
        
        # GBM parameters
        self.gbm_params_ = self._optimize_gbm_params(close)
        
        # Neural jump detection
        self.jump_prob_, self.jump_mu_, self.jump_sigma_ = \
            self._detect_jumps_neural(close)
        self.jump_params_ = (
            self.jump_prob_ * 10,  # Annualized jump frequency
            self.jump_mu_,
            self.jump_sigma_
        )
        
        # Volatility regimes
        self.vol_regimes_ = GaussianMixture(n_components=2).fit(
            returns.rolling(5).std().dropna().values.reshape(-1,1))
        
        self._is_fitted = True
        return self

    def predict(self, horizon=5, n_sims=5000):
        """Generate probabilistic trend forecast"""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
            
        def run_simulation(i):
            np.random.seed(i)
            paths = self._simulate_jump_diffusion(
                self.close_.iloc[-1], 
                (*self.gbm_params_, *self.jump_params_),
                T=horizon/252,
                n_sims=1
            )
            return linregress(np.arange(horizon+1), paths[0]).slope
        
        slopes = Parallel(n_jobs=self.n_jobs)(
            delayed(run_simulation)(i) for i in tqdm(range(n_sims))
        )
        # Regime adjustment
        current_vol = np.log(self.close_.iloc[-1]/self.close_.iloc[-2])
        regime = self.vol_regimes_.predict([[current_vol]])[0]
        vol_weight = 1.5 if regime == 1 else 0.8
        
        # Trend metrics
        up_prob = np.mean(np.array(slopes) > (0.01/vol_weight))
        down_prob = np.mean(np.array(slopes) < (-0.01/vol_weight))
        trend_strength = np.tanh(np.mean(slopes) * 10 * vol_weight)
        
        return {
            'up_prob': up_prob,
            'down_prob': down_prob,
            'trend_strength': trend_strength,
            'confidence_interval': np.percentile(slopes, [5, 95])
        }

    def get_ensemble_signal(self, lookback=21):
        """Composite signal from multiple time horizons"""
        horizons = [5, 10, 21]  # Short/medium/long
        results = [self.predict(h) for h in horizons]
        
        # Volatility-weighted averaging
        vol = np.log(self.close_[-1] / self.close_[-lookback]).std()
        weights = np.exp(-np.array(horizons)/(vol*100))
        weights /= weights.sum()
        
        composite = sum(r['trend_strength']*w for r,w in zip(results,weights))
        return np.clip(composite, -1, 1)
    
#    Attention mechanisms for interpretable jump detection

#    Reinforcement learning for adaptive simulation counts

#    Quantum-inspired neural layers for fat-tail modeling