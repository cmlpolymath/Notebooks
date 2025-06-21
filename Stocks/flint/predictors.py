# predictors.py
# predictors.py
import initialize
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import Dense, Input, concatenate, MultiHeadAttention, LayerNormalization, Layer, Lambda
from keras.models import Model
from keras.optimizers import Adam
from tensorflow_probability import distributions as tfd
from scipy.stats import linregress
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from joblib import Parallel, delayed
from tqdm import tqdm
import config

# ==============================================================================
# === STEP 1: CREATE A NEW, SELF-CONTAINED FUNCTION FOR PARALLEL EXECUTION ===
# ==============================================================================
def _run_single_simulation(seed, S0, gbm_params, jump_params, horizon, dt):
    """
    A self-contained function that runs one full Monte Carlo path.
    This is what will be sent to each parallel worker. It has no dependency
    on the MonteCarloTrendFilter class instance.
    """
    np.random.seed(seed)
    mu, sigma = gbm_params
    lam, jump_mu, jump_sigma = jump_params
    
    steps = int(horizon / dt)
    dw = np.random.normal(scale=np.sqrt(dt), size=steps)
    jumps = np.random.poisson(lam * dt, size=steps)
    jump_sizes = np.random.normal(jump_mu, jump_sigma, size=steps) * jumps
    
    path = np.zeros(steps + 1)
    path[0] = S0
    
    # Add a cap for the maximum simulated price to prevent overflow
    # Set it to a very large number, e.g., 1000x the starting price
    max_price = S0 * 1000

    for t in range(1, steps + 1):
        # Calculate the next price
        next_price = path[t-1] * np.exp((mu - 0.5 * sigma**2) * dt +
                                        sigma * dw[t-1] + jump_sizes[t-1])
        
        # If the price overflows or becomes absurdly large, cap it.
        if not np.isfinite(next_price) or next_price > max_price:
            path[t] = max_price
        else:
            path[t] = next_price                                     
    return linregress(np.arange(horizon + 1), path).slope

# The loss function is still needed for the main process training
def jump_size_loss(y_true, y_pred):
    mu, log_sigma = tf.split(y_pred, 2, axis=-1)
    sigma = tf.exp(log_sigma) + 1e-6
    dist = tfd.Normal(loc=mu, scale=sigma)
    return -tf.reduce_mean(dist.log_prob(y_true))

class QuantumInspiredDense(Layer):
    # This class is correct, no changes needed
    def __init__(self, units, **kwargs):
        super(QuantumInspiredDense, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.theta_layer = Dense(self.units, use_bias=False, name='theta_matmul')
        self.phi_layer = Dense(self.units, use_bias=False, name='phi_matmul')
        super(QuantumInspiredDense, self).build(input_shape)
    def call(self, inputs):
        theta_prod = self.theta_layer(inputs)
        phi_prod = self.phi_layer(inputs)
        cos_layer = Lambda(lambda x: tf.math.cos(x))
        sin_layer = Lambda(lambda x: tf.math.sin(x))
        real_part = cos_layer(theta_prod)
        imag_part = sin_layer(phi_prod)
        return concatenate([real_part, imag_part])
    def get_config(self):
        config = super(QuantumInspiredDense, self).get_config()
        config.update({"units": self.units})
        return config

class MonteCarloTrendFilter:
    def __init__(self, gbm_lookback=63, jump_window=21, n_optimize=50, base_n_sims=1000, n_jobs=-1):
        self.gbm_lookback = gbm_lookback
        self.jump_window = jump_window
        self.n_optimize = n_optimize
        self.base_n_sims = base_n_sims
        self.n_jobs = n_jobs
        self._is_fitted = False
        self.policy_net = self._build_policy_network(input_dim=2)

    def _build_policy_network(self, input_dim):
        inp = Input(shape=(input_dim,))
        x = Dense(32, activation='relu')(inp)
        x = Dense(16, activation='relu')(x)
        out = Dense(1, activation='softplus', name='sim_count')(x)
        return Model(inp, out)

    def _select_simulation_count(self, vol, jump_prob):
        state = np.array([[vol, jump_prob]]).astype(np.float32)
        mult = self.policy_net.predict(state, verbose=0)[0,0]
        return max(100, int(self.base_n_sims * mult)) # Ensure at least 100 sims

    def _build_jump_network(self, input_shape):
        # This is correct, no changes needed
        price_input = Input(shape=input_shape, name='price_seq')
        vol_input = Input(shape=(input_shape[0],), name='vol_seq')
        attn = MultiHeadAttention(num_heads=4, key_dim=8)(price_input, price_input)
        attn = LayerNormalization()(attn)
        attn_out = Lambda(lambda x: tf.reduce_mean(x, axis=1))(attn)
        vol_dense = Dense(16, activation='softplus')(vol_input)
        merged = concatenate([attn_out, vol_dense])
        jump_prob = Dense(1, activation='sigmoid', name='jump_prob')(merged)
        jump_size = Dense(2, activation='linear', name='jump_size')(merged)
        return Model(inputs=[price_input, vol_input], outputs=[jump_prob, jump_size])

    def _train_jump_model(self, X_prices, X_vol, y_jumps):
        # We pass the function object directly, no need for serialization tricks
        jump_model = self._build_jump_network(X_prices.shape[1:])
        jump_model.compile(
            optimizer=Adam(1e-3),
            loss={'jump_prob': 'binary_crossentropy', 'jump_size': jump_size_loss},
            metrics={'jump_prob': 'accuracy'}
        )
        jump_model.fit([X_prices, X_vol], y_jumps,
                             epochs=30, batch_size=32,
                             validation_split=0.2, verbose=0)
        return jump_model # Return the trained model

    def _detect_jumps_neural(self, close, lookback=None):
        lookback = lookback or self.jump_window
        returns = np.log(close/close.shift(1)).dropna()
        volatilities = returns.rolling(5).std().dropna()
        aligned_data = pd.concat([returns, volatilities], axis=1, keys=['returns', 'vol']).dropna()
        aligned_returns = aligned_data['returns']
        aligned_vol = aligned_data['vol']
        Xp, Xv, y_prob, y_size = [], [], [], []
        std = returns.std()
        for i in range(lookback, len(aligned_returns)):
            Xp.append(aligned_returns.values[i-lookback:i].reshape(-1,1))
            Xv.append(aligned_vol.values[i-lookback:i])
            is_jump = abs(aligned_returns.values[i]) > 3*std
            y_prob.append(int(is_jump))
            y_size.append(aligned_returns.values[i] if is_jump else 0.0)
        y_jumps = {
            'jump_prob': np.array(y_prob, dtype=np.float32),
            'jump_size': np.array(y_size, dtype=np.float32)
        }
        
        # Train the model here
        jump_model = self._train_jump_model(np.array(Xp), np.array(Xv), y_jumps)

        rp = returns.values[-lookback:].reshape(1, -1, 1)
        rv = volatilities.values[-lookback:].reshape(1, -1)
        predictions = jump_model.predict([rp, rv], verbose=0)
        prob = predictions[0][0, 0]
        mu = predictions[1][0, 0]
        log_sigma = predictions[1][0, 1]
        return prob, mu, np.exp(log_sigma)

    def _optimize_gbm_params(self, close):
        returns = np.log(close/close.shift(1)).dropna().values
        kernel = RBF(1.0) + WhiteKernel(1.0)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        X = np.arange(len(returns)).reshape(-1,1)
        gpr.fit(X, returns)
        mu = np.mean(gpr.predict(X))
        sigma = np.std(returns)
        return mu, sigma

    def fit(self, data, warmup=252):
        if len(data) < warmup:
            raise ValueError(f"Need >= {warmup} days for calibration")
        self.close_ = data['Close']
        hist = self.close_.iloc[-warmup:]
        self.gbm_params_ = self._optimize_gbm_params(hist)
        jp, jm, js = self._detect_jumps_neural(hist)
        self.jump_params_ = (jp * 252, jm, js)
        returns = np.log(hist/hist.shift(1)).dropna()
        self.vol_regimes_ = GaussianMixture(2).fit(returns.rolling(5).std().dropna().values.reshape(-1,1))
        self._is_fitted = True
        return self

    # ==============================================================================
    # === STEP 2: MODIFY THE PREDICT METHOD TO USE THE NEW STANDALONE FUNCTION ===
    # ==============================================================================
    def predict(self, horizon=21):
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        # Use a fixed dt for the prediction horizon
        dt = 1.0
        prediction_horizon_steps = horizon

        state_vol = np.std(np.log(self.close_[-self.jump_window:] / self.close_[-self.jump_window-1:-1]))
        n_sims = self._select_simulation_count(state_vol, self.jump_params_[0])

        # We pass simple, picklable Python objects (tuples, floats) to the parallel function
        slopes = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_single_simulation)(
                seed=i,
                S0=self.close_.iloc[-1],
                gbm_params=self.gbm_params_,
                jump_params=self.jump_params_,
                horizon=prediction_horizon_steps,
                dt=dt
            ) for i in tqdm(range(n_sims))
        )
        
        slopes = [s for s in slopes if not np.isnan(s)] # Filter out potential NaNs
        if not slopes:
            return {'up_prob': 0.5, 'down_prob': 0.5, 'trend_strength': 0.0, 'ci': [0,0], 'n_sims': n_sims, 'simulated_slopes': []}

        up_prob = np.mean(np.array(slopes) > 0)
        down_prob = np.mean(np.array(slopes) < 0)
        strength = np.tanh(np.mean(slopes) * 10)

        q_in = tf.constant(np.array(slopes).reshape(-1,1), dtype=tf.float32)
        q_layer = QuantumInspiredDense(8)
        q_out = q_layer(q_in)
        tail_factor = tf.reduce_mean(q_out).numpy()
        strength *= (1 + 0.1 * tail_factor)

        result = {
            'up_prob': up_prob,
            'down_prob': down_prob,
            'trend_strength': strength,
            'ci': np.percentile(slopes, [5, 95]).tolist(),
            'n_sims': n_sims,
            'simulated_slopes': slopes
        }
        return result

    def get_ensemble_signal(self, lookback=21):
        # This method is no longer used in the main run.py but is kept for potential future use
        horizons = [5, 10, 21]
        outs = [self.predict(h) for h in horizons]
        vols = np.log(self.close_[-1] / self.close_[-lookback]).std()
        weights = np.exp(-np.array(horizons) / (vols * 100))
        weights /= weights.sum()
        comp = sum(o['trend_strength'] * w for o, w in zip(outs, weights))
        return float(np.clip(comp, -1, 1))

def add_mc_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates historical trend strength from the Monte Carlo filter
    and adds it as a feature. This is computationally intensive.
    """
    print("Generating Monte Carlo trend feature... (this may take a while)")
    mc_filter = MonteCarloTrendFilter(**config.MC_FILTER_PARAMS)
    
    # We need a warm-up period for the first prediction
    warmup = 252 
    trend_strengths = [np.nan] * warmup

    # Iterate through the dataframe to generate a signal for each day
    for i in tqdm(range(warmup, len(df))):
        # Use historical data up to the current point
        historical_data = df.iloc[:i]
        
        # Fit and get the signal
        mc_filter.fit(historical_data, warmup=warmup)
        # Using the ensemble signal for a robust measure
        signal = mc_filter.get_ensemble_signal(lookback=21)
        trend_strengths.append(signal)

    df_copy = df.copy()
    # df_copy['mc_trend_strength'] = trend_strengths
    return df_copy
