# economics.py

import os
import asyncio
import uvloop
import aiohttp
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from config import FRED_INDICATORS

# ─── Configuration ────────────────────────────────────────────────────────────
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
load_dotenv(dotenv_path='/workspaces/smart_dev/ml-unified/vars.env')

API_KEY   = os.getenv('FRED_API_KEY')
OBS_START = '2000-01-01'
OBS_URL   = 'https://api.stlouisfed.org/fred/series/observations'

if not API_KEY:
    raise RuntimeError("FRED_API_KEY not set in vars.env")

# ─── Internal helper ─────────────────────────────────────────────────────────
async def _fetch_series(session: aiohttp.ClientSession, name: str, code: str) -> pd.Series:
    """Fetch full history for one FRED series as a pandas Series (float dtype)."""
    params = {
        'series_id':         code,
        'api_key':           API_KEY,
        'file_type':         'json',
        'observation_start': OBS_START
    }
    async with session.get(OBS_URL, params=params) as resp:
        resp.raise_for_status()
        data = await resp.json()

    dates  = [obs['date'] for obs in data['observations']]
    values = [np.nan if obs['value']=='.' else float(obs['value'])
              for obs in data['observations']]

    return pd.Series(values,
                     index=pd.to_datetime(dates),
                     name=name)

# ─── Public API ──────────────────────────────────────────────────────────────
async def fetch_macro_indicators(indicators: dict) -> pd.DataFrame:
    """
    Concurrently downloads all configured FRED series, reindexes them to daily,
    forward-fills missing values, and appends an '<indicator>_is_update' flag
    for days of real publication.

    Returns
    -------
    df : pandas.DataFrame
        Daily-indexed frame with 2× columns per indicator: the value itself,
        and a boolean flag marking release dates.
    """
    connector = aiohttp.TCPConnector(limit=0)
    timeout   = aiohttp.ClientTimeout(total=15)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [_fetch_series(session, name, code) for name, code in indicators.items()]
        series_list = await asyncio.gather(*tasks)

    # 1. Merge, sort, reindex to daily + forward-fill (as before)
    df = pd.concat(series_list, axis=1).sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df_daily = df.reindex(full_idx).ffill()

    # --- 2. FEATURE ENGINEERING STAGE ---
    
    # Store original observation dates to create flags
    obs_dates = {s.name: set(s.index.date) for s in series_list}
    idx_dates = pd.Series(df_daily.index.date, index=df_daily.index)
    
    engineered_features = {}

    for name in indicators:
        # Create the original boolean flag for 'is update day'
        is_update_col = idx_dates.isin(obs_dates[name])
        
        # Feature 1: Is_Update_Day (renamed from _is_update for clarity)
        # This is the base for the other features.
        engineered_features[f"{name}_IsUpdateDay"] = is_update_col.astype(int)

        # Feature 2: Days_Since_Last_Update
        # A counter that increments daily and resets to 0 on an update day.
        # This is a clever pandas trick: groupby a cumulative sum of boolean flags.
        days_since_update = is_update_col.groupby((is_update_col).cumsum()).cumcount()
        engineered_features[f"{name}_DaysSinceUpdate"] = days_since_update

        # Feature 3: Event_Proximity_Flag (e.g., is it within 3 days of an update?)
        # We achieve this by forward-filling the update flag over a 3-day window.
        event_window = is_update_col.rolling(window=3, min_periods=1).sum() > 0
        engineered_features[f"{name}_InEventWindow"] = event_window.astype(int)

    # Combine engineered features into a new DataFrame
    df_engineered = pd.DataFrame(engineered_features, index=df_daily.index)

    # Concatenate the original data with the new engineered features
    final_df = pd.concat([df_daily, df_engineered], axis=1)
    
    # Clean up column names just in case
    final_df.columns = final_df.columns.str.replace(' ', '_')
    
    return final_df


# ─── Convenience entrypoint ─────────────────────────────────────────────────
def main():
    """Standalone CLI: fetch and save parquet to disk."""
    df = asyncio.run(fetch_macro_indicators(FRED_INDICATORS))
    print("Generated DataFrame Head:")
    print(df.head())
    print("\nColumns:")
    print(df.columns)
    df.to_parquet('data/macro_indicators_daily.parquet')
    print("\nSaved to macro_indicators_daily.parquet")

if __name__ == '__main__':
    main()
