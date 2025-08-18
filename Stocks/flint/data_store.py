# FILE: data_store.py
"""
A module for efficiently storing and retrieving processed data packages
using a hybrid Parquet, Blosc2, and Pickle format.
"""
import time
import pickle
import json
from pathlib import Path
from datetime import date
from typing import Dict, Any, Optional

import blosc2
import numpy as np
import pandas as pd
import psutil
import structlog

logger = structlog.get_logger(__name__)

class B2Data:
    """
    Manages the storage and loading of data packages using a structured,
    highly compressed directory format.
    """
    def __init__(self, base_dir: str = "data/processed"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        blosc2.set_nthreads(psutil.cpu_count())
        self.compression_kwargs = {'codec': blosc2.Codec.ZSTD, 'clevel': 5}
        self.int_compression_kwargs = {'codec': blosc2.Codec.ZSTD, 'clevel': 9}

    def get_ticker_dir(self, safe_ticker: str) -> Path:
        """Returns the specific directory for a given ticker."""
        return self.base_dir / safe_ticker

    def save_data_package(self, data_package: Dict[str, Any], safe_ticker: str) -> None:
        """Saves a data package using a structured directory with JSON for metadata."""
        ticker_dir = self.get_ticker_dir(safe_ticker)
        ticker_dir.mkdir(parents=True, exist_ok=True)
        log = logger.bind(ticker=safe_ticker, directory=str(ticker_dir))
        log.info("data_package_save_start")

        metadata = {
            'created_date': date.today().isoformat(),
            'data_keys': list(data_package.keys()),
            'numpy_arrays': {},
            'other_data': {}
        }

        for key, value in data_package.items():
            file_path = ticker_dir / key
            if isinstance(value, pd.DataFrame):
                value.to_parquet(file_path.with_suffix('.parquet'))
            elif isinstance(value, pd.Series):
                value.to_frame().to_parquet(file_path.with_suffix('.parquet'))
            elif isinstance(value, np.ndarray):
                compressed = blosc2.pack_array(value, **self.compression_kwargs)
                with open(file_path.with_suffix('.blosc2'), 'wb') as f:
                    f.write(compressed)
                metadata['numpy_arrays'][key] = {'shape': str(value.shape), 'dtype': str(value.dtype)}
            else:
                metadata['other_data'][key] = value
        
        # Save the single, comprehensive metadata JSON file.
        with open(ticker_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        log.info("data_package_save_complete")

    def load_data_package(self, safe_ticker: str) -> Optional[Dict[str, Any]]:
        """Loads a data package from a structured directory."""
        ticker_dir = self.get_ticker_dir(safe_ticker)
        metadata_path = ticker_dir / "metadata.json"
        if not metadata_path.exists():
            return None

        log = logger.bind(ticker=safe_ticker, directory=str(ticker_dir))
        log.info("data_package_load_start")
        data_package = {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load the miscellaneous data from the metadata file itself.
        data_package.update(metadata.get('other_data', {}))

        # Load the larger artifacts from their respective files.
        for key in metadata.get('data_keys', []):
            parquet_path = ticker_dir / f"{key}.parquet"
            blosc2_path = ticker_dir / f"{key}.blosc2"
            
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                data_package[key] = df[key] if len(df.columns) == 1 and df.columns[0] == key else df
            elif blosc2_path.exists():
                with open(blosc2_path, 'rb') as f:
                    data_package[key] = blosc2.unpack_array(f.read())
        
        log.info("data_package_load_complete")
        return data_package


    def is_cache_valid(self, safe_ticker: str) -> bool:
        """Checks if a valid cache for the ticker exists and is from today."""
        metadata_path = self.get_ticker_dir(safe_ticker) / "metadata.json"
        if not metadata_path.exists():
            return False
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return date.fromisoformat(metadata.get('created_date')) >= date.today()
        except Exception:
            return False
