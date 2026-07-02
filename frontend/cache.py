"""
Dash server-side dataframe cache.

This module belongs to the frontend layer because it adapts pandas dataframes to
Dash's dcc.Store payload model. Backend modules should accept dataframes or file
paths directly and should not know about cache IDs or Dash session state.
"""

import atexit
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd

CACHE_DIR = Path(tempfile.mkdtemp(prefix="dash_df_cache_"))
CACHE_DIR.chmod(0o700)

_MEMORY_CACHE: dict[str, pd.DataFrame] = {}


def _cleanup_cache():
    _MEMORY_CACHE.clear()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

atexit.register(_cleanup_cache)


def cache_dataframe(df: pd.DataFrame, prefix: str = "df") -> Optional[str]:
    """
    Cache DataFrame to disk and in-memory, return a reference ID.
    
    Uses Parquet format on disk and in-memory dict for fast access.
    
    Args:
        df: DataFrame to cache
        prefix: Prefix for the cache file
    
    Returns:
        Cache ID string to store in dcc.Store
    """
    if df is None or df.empty:
        return None
    
    cache_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
    cache_path = CACHE_DIR / f"{cache_id}.parquet"
    
    # Persist to disk (backup) and keep in memory (fast access)
    df.to_parquet(cache_path, engine="pyarrow", index=False)
    _MEMORY_CACHE[cache_id] = df
    
    return cache_id


def load_cached_dataframe(cache_id: Optional[str]) -> pd.DataFrame:
    """Load DataFrame from cache by ID.
    
    Checks in-memory cache first (instant), falls back to disk Parquet.
    
    Args:
        cache_id: Cache ID returned by cache_dataframe()
    
    Returns:
        Cached DataFrame or empty DataFrame if not found
    """
    if not cache_id:
        return pd.DataFrame()
    
    # Check in-memory cache first (instant, no disk I/O)
    if cache_id in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_id]
    
    # Fall back to disk Parquet and promote to memory
    cache_path = CACHE_DIR / f"{cache_id}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path, engine="pyarrow")
        _MEMORY_CACHE[cache_id] = df  # Promote to memory for next access
        return df
    
    # Cache miss
    return pd.DataFrame({"__cache_miss__": [True]})


def is_cache_miss(df: pd.DataFrame) -> bool:
    """True when df_from_store returned a sentinel for a stale server-side cache."""
    return not df.empty and "__cache_miss__" in df.columns


def df_from_store(store_data: Any) -> pd.DataFrame:
    """
    Reconstruct DataFrame from dcc.Store data.
    
    Handles:
    - Cache ID string (new optimized approach for large data)
    - 'split' format dict (medium datasets)
    - 'records' format (legacy)
    """
    if store_data is None:
        return pd.DataFrame()
    
    # Check if it's a cache ID (string starting with prefix)
    if isinstance(store_data, str):
        return load_cached_dataframe(store_data)
    
    # Check if it's split format (has 'columns', 'data' keys)
    if isinstance(store_data, dict) and "columns" in store_data and "data" in store_data:
        return pd.DataFrame(store_data["data"], columns=store_data["columns"])
    else:
        # Legacy 'records' format or direct dict
        return pd.DataFrame(store_data)
