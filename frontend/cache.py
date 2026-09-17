"""
Dash server-side dataframe cache.

This module belongs to the frontend layer because it adapts pandas dataframes to
Dash's dcc.Store payload model. Backend modules should accept dataframes or file
paths directly and should not know about cache IDs or Dash session state.

Default behavior keeps frames in a bounded in-memory LRU only.
Pass `persist_to_disk=True` when a Parquet backup is needed (e.g. tests that
evict memory and reload from disk).
"""
import atexit
import shutil
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

CACHE_DIR = Path(tempfile.mkdtemp(prefix="dash_df_cache_"))
CACHE_DIR.chmod(0o700)

_DEFAULT_MAX_ENTRIES = 32
_MAX_ENTRIES = _DEFAULT_MAX_ENTRIES
# OrderedDict (LRU order: oldest at the front)
_MEMORY_CACHE: OrderedDict[str, pd.DataFrame] = OrderedDict()


def _cache_path(cache_id: str) -> Path:
    return CACHE_DIR / f"{cache_id}.parquet"


def _evict_overflow() -> None:
    while len(_MEMORY_CACHE) > _MAX_ENTRIES:
        old_id, _ = _MEMORY_CACHE.popitem(last=False)
        path = _cache_path(old_id)
        if path.exists():
            path.unlink(missing_ok=True)


def _cleanup_cache() -> None:
    _MEMORY_CACHE.clear()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


atexit.register(_cleanup_cache)


def cache_id_from_store(store_data: Any) -> Optional[str]:
    """Extract a cache id from a dcc.Store payload (raw id or filtered dict)."""
    if isinstance(store_data, str) and store_data:
        return store_data
    if isinstance(store_data, dict):
        cache_id = store_data.get("cache_id")
        if isinstance(cache_id, str) and cache_id:
            return cache_id
    return None


def delete_cached_dataframe(cache_id: Optional[str]) -> None:
    """Remove one entry from memory and disk (no-op if missing)."""
    if not cache_id:
        return
    _MEMORY_CACHE.pop(cache_id, None)
    path = _cache_path(cache_id)
    if path.exists():
        path.unlink(missing_ok=True)


def clear_dataframe_cache() -> None:
    """Drop all cached frames and remove Parquet files under CACHE_DIR."""
    _MEMORY_CACHE.clear()
    if CACHE_DIR.exists():
        for path in CACHE_DIR.glob("*.parquet"):
            path.unlink(missing_ok=True)


def cache_dataframe(
    df: pd.DataFrame,
    prefix: str = "df",
    *,
    persist_to_disk: bool = False,
) -> Optional[str]:
    """
    Cache a DataFrame and return a reference ID for dcc.Store.

    By default stores an owned copy in the in-memory LRU only. When
    ``persist_to_disk`` is True, also writes ``{cache_id}.parquet`` under
    ``CACHE_DIR`` so the entry can be reloaded after memory eviction.
    """
    if df is None or df.empty:
        return None

    cache_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
    owned = df.copy()
    if persist_to_disk:
        owned.to_parquet(_cache_path(cache_id), engine="pyarrow", index=False)

    _MEMORY_CACHE[cache_id] = owned
    _MEMORY_CACHE.move_to_end(cache_id)
    _evict_overflow()
    return cache_id


def load_cached_dataframe(cache_id: Optional[str]) -> pd.DataFrame:
    """Load DataFrame from cache by ID.

    Checks in-memory LRU first (and marks the entry as most recently used),
    then falls back to disk Parquet when present. Always returns a copy so
    Dash callbacks cannot mutate shared cache state.
    """
    if not cache_id:
        return pd.DataFrame()

    if cache_id in _MEMORY_CACHE:
        _MEMORY_CACHE.move_to_end(cache_id)
        return _MEMORY_CACHE[cache_id].copy()

    cache_path = _cache_path(cache_id)
    if cache_path.exists():
        df = pd.read_parquet(cache_path, engine="pyarrow")
        _MEMORY_CACHE[cache_id] = df
        _MEMORY_CACHE.move_to_end(cache_id)
        _evict_overflow()
        return df.copy()

    return pd.DataFrame({"__cache_miss__": [True]})


def is_cache_miss(df: pd.DataFrame) -> bool:
    """True when df_from_store returned a sentinel for a stale server-side cache."""
    return not df.empty and "__cache_miss__" in df.columns


def df_from_store(store_data: Any) -> pd.DataFrame:
    """
    Reconstruct DataFrame from dcc.Store data.

    Handles:
    - Cache ID string (server-side cache reference)
    - Filtered-store dict with a ``cache_id`` field
    - 'split' format dict (medium datasets)
    - 'records' format (legacy)
    """
    if store_data is None:
        return pd.DataFrame()

    if isinstance(store_data, str):
        return load_cached_dataframe(store_data)

    if isinstance(store_data, dict) and "cache_id" in store_data and "columns" not in store_data:
        return load_cached_dataframe(store_data.get("cache_id"))

    if isinstance(store_data, dict) and "columns" in store_data and "data" in store_data:
        return pd.DataFrame(store_data["data"], columns=store_data["columns"])

    return pd.DataFrame(store_data)
