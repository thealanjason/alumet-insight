"""Stateless view operations on processed DataFrames."""

from __future__ import annotations

from typing import Optional

import pandas as pd


# Maps metric name suffix → (scale factor, SI replacement suffix).
# Add a line here when any plugin starts reporting a new non-SI unit.
_SI_RESCALE: dict[str, tuple[float, str]] = {
    "_mJ": (1e-3, "_J"),
    "_mW": (1e-3, "_W"),
}


def normalize_to_si(df: pd.DataFrame, col: str = "metric") -> pd.DataFrame:
    """Rescale values and rename metric suffixes to SI units (e.g. _mW→_W, _mJ→_J)."""
    df = df.copy()
    # Cast to object so renamed values can be assigned without category constraints
    if isinstance(df[col].dtype, pd.CategoricalDtype):
        df[col] = df[col].astype(object)

    for from_suffix, (factor, to_suffix) in _SI_RESCALE.items():
        mask = df[col].str.endswith(from_suffix, na=False)
        if mask.any():
            df.loc[mask, "value"] = df.loc[mask, "value"] * factor
            df.loc[mask, col] = df.loc[mask, col].str[: -len(from_suffix)] + to_suffix

    return df


def filter_to_time_range(
    df: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    *,
    timestamp_col: str = "timestamp",
    require_bounds: bool = True,
) -> pd.DataFrame:
    """
    Return rows whose timestamp falls between start and end.

    When require_bounds is true, missing bounds produce an empty dataframe.
    When false, missing bounds leave the dataframe unfiltered.
    """
    if df.empty:
        return df.copy()
    if start is None or end is None:
        return df.iloc[0:0].copy() if require_bounds else df.copy()
    if timestamp_col not in df.columns:
        raise ValueError(f"Cannot filter by time range: missing '{timestamp_col}' column")

    filtered = df.copy()
    filtered[timestamp_col] = pd.to_datetime(filtered[timestamp_col], errors="coerce")
    return filtered[(filtered[timestamp_col] >= start) & (filtered[timestamp_col] <= end)].copy()


def get_process_time_range_from_df(df: pd.DataFrame) -> tuple:
    """Get the process active time range from the dataframe.

    Finds the actual process execution period by looking at process-level metrics
    (consumer_kind='process') that have non-zero values.
    """
    if df.empty or "timestamp" not in df.columns:
        return None, None

    process_mask = df["consumer_kind"] == "process"

    if not process_mask.any():
        return df["timestamp"].min(), df["timestamp"].max()

    active_mask = process_mask & (df["value"] > 0)

    if not active_mask.any():
        process_timestamps = df.loc[process_mask, "timestamp"]
        return process_timestamps.min(), process_timestamps.max()

    active_timestamps = df.loc[active_mask, "timestamp"]
    return active_timestamps.min(), active_timestamps.max()
