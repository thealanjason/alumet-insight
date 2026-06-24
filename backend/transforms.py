"""Stateless view operations on processed DataFrames."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from backend.formatting import get_bytes_tickvals_ticktext


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


def align_xrange_tz(
    x_min: pd.Timestamp,
    x_max: pd.Timestamp,
    df_tz,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Align x_min and x_max timezone to match the DataFrame timestamp column tz."""
    if df_tz is not None:
        if x_min.tz is None:
            x_min = x_min.tz_localize(df_tz)
        if x_max.tz is None:
            x_max = x_max.tz_localize(df_tz)
    else:
        if x_min.tz is not None:
            x_min = x_min.tz_convert(None) if hasattr(x_min, "tz_convert") else x_min.replace(tzinfo=None)
        if x_max.tz is not None:
            x_max = x_max.tz_convert(None) if hasattr(x_max, "tz_convert") else x_max.replace(tzinfo=None)
    return x_min, x_max


def _padded_range(y_min: float, y_max: float, *, clamp_zero: bool = False) -> tuple[float, float]:
    y_range = y_max - y_min if y_max != y_min else abs(y_max) if y_max != 0 else 1
    y_padding = 0.1 * y_range if y_range > 0 else 0.1

    calc_min = y_min - y_padding
    calc_max = y_max + y_padding
    if calc_min >= calc_max:
        calc_min = y_min - 0.1 if y_min != 0 else -0.1
        calc_max = y_max + 0.1 if y_max != 0 else 0.1

    if clamp_zero:
        calc_min = max(0, calc_min)
    return calc_min, calc_max


def compute_yaxis_ranges(
    visible_data: pd.DataFrame,
    metric_order: list[str],
    share_yaxis: bool,
    is_memory: bool,
) -> dict:
    """Compute Y-axis range/tick settings for each subplot.

    Returns a dict keyed by yaxis_key ("yaxis", "yaxis2", …)
    with range, autorange, and optional tickvals/ticktext.
    """
    result: dict[str, dict] = {}

    if share_yaxis:
        global_y_min = visible_data["value"].min()
        global_y_max = visible_data["value"].max()
        calc_min, calc_max = _padded_range(global_y_min, global_y_max, clamp_zero=is_memory)

        shared_tickvals = None
        shared_ticktext = None
        if is_memory:
            shared_tickvals, shared_ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)

        for subplot_idx in range(len(metric_order)):
            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            entry: dict = {"range": [calc_min, calc_max], "autorange": False}
            if is_memory and shared_tickvals is not None:
                entry["tickvals"] = shared_tickvals
                entry["ticktext"] = shared_ticktext
            result[yaxis_key] = entry
    else:
        for subplot_idx, metric_id in enumerate(metric_order):
            metric_visible = visible_data[visible_data["metric_id"] == metric_id]
            if metric_visible.empty:
                continue
            y_min_val = metric_visible["value"].min()
            y_max_val = metric_visible["value"].max()
            calc_min, calc_max = _padded_range(y_min_val, y_max_val, clamp_zero=is_memory)

            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            entry = {"range": [calc_min, calc_max], "autorange": False}
            if is_memory:
                tickvals, ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)
                entry["tickvals"] = tickvals
                entry["ticktext"] = ticktext
            result[yaxis_key] = entry

    return result


def comparative_metric_ids(
    df_processed: pd.DataFrame,
    proc_start: Optional[pd.Timestamp],
    proc_end: Optional[pd.Timestamp],
) -> list[str]:
    """Metric IDs that have samples inside the process active window."""
    if df_processed.empty:
        return []
    df_window = filter_to_time_range(df_processed, proc_start, proc_end)
    if df_window.empty:
        return []
    return sorted(df_window["metric_id"].dropna().astype(str).unique().tolist())


def align_xy_metrics(
    df_processed: pd.DataFrame,
    x_metric_id: str,
    y_metric_id: str,
    proc_start: pd.Timestamp,
    proc_end: pd.Timestamp,
) -> pd.DataFrame:
    """Align two metric series on timestamps within the process window.

    Returns a DataFrame with columns ``timestamp``, ``x``, ``y``.
    Uses nearest-neighbour matching (merge_asof) — avoids synthetic data points.
    """
    dfw = filter_to_time_range(df_processed, proc_start, proc_end)

    dfx = dfw[dfw["metric_id"].astype(str) == str(x_metric_id)][["timestamp", "value"]].rename(columns={"value": "x"})
    dfy = dfw[dfw["metric_id"].astype(str) == str(y_metric_id)][["timestamp", "value"]].rename(columns={"value": "y"})

    if dfx.empty or dfy.empty:
        return pd.DataFrame(columns=["timestamp", "x", "y"])

    dfx = dfx.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp", ignore_index=True)
    dfy = dfy.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp", ignore_index=True)

    dx = dfx["timestamp"].diff().median()
    dy = dfy["timestamp"].diff().median()
    tol = None
    if pd.notna(dx) or pd.notna(dy):
        base = max([v for v in [dx, dy] if pd.notna(v)], default=pd.Timedelta(milliseconds=0))
        tol = base * 2 if base > pd.Timedelta(0) else pd.Timedelta(seconds=1)

    dfxy = pd.merge_asof(dfx, dfy, on="timestamp", direction="nearest", tolerance=tol).dropna(subset=["y"])
    return dfxy.sort_values("timestamp", ignore_index=True)


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
