"""
Backend helpers for comparative analysis tab.

Handles metric ID selection within a process window, process-only filtering,
X/Y dropdown value selection, timestamp alignment via merge_asof, and
CSV export preparation.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from backend.metrics import metric_id_is_process_consumer
from backend.transforms import filter_to_time_range
from backend.utils import safe_filename


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


def filter_process_metric_ids(metric_ids: list[str], process_only: bool) -> list[str]:
    """Restrict to process-attributed series when *process_only* is True."""
    if not process_only:
        return list(metric_ids)
    return [m for m in metric_ids if metric_id_is_process_consumer(m)]


def pick_xy_values(filtered: list[str], cur_x: Any, cur_y: Any) -> tuple[Any, Any]:
    """Pick valid X/Y dropdown values, preserving current selection when possible."""
    if not filtered:
        return None, None
    if len(filtered) == 1:
        return filtered[0], filtered[0]
    x_val = cur_x if cur_x in filtered else filtered[0]
    others = [m for m in filtered if m != x_val]
    y_val = cur_y if cur_y in others else others[0]
    return x_val, y_val


def align_xy_metrics(
    df_processed: pd.DataFrame,
    x_metric_id: str,
    y_metric_id: str,
    proc_start: pd.Timestamp,
    proc_end: pd.Timestamp,
) -> pd.DataFrame:
    """Align two metric series on timestamps within the process window.

    Returns a DataFrame with columns ``timestamp``, ``x``, ``y``.
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


def prepare_xy_download(
    dfxy: pd.DataFrame,
    x_metric_id: str,
    y_metric_id: str,
) -> tuple[pd.DataFrame, str]:
    """Rename columns for CSV export and compute a safe filename.

    Returns ``(df_renamed, filename)``.
    """
    df_out = dfxy.rename(columns={"x": x_metric_id, "y": y_metric_id})
    filename = safe_filename(f"xy_{x_metric_id}_vs_{y_metric_id}.csv")
    return df_out, filename
