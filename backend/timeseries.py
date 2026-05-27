"""
Backend helpers for time-series tab data processing.

Handles Y-axis shareability, range calculations, timezone alignment, and
category label resolution.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from backend.metrics import get_bytes_tickvals_ticktext

YAXIS_SHAREABLE_CATEGORIES = frozenset(
    {"energy", "power", "utilization", "temperature", "memory", "kernel_cpu_time"}
)


def is_yaxis_shareable(category: str) -> bool:
    """Return True when all metrics in *category* share the same unit."""
    return category in YAXIS_SHAREABLE_CATEGORIES


def category_yaxis_label(category: Optional[str]) -> str:
    """Return the Y-axis label string for a time-series category."""
    labels = {
        "energy": "Value (J)",
        "power": "Value (W)",
        "memory": "Value (B)",
        "utilization": "Value (%)",
        "temperature": "Value (°C)",
        "perf_counters": "Value (count)",
        "kernel_cpu_time": "Value (ms)",
    }
    return labels.get(category, "Value")


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


def compute_yaxis_ranges(
    visible_data: pd.DataFrame,
    metric_order: list[str],
    share_yaxis: bool,
    is_memory: bool,
) -> dict:
    """
    Compute Y-axis range/tick settings for each subplot.

    Returns a dict keyed by yaxis_key ("yaxis", "yaxis2", etc.)
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
            entry: dict = {
                "range": [calc_min, calc_max],
                "autorange": False,
            }
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
            entry = {
                "range": [calc_min, calc_max],
                "autorange": False,
            }
            if is_memory:
                tickvals, ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)
                entry["tickvals"] = tickvals
                entry["ticktext"] = ticktext
            result[yaxis_key] = entry

    return result


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
