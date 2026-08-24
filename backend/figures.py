"""Static matplotlib exports for CLI time-series figures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.categories import category_yaxis_label
from backend.counterdiff import (
    build_counterdiff_spike_coordinates,
    build_step_power_coordinates,
    counterdiff_spike_marker_sizes,
    sort_for_plotting,
)
from backend.formatting import format_bytes_ticklabel
from backend.metrics import (
    get_metric_unit,
    is_memory_metric,
    is_spike_metric,
    is_step_power_metric,
)
from backend.utils import safe_filename

SUPPORTED_FIGURE_FORMATS = ("png", "pdf", "svg")


def _series_title(metric_id: str) -> str:
    """Make a long metric_id a little easier to read in static figure titles."""
    return str(metric_id).replace("_R_", " | R=").replace("_C_", " | C=").replace("_A_", " | A=")


def _ylabel(metric_id: str, category: str) -> str:
    """Return a readable Y-axis label for a static export."""
    label = category_yaxis_label(category)
    if label != "Value":
        return label
    unit = get_metric_unit(metric_id)
    return f"Value ({unit})" if unit else "Value"


def _format_memory_axis(ax) -> None:
    ax.yaxis.set_major_formatter(lambda value, _pos: format_bytes_ticklabel(float(value)))


def _plot_segments(ax, x_coords, y_coords, *, drawstyle: str | None = None) -> None:
    """Plot transient segments as one artist with NaN line breaks."""
    xs = [np.nan if x is None else mdates.date2num(pd.Timestamp(x).to_pydatetime()) for x in x_coords]
    ys = [np.nan if y is None else y for y in y_coords]
    kwargs = {"linewidth": 1.8}
    if drawstyle is not None:
        kwargs["drawstyle"] = drawstyle
    ax.plot(xs, ys, **kwargs)
    ax.xaxis_date()


def _plot_counterdiff_spike_markers(ax, x_coords, y_coords, *, marker_size: float = 6) -> None:
    """Draw dots only at observed CounterDiff peaks (including zero)."""
    sizes = counterdiff_spike_marker_sizes(x_coords, marker_size=marker_size)
    peak_x = []
    peak_y = []
    for x, y, size in zip(x_coords, y_coords, sizes):
        if size <= 0 or x is None or y is None:
            continue
        peak_x.append(mdates.date2num(pd.Timestamp(x).to_pydatetime()))
        peak_y.append(y)
    if peak_x:
        # Matplotlib scatter `s` is marker area in points^2.
        ax.scatter(peak_x, peak_y, s=marker_size**2, zorder=3)


def save_metric_time_series_figure(
    df_metric: pd.DataFrame,
    path: Path,
    category: str,
    metric_id: str,
    proc_start: Optional[pd.Timestamp] = None,
    proc_end: Optional[pd.Timestamp] = None,
    dpi: int = 150,
) -> Path:
    """Save one static time-series figure for a single metric_id."""
    df_metric = sort_for_plotting(df_metric.copy())
    df_metric["timestamp"] = pd.to_datetime(df_metric["timestamp"], errors="coerce")
    df_metric["value"] = pd.to_numeric(df_metric["value"], errors="coerce")
    df_metric = df_metric.dropna(subset=["timestamp", "value"])

    fig, ax = plt.subplots(figsize=(12, 5))

    roles = df_metric["point_role"] if "point_role" in df_metric.columns else None
    orders = df_metric["point_order"] if "point_order" in df_metric.columns else None
    interval_starts = df_metric["interval_start"] if "interval_start" in df_metric.columns else None

    if is_spike_metric(metric_id):
        x_coords, y_coords = build_counterdiff_spike_coordinates(
            df_metric["timestamp"],
            df_metric["value"],
            point_roles=roles,
            point_orders=orders,
        )
        _plot_segments(ax, x_coords, y_coords)
        _plot_counterdiff_spike_markers(ax, x_coords, y_coords)
    elif is_step_power_metric(metric_id):
        x_coords, y_coords = build_step_power_coordinates(
            df_metric["timestamp"],
            df_metric["value"],
            point_roles=roles,
            interval_starts=interval_starts,
        )
        _plot_segments(ax, x_coords, y_coords, drawstyle="steps-post")
    else:
        ax.plot(df_metric["timestamp"], df_metric["value"], linewidth=1.8)

    if proc_start is not None and proc_end is not None:
        ax.axvspan(proc_start, proc_end, color="#88C0D0", alpha=0.15, label="Process active")
        ax.legend(loc="best")

    ax.set_title(_series_title(metric_id), fontsize=10, wrap=True)
    ax.set_xlabel("Time")
    ax.set_ylabel(_ylabel(metric_id, category))
    ax.grid(True, alpha=0.25)

    if category == "memory" or is_memory_metric(metric_id):
        _format_memory_axis(ax)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def export_category_figures(
    df_category: pd.DataFrame,
    output_dir: Path,
    category: str,
    figure_format: str = "png",
    proc_start: Optional[pd.Timestamp] = None,
    proc_end: Optional[pd.Timestamp] = None,
    dpi: int = 150,
) -> list[Path]:
    """Export one static figure per metric_id for a filtered category DataFrame."""
    if figure_format not in SUPPORTED_FIGURE_FORMATS:
        raise ValueError(f"Unsupported figure format: {figure_format}")
    if df_category.empty:
        return []

    created: list[Path] = []
    for metric_id, df_metric in df_category.groupby("metric_id", sort=True):
        safe_metric = safe_filename(str(metric_id))
        path = output_dir / f"{safe_metric}.{figure_format}"
        created.append(
            save_metric_time_series_figure(
                df_metric,
                path,
                category,
                str(metric_id),
                proc_start=proc_start,
                proc_end=proc_end,
                dpi=dpi,
            )
        )
    return created
