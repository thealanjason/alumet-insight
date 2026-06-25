"""Static matplotlib exports for CLI time-series figures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from backend.categories import category_yaxis_label
from backend.formatting import format_bytes_ticklabel
from backend.metrics import get_metric_unit, is_memory_metric
from backend.utils import safe_filename


SUPPORTED_FIGURE_FORMATS = ("png", "pdf", "svg")


def _series_title(metric_id: str) -> str:
    """Make a long metric_id a little easier to read in static figure titles."""
    return (
        str(metric_id)
        .replace("_R_", " | R=")
        .replace("_C_", " | C=")
        .replace("_A_", " | A=")
    )


def _ylabel(metric_id: str, category: str) -> str:
    """Return a readable Y-axis label for a static export."""
    label = category_yaxis_label(category)
    if label != "Value":
        return label
    unit = get_metric_unit(metric_id)
    return f"Value ({unit})" if unit else "Value"


def _format_memory_axis(ax) -> None:
    ax.yaxis.set_major_formatter(lambda value, _pos: format_bytes_ticklabel(float(value)))


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
    df_metric = df_metric.copy()
    df_metric["timestamp"] = pd.to_datetime(df_metric["timestamp"], errors="coerce")
    df_metric["value"] = pd.to_numeric(df_metric["value"], errors="coerce")
    df_metric = df_metric.dropna(subset=["timestamp", "value"]).sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12, 5))
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
