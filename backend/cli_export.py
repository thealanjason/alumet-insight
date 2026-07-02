"""
CLI-specific export and discovery operations on loaded Alumet data.

These are higher-level workflow functions used by the CLI. They accept a loaded
AlumetData instance and orchestrate filtering, formatting, and file I/O.
Backend modules and the Dash frontend should not import from this module.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from backend.categories import (
    available_category_values,
    category_for_metric_id,
    filter_time_series_category,
    validate_metric_id_in_category,
)
from backend.figures import export_category_figures
from backend.metrics import (
    filter_by_base_metric,
    filter_by_metric_id,
    format_metric_id_list,
    metric_ids_from_df,
)
from backend.transforms import filter_to_time_range, validate_time_range
from backend.utils import safe_filename
from backend.data import AlumetData

def _selected_categories(data: AlumetData, category: Optional[str]) -> list[str]:
    if category:
        return [category]
    return available_category_values(data.processed_df)


def _prepare_df_for_export(
    data: AlumetData,
    df: pd.DataFrame,
    *,
    process_specific: bool = False,
    start_time: Optional[str | pd.Timestamp] = None,
    end_time: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    if process_specific:
        df = filter_to_time_range(df, *data.process_time_range)
    start_ts, end_ts = validate_time_range(start_time, end_time, *data.data_time_range)
    if start_ts is None and end_ts is None:
        return df.copy()
    return filter_to_time_range(df, start_ts, end_ts)


def summary(data: AlumetData) -> str:
    """Return a compact CLI summary of the loaded dataset."""
    proc_start, proc_end = data.process_time_range
    data_start, data_end = data.data_time_range
    df = data.processed_df
    categories = available_category_values(df)
    summary_width = 80
    rule = "-" * summary_width
    title = f" Alumet Measurement Summary: {Path(data.directory).name} "
    header = title.center(summary_width, "=")

    def section(name: str) -> list[str]:
        return ["", name, rule]

    def field(label: str, value) -> str:
        return f"  {label:<28} {value}"

    def bullet(value: str) -> str:
        return f"  - {value}"

    available_metrics = []
    for metric in data.metrics:
        series_count = df[df["base_metric"] == metric]["metric_id"].nunique()
        available_metrics.append(bullet(f"{metric} ({series_count} series)"))

    lines = [
        header,
        *section("Overview"),
        field("Experiment name", Path(data.directory).name),
        field("Process ID", data.pid or "N/A"),
        field("Device type", data.device),
        field("Metric categories", len(categories)),
        field("Base metrics", len(data.metrics)),
        field("Total metric series", len(data.metric_ids)),
        *section("Time Ranges"),
        field("Start time", data_start or "N/A"),
        field("End time", data_end or "N/A"),
        field("Process start time", proc_start or "N/A"),
        field("Process end time", proc_end or "N/A"),
        *section("Available Categories"),
        *(
            [bullet(cat) for cat in categories] if categories else [bullet("N/A")]
        ),
        *section("Base Metrics"),
        *(available_metrics or [bullet("N/A")]),
        *section("Next Steps"),
        "  List exact metric IDs:   --list-metric-ids [--category <cat>] [--metric-name <name>]",
        "  Export CSV:              --export-csv <dir> [--category <cat> | --metric-id <id>]",
        "  Export figures:          --export-figures <dir> [--category <cat> | --metric-id <id>]",
        "  Custom time window:      --start-time 2026-03-24T23:51:41+00:00 --end-time <timestamp>",
        rule,
    ]
    return "\n".join(lines)


def build_metric_id_listing(
    data: AlumetData,
    category: Optional[str] = None,
    metric_name: Optional[str] = None,
    cpu_core: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """Return a formatted listing of metric IDs, optionally filtered and limited."""
    if metric_name:
        metric_ids = metric_ids_from_df(filter_by_base_metric(data.processed_df, metric_name))
        heading = f"Metric IDs for base metric: {metric_name}"
    elif category:
        metric_ids = metric_ids_from_df(
            filter_time_series_category(data.processed_df, category, selected_cpu_core=cpu_core)
        )
        heading = f"Metric IDs in category: {category}"
    else:
        metric_ids = data.metric_ids
        heading = "All metric IDs"
    return format_metric_id_list(metric_ids, heading, limit=limit)


def export_csvs(
    data: AlumetData,
    output_root: Path,
    category: Optional[str] = None,
    cpu_core: Optional[str] = None,
    process_specific: bool = False,
    metric_id: Optional[str] = None,
    start_time: Optional[str | pd.Timestamp] = None,
    end_time: Optional[str | pd.Timestamp] = None,
) -> list[Path]:
    """Export category CSV files under ``output_root/<category>/csv/``."""
    output_root = Path(output_root)
    df_processed = data.processed_df
    created = []

    if metric_id:
        validate_metric_id_in_category(df_processed, metric_id, category, selected_cpu_core=cpu_core)
        df = _prepare_df_for_export(
            data, filter_by_metric_id(df_processed, metric_id),
            process_specific=process_specific, start_time=start_time, end_time=end_time,
        )
        if df.empty:
            return created
        metric_dir = output_root / (category or "metrics") / "csv"
        metric_dir.mkdir(parents=True, exist_ok=True)
        path = metric_dir / f"{safe_filename(metric_id)}.csv"
        df.to_csv(path, index=False)
        return [path]

    for category_value in _selected_categories(data, category):
        df = _prepare_df_for_export(
            data,
            filter_time_series_category(df_processed, category_value, selected_cpu_core=cpu_core),
            process_specific=process_specific, start_time=start_time, end_time=end_time,
        )
        if df.empty:
            continue
        category_dir = output_root / category_value / "csv"
        category_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_core_{cpu_core}" if category_value == "kernel_cpu_time" and cpu_core else ""
        path = category_dir / f"{safe_filename(category_value + suffix)}.csv"
        df.to_csv(path, index=False)
        created.append(path)
    return created


def export_figures(
    data: AlumetData,
    output_root: Path,
    category: Optional[str] = None,
    cpu_core: Optional[str] = None,
    figure_format: str = "png",
    dpi: int = 150,
    process_specific: bool = False,
    metric_id: Optional[str] = None,
    start_time: Optional[str | pd.Timestamp] = None,
    end_time: Optional[str | pd.Timestamp] = None,
) -> list[Path]:
    """Export static time-series figures under ``output_root/<category>/plots/``."""
    output_root = Path(output_root)
    df_processed = data.processed_df
    proc_start, proc_end = data.process_time_range
    created = []

    if metric_id:
        validate_metric_id_in_category(df_processed, metric_id, category, selected_cpu_core=cpu_core)
        df = _prepare_df_for_export(
            data, filter_by_metric_id(df_processed, metric_id),
            process_specific=process_specific, start_time=start_time, end_time=end_time,
        )
        if df.empty:
            return created
        return export_category_figures(
            df,
            output_root / (category or "metrics") / "plots",
            category_for_metric_id(df_processed, metric_id, category),
            figure_format=figure_format,
            proc_start=proc_start,
            proc_end=proc_end,
            dpi=dpi,
        )

    for category_value in _selected_categories(data, category):
        df = _prepare_df_for_export(
            data,
            filter_time_series_category(df_processed, category_value, selected_cpu_core=cpu_core),
            process_specific=process_specific, start_time=start_time, end_time=end_time,
        )
        if df.empty:
            continue
        plots_dir = output_root / category_value / "plots"
        created.extend(
            export_category_figures(
                df,
                plots_dir,
                category_value,
                figure_format=figure_format,
                proc_start=proc_start,
                proc_end=proc_end,
                dpi=dpi,
            )
        )
    return created