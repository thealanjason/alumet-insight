"""
Data loading and preprocessing.
"""

import re
import warnings
import numpy as np
import pandas as pd
import polars as pl
from typing import List, Optional
from pathlib import Path

from backend.categories import available_category_values, filter_time_series_category


# CSV / Parquet I/O
def _read_csv_with_polars(csv_path: Path) -> pl.DataFrame:
    """
    Read CSV with Polars multi-threaded reader, with Parquet sidecar caching.

    On first load: reads CSV with Polars (multi-threaded) and saves a .parquet sidecar.
    On subsequent loads: reads the Parquet sidecar directly (instant).
    """
    parquet_path = csv_path.with_suffix(".parquet")
    if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
        cached = pl.read_parquet(parquet_path)
        id_cols = [c for c in ("resource_id", "consumer_id") if c in cached.columns]
        if all(cached[c].dtype == pl.Utf8 for c in id_cols):
            return cached

    df_pl = pl.read_csv(
        csv_path,
        separator=";",
        try_parse_dates=True,
        infer_schema_length=10000,
        schema_overrides={
            "value": pl.Float64,
            "resource_id": pl.Utf8,
            "consumer_id": pl.Utf8,
        },
    )
    df_pl.write_parquet(parquet_path)
    return df_pl


def load_csv_from_path(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV data from file path with Polars multi-threaded reader and Parquet sidecar caching.
    """
    if not csv_path.exists():
        raise ValueError(f"CSV file not found: {csv_path}")

    try:
        df_pl = _read_csv_with_polars(csv_path)
        df = df_pl.to_pandas()
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")

    if df.empty:
        raise ValueError("No data found in CSV.")

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    category_cols = ["metric", "resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def preprocess_dataframe_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataframe to have metric_id, base_metric, timestamp, and value columns.
    """
    df_pl = pl.from_pandas(df)

    col_exprs = {}
    for col_name in ["metric", "resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]:
        if col_name in df_pl.columns:
            col_exprs[col_name] = pl.col(col_name).cast(pl.Utf8).fill_null("")
        else:
            col_exprs[col_name] = pl.lit("")

    result_pl = df_pl.select([
        (
            col_exprs["metric"] + "_R_" +
            col_exprs["resource_kind"] + "_" +
            col_exprs["resource_id"] + "_C_" +
            col_exprs["consumer_kind"] + "_" +
            col_exprs["consumer_id"] + "_A_" +
            col_exprs["__late_attributes"]
        ).alias("metric_id"),
        col_exprs["metric"].alias("base_metric"),
        pl.col("timestamp"),
        pl.col("value"),
    ])

    return result_pl.to_pandas()


# File directory helpers
def find_measurement_file_in_directory(directory_path: str, extensions: List[str]) -> Path:
    """
    Find measurement file with specified extensions in a directory.

    Args:
        directory_path: Path to the directory
        extensions: List of file extensions to search for

    Returns:
        Path object for matching file
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    found_files = []
    for ext in extensions:
        found_files.extend(dir_path.glob(f"*{ext}"))
    if not found_files:
        raise ValueError(f"No files found with extensions: {extensions} in directory: {directory_path}")
    if len(found_files) > 1:
        warnings.warn(f"Multiple files found with extensions: {extensions} in directory: {directory_path}. Returning the first one.")
    return sorted(found_files)[0]


def read_file_content(file_path: Path) -> str:
    """Read file content as string."""
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


# Log content extraction for runned process information
def extract_pid_from_content(log_content: str) -> Optional[int]:
    """Extract process ID from Alumet log file content."""
    if not log_content:
        return None
    for line in log_content.split("\n"):
        if "pid" in line:
            match = re.search(r"pid (\d+)", line)
            if match:
                return int(match.group(1))
    return None


def _extract_process_pid(metric_id: str) -> Optional[str]:
    pid_pattern = re.compile(r"_C_process_(\d+)")
    m = pid_pattern.search(str(metric_id))
    return m.group(1) if m else None


def is_gpu_from_content(log_content: str) -> bool:
    """Detect whether the run used GPU-related Alumet plugins (NVML) from agent log text."""
    if not log_content:
        return False
    for line in log_content.split("\n"):
        if re.search(r"nvml", line, re.IGNORECASE):
            return True
    return False


def is_cpu_from_content(log_content: str) -> bool:
    """Detect whether the run used CPU-related Alumet plugins from agent log text."""
    if not log_content:
        return False
    for line in log_content.split("\n"):
        if re.search(r"rapl", line, re.IGNORECASE):
            return True
    return False


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


# Time range extraction from dataframe
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


# Total energy synthesis from cpu/gpu energy metrics
def _align_cumulative_energy_to_timeline(
    df_metric: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    value_name: str,
) -> pd.Series:
    """
    Align a cumulative metric to a shared timeline.

    Before the first observed sample the value is treated as 0. 
    Between observed samples, values are linearly interpolated on timestamp. 
    After the last observed sample, values remain N/A so downstream synthesis can dropna to truncate the timeline.
    """
    if df_metric.empty:
        return pd.Series(0.0, index=timeline, name=value_name)

    aligned = (
        df_metric.groupby("timestamp", as_index=True)["value"]
        .sum()
        .sort_index()
        .reindex(timeline)
        .astype(float)
    )

    first_valid = aligned.first_valid_index()
    if first_valid is None:
        return pd.Series(0.0, index=timeline, name=value_name)

    aligned.loc[aligned.index < first_valid] = 0.0
    aligned = aligned.interpolate(method="time", limit_area="inside")
    aligned = aligned.fillna(0.0)
    last_valid = df_metric["timestamp"].max()
    aligned.loc[aligned.index > last_valid] = np.nan
    aligned.name = value_name
    return aligned


def synthesize_attributed_energy_total(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize attributed_energy_total_J metric from attributed_energy_cpu and attributed_energy_gpu metrics.

    Creates two synthetic metrics:
    1. attributed_energy_gpu_total_J: sum of attributed GPU energy across all GPUs per process (pid).
    2. attributed_energy_total_J: CPU + GPU total per process on the union of timestamps.

    Args:
        df_processed: Preprocessed DataFrame with metric_id, base_metric, timestamp, value columns.

    Returns:
        DataFrame of synthetic rows or empty DataFrame when source data is missing.
    """
    cpu_mask = df_processed["base_metric"].str.contains("attributed_energy_cpu", case=False, na=False)
    gpu_mask = df_processed["base_metric"].str.contains("attributed_energy_gpu", case=False, na=False)

    df_cpu = df_processed.loc[cpu_mask, ["metric_id", "timestamp", "value"]].copy()
    df_gpu = df_processed.loc[gpu_mask, ["metric_id", "timestamp", "value"]].copy()

    if df_cpu.empty and df_gpu.empty:
        return pd.DataFrame(columns=df_processed.columns)

    if not df_cpu.empty:
        df_cpu["pid"] = df_cpu["metric_id"].map(_extract_process_pid)
        df_cpu["timestamp"] = pd.to_datetime(df_cpu["timestamp"], errors="coerce")
        df_cpu.dropna(subset=["pid", "timestamp"], inplace=True)

    if not df_gpu.empty:
        df_gpu["pid"] = df_gpu["metric_id"].map(_extract_process_pid)
        df_gpu["timestamp"] = pd.to_datetime(df_gpu["timestamp"], errors="coerce")
        df_gpu.dropna(subset=["pid", "timestamp"], inplace=True)

    if df_cpu.empty and df_gpu.empty:
        return pd.DataFrame(columns=df_processed.columns)

    df_gpu_summed = (
        df_gpu.groupby(["timestamp", "pid"], as_index=False)["value"].sum()
        if not df_gpu.empty
        else pd.DataFrame(columns=["timestamp", "pid", "value"])
    )
    df_cpu_summed = (
        df_cpu.groupby(["timestamp", "pid"], as_index=False)["value"].sum()
        if not df_cpu.empty
        else pd.DataFrame(columns=["timestamp", "pid", "value"])
    )

    synthetic_parts: list[pd.DataFrame] = []

    for pid in df_gpu_summed["pid"].unique():
        gpu_pid = df_gpu_summed.loc[df_gpu_summed["pid"] == pid, ["timestamp", "value"]].copy()
        gpu_pid["metric_id"] = f"attributed_energy_gpu_total_J_R_gpu_all__C_process_{pid}_A_"
        gpu_pid["base_metric"] = "attributed_energy_gpu_total_J"
        synthetic_parts.append(gpu_pid[["metric_id", "base_metric", "timestamp", "value"]])

    shared_pids = sorted(set(df_cpu_summed["pid"]) & set(df_gpu_summed["pid"]))
    for pid in shared_pids:
        cpu_pid = df_cpu_summed.loc[df_cpu_summed["pid"] == pid, ["timestamp", "value"]].copy()
        gpu_pid = df_gpu_summed.loc[df_gpu_summed["pid"] == pid, ["timestamp", "value"]].copy()
        if cpu_pid.empty or gpu_pid.empty:
            continue

        timeline = pd.DatetimeIndex(
            pd.Index(cpu_pid["timestamp"]).union(pd.Index(gpu_pid["timestamp"])).sort_values()
        )
        if timeline.empty:
            continue

        cpu_aligned = _align_cumulative_energy_to_timeline(cpu_pid, timeline, "cpu_value")
        gpu_aligned = _align_cumulative_energy_to_timeline(gpu_pid, timeline, "gpu_value")

        total_pid = pd.DataFrame({
            "timestamp": timeline,
            "cpu_value": cpu_aligned.to_numpy(),
            "gpu_value": gpu_aligned.to_numpy(),
        })
        total_pid.dropna(subset=["cpu_value", "gpu_value"], inplace=True)
        if total_pid.empty:
            continue
        total_pid["value"] = total_pid["cpu_value"] + total_pid["gpu_value"]
        total_pid["metric_id"] = f"attributed_energy_total_J_R_total__C_process_{pid}_A_"
        total_pid["base_metric"] = "attributed_energy_total_J"
        synthetic_parts.append(total_pid[["metric_id", "base_metric", "timestamp", "value"]])

    if not synthetic_parts:
        return pd.DataFrame(columns=df_processed.columns)

    return pd.concat(synthetic_parts, ignore_index=True)


# OOP wrapper for all data operations
class AlumetData:
    """
    Load, preprocess, query, and export data from an Alumet measurement directory.

    Args:
        directory: Path to the directory containing ``.csv`` and ``.log/.txt`` files.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._df_raw: pd.DataFrame = pd.DataFrame()
        self._df_processed: pd.DataFrame = pd.DataFrame()
        self._log_content: str = ""
        self._csv_path: Optional[Path] = None
        self._log_path: Optional[Path] = None
        self._load()

    def _load(self) -> None:
        self._csv_path = find_measurement_file_in_directory(str(self.directory), [".csv"])
        try:
            self._log_path = find_measurement_file_in_directory(str(self.directory), [".log", ".txt"])
            self._log_content = read_file_content(self._log_path)
        except ValueError:
            self._log_path = None
            self._log_content = ""

        self._df_raw = load_csv_from_path(self._csv_path)
        self._df_processed = preprocess_dataframe_for_visualization(self._df_raw)

    # ==========
    # Properties
    # ==========
    @property
    def pid(self) -> Optional[int]:
        """Process ID extracted from the log file, or ``None``."""
        return extract_pid_from_content(self._log_content)

    @property
    def device(self) -> str:
        """``"CPU"``, ``"GPU"``, or ``"CPU + GPU"`` based on log content."""
        gpu = is_gpu_from_content(self._log_content)
        cpu = is_cpu_from_content(self._log_content)
        if gpu and not cpu:
            return "GPU"
        if cpu and not gpu:
            return "CPU"
        return "CPU + GPU"

    @property
    def process_time_range(self) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """First and last timestamps where the measured process was active."""
        return get_process_time_range_from_df(self._df_raw)

    @property
    def metrics(self) -> list[str]:
        """Unique base metric names."""
        if "base_metric" in self._df_processed.columns:
            return sorted(self._df_processed["base_metric"].dropna().unique().tolist())
        return []

    @property
    def metric_ids(self) -> list[str]:
        """Unique full metric_id strings."""
        if "metric_id" in self._df_processed.columns:
            return sorted(self._df_processed["metric_id"].dropna().unique().tolist())
        return []

    @property
    def raw_df(self) -> pd.DataFrame:
        """The raw (original) DataFrame as loaded from CSV."""
        return self._df_raw.copy()

    @property
    def processed_df(self) -> pd.DataFrame:
        """The preprocessed DataFrame (metric_id, base_metric, timestamp, value)."""
        return self._df_processed.copy()

    # ==========
    #   Query
    # ==========
    def filter_by_metric(self, metric: str) -> pd.DataFrame:
        """Return rows whose ``base_metric`` matches *metric*."""
        return self._df_processed[self._df_processed["base_metric"] == metric].copy()

    def filter_process_metrics(self) -> pd.DataFrame:
        """Return only rows attributed to a process (``_C_process_`` in metric_id)."""
        from backend.metrics import metric_id_is_process_consumer
        mask = self._df_processed["metric_id"].apply(metric_id_is_process_consumer)
        return self._df_processed[mask].copy()

    def filter_by_category(self, category: Optional[str], cpu_core: Optional[str] = None) -> pd.DataFrame:
        """Return rows matching a dashboard time-series category."""
        return filter_time_series_category(self._df_processed, category, selected_cpu_core=cpu_core)

    def filter_to_process_time_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return rows from dataframe that fall within the measured process active time range."""
        return filter_to_time_range(df, *self.process_time_range)

    # ==========
    #   Export
    # ==========
    def summary(self) -> str:
        """Return a human-readable text summary of the dataset."""
        proc_start, proc_end = self.process_time_range
        categories = available_category_values(self._df_processed)
        lines = [
            f"Experiment name              : {Path(self.directory).name}",
            f"Process ID                   : {self.pid or 'N/A'}",
            f"Device type                  : {self.device}",
            f"Number of metric categories  : {len(categories)}",
            f"Number of metrics            : {len(self.metrics)}",
            f"Number of metric series IDs  : {len(self.metric_ids)}",
            f"Time range                   : {proc_start} - {proc_end}",
        ]
        return "\n".join(lines)

    def _selected_categories(self, category: Optional[str]) -> list[str]:
        """Return selected categories, or all available categories when not specified."""
        if category:
            return [category]
        return available_category_values(self._df_processed)

    @staticmethod
    def _safe_filename(value: str) -> str:
        """Return a filesystem-safe filename stem."""
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)

    def export_csvs(
        self,
        output_root: Path,
        category: Optional[str] = None,
        cpu_core: Optional[str] = None,
        process_specific: bool = False,
    ) -> list[Path]:
        """Export category CSV files under ``output_root/<category>/csv/``."""
        output_root = Path(output_root)
        created = []
        for category_value in self._selected_categories(category):
            df = self.filter_by_category(category_value, cpu_core=cpu_core)
            if process_specific:
                df = self.filter_to_process_time_range(df)
            if df.empty:
                continue
            category_dir = output_root / category_value / "csv"
            category_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_core_{cpu_core}" if category_value == "kernel_cpu_time" and cpu_core else ""
            path = category_dir / f"{self._safe_filename(category_value + suffix)}.csv"
            df.to_csv(path, index=False)
            created.append(path)
        return created

