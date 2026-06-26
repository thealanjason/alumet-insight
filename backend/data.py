"""
Data loading and preprocessing.
"""

import pandas as pd
import polars as pl
from typing import Optional
from pathlib import Path

from backend.synthesis import synthesize_attributed_energy_total
from backend.transforms import (
    get_process_time_range_from_df,
    get_time_range_from_df,
    normalize_to_si,
)
from backend.utils import (
    extract_pid_from_content,
    find_measurement_file_in_directory,
    is_cpu_from_content,
    is_cpu_from_metrics,
    is_gpu_from_content,
    is_gpu_from_metrics,
    read_file_content,
)


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


class AlumetData:
    """
    Load and expose measurement data from an Alumet measurement directory.

    Args:
        directory: Path to the directory containing ``.csv`` and ``.log/.txt`` files.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._df_source: pd.DataFrame = pd.DataFrame()
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

        # _df_source: non-SI units rescaled to SI and metric names updated — done once here
        self._df_source = normalize_to_si(load_csv_from_path(self._csv_path), col="metric")
        self._df_processed = preprocess_dataframe_for_visualization(self._df_source)
        synthetic = synthesize_attributed_energy_total(self._df_processed)
        if not synthetic.empty:
            self._df_processed = pd.concat([self._df_processed, synthetic], ignore_index=True)

    # ==========
    # Properties
    # ==========
    @property
    def pid(self) -> Optional[int]:
        """Process ID extracted from the log file, or ``None``."""
        return extract_pid_from_content(self._log_content)

    @property
    def device(self) -> str:
        """``"CPU"``, ``"GPU"``, ``"CPU + GPU"``, or ``"N/A"`` based on log content or metric names."""
        gpu = is_gpu_from_content(self._log_content)
        cpu = is_cpu_from_content(self._log_content)
        if not gpu and not cpu:
            gpu = is_gpu_from_metrics(self._df_processed)
            cpu = is_cpu_from_metrics(self._df_processed)
        if gpu and cpu:
            return "CPU + GPU"
        if gpu:
            return "GPU"
        if cpu:
            return "CPU"
        return "N/A"

    @property
    def process_time_range(self) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """First and last timestamps where the measured process was active."""
        return get_process_time_range_from_df(self._df_source)

    @property
    def data_time_range(self) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """First and last timestamps present in the measurement CSV."""
        return get_time_range_from_df(self._df_processed)

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
    def source_df(self) -> pd.DataFrame:
        """The source DataFrame with NVML units corrected (mW→W, mJ→J) and metric names updated."""
        return self._df_source.copy()

    @property
    def processed_df(self) -> pd.DataFrame:
        """The preprocessed DataFrame (metric_id, base_metric, timestamp, value)."""
        return self._df_processed.copy()

