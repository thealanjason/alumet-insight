import warnings
import re
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from typing import List, Optional
from pathlib import Path

# ================================================
# CSV dataframe helper functions
# ================================================
def _read_csv_with_polars(csv_path: Path) -> pl.DataFrame:
    """Read CSV with Polars multi-threaded reader, with Parquet sidecar caching.
    
    On first load: reads CSV with Polars (multi-threaded) and saves a .parquet sidecar.
    On subsequent loads: reads the Parquet sidecar directly (instant).
    """
    parquet_path = csv_path.with_suffix(".parquet")
    if parquet_path.exists() and parquet_path.stat().st_mtime >= csv_path.stat().st_mtime:
        cached = pl.read_parquet(parquet_path)
        id_cols = [c for c in ("resource_id", "consumer_id") if c in cached.columns]
        if all(cached[c].dtype == pl.Utf8 for c in id_cols):
            return cached
    
    # Use Polars multi-threaded CSV reader (significantly faster than pandas)
    # Explicit dtypes:
    # - value: Float64 — early rows may look integer-like
    # - resource_id / consumer_id: Utf8 — NVML uses PCI bus IDs (e.g. 00000000:3F:00.0)
    #   which are not valid i64; empty early rows make Polars infer i64 otherwise
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
    # Save Parquet sidecar for instant future loads
    df_pl.write_parquet(parquet_path)
    return df_pl


def load_csv_from_path(csv_path: Path) -> pd.DataFrame:
    """Load CSV data from file path.
    
    Optimized for large files:
    - Uses Polars multi-threaded CSV reader
    - Caches a Parquet file next to the CSV for subsequent re-loads
    - Uses category dtype for string columns with limited unique values
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
    
    # Ensure timestamp is datetime
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Convert string columns to category dtype for memory efficiency
    category_cols = ["metric", "resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]
    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    return df

def preprocess_dataframe_for_visualization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataframe to have metric, timestamp, and value columns.
    Metric column format: f"{metric}_R_{resource_kind}_{resource_id}_C_{consumer_kind}_{consumer_id}_A_{late_attributes}"
    
    Uses Polars for string concatenation,
    then converts back to pandas for downstream compatibility.
    Also pre-computes base_metric to avoid repeated .str.split() in callbacks.
    
    Returns:
        Preprocessed DataFrame with metric_id, base_metric, timestamp, value columns
    """
    # Convert to Polars for fast string operations
    df_pl = pl.from_pandas(df)
    
    # Build fill_null expressions for each column (replaces NaN/"null" with "")
    col_exprs = {}
    for col_name in ["metric", "resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]:
        if col_name in df_pl.columns:
            col_exprs[col_name] = pl.col(col_name).cast(pl.Utf8).fill_null("")
        else:
            col_exprs[col_name] = pl.lit("")
    
    # Polars string concat is multi-threaded and much faster than pandas
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

# ================================================
# Directory and file path helper functions
# ================================================
def find_files_in_directory(directory_path: str, extensions: List[str]) -> List[Path]:
    """Find files with specified extensions in a directory.
    
    Args:
        directory_path: Path to the directory
        extensions: List of file extensions to search for
    
    Returns:
        List of Path objects for matching files
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
    return file_path.read_text(encoding='utf-8')

# ================================================
# Information extraction helper functions
# ================================================
def extract_pid_from_content(log_content: str) -> Optional[int]:
    """Extract process ID from Alumet log file content."""
    if not log_content:
        return None
    for line in log_content.split('\n'):
        if 'pid' in line:
            match = re.search(r'pid (\d+)', line)
            if match:
                return int(match.group(1))
    return None

def metric_id_is_process_consumer(metric_id: str) -> bool:
    """
    True if the metric row was attributed to a process (consumer_kind == 'process').
    
    Preprocessed metric_id embeds consumers as ..._C_{consumer_kind}_{consumer_id}_A_{late_attributes}....
    Process rows contain _C_process_<pid>_ (e.g. _C_process_172681_A_).
    """
    return "_C_process_" in str(metric_id)

def _extract_process_pid(metric_id: str) -> Optional[str]:
    pid_pattern = re.compile(r"_C_process_(\d+)")
    m = pid_pattern.search(str(metric_id))
    return m.group(1) if m else None


def _align_cumulative_energy_to_timeline(
    df_metric: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    value_name: str,
) -> pd.Series:
    """
    Align a cumulative metric to a shared timeline.

    Missing values between the earlier start/end of this stream and later start/end of the other stream are returned as 0.
    Mask timestamps outside the combined CPU/GPU observation window as NaN.
    Between observed samples, values are linearly interpolated on timestamp.
    """
    if df_metric.empty:
        return pd.Series(0.0, index=timeline, name=value_name)

    source = (
        df_metric.groupby("timestamp", as_index=True)["value"]
        .sum()
        .sort_index()
        .astype(float)
    )
    interpolation_index = pd.DatetimeIndex(source.index.union(timeline).sort_values())
    aligned = source.reindex(interpolation_index)

    first_valid = aligned.first_valid_index()
    if first_valid is None:
        return pd.Series(0.0, index=timeline, name=value_name)

    aligned.loc[aligned.index < first_valid] = 0.0
    aligned = aligned.interpolate(method="time", limit_area="inside")
    aligned = aligned.fillna(0.0)
    aligned = aligned.reindex(timeline)
    aligned.name = value_name
    return aligned


def _build_total_energy_timeline(
    cpu_pid: pd.DataFrame,
    gpu_pid: pd.DataFrame,
) -> pd.DatetimeIndex:
    """
    Build timestamps for attributed total energy.

    Mirrors Alumet's energy-attribution interpolation plug-in (https://github.com/alumet-dev/alumet/tree/main/plugins/energy-attribution): 
    one timeseries is the reference and remains unchanged, while other timeseries are interpolated onto its timestamps.
    CPU timestamps are the reference when CPU data exists; GPU timestamps are used only for GPU-only data.
    """
    cpu_index = pd.DatetimeIndex(pd.Index(cpu_pid["timestamp"]).unique()).sort_values()
    gpu_index = pd.DatetimeIndex(pd.Index(gpu_pid["timestamp"]).unique()).sort_values()

    if cpu_index.empty:
        return gpu_index
    return cpu_index

def synthesize_attributed_energy_total(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize attributed_energy_total_J metric from attributed_energy_cpu and attributed_energy_gpu.
    
    The energy-attribution plugin produces one row per (GPU, process) pair.
    This function creates two synthetic metrics in post-processing:
    
    1. attributed_energy_gpu_total_J — sum of attributed GPU energy across
       all GPUs for each process (PID).  Useful when a process spans multiple GPUs.
    2. attributed_energy_total_J — attributed_energy_gpu_total + attributed_energy_cpu
       for each process with either CPU or GPU data. CPU timestamps are the reference
       when both streams exist; GPU values are linearly interpolated onto that
       reference. If only GPU data exists, GPU timestamps are the reference.
       Missing non-reference values outside their observed range contribute 0.
    
    Args:
        df_processed: Preprocessed DataFrame with metric_id, base_metric, timestamp, value columns.
    
    Returns:
        DataFrame of synthetic rows (both metrics) in the same column format,
        or empty DataFrame when source data is missing.
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

    # Sum GPU attributed energy across all GPUs per (timestamp, pid).
    df_gpu_summed = (
        df_gpu.groupby(["timestamp", "pid"], as_index=False)["value"].sum()
        if not df_gpu.empty
        else pd.DataFrame(columns=["timestamp", "pid", "value"])
    )
    # Sum CPU attributed energy per (timestamp, pid).
    df_cpu_summed = (
        df_cpu.groupby(["timestamp", "pid"], as_index=False)["value"].sum()
        if not df_cpu.empty
        else pd.DataFrame(columns=["timestamp", "pid", "value"])
    )

    synthetic_parts: list[pd.DataFrame] = []

    # attributed_energy_gpu_total_J (always produced when GPU data exists)
    for pid in df_gpu_summed["pid"].unique():
        gpu_pid = df_gpu_summed.loc[df_gpu_summed["pid"] == pid, ["timestamp", "value"]].copy()
        gpu_pid["metric_id"] = f"attributed_energy_gpu_total_J_R_gpu_all__C_process_{pid}_A_"
        gpu_pid["base_metric"] = "attributed_energy_gpu_total_J"
        synthetic_parts.append(gpu_pid[["metric_id", "base_metric", "timestamp", "value"]])

    # attributed_energy_total_J for any PID that has CPU or GPU attributed energy.
    total_pids = sorted(set(df_cpu_summed["pid"]) | set(df_gpu_summed["pid"]))
    for pid in total_pids:
        cpu_pid = df_cpu_summed.loc[df_cpu_summed["pid"] == pid, ["timestamp", "value"]].copy()
        gpu_pid = df_gpu_summed.loc[df_gpu_summed["pid"] == pid, ["timestamp", "value"]].copy()

        timeline = _build_total_energy_timeline(cpu_pid, gpu_pid)
        if timeline.empty:
            continue

        cpu_aligned = _align_cumulative_energy_to_timeline(cpu_pid, timeline, "cpu_value")
        gpu_aligned = _align_cumulative_energy_to_timeline(gpu_pid, timeline, "gpu_value")

        total_pid = pd.DataFrame({
            "timestamp": timeline,
            "cpu_value": cpu_aligned.to_numpy(),
            "gpu_value": gpu_aligned.to_numpy(),
        })
        observed_starts = []
        observed_ends = []
        if not cpu_pid.empty:
            observed_starts.append(cpu_pid["timestamp"].min())
            observed_ends.append(cpu_pid["timestamp"].max())
        if not gpu_pid.empty:
            observed_starts.append(gpu_pid["timestamp"].min())
            observed_ends.append(gpu_pid["timestamp"].max())
        combined_window = total_pid["timestamp"].between(min(observed_starts), max(observed_ends))
        total_pid.loc[~combined_window, ["cpu_value", "gpu_value"]] = np.nan
        total_pid.dropna(subset=["cpu_value", "gpu_value"], inplace=True)
        if total_pid.empty:
            continue
        total_pid["value"] = total_pid["cpu_value"] + total_pid["gpu_value"]
        total_pid["metric_id"] = f"attributed_energy_total_J_R_node__C_process_{pid}_A_"
        total_pid["base_metric"] = "attributed_energy_total_J"
        synthetic_parts.append(total_pid[["metric_id", "base_metric", "timestamp", "value"]])

    if not synthetic_parts:
        return pd.DataFrame(columns=df_processed.columns)

    return pd.concat(synthetic_parts, ignore_index=True)


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

def get_process_time_range_from_df(df: pd.DataFrame) -> tuple:
    """Get the process active time range from the dataframe.
    
    Finds the actual process execution period by looking at process-level metrics
    (consumer_kind='process') that have non-zero values. These metrics only have
    values when the process is actually running.
    Returns the first and last timestamps where the process was active.
    
    Optimized: removed unnecessary .copy() calls for read-only operations.
    """
    if df.empty or "timestamp" not in df.columns:
        return None, None
    
    # Filter for process-level data (consumer_kind == 'process')
    # No .copy() needed - we're only reading, not modifying
    process_mask = df["consumer_kind"] == "process"
    
    if not process_mask.any():
        # Fallback: use all timestamps if no process-level data found
        return df["timestamp"].min(), df["timestamp"].max()
    
    # Filter for any process-level metric with non-zero values
    # Non-zero values indicate the process was active at that time
    active_mask = process_mask & (df["value"] > 0)
    
    if not active_mask.any():
        # Fallback: use all process-level timestamps
        process_timestamps = df.loc[process_mask, "timestamp"]
        return process_timestamps.min(), process_timestamps.max()
    
    # Get the first and last timestamps where process was active
    active_timestamps = df.loc[active_mask, "timestamp"]
    return active_timestamps.min(), active_timestamps.max()
    
# ================================================
# Metric classification helper functions
# ================================================
def metric_id_is_process_consumer(metric_id: str) -> bool:
    """
    True if the metric row was attributed to a process (consumer_kind == 'process').
    
    Preprocessed metric_id embeds consumers as ..._C_{consumer_kind}_{consumer_id}_A_{late_attributes}....
    Process rows contain _C_process_<pid>_ (e.g. _C_process_172681_A_).
    """
    return "_C_process_" in str(metric_id)

def is_cumulative_metric(metric_id: str) -> bool:
    """
    Determine if a metric represents a cumulative quantity that should be summed over time.
    
    Based on analysis of Alumet CSV outputs, metrics fall into two categories:
    
    CUMULATIVE (values accumulate over time, summing makes sense):
    - Energy metrics: attributed_energy_J, rapl_consumed_energy_J
    - Time metrics: cpu_time_delta_ns, kernel_cpu_time_ms
    - Hardware perf counters: perf_hardware_* (INSTRUCTIONS, CPU_CYCLES, CACHE_*, BRANCH_*, etc.)
    - Software perf counters: perf_software_* (PAGE_FAULTS, CONTEXT_SWITCHES, etc.)
    - Kernel counters: kernel_context_switches, kernel_new_forks
    
    NON-CUMULATIVE (instantaneous state or rate values):
    - Memory metrics: mem_*, *_kB, memory_usage_B (current state snapshots)
    - Rate metrics: cpu_percent, cpu_percent_% (instantaneous utilization)
    - State counters: kernel_n_procs_running, kernel_n_procs_blocked (current count)
    """
    metric_lower = str(metric_id).lower()
    
    # Non-cumulative metrics
    non_cumulative_patterns = [
        "percent",          # cpu_percent, cpu_percent_% - rates
        "n_procs",          # kernel_n_procs_running, kernel_n_procs_blocked - current state
        "mem_total",        # Total memory (constant)
        "mem_free",         # Current free memory
        "mem_available",    # Current available memory
        "memory_usage",     # Current memory usage
        "active_kb",        # Current active memory
        "inactive_kb",      # Current inactive memory
        "cached_kb",        # Current cached memory  
        "mapped_kb",        # Current mapped memory
        "swap_cached",      # Current swap cached
        # GPU (NVML) instantaneous metrics
        "nvml_instant_power",       # instantaneous power reading (mW)
        "nvml_temperature",         # instantaneous temperature (°C)
        "nvml_gpu_utilization",     # instantaneous utilization (%)
        "nvml_memory_utilization",  # instantaneous memory utilization (%)
        "nvml_encoder_utilization", # instantaneous encoder utilization (%)
        "nvml_decoder_utilization", # instantaneous decoder utilization (%)
        "nvml_sm_utilization",      # instantaneous SM utilization (%)
        "nvml_n_compute",           # current count of compute processes
        "nvml_n_graphic",           # current count of graphic processes
        "nvml_encoder_sampling",    # sampling period (μs)
        "nvml_decoder_sampling",    # sampling period (μs)
    ]
    
    for pattern in non_cumulative_patterns:
        if pattern in metric_lower:
            return False
    
    # Cumulative metrics
    cumulative_patterns = [
        # Energy metrics (Joules)
        "energy",           # attributed_energy_J, rapl_consumed_energy_J, nvml_energy_consumption_mJ
        "_j",               # Joules unit suffix
        
        # Time metrics
        "cpu_time",         # cpu_time_delta_ns, kernel_cpu_time_ms
        "time_delta",       # Time deltas
        "time_ms",          # Time in milliseconds
        "time_ns",          # Time in nanoseconds
        
        # Hardware performance counters (perf_hardware_*)
        "perf_hardware",    # All hardware perf counters are cumulative
        "instruction",      # perf_hardware_INSTRUCTIONS, perf_hardware_BRANCH_INSTRUCTIONS
        "cpu_cycles",       # perf_hardware_CPU_CYCLES, perf_hardware_REF_CPU_CYCLES
        "bus_cycles",       # perf_hardware_BUS_CYCLES
        "cache_miss",       # perf_hardware_CACHE_MISSES
        "cache_ref",        # perf_hardware_CACHE_REFERENCES
        "branch_miss",      # perf_hardware_BRANCH_MISSES
        
        # Software performance counters (perf_software_*)
        "perf_software",    # All software perf counters are cumulative
        "page_fault",       # perf_software_PAGE_FAULTS*
        "context_switch",   # perf_software_CONTEXT_SWITCHES, kernel_context_switches
        "cpu_migration",    # perf_software_CPU_MIGRATIONS
        "cgroup_switch",    # perf_software_CGROUP_SWITCHES
        "alignment_fault",  # perf_software_ALIGNMENT_FAULTS
        "emulation_fault",  # perf_software_EMULATION_FAULTS
        
        # Kernel counters
        "new_forks",        # kernel_new_forks
        
        # General patterns for other potential metrics
        "flop",             # Floating point operations
        "bytes_read",       # Data read (I/O)
        "bytes_written",    # Data written (I/O)
        "bytes_transfer",   # Data transferred
        "packets",          # Network packets
    ]
    
    for pattern in cumulative_patterns:
        if pattern in metric_lower:
            return True
    
    return False


def get_metric_unit(metric_name: str) -> str:
    """
    Extract the unit from a metric name.
    
    Note: Memory metrics with "_kB" suffix actually store values in Bytes, not kiloBytes.
    This is a known issue with the naming convention.
    
    Returns:
        Unit string (e.g., "J", "B", "ns", "ms", "%", "mW", "mJ", "°C", "μs")
    """
    metric_lower = str(metric_name).lower()
    
    # GPU power metrics (milliWatts) — check before generic energy match
    if "_mw" in metric_lower or "instant_power" in metric_lower:
        return "mW"
    
    # GPU energy metrics (milliJoules) — check before generic "_j" match
    if "_mj" in metric_lower:
        return "mJ"
    
    # GPU temperature metrics
    if "°c" in metric_lower or "temperature" in metric_lower:
        return "°C"
    
    # GPU sampling period metrics (microseconds)
    if "μs" in metric_lower or "_μs" in metric_name or "sampling_period" in metric_lower:
        return "μs"
    
    # Energy metrics (Joules)
    if "_j" in metric_lower or "energy" in metric_lower:
        return "J"
    
    # Memory metrics - values are in Bytes despite "_kB" in name
    if "_kb" in metric_lower or "memory_usage" in metric_lower:
        return "B"
    
    # Time metrics
    if "_ns" in metric_lower or "delta_ns" in metric_lower:
        return "ns"
    if "_ms" in metric_lower or "time_ms" in metric_lower:
        return "ms"
    
    # Percentage metrics
    if "percent" in metric_lower or metric_lower.endswith("_%"):
        return "%"
    
    # Count metrics (no unit)
    return ""


def is_memory_metric(metric_name: str) -> bool:
    """Check if a metric is a memory-related metric (values in Bytes).
    
    Excludes nvml_memory_utilization_% which is a percentage, not a byte count.
    """
    metric_lower = str(metric_name).lower()
    # Exclude NVML memory utilization (it's a percentage, not bytes)
    if "nvml_memory" in metric_lower:
        return False
    memory_patterns = [
        "mem_", "memory", "_kb", "active_kb", "inactive_kb", 
        "cached_kb", "mapped_kb", "swap_cached"
    ]
    return any(p in metric_lower for p in memory_patterns)


def format_bytes_ticklabel(value: float, decimals: int = 1) -> str:
    """
    Format a byte value with appropriate unit (B, KB, MB, GB, TB).
    Uses binary prefixes (1024-based).
    
    Args:
        value: Value in bytes
        decimals: Decimal places for scaled units (KB-TB). Use 2-3 when the axis span is
            narrow magnitude so adjacent ticks don't format to identical labels.
    
    Returns:
        Formatted string with appropriate unit
    """
    d = max(0, min(6, int(decimals)))
    if abs(value) < 1024:
        return f"{value:.0f} B"
    elif abs(value) < 1024 ** 2:
        return f"{value / 1024:.{d}f} KB"
    elif abs(value) < 1024 ** 3:
        return f"{value / (1024 ** 2):.{d}f} MB"
    elif abs(value) < 1024 ** 4:
        return f"{value / (1024 ** 3):.{d}f} GB"
    else:
        return f"{value / (1024 ** 4):.{d}f} TB"


def get_bytes_tickvals_ticktext(y_min: float, y_max: float, num_ticks: int = 5) -> tuple:
    """
    Generate tick values and formatted tick text for byte-valued axes.
    
    Uses adaptive precision and label deduplication so narrow ranges
    don't produce multiple overlapping labels from linspace in raw bytes.
    
    Args:
        y_min: Minimum y value in bytes (will be clamped to 0 if negative)
        y_max: Maximum y value in bytes
        num_ticks: Approximate number of ticks to generate
    
    Returns:
        Tuple of (tickvals, ticktext) lists for Plotly axis configuration
    """
    # Ensure y_min is not negative (memory can't be negative)
    y_min = max(0, float(y_min))
    y_max = float(y_max)
    
    if y_max <= y_min:
        y_max = y_min + 1
    
    span = y_max - y_min
    span_ratio = span / max(abs(y_max), abs(y_min), 1e-12)

    # Prefer fewer ticks when relative span is tiny (reduces collision risk)
    nt_candidates = [num_ticks, max(3, num_ticks - 1), 3]
    if span_ratio < 0.001:
        nt_candidates = [4, 3]
        
    for nt in nt_candidates:
        tickvals = np.linspace(y_min, y_max, nt)
        for dec in (1, 2, 3, 4):
            ticktext = [format_bytes_ticklabel(float(v), decimals=dec) for v in tickvals]
            if len(set(ticktext)) == len(ticktext):
                return list(tickvals), ticktext
        # Dedupe same-formatted labels while keeping spread
        ticktext = [format_bytes_ticklabel(float(v), decimals=4) for v in tickvals]
        out_vals: list = []
        out_text: list = []
        seen: set = set()
        for v, t in zip(tickvals, ticktext):
            if t not in seen:
                seen.add(t)
                out_vals.append(float(v))
                out_text.append(t)
        if len(out_vals) >= 2:
            return out_vals, out_text
        
    # Last resort: endpoints only
    return (
        [y_min, y_max],
        [
            format_bytes_ticklabel(y_min, decimals=4),
            format_bytes_ticklabel(y_max, decimals=4),
        ],
    )



# ================================================
# Metric plot label, title, and hovertemplate formatting helper functions
# ================================================
def _split_kind_id(part: str) -> tuple:
    if not part:
        return "", ""
    # Split on last underscore if the last part appears to be an ID
    if "_" in part:
        parts_list = part.rsplit("_", 1)
        if len(parts_list) == 2:
            potential_id = parts_list[1]
            # Check if it appears to be an ID (number or decimal number value)
            if (potential_id.replace(".", "").replace("-", "").isdigit() or 
                potential_id in ["total", "0", "1", ""] or
                (len(potential_id) <= 15 and "_" not in potential_id)):
                kind = parts_list[0].replace("_", " ") if parts_list[0] else ""
                return kind, potential_id
    # If we can't determine, show the whole string with underscores replaced by spaces for readability
    return part.replace("_", " "), ""

def _format_id(id_str: str) -> str:
    # Convert ID to int if it's a whole number
    if not id_str:
        return ""
    try:
        float_val = float(id_str)
        if float_val.is_integer():
            return str(int(float_val))
        else:
            return id_str  
    except (ValueError, TypeError):
        return id_str

def _format_metric_title(metric_id: str) -> str:
    """Format metric_id into more well-structured plot title.
    
    Parses metric_id format: {base_metric}_R_{resource_kind}_{resource_id}_C_{consumer_kind}_{consumer_id}_A_{late_attributes}
    Returns formatted string: "{base_metric} R: {resource_kind} {resource_id} C: {consumer_kind} {consumer_id} A: {__late_attributes}"
    """
    try:
        # Split by the delimiter "_R_" to get the base metric
        if "_R_" not in metric_id:
            return metric_id  # Return as-is if format doesn't match
        
        parts = metric_id.split("_R_", 1)
        base_metric = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        
        
        if "_C_" not in rest:
            # No consumer part, just resource part
            # Split by the delimiter "_A_" to get the late attributes if present
            resource_part = rest.split("_A_")[0] if "_A_" in rest else rest
            consumer_part = ""
            late_attr = rest.split("_A_", 1)[1] if "_A_" in rest else ""
        else:
            # Split by the delimiter "_C_" to get the resource part, consumer part and rest
            resource_consumer = rest.split("_C_", 1)
            resource_part = resource_consumer[0]
            rest = resource_consumer[1] if len(resource_consumer) > 1 else ""
            # Split by the delimiter "_A_" to get the late attributes if present
            if "_A_" not in rest:
                consumer_part = rest
                late_attr = ""
            else:
                consumer_late = rest.split("_A_", 1)
                consumer_part = consumer_late[0]
                late_attr = consumer_late[1] if len(consumer_late) > 1 else ""
        
        resource_kind, resource_id = _split_kind_id(resource_part)
        consumer_kind, consumer_id = _split_kind_id(consumer_part)
        
        # Format IDs as integers if they're numeric
        resource_id_formatted = _format_id(resource_id)
        consumer_id_formatted = _format_id(consumer_id)
        
        # Build formatted title
        title_parts = [base_metric]
        
        # Add resource info if present
        if resource_kind or resource_id_formatted:
            resource_str = f"R: {resource_kind}"
            if resource_id_formatted:
                resource_str += f" {resource_id_formatted}"
            title_parts.append(resource_str)
        
        # Add consumer info if present
        if consumer_kind or consumer_id_formatted:
            consumer_str = f"C: {consumer_kind}"
            if consumer_id_formatted:
                consumer_str += f" {consumer_id_formatted}"
            title_parts.append(consumer_str)
        
        # Add late attributes if present with underscores replaced by spaces for readability
        if late_attr:
            late_attr_formatted = late_attr.replace("_", " ")
            title_parts.append(f"A: {late_attr_formatted}")
        
        return " ".join(title_parts)
    except Exception:
        # If parsing fails, return the original metric_id with underscores replaced by spaces for readability
        return metric_id.replace("_", " ")

def metric_id_to_plot_label(metric_id: str, max_len: int = 60) -> str:
    if not metric_id:
        return ""
    s = str(metric_id)

    # normalizations
    s = s.replace("_R_", " | R=")
    s = s.replace("_C_", " | C=")
    s = s.replace("_A_", " | ")
    s = s.replace("__", "_")

    # shorten some metric names
    s = s.replace("local_machine", "local")
    s = s.replace("cpu_percent_%", "cpu%")
    s = s.replace("kernel_cpu_time_ms", "kernel_cpu_ms")

    # remove process IDs
    s = re.sub(r"\| C=process_\d+(\.\d+)?", "| C=process", s)

    # truncate for display
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s

# ================================================
# Helper functions for MATCH callbacks
# ================================================
def norm(x):
    """Normalize a value to a clean string or None"""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s

def uniq_str(series: pd.Series) -> list:
    """Get unique non-empty string values from a series.
    
    Optimized using vectorized operations instead of loop.
    """
    # astype(object) first to escape category dtype (fillna requires the fill value to be an
    # existing category, but "" is not). After object cast, float NaN can be filled safely.
    str_series = series.astype(object).fillna("").astype(str).str.strip()
    # Filter out empty strings using boolean indexing (vectorized)
    mask = str_series != ""
    # Get unique values and sort
    unique_vals = str_series[mask].unique()
    return sorted(unique_vals)


# ================================================
# Plotting helper functions
# ================================================
def get_color_palette(n_colors: int) -> List[str]:
    """Get a color palette for n_colors time series.
    Uses Plotly's qualitative color palettes and cycles through them if needed.
    """
    # Use Plotly's qualitative palettes
    # Combine multiple palettes for more colors
    palettes = [
        pc.qualitative.Plotly,      # 10 colors
        pc.qualitative.Set2,        # 8 colors
        pc.qualitative.Set3,        # 12 colors
        pc.qualitative.Pastel,      # 10 colors
        pc.qualitative.Dark2,       # 8 colors
        pc.qualitative.Pastel1,     # 9 colors
        pc.qualitative.Pastel2,     # 8 colors
    ]
    
    colors = []
    for palette in palettes:
        colors.extend(palette)
        if len(colors) >= n_colors:
            break
    
    # If still not enough, cycle through the colors
    while len(colors) < n_colors:
        colors.extend(colors[:min(len(colors), n_colors - len(colors))])
    
    return colors[:n_colors]

def create_all_timeseries_plots(df_processed: pd.DataFrame, proc_start: Optional[pd.Timestamp] = None, proc_end: Optional[pd.Timestamp] = None, full_time_range: Optional[tuple] = None, category: Optional[str] = None, share_yaxis: bool = False) -> go.Figure:
    """Create all time series as scrollable subplots.
    
    Args:
        df_processed: Filtered dataframe with metric_id, timestamp, and value
        proc_start: Process start time for gray highlight
        proc_end: Process end time for gray highlight
        full_time_range: Tuple of (min_time, max_time) for full measurement range to fix x-axis
        category: Metric category ("energy", "utilization", "temperature", "memory", "perf_counters", "kernel_cpu_time", "kernel_system", "miscellaneous") to set appropriate Y-axis label
        share_yaxis: If True, all subplots share the same Y-axis range for easier comparison
    """
    if df_processed.empty:
        return go.Figure()
    
    # Coerce timestamps
    df_processed = df_processed.copy()
    df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"], errors="coerce")
    
    # Get unique metric_ids
    unique_metrics = df_processed["metric_id"].unique()
    n_metrics = len(unique_metrics)

    if n_metrics == 0:
        return go.Figure()
    
    # Get full time range for x-axis (calculate from data if provided range is not provided)
    if full_time_range:
        x_min, x_max = full_time_range
    else:
        x_min = df_processed["timestamp"].min()
        x_max = df_processed["timestamp"].max()
    x_min = pd.Timestamp(x_min)
    x_max = pd.Timestamp(x_max)
    
    # Get color palette
    colors = get_color_palette(n_metrics)
    color_map = {metric: colors[i] for i, metric in enumerate(unique_metrics)}
    
    # Vertical spacing between subplots
    MIN_SUBPLOT_HEIGHT = 240  # Pixels per subplot row (scroll container avoids squashing)
    SUBPLOT_GAP_PX = 52       # Fixed pixel gap between subplot rows
    total_height = MIN_SUBPLOT_HEIGHT * n_metrics + SUBPLOT_GAP_PX * max(n_metrics - 1, 0)
    vertical_spacing = (SUBPLOT_GAP_PX / total_height) if n_metrics > 1 else 0.05
    
    # Create subplots with formatted titles
    # Using shared_xaxes=True so zooming one subplot zooms all others to the same time region
    formatted_titles = [_format_metric_title(metric_id) for metric_id in unique_metrics]
    fig = make_subplots(
        rows=n_metrics, cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=[f"<b>{title}</b>" for title in formatted_titles],
    )

    # Pre-calculate y-axis ranges for each metric using vectorized groupby (much faster than loop)
    y_stats = df_processed.groupby("metric_id")["value"].agg(["min", "max"])
    
    # Check if this is a memory category
    is_memory_category = category == "memory"
    
    y_ranges = {}
    for metric_id in unique_metrics:
        if metric_id not in y_stats.index:
            y_ranges[metric_id] = {"min": 0 if is_memory_category else -1, "max": 1}
            continue
        
        y_min = y_stats.loc[metric_id, "min"]
        y_max = y_stats.loc[metric_id, "max"]
        y_range = y_max - y_min if y_max != y_min else abs(y_max) if y_max != 0 else 1
        y_padding = 0.1 * y_range if y_range > 0 else 0.1
        
        # Ensure valid range (min < max)
        calculated_min = y_min - y_padding
        calculated_max = y_max + y_padding
        if calculated_min >= calculated_max:
            calculated_min = y_min - 0.1 if y_min != 0 else -0.1
            calculated_max = y_max + 0.1 if y_max != 0 else 0.1
        
        # For memory metrics, ensure minimum is not negative
        if is_memory_category:
            calculated_min = max(0, calculated_min)
        
        y_ranges[metric_id] = {
            "min": calculated_min,
            "max": calculated_max
        }
    
    # If sharing Y-axis, calculate global range across all metrics
    shared_y_range = None
    if share_yaxis and y_ranges:
        global_min = min(r["min"] for r in y_ranges.values())
        global_max = max(r["max"] for r in y_ranges.values())
        # For memory metrics, ensure minimum is not negative
        if is_memory_category:
            global_min = max(0, global_min)
        shared_y_range = {"min": global_min, "max": global_max}

    # Gray highlighted zone for process active period
    # Use layout shapes (vrect) instead of traces — these always span the full Y-axis
    # regardless of zoom level, and don't clutter the legend
    if proc_start and proc_end:
        for idx in range(1, n_metrics + 1):
            # yref="y{n} domain" makes the shape span 0-100% of the subplot's Y-axis
            yref = f"y{idx} domain" if idx > 1 else "y domain"
            xref = f"x{idx}" if idx > 1 else "x"
            fig.add_shape(
                type="rect",
                x0=proc_start, x1=proc_end,
                y0=0, y1=1,
                xref=xref,
                yref=yref,
                fillcolor="rgba(136, 192, 208, 0.12)",
                line=dict(width=0),
                layer="below",
            )
        # Add a single trace for the legend entry without plotting a data point.
        # Using visible="legendonly" makes Plotly dim the label, which has poor
        # contrast in both light and dark themes.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="rgba(136, 192, 208, 0.4)", symbol="square"),
                name="Process Active",
                showlegend=True,
                legendgroup="process_active",
            ),
            row=1, col=1,
        )

    # Pre-group data by metric_id for faster iteration (sort once, not per metric)
    df_sorted = df_processed.sort_values(["metric_id", "timestamp"])
    grouped = {mid: grp for mid, grp in df_sorted.groupby("metric_id", observed=True, sort=False)}
    
    # Determine total points to decide rendering strategy
    # Per-metric rendering: use WebGL when the series is huge.
    total_points = len(df_processed)
    use_webgl = total_points > 10000  # Use WebGL for large datasets
    show_markers_global = total_points < 5000  # Only show markers for smaller datasets
    
    # Add traces for each metric
    for idx, metric_id in enumerate(unique_metrics, start=1):
        metric_data = grouped.get(metric_id, pd.DataFrame())
        if metric_data.empty:
            continue

        n_pts = len(metric_data)
        use_webgl = n_pts > 10000  # Use WebGL for large series based on total number of points in each metric
        show_markers = show_markers_global and n_pts < 5000
            
        color = color_map[metric_id]

        # Convert hex color to rgba for fillcolor
        if color.startswith('#'):
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgba_fill = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)"
        elif color.startswith('rgba'):
            # Already rgba, just adjust opacity
            rgba_fill = color.rsplit(',', 1)[0] + ', 0.15)'
        elif color.startswith('rgb'):
            # Convert rgb to rgba
            rgba_fill = color.replace('rgb', 'rgba').replace(')', ', 0.15)')
        else:
            # Default fallback
            rgba_fill = "rgba(136, 192, 208, 0.15)"

        # Choose scatter type: Scattergl (WebGL) for large series, Scatter otherwise
        ScatterClass = go.Scattergl if use_webgl else go.Scatter
        
        # Build trace config - markers only for smaller datasets (performance)
        trace_config = dict(
            x=pd.to_datetime(metric_data["timestamp"], errors="coerce"),
            y=metric_data["value"],
            mode="lines+markers" if show_markers else "lines",
            name=metric_id,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{metric_id}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
            showlegend=False,
        )
        
        # Add markers only for smaller datasets
        if show_markers:
            trace_config["marker"] = dict(
                color=color,
                size=6,
                symbol="circle",
                line=dict(width=1, color="rgba(255, 255, 255, 0.5)"),
            )
        
        # Fill only works well with regular Scatter (not Scattergl)
        if not use_webgl:
            trace_config["fill"] = "tozeroy"
            trace_config["fillcolor"] = rgba_fill

        fig.add_trace(ScatterClass(**trace_config), row=idx, col=1)
        
        # Fix x-axis range to full measurement time
        # With shared_xaxes=True, zooming one subplot zooms all others to the same time region
        fig.update_xaxes(
            type="date",
            range=[x_min, x_max],
            gridcolor="rgba(76, 86, 106, 0.2)",
            showticklabels=True,  # Show tick labels on each subplot (overrides shared_xaxes default)
            # White dotted spike line on hover - only at data points
            showspikes=True,
            spikemode="across",
            spikesnap="data",  # Only show spike when hovering near actual data points
            spikethickness=1,
            spikecolor="white",
            spikedash="dot",
            row=idx, col=1
        )
        # Determine Y-axis label based on category
        if category == "energy":
            y_axis_label = "Value (J)"
        elif category == "power":
            y_axis_label = "Value (W)"
        elif category == "memory":
            y_axis_label = "Value (B)"
        elif category == "utilization":
            y_axis_label = "Value (%)"
        elif category == "temperature":
            y_axis_label = "Value (°C)"
        elif category == "perf_counters":
            y_axis_label = "Value (count)"
        else:
            y_axis_label = "Value"
        
        # Build y-axis configuration
        yaxis_config = dict(
            title_text=y_axis_label,
            fixedrange=False,
            gridcolor="rgba(76, 86, 106, 0.2)",
        )
        
        # If sharing Y-axis, set fixed range; otherwise use autorange
        if shared_y_range:
            yaxis_config["range"] = [shared_y_range["min"], shared_y_range["max"]]
            yaxis_config["autorange"] = False
        else:
            yaxis_config["autorange"] = True
        
        # For memory metrics, add custom tick formatting
        if category == "memory" and not metric_data.empty:
            if shared_y_range:
                # Use global range for tick formatting when sharing Y-axis
                tickvals, ticktext = get_bytes_tickvals_ticktext(shared_y_range["min"], shared_y_range["max"], num_ticks=5)
            else:
                y_min, y_max = metric_data["value"].min(), metric_data["value"].max()
                tickvals, ticktext = get_bytes_tickvals_ticktext(y_min, y_max, num_ticks=5)
            yaxis_config["tickvals"] = tickvals
            yaxis_config["ticktext"] = ticktext
        
        fig.update_yaxes(**yaxis_config, row=idx, col=1)
    
    # Add "Time" label only to the bottom subplot
    fig.update_xaxes(title_text="Time", row=n_metrics, col=1)

    # total_height was already calculated above from MIN_SUBPLOT_HEIGHT * n_metrics + gaps
    fig.update_layout(
        height=total_height,  # Total height for all subplots
        title=dict(text="<b>📈 Time series of all metrics</b>", x=0.5, font=dict(size=16)),
        paper_bgcolor="rgba(46, 52, 64, 0.95)",
        plot_bgcolor="rgba(59, 66, 82, 0.7)",
        font=dict(color="#d8dee9"),
        hovermode="closest",  # Show hover info for closest point with spike line
        margin=dict(l=50, r=20, t=60, b=40),  # Reduced margins for wider plots
        autosize=True,  # Enable autosize to fill container width
        width=None,  # Let it fill the container
        showlegend=True,  # Ensure legend is visible
        legend=dict[str, str | int | dict[str, str]](
            bgcolor="rgba(46, 52, 64, 0.8)",
            bordercolor="rgba(136, 192, 208, 0.3)",
            borderwidth=1,
            font=dict(color="#d8dee9"),
        ),
    )
    fig.update_xaxes(type="date", rangeslider=dict(visible=False), row=n_metrics, col=1)
    
    return fig