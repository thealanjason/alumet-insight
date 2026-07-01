import pandas as pd


def base_metric_from_id(metric_id: str) -> str:
    """Return the base metric name from a full metric_id (part before the first ``_R_``)."""
    s = str(metric_id)
    return s.split("_R_")[0] if "_R_" in s else s


def metric_ids_from_df(df: pd.DataFrame) -> list[str]:
    """Return sorted metric_id strings from a dataframe."""
    if df.empty or "metric_id" not in df.columns:
        return []
    return sorted(df["metric_id"].dropna().astype(str).unique().tolist())


def filter_by_metric_id(df: pd.DataFrame, metric_id: str) -> pd.DataFrame:
    """Return rows whose full ``metric_id`` matches *metric_id*."""
    return df[df["metric_id"].astype(str) == str(metric_id)].copy()


def filter_by_base_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return rows whose ``base_metric`` matches *metric*."""
    return df[df["base_metric"] == metric].copy()


def format_metric_id_list(
    metric_ids: list[str],
    heading: str,
    *,
    limit: int | None = None,
) -> str:
    """Return a formatted metric-id listing."""
    total = len(metric_ids)
    visible_ids = metric_ids
    truncated = False
    if limit is not None and limit < total:
        visible_ids = metric_ids[:limit]
        truncated = True

    lines = [heading, "-" * 80]
    lines.extend(f"  - {metric_id}" for metric_id in visible_ids)
    if truncated:
        lines.append(
            f"  ... and {total - limit} more hidden. "
            "Omit --limit or pass a larger value to show all."
        )
    lines.append(f"Total: {total}")
    return "\n".join(lines)


def filter_process_metric_ids(metric_ids: list[str], process_only: bool) -> list[str]:
    """Restrict to process-attributed series when *process_only* is True."""
    if not process_only:
        return list(metric_ids)
    return [m for m in metric_ids if metric_id_is_process_consumer(m)]


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
