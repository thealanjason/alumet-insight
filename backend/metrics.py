import numpy as np


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
