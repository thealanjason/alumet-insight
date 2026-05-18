"""Groups Alumet time series output into defined categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesCategory:
    value: str
    label: str


TIME_SERIES_CATEGORIES: tuple[TimeSeriesCategory, ...] = (
    TimeSeriesCategory("energy", "Energy (J)"),
    TimeSeriesCategory("power", "Power (W)"),
    TimeSeriesCategory("utilization", "Utilization"),
    TimeSeriesCategory("temperature", "Temperature"),
    TimeSeriesCategory("memory", "Memory"),
    TimeSeriesCategory("perf_counters", "Perf Counters"),
    TimeSeriesCategory("kernel_cpu_time", "Kernel CPU Time"),
    TimeSeriesCategory("kernel_system", "Kernel/System"),
    TimeSeriesCategory("miscellaneous", "Miscellaneous"),
)

CATEGORY_LABELS = {category.value: category.label for category in TIME_SERIES_CATEGORIES}
CATEGORY_VALUES = tuple(category.value for category in TIME_SERIES_CATEGORIES)


def _ensure_base_metric(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a ``base_metric`` column."""
    if "base_metric" in df_processed.columns:
        return df_processed
    df = df_processed.copy()
    df["base_metric"] = df["metric_id"].str.split("_R_").str[0]
    return df


def available_category_values(df_processed: pd.DataFrame) -> list[str]:
    """Return dashboard category values that have data in ``df_processed``."""
    df = _ensure_base_metric(df_processed)
    base_metrics = df["base_metric"].dropna().unique()

    buckets = {
        "energy": set(),
        "power": set(),
        "utilization": set(),
        "temperature": set(),
        "memory": set(),
        "perf_counters": set(),
        "kernel_cpu_time": set(),
        "kernel_system": set(),
    }
    all_categorized = set()

    for metric in base_metrics:
        metric_lower = str(metric).lower()
        if "nvml_instant_power" in metric_lower:
            buckets["power"].add(metric)
            all_categorized.add(metric)
        elif "energy" in metric_lower or "rapl" in metric_lower or "attributed" in metric_lower:
            buckets["energy"].add(metric)
            all_categorized.add(metric)
        elif (
            "cpu_percent" in metric_lower
            or "nvml_gpu_utilization" in metric_lower
            or "nvml_sm_utilization" in metric_lower
            or "nvml_encoder_utilization" in metric_lower
            or "nvml_decoder_utilization" in metric_lower
            or "nvml_memory_utilization" in metric_lower
        ):
            buckets["utilization"].add(metric)
            all_categorized.add(metric)
        elif "temperature" in metric_lower:
            buckets["temperature"].add(metric)
            all_categorized.add(metric)
        elif ("mem" in metric_lower or "memory" in metric_lower or "kb" in metric_lower) and "nvml" not in metric_lower:
            buckets["memory"].add(metric)
            all_categorized.add(metric)
        elif metric_lower.startswith("perf_hardware") or metric_lower.startswith("perf_software"):
            buckets["perf_counters"].add(metric)
            all_categorized.add(metric)
        elif "kernel_cpu_time" in metric_lower:
            buckets["kernel_cpu_time"].add(metric)
            all_categorized.add(metric)
        elif metric_lower.startswith("kernel_") or metric_lower.startswith("network_"):
            buckets["kernel_system"].add(metric)
            all_categorized.add(metric)

    values = [category.value for category in TIME_SERIES_CATEGORIES if category.value in buckets and buckets[category.value]]
    if set(base_metrics) - all_categorized:
        values.append("miscellaneous")
    return values


def available_cpu_cores(df_processed: pd.DataFrame) -> list[str]:
    """Return CPU core identifiers available for ``kernel_cpu_time_ms``."""
    df = _ensure_base_metric(df_processed)
    kernel_metrics = df[df["base_metric"] == "kernel_cpu_time_ms"]
    if kernel_metrics.empty:
        return []

    cpu_cores = set()
    for metric_id in kernel_metrics["metric_id"]:
        if "_R_cpu_core_" not in str(metric_id):
            continue
        try:
            core_part = str(metric_id).split("_R_cpu_core_")[1].split("_")[0]
        except IndexError:
            continue
        core = core_part.replace(".0", "")
        if core:
            cpu_cores.add(core)
    return sorted(cpu_cores)


def filter_time_series_category(
    df_processed: pd.DataFrame,
    category: Optional[str],
    selected_cpu_core: Optional[str] = None,
) -> pd.DataFrame:
    """Filter processed data exactly like the dashboard time-series category dropdown."""
    df = _ensure_base_metric(df_processed)
    if not category:
        return df.copy()

    if category == "energy":
        energy_mask = df["base_metric"].str.contains("energy|rapl|attributed", case=False, na=False)
        not_power = ~df["base_metric"].str.contains("nvml_instant_power", case=False, na=False)
        filtered = df[energy_mask & not_power].copy()
        nvml_energy_mask = filtered["base_metric"].str.contains("nvml_energy_consumption", case=False, na=False)
        if nvml_energy_mask.any():
            filtered.loc[nvml_energy_mask, "value"] = filtered.loc[nvml_energy_mask, "value"] / 1000.0
        from backend.data import synthesize_attributed_energy_total

        synthetic_total = synthesize_attributed_energy_total(filtered)
        if not synthetic_total.empty:
            filtered = pd.concat([filtered, synthetic_total], ignore_index=True)
        return filtered

    if category == "power":
        filtered = df[df["base_metric"].str.contains("nvml_instant_power", case=False, na=False)].copy()
        nvml_power_mask = filtered["base_metric"].str.contains("nvml_instant_power", case=False, na=False)
        if nvml_power_mask.any():
            filtered.loc[nvml_power_mask, "value"] = filtered.loc[nvml_power_mask, "value"] / 1000.0
        return filtered

    if category == "utilization":
        return df[
            df["base_metric"].str.contains(
                "cpu_percent|nvml_gpu_utilization|nvml_sm_utilization|nvml_encoder_utilization|nvml_decoder_utilization|nvml_memory_utilization",
                case=False,
                na=False,
            )
        ].copy()

    if category == "temperature":
        return df[df["base_metric"].str.contains("temperature", case=False, na=False)].copy()

    if category == "memory":
        mem_mask = df["base_metric"].str.contains("mem|memory|kb", case=False, na=False)
        nvml_mask = df["base_metric"].str.contains("nvml", case=False, na=False)
        return df[mem_mask & ~nvml_mask].copy()

    if category == "perf_counters":
        return df[
            df["base_metric"].str.contains("^perf_hardware|^perf_software", case=False, na=False, regex=True)
        ].copy()

    if category == "kernel_cpu_time":
        filtered = df[df["base_metric"] == "kernel_cpu_time_ms"].copy()
        if selected_cpu_core:
            core_patterns = [
                f"_R_cpu_core_{selected_cpu_core}.0_",
                f"_R_cpu_core_{selected_cpu_core}_",
                f"_R_cpu_core_{selected_cpu_core}.",
            ]
            mask = pd.Series([False] * len(filtered), index=filtered.index)
            for pattern in core_patterns:
                mask |= filtered["metric_id"].str.contains(pattern, na=False, regex=False)
            filtered = filtered[mask]
        return filtered

    if category == "kernel_system":
        kernel_mask = df["base_metric"].str.startswith("kernel_")
        kernel_cpu_time_mask = df["base_metric"].str.contains("kernel_cpu_time", na=False)
        network_mask = df["base_metric"].str.startswith("network_")
        return df[(kernel_mask & ~kernel_cpu_time_mask) | network_mask].copy()

    if category == "miscellaneous":
        energy_pat = "energy|rapl|attributed"
        power_pat = "nvml_instant_power"
        util_pat = "cpu_percent|nvml_gpu_utilization|nvml_sm_utilization|nvml_encoder_utilization|nvml_decoder_utilization|nvml_memory_utilization"
        temp_pat = "temperature"
        mem_pat = "mem|memory|kb"
        perf_pat = "^perf_hardware|^perf_software"
        kernel_pat = "^kernel_"
        network_pat = "^network_"

        is_energy = df["base_metric"].str.contains(energy_pat, case=False, na=False) & ~df["base_metric"].str.contains(power_pat, case=False, na=False)
        is_power = df["base_metric"].str.contains(power_pat, case=False, na=False)
        is_util = df["base_metric"].str.contains(util_pat, case=False, na=False)
        is_temp = df["base_metric"].str.contains(temp_pat, case=False, na=False)
        is_mem = df["base_metric"].str.contains(mem_pat, case=False, na=False) & ~df["base_metric"].str.contains("nvml", case=False, na=False)
        is_perf = df["base_metric"].str.contains(perf_pat, case=False, na=False, regex=True)
        is_kernel = df["base_metric"].str.contains(kernel_pat, case=False, na=False, regex=True)
        is_network = df["base_metric"].str.contains(network_pat, case=False, na=False, regex=True)
        return df[~(is_energy | is_power | is_util | is_temp | is_mem | is_perf | is_kernel | is_network)].copy()

    raise ValueError(f"Unknown time-series category: {category}")

