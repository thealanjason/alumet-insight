"""Groups Alumet time series output into defined categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backend.metrics import (
    MetricId,
    is_memory_metric,
    is_power_metric,
    is_raw_counter_base_metric,
    metric_ids_from_df,
)

YAXIS_SHAREABLE_CATEGORIES: frozenset[str] = frozenset(
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


@dataclass(frozen=True)
class TimeSeriesCategory:
    value: str
    label: str


TIME_SERIES_CATEGORIES: tuple[TimeSeriesCategory, ...] = (
    TimeSeriesCategory("energy", "Energy (J)"),
    TimeSeriesCategory("power", "Power (W)"),
    TimeSeriesCategory("utilization", "Utilization"),
    TimeSeriesCategory("temperature", "Temperature"),
    TimeSeriesCategory("memory", "Memory (B)"),
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
    df["base_metric"] = df["metric_id"].map(lambda metric_id: MetricId.parse(metric_id).base_metric)
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
        if is_power_metric(str(metric)):
            buckets["power"].add(metric)
            all_categorized.add(metric)
        elif "energy" in metric_lower or "rapl" in metric_lower or "attributed_energy" in metric_lower:
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
        elif is_memory_metric(str(metric)):
            buckets["memory"].add(metric)
            all_categorized.add(metric)
        elif is_raw_counter_base_metric(str(metric)):
            buckets["perf_counters"].add(metric)
            all_categorized.add(metric)
        elif "kernel_cpu_time" in metric_lower:
            buckets["kernel_cpu_time"].add(metric)
            all_categorized.add(metric)
        elif metric_lower.startswith("kernel_") or metric_lower.startswith("network_"):
            buckets["kernel_system"].add(metric)
            all_categorized.add(metric)

    values = [
        category.value for category in TIME_SERIES_CATEGORIES if category.value in buckets and buckets[category.value]
    ]
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
        core = MetricId.parse(metric_id).resource_id("cpu_core")
        if core is not None:
            cpu_cores.add(core.removesuffix(".0"))
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
        energy_mask = df["base_metric"].str.contains("energy|rapl", case=False, na=False) | df[
            "base_metric"
        ].str.contains("attributed_energy", case=False, na=False)
        not_power = ~df["base_metric"].map(is_power_metric)
        return df[energy_mask & not_power].copy()

    if category == "power":
        power_mask = df["base_metric"].map(is_power_metric)
        return df[power_mask].copy()

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
        return df[df["base_metric"].map(is_memory_metric)].copy()

    if category == "perf_counters":
        return df[df["base_metric"].map(is_raw_counter_base_metric)].copy()

    if category == "kernel_cpu_time":
        filtered = df[df["base_metric"] == "kernel_cpu_time_ms"].copy()
        if selected_cpu_core:
            selected = str(selected_cpu_core).removesuffix(".0")
            resource_cores = filtered["metric_id"].map(
                lambda metric_id: MetricId.parse(metric_id).resource_id("cpu_core")
            )
            filtered = filtered[resource_cores.fillna("").str.removesuffix(".0") == selected]
        return filtered

    if category == "kernel_system":
        kernel_mask = df["base_metric"].str.startswith("kernel_")
        kernel_cpu_time_mask = df["base_metric"].str.contains("kernel_cpu_time", na=False)
        network_mask = df["base_metric"].str.startswith("network_")
        return df[(kernel_mask & ~kernel_cpu_time_mask) | network_mask].copy()

    if category == "miscellaneous":
        energy_pat = "energy|rapl|attributed_energy"
        util_pat = "cpu_percent|nvml_gpu_utilization|nvml_sm_utilization|nvml_encoder_utilization|nvml_decoder_utilization|nvml_memory_utilization"
        temp_pat = "temperature"
        perf_pat = "^perf_hardware|^perf_software"
        kernel_pat = "^kernel_"
        network_pat = "^network_"

        is_power = df["base_metric"].map(is_power_metric)
        is_energy = (
            df["base_metric"].str.contains(
                energy_pat,
                case=False,
                na=False,
            )
            & ~is_power
        )
        is_util = df["base_metric"].str.contains(util_pat, case=False, na=False)
        is_temp = df["base_metric"].str.contains(temp_pat, case=False, na=False)
        is_mem = df["base_metric"].map(is_memory_metric)
        is_perf = df["base_metric"].str.contains(perf_pat, case=False, na=False, regex=True)
        is_kernel = df["base_metric"].str.contains(kernel_pat, case=False, na=False, regex=True)
        is_network = df["base_metric"].str.contains(network_pat, case=False, na=False, regex=True)
        return df[~(is_energy | is_power | is_util | is_temp | is_mem | is_perf | is_kernel | is_network)].copy()

    raise ValueError(f"Unknown time-series category: {category}")


def category_for_metric_id(
    df_processed: pd.DataFrame,
    metric_id: str,
    category: Optional[str] = None,
) -> str:
    """Return the corresponding category for the given metric_id."""
    if category:
        return category
    for category_value in available_category_values(df_processed):
        df_category = filter_time_series_category(df_processed, category_value)
        if str(metric_id) in metric_ids_from_df(df_category):
            return category_value
    return "miscellaneous"


def validate_metric_id_in_category(
    df_processed: pd.DataFrame,
    metric_id: str,
    category: Optional[str],
    selected_cpu_core: Optional[str] = None,
) -> None:
    """Raise ValueError when a metric-id/category combination is inconsistent."""
    if category is None:
        return
    df_category = filter_time_series_category(df_processed, category, selected_cpu_core=selected_cpu_core)
    if str(metric_id) not in metric_ids_from_df(df_category):
        raise ValueError(f"Metric '{metric_id}' is not in category '{category}'; use --summary or omit --category.")
