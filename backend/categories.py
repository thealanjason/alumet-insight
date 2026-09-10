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

_UTILIZATION_SUBSTRINGS: tuple[str, ...] = (
    "cpu_percent",
    "nvml_gpu_utilization",
    "nvml_sm_utilization",
    "nvml_encoder_utilization",
    "nvml_decoder_utilization",
    "nvml_memory_utilization",
)


def classify_metric(base_metric: str) -> str:
    """Classify a base metric into a time-series category value.

    Power is decided first via ``is_power_metric`` so watt series are not
    swept into energy. Memory uses ``is_memory_metric`` (bytes, including
    GPU info; excludes utilization percentages). Perf counters use
    ``is_raw_counter_base_metric``.
    """
    metric = str(base_metric)
    metric_lower = metric.lower()

    if is_power_metric(metric):
        return "power"
    if "energy" in metric_lower or "rapl" in metric_lower or "attributed_energy" in metric_lower:
        return "energy"
    if any(token in metric_lower for token in _UTILIZATION_SUBSTRINGS):
        return "utilization"
    if "temperature" in metric_lower:
        return "temperature"
    if is_memory_metric(metric):
        return "memory"
    if is_raw_counter_base_metric(metric):
        return "perf_counters"
    if "kernel_cpu_time" in metric_lower:
        return "kernel_cpu_time"
    if metric_lower.startswith("kernel_") or metric_lower.startswith("network_"):
        return "kernel_system"
    return "miscellaneous"


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
    buckets = {classify_metric(metric) for metric in base_metrics}
    return [category.value for category in TIME_SERIES_CATEGORIES if category.value in buckets]


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

    if category not in CATEGORY_VALUES:
        raise ValueError(f"Unknown time-series category: {category}")

    filtered = df[df["base_metric"].map(classify_metric) == category].copy()

    if category == "kernel_cpu_time" and selected_cpu_core:
        selected = str(selected_cpu_core).removesuffix(".0")
        resource_cores = filtered["metric_id"].map(
            lambda metric_id: MetricId.parse(metric_id).resource_id("cpu_core")
        )
        filtered = filtered[resource_cores.fillna("").str.removesuffix(".0") == selected]

    return filtered


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
