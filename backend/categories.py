"""Groups Alumet time series output into defined categories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backend.metrics import metric_ids_from_df


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
    TimeSeriesCategory("memory", "Memory"),
    TimeSeriesCategory("perf_counters", "Perf Counters"),
    TimeSeriesCategory("kernel_cpu_time", "Kernel CPU Time"),
    TimeSeriesCategory("kernel_system", "Kernel/System"),
    TimeSeriesCategory("miscellaneous", "Miscellaneous"),
)

CATEGORY_LABELS = {category.value: category.label for category in TIME_SERIES_CATEGORIES}
CATEGORY_VALUES = tuple(category.value for category in TIME_SERIES_CATEGORIES)

UTILIZATION_SUBSTRINGS = (
    "cpu_percent",
    "nvml_gpu_utilization",
    "nvml_sm_utilization",
    "nvml_encoder_utilization",
    "nvml_decoder_utilization",
    "nvml_memory_utilization",
)


def classify_metric(base_metric: str) -> str:
    """Classify a base metric name into a time-series category value."""
    metric_lower = str(base_metric).lower()

    if "nvml_instant_power" in metric_lower:
        return "power"
    if "energy" in metric_lower or "rapl" in metric_lower or "attributed" in metric_lower:
        return "energy"
    if any(token in metric_lower for token in UTILIZATION_SUBSTRINGS):
        return "utilization"
    if "temperature" in metric_lower:
        return "temperature"
    if ("mem" in metric_lower or "memory" in metric_lower or "kb" in metric_lower) and "nvml" not in metric_lower:
        return "memory"
    if metric_lower.startswith("perf_hardware") or metric_lower.startswith("perf_software"):
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
    df["base_metric"] = df["metric_id"].str.split("_R_").str[0]
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

    if category not in CATEGORY_VALUES:
        raise ValueError(f"Unknown time-series category: {category}")

    mask = df["base_metric"].map(classify_metric) == category
    filtered = df[mask].copy()

    if category == "kernel_cpu_time" and selected_cpu_core:
        core_patterns = [
            f"_R_cpu_core_{selected_cpu_core}.0_",
            f"_R_cpu_core_{selected_cpu_core}_",
            f"_R_cpu_core_{selected_cpu_core}.",
        ]
        core_mask = pd.Series(False, index=filtered.index)
        for pattern in core_patterns:
            core_mask |= filtered["metric_id"].str.contains(pattern, na=False, regex=False)
        filtered = filtered[core_mask]

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
