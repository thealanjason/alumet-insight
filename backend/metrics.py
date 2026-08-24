"""
Metric identity, classification, dataframe helpers, and display units.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, StrEnum

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Types: MetricType, MetricOrigin, PowerKind, MetricKindInfo, MetricId
# ---------------------------------------------------------------------------

class MetricType(str, Enum):
    """Semantics of Alumet metric types."""

    GAUGE = "gauge"
    COUNTER_DIFF = "counter_diff"
    RAW_COUNTER = "raw_counter"


class MetricOrigin(StrEnum):
    """Whether a series came from Alumet measurements or post-processing."""

    MEASURED = "measured"
    DERIVED = "derived"


class PowerKind(str, Enum):
    """Power subtype for gauge metrics; orthogonal to MetricOrigin provenance."""

    NONE = "none"
    MEASURED = "measured"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class MetricKindInfo:
    """Classification result for one base metric name."""

    metric_type: MetricType
    power_kind: PowerKind = PowerKind.NONE


@dataclass(frozen=True, slots=True)
class MetricId:
    """
    Structured form of an Alumet metric id string.

    A full metric id looks like:
        ``base_metric_R_<resource>_C_<consumer>_A_<late_attributes>``

    Each of ``resource``, ``consumer``, and ``late_attributes`` may be:
    - ``None``: that section marker was absent
    - ``""``: the marker was present, but its value was empty

    Keeping that distinction lets ``parse`` → ``serialized`` rebuild the
    exact original string.
    """

    base_metric: str
    resource: str | None = None
    consumer: str | None = None
    late_attributes: str | None = None

    @classmethod
    def parse(cls, metric_id: str) -> "MetricId":
        """Parse a serialized metric id."""
        serialized = str(metric_id)
        base_metric, resource_marker, remainder = serialized.partition("_R_")
        if not resource_marker:
            return cls(base_metric=serialized)

        before_attributes, attributes_marker, late_attributes = remainder.partition("_A_")
        resource, consumer_marker, consumer = before_attributes.partition("_C_")
        return cls(
            base_metric=base_metric,
            resource=resource,
            consumer=consumer if consumer_marker else None,
            late_attributes=late_attributes if attributes_marker else None,
        )

    @property
    def serialized(self) -> str:
        """Return the canonical serialized representation."""
        result = self.base_metric
        if self.resource is not None:
            result += f"_R_{self.resource}"
        if self.consumer is not None:
            result += f"_C_{self.consumer}"
        if self.late_attributes is not None:
            result += f"_A_{self.late_attributes}"
        return result

    @property
    def series_key(self) -> tuple[str | None, str | None, str | None]:
        """Return the resource, consumer, and attributes identity."""
        return self.resource, self.consumer, self.late_attributes

    @property
    def is_process_consumer(self) -> bool:
        return self.component_id("consumer", "process") is not None

    @property
    def process_id(self) -> str | None:
        return self.component_id("consumer", "process")

    def resource_id(self, resource_kind: str) -> str | None:
        """Return the id portion when the resource has the requested kind."""
        return self.component_id("resource", resource_kind)

    def component_id(self, component_name: str, kind: str) -> str | None:
        """Extract an id from a resource or consumer of a known kind.

        Kind/id boundaries cannot be inferred generically because kinds such as
        ``local_machine`` contain underscores. Callers therefore provide the
        expected kind explicitly.
        """
        if component_name not in {"resource", "consumer"}:
            raise ValueError(f"Unknown metric id component: {component_name}")
        component = getattr(self, component_name)
        if component is None:
            return None
        prefix = f"{kind}_"
        return component[len(prefix) :] if component.startswith(prefix) else None

    def with_base_metric(self, base_metric: str) -> "MetricId":
        """Return this series id with a different base metric."""
        return replace(self, base_metric=str(base_metric))

    def __str__(self) -> str:
        return self.serialized


# ---------------------------------------------------------------------------
# 2. Classification tables: exact name and prefix registries
# ---------------------------------------------------------------------------

# Alumet CSV names often append a unit suffix. 
# Classification matches the unit-free stem.
_CLASSIFICATION_UNIT_SUFFIXES: tuple[str, ...] = (
    "_μs",
    "_us",
    "_ms",
    "_ns",
    "_mJ",
    "_mW",
    "_μW",
    "_kB",
    "_°C",
    "_%",
    "_J",
    "_W",
    "_B",
)


def classification_stem(base_metric: str | None) -> str:
    """Return a unit-stripped base metric stem."""
    if base_metric is None:
        return ""
    name = str(base_metric).strip()
    if not name:
        return ""
    for suffix in _CLASSIFICATION_UNIT_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


EXACT_COUNTERDIFF_STEMS: frozenset[str] = frozenset(
    {
        "rapl_consumed_energy",
        "nvml_energy_consumption",
        "amd_gpu_energy_consumption",
        "grace_energy_consumption",
        "attributed_energy",
        "attributed_energy_cpu",
        "attributed_energy_gpu",
        "attributed_energy_gpu_total",
        "attributed_energy_total",
        "cpu_time_delta",
        "kernel_cpu_time",
        "kernel_context_switches",
        "kernel_new_forks",
        "network_bytes",
        "network_errors",
        "network_packet_drops",
        "network_packets",
    }
)

RAW_COUNTER_PREFIXES: tuple[str, ...] = (
    "perf_hardware_",
    "perf_software_",
    "perf_cache_",
)

EXACT_MEASURED_POWER_STEMS: frozenset[str] = frozenset(
    {
        "nvml_instant_power",
        "amd_gpu_power_consumption",
        "grace_instant_power",
        "input_power",
        "disk_power",
        "wattmetre_power",
        "wattmeter_power",
    }
)

EXACT_DERIVED_POWER_STEMS: frozenset[str] = frozenset(
    {
        "rapl_average_power",
        "nvml_average_power",
        "attributed_power",
        "attributed_power_cpu",
        "attributed_power_gpu",
        "attributed_power_gpu_total",
        "attributed_power_total",
        "amd_gpu_average_power",
        "grace_average_power",
    }
)

_DEFAULT_METRIC_KIND = MetricKindInfo(metric_type=MetricType.GAUGE)


# ---------------------------------------------------------------------------
# 3. Identity helpers: parse / serialize / series accessors
# ---------------------------------------------------------------------------

def base_metric_from_id(metric_id: str) -> str:
    """Return the base metric name from a full metric id."""
    return MetricId.parse(metric_id).base_metric


def metric_id_is_process_consumer(metric_id: str) -> bool:
    """Return whether the metric id is attributed to a process consumer."""
    return MetricId.parse(metric_id).is_process_consumer


def filter_process_metric_ids(
    metric_ids: list[str],
    process_only: bool,
) -> list[str]:
    """Restrict to process-attributed series when process_only is true."""
    if not process_only:
        return list(metric_ids)
    return [m for m in metric_ids if metric_id_is_process_consumer(m)]


# ---------------------------------------------------------------------------
# 4. Classification API: predicates used by synthesis and UI
# ---------------------------------------------------------------------------

def classify_base_metric(base_metric: str | None) -> MetricKindInfo:
    """Classify a metric; unknown names render as Gauge by default."""
    stem = classification_stem(base_metric)
    if not stem:
        return _DEFAULT_METRIC_KIND

    if stem in EXACT_DERIVED_POWER_STEMS:
        return MetricKindInfo(
            metric_type=MetricType.GAUGE,
            power_kind=PowerKind.DERIVED,
        )
    if stem in EXACT_MEASURED_POWER_STEMS:
        return MetricKindInfo(
            metric_type=MetricType.GAUGE,
            power_kind=PowerKind.MEASURED,
        )
    if stem in EXACT_COUNTERDIFF_STEMS:
        return MetricKindInfo(metric_type=MetricType.COUNTER_DIFF)
    if stem.startswith(RAW_COUNTER_PREFIXES):
        return MetricKindInfo(metric_type=MetricType.RAW_COUNTER)
    return _DEFAULT_METRIC_KIND


def is_counterdiff_base_metric(base_metric: str | None) -> bool:
    return classify_base_metric(base_metric).metric_type is MetricType.COUNTER_DIFF


def is_raw_counter_base_metric(base_metric: str | None) -> bool:
    """Return whether source values are cumulative counter readings."""
    return classify_base_metric(base_metric).metric_type is MetricType.RAW_COUNTER


def power_kind(metric_id: str) -> PowerKind:
    """Return the power subtype for a metric id."""
    return classify_base_metric(base_metric_from_id(metric_id)).power_kind


def is_counterdiff_metric(metric_id: str) -> bool:
    """Return whether this metric contains interval-delta CounterDiff values."""
    return is_counterdiff_base_metric(base_metric_from_id(metric_id))


def is_raw_counter_metric(metric_id: str) -> bool:
    """Return whether this metric contains undifferenced cumulative readings."""
    return is_raw_counter_base_metric(base_metric_from_id(metric_id))


def is_measured_power_metric(metric_id: str) -> bool:
    """Return whether this is an Alumet-measured instantaneous power gauge."""
    return power_kind(metric_id) is PowerKind.MEASURED


def is_derived_power_metric(metric_id: str) -> bool:
    """Return whether this is post-processed interval-average power."""
    return power_kind(metric_id) is PowerKind.DERIVED


def is_power_metric(metric_id: str) -> bool:
    """Return whether this is a supported measured or derived power gauge."""
    return power_kind(metric_id) is not PowerKind.NONE


def is_cumulative_metric(metric_id: str) -> bool:
    """Return whether comparative aggregation should sum values over time."""
    return is_counterdiff_metric(metric_id)


def is_spike_metric(metric_id: str) -> bool:
    """True when a metric should render as isolated CounterDiff spikes."""
    return is_counterdiff_metric(metric_id)


def is_step_power_metric(metric_id: str) -> bool:
    """True when a metric should render as an interval-aware power step line."""
    return is_derived_power_metric(metric_id)


def metric_type(metric_id: str) -> MetricType:
    """Return the value semantics for a metric id."""
    return classify_base_metric(base_metric_from_id(metric_id)).metric_type


# ---------------------------------------------------------------------------
# 5. Power derivation helpers: energy → derived average-power naming
# ---------------------------------------------------------------------------

def derived_power_base_metric(energy_base_metric: str) -> str:
    """Map an energy base metric to a derived interval-average power name."""
    name = str(energy_base_metric)
    replacements = (
        ("attributed_energy_total_J", "attributed_power_total_W"),
        ("attributed_energy_gpu_total_J", "attributed_power_gpu_total_W"),
        ("attributed_energy_cpu_J", "attributed_power_cpu_W"),
        ("attributed_energy_gpu_J", "attributed_power_gpu_W"),
        ("attributed_energy_J", "attributed_power_W"),
        ("rapl_consumed_energy_J", "rapl_average_power_W"),
        ("rapl_consumed_energy", "rapl_average_power_W"),
        ("nvml_energy_consumption_J", "nvml_average_power_W"),
        ("nvml_energy_consumption", "nvml_average_power_W"),
        ("_energy_J", "_average_power_W"),
        ("_energy", "_average_power_W"),
        ("_J", "_W"),
    )
    for old, new in replacements:
        if old in name:
            return name.replace(old, new, 1)
    if name.endswith("_J"):
        return name[:-2] + "_average_power_W"
    return f"{name}_average_power_W"


def should_derive_power_from_energy(
    energy_metric_id: str,
    available_metric_ids: list[str] | set[str],
) -> bool:
    """Prefer measured power gauges when available; otherwise derive from energy."""
    energy_identity = MetricId.parse(energy_metric_id)
    energy_base = energy_identity.base_metric
    energy_lower = energy_base.lower()
    if "energy" not in energy_lower and "rapl" not in energy_lower:
        return False

    available = {str(metric_id) for metric_id in available_metric_ids}
    available_identities = [MetricId.parse(metric_id) for metric_id in available]
    derived_id = energy_identity.with_base_metric(derived_power_base_metric(energy_base)).serialized
    if derived_id in available:
        return False

    series_key = energy_identity.series_key
    measured_counterparts = {
        "nvml_energy": ("nvml_instant_power",),
        "amd_gpu_energy": ("amd_gpu_power_consumption",),
        "grace_energy": ("grace_instant_power",),
    }
    for energy_pattern, power_patterns in measured_counterparts.items():
        if energy_pattern not in energy_lower:
            continue
        return not any(
            identity.series_key == series_key
            and any(power_pattern in identity.base_metric.lower() for power_pattern in power_patterns)
            for identity in available_identities
        )

    return (
        "rapl" in energy_lower
        or "attributed_energy" in energy_lower
    )


# ---------------------------------------------------------------------------
# 6. Dataframe helpers: filtering and measured/derived provenance
# ---------------------------------------------------------------------------

def metric_ids_from_df(df: pd.DataFrame) -> list[str]:
    """Return sorted metric-id strings from a dataframe."""
    if df.empty or "metric_id" not in df.columns:
        return []
    return sorted(df["metric_id"].dropna().astype(str).unique().tolist())


def filter_by_metric_id(df: pd.DataFrame, metric_id: str) -> pd.DataFrame:
    """Return rows whose full metric id matches *metric_id*."""
    return df[df["metric_id"].astype(str) == str(metric_id)].copy()


def filter_by_base_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return rows whose base metric matches *metric*."""
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
        lines.append(f"  ... and {total - limit} more hidden. Omit --limit or pass a larger value to show all.")
    lines.append(f"Total: {total}")
    return "\n".join(lines)


def mark_as_measured(df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows as coming from Alumet measurements (not post-processed)."""
    if df.empty:
        out = df.copy()
        if "metric_origin" not in out.columns:
            out["metric_origin"] = pd.Series(dtype="string")
        return out

    out = df.copy()
    out["metric_origin"] = MetricOrigin.MEASURED.value
    return out


def mark_as_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Mark rows as post-processed / synthesized (not raw Alumet samples)."""
    if df.empty:
        return df.copy()
    out = df.copy()
    out["metric_origin"] = MetricOrigin.DERIVED.value
    return out


# ---------------------------------------------------------------------------
# 7. Display units: axis / label unit inference
# ---------------------------------------------------------------------------

def get_metric_unit(metric_name: str) -> str:
    """Infer the display unit from a metric name."""
    metric_lower = base_metric_from_id(metric_name).lower()

    # CPU power metrics (Watts)
    if (metric_lower.endswith("_w") or "average_power" in metric_lower or "attributed_power" in metric_lower) and not metric_lower.endswith("_mw"):
        return "W"

    # GPU power metrics (milliWatts)
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
    metric_lower = base_metric_from_id(metric_name).lower()
    # Exclude NVML memory utilization (it's a percentage, not bytes)
    if "nvml_memory" in metric_lower:
        return False
    memory_patterns = [
        "mem_", "memory", "_kb", "active_kb", "inactive_kb", 
        "cached_kb", "mapped_kb", "swap_cached"
    ]
    return any(p in metric_lower for p in memory_patterns)
