"""Derived metric synthesis — creates new rows that don't exist in the raw CSV."""

from __future__ import annotations

import re

import pandas as pd

from backend.counterdiff import (
    derive_interval_average_power,
    ensure_point_metadata,
    expand_counterdiff_rows,
    interpolate_counterdiff_at_timeline,
    normalize_observed_rows,
    observed_only,
)
from backend.metrics import (
    MetricId,
    should_derive_power_from_energy,
    mark_as_derived,
)

# Alumet energy-attribution configs typically pin CPU energy to RAPL package_total.
RAPL_PACKAGE_TOTAL_LATE_ATTR = "domain=package_total"
CPU_KIND_TOTAL_LATE_ATTR = "kind=total"
PACKAGE_TOTAL_REGEX = re.compile(
    rf"(?:^|,){re.escape(RAPL_PACKAGE_TOTAL_LATE_ATTR)}(?:$|,)"
)
KIND_TOTAL_REGEX = re.compile(
    rf"(?:^|,){re.escape(CPU_KIND_TOTAL_LATE_ATTR)}(?:$|,)"
)


def _attach_process_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Add structured process-consumer and late-attribute columns."""
    if df.empty:
        return df.copy()
    out = df.copy()
    identities = out["metric_id"].map(MetricId.parse)
    out["consumer"] = identities.map(lambda identity: identity.consumer if identity.is_process_consumer else None)
    out["late_attributes"] = identities.map(
        lambda identity: identity.late_attributes if identity.is_process_consumer else None
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out.dropna(subset=["consumer", "late_attributes", "timestamp"])


def _split_kind_id(component: str | None, default_kind: str) -> tuple[str, str]:
    """Best-effort split of a MetricId resource/consumer blob into kind/id."""
    if component is None:
        return default_kind, ""
    text = str(component)
    if "_" not in text:
        return text or default_kind, ""
    kind, _, remainder = text.partition("_")
    return kind or default_kind, remainder


def _sum_observed_by_timestamp_consumer(df: pd.DataFrame) -> pd.DataFrame:
    """Sum observed resources per process/consumer, collapsing resource splits."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "consumer", "value"])

    return df.groupby(
        ["timestamp", "consumer"],
        as_index=False,
        dropna=False,
    )["value"].sum()


def _align_counterdiff_energy_to_timeline(
    df_metric: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    value_name: str,
) -> pd.Series:
    """
    Align CounterDiff energy to a shared timeline with phase-aware interpolation.

    Before the first observation the value is `0`. Between samples the value rises
    from the post-sample zero toward the next observation. After the last observation
    values are `NaN` so downstream synthesis can truncate with `dropna`.
    """
    if df_metric.empty:
        return pd.Series(0.0, index=timeline, name=value_name)

    if df_metric["timestamp"].duplicated().any():
        raise ValueError(
            "CounterDiff alignment requires unique aggregated timestamps; "
            f"found duplicates: {df_metric.loc[df_metric['timestamp'].duplicated(), 'timestamp'].tolist()}"
        )

    aligned_values = interpolate_counterdiff_at_timeline(
        df_metric["timestamp"],
        df_metric["value"],
        timeline,
    )
    aligned = pd.Series(aligned_values.to_numpy(), index=timeline, name=value_name, dtype="float64")
    return aligned


def _select_attributed_cpu_rows(df_cpu: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict attributed CPU energy to the RAPL package_total attribution slice.

    Real Alumet configs attribute process CPU energy from `rapl_consumed_energy` with `domain=package_total`. 
    Summing every CPU late_attribute would mix RAPL scopes. 
    If package_total is absent:
    - a single late-attribute value (including empty) is accepted as unambiguous;
    - `kind=total` is preferred over user/system splits;
    - otherwise return empty rather than blindly summing mixed domains.
    """
    if df_cpu.empty:
        return df_cpu.copy()

    late = df_cpu["late_attributes"].fillna("").astype(str)
    package_mask = late.map(lambda value: PACKAGE_TOTAL_REGEX.search(value) is not None)
    package_total = df_cpu.loc[package_mask]
    if not package_total.empty:
        return package_total.copy()

    unique_late = sorted(set(late.tolist()))
    if len(unique_late) == 1:
        return df_cpu.copy()

    kind_mask = late.map(lambda value: KIND_TOTAL_REGEX.search(value) is not None)
    kind_total = df_cpu.loc[kind_mask]
    if not kind_total.empty:
        return kind_total.copy()

    return df_cpu.iloc[0:0].copy()


def _build_aligned_sum_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    """
    Align two CounterDiff energy series and return timestamp/value sums.

    Missing values are filled with 0 before the first sample and after the last
    so the total spans the union timeline. Appropriate only for process-level
    attributed CPU+GPU totals, where an absent side means that process used 0 J.
    """
    if left.empty or right.empty:
        return pd.DataFrame(columns=["timestamp", "value"])

    timeline = pd.DatetimeIndex(pd.Index(left["timestamp"]).union(pd.Index(right["timestamp"])).sort_values())
    if timeline.empty:
        return pd.DataFrame(columns=["timestamp", "value"])

    left_aligned = _align_counterdiff_energy_to_timeline(left, timeline, left_name)
    right_aligned = _align_counterdiff_energy_to_timeline(right, timeline, right_name)
    out = pd.DataFrame(
        {
            "timestamp": timeline,
            left_name: left_aligned.to_numpy(),
            right_name: right_aligned.to_numpy(),
        }
    )

    out[left_name] = out[left_name].fillna(0.0)
    out[right_name] = out[right_name].fillna(0.0)

    if out.empty:
        return pd.DataFrame(columns=["timestamp", "value"])
    out["value"] = out[left_name] + out[right_name]
    return out.drop(columns=[left_name, right_name])


def _build_cpu_gpu_total_rows(
    cpu_pid: pd.DataFrame,
    gpu_pid: pd.DataFrame,
    *,
    consumer: str,
) -> pd.DataFrame:
    """Align CPU/GPU energy on a shared timeline and emit attributed_energy_total_J rows."""
    total_pid = _build_aligned_sum_rows(
        cpu_pid,
        gpu_pid,
        left_name="cpu_value",
        right_name="gpu_value",
    )
    if total_pid.empty:
        return total_pid

    consumer_kind, consumer_id = _split_kind_id(consumer, "process")
    # Process totals are identified by pid; RAPL/GPU late attrs are not a shared key.
    late_attributes = ""
    total_pid["metric_id"] = MetricId(
        base_metric="attributed_energy_total_J",
        resource="total_",
        consumer=consumer,
        late_attributes=late_attributes,
    ).serialized
    total_pid["base_metric"] = "attributed_energy_total_J"
    total_pid["metric"] = "attributed_energy_total_J"
    total_pid["resource_kind"] = "total"
    total_pid["resource_id"] = ""
    total_pid["consumer_kind"] = consumer_kind
    total_pid["consumer_id"] = consumer_id
    total_pid["__late_attributes"] = late_attributes
    return total_pid


def synthesize_attributed_energy_total(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize process-scoped attributed energy totals.

    Creates:
    1. `attributed_energy_gpu_total_J`: sum of attributed GPU energy across GPUs per pid.
    2. `attributed_energy_total_J`: package_total-attributed CPU + GPU total per pid.

    CPU+GPU totals use the ``fill_zero`` boundary policy: a missing side (not yet
    started or already ended) counts as 0 J so the process total covers the union
    of CPU/GPU activity (e.g. CPU-only tail after GPU attribution stops).
    """
    observed = observed_only(df_processed)
    if observed.empty:
        return pd.DataFrame(
            columns=list(df_processed.columns)
            if len(df_processed.columns)
            else [
                "metric_id",
                "base_metric",
                "timestamp",
                "value",
                "point_role",
                "point_order",
                "sample_id",
            ]
        )

    cpu_mask = observed["base_metric"].str.contains("attributed_energy_cpu", case=False, na=False)
    gpu_mask = observed["base_metric"].str.contains("attributed_energy_gpu", case=False, na=False)
    # Exclude already-synthesized gpu_total from re-aggregation.
    gpu_mask = gpu_mask & ~observed["base_metric"].str.contains("attributed_energy_gpu_total", case=False, na=False)

    df_cpu = observed.loc[cpu_mask, ["metric_id", "timestamp", "value"]].copy()
    df_gpu = observed.loc[gpu_mask, ["metric_id", "timestamp", "value"]].copy()

    if df_cpu.empty and df_gpu.empty:
        return pd.DataFrame(columns=observed.columns)

    df_cpu = _attach_process_identity(df_cpu)
    df_gpu = _attach_process_identity(df_gpu)

    if df_cpu.empty and df_gpu.empty:
        return pd.DataFrame(columns=observed.columns)

    df_cpu = _select_attributed_cpu_rows(df_cpu)
    df_gpu_summed = _sum_observed_by_timestamp_consumer(df_gpu)
    df_cpu_summed = _sum_observed_by_timestamp_consumer(df_cpu)

    synthetic_parts: list[pd.DataFrame] = []

    for consumer in sorted(df_gpu_summed["consumer"].dropna().unique()):
        gpu_pid = df_gpu_summed.loc[
            df_gpu_summed["consumer"] == consumer,
            ["timestamp", "value"],
        ].copy()
        consumer_kind, consumer_id = _split_kind_id(consumer, "process")
        gpu_pid["metric_id"] = MetricId(
            base_metric="attributed_energy_gpu_total_J",
            resource="gpu_all_",
            consumer=consumer,
            late_attributes="",
        ).serialized
        gpu_pid["base_metric"] = "attributed_energy_gpu_total_J"
        gpu_pid["metric"] = "attributed_energy_gpu_total_J"
        gpu_pid["resource_kind"] = "gpu"
        gpu_pid["resource_id"] = "all"
        gpu_pid["consumer_kind"] = consumer_kind
        gpu_pid["consumer_id"] = consumer_id
        gpu_pid["__late_attributes"] = ""
        synthetic_parts.append(gpu_pid)

    cpu_consumers = set(df_cpu_summed["consumer"].unique()) if not df_cpu_summed.empty else set()
    gpu_consumers = set(df_gpu_summed["consumer"].unique()) if not df_gpu_summed.empty else set()
    for consumer in sorted(cpu_consumers & gpu_consumers):
        total_pid = _build_cpu_gpu_total_rows(
            df_cpu_summed.loc[df_cpu_summed["consumer"] == consumer, ["timestamp", "value"]],
            df_gpu_summed.loc[df_gpu_summed["consumer"] == consumer, ["timestamp", "value"]],
            consumer=consumer,
        )
        if not total_pid.empty:
            synthetic_parts.append(total_pid)

    if not synthetic_parts:
        return pd.DataFrame(columns=observed.columns)

    derived = pd.concat(synthetic_parts, ignore_index=True)
    return ensure_point_metadata(mark_as_derived(derived))


def synthesize_derived_power(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Derive interval-average power (W) from CounterDiff energy when no measured power exists.

    Prefers existing watt Gauges (e.g. `nvml_instant_power`). Derives RAPL and
    attributed power always; derives NVML/AMD/Grace energy power only when a matching
    measured power series is absent.
    """
    observed = observed_only(df_processed)
    if observed.empty or "metric_id" not in observed.columns:
        return pd.DataFrame(columns=list(df_processed.columns))

    available_ids = set(observed["metric_id"].astype(str).unique())
    energy_ids = [mid for mid in available_ids if should_derive_power_from_energy(mid, available_ids)]
    if not energy_ids:
        return pd.DataFrame(columns=observed.columns)

    frames: list[pd.DataFrame] = []
    for energy_id in sorted(energy_ids):
        energy_df = observed[observed["metric_id"].astype(str) == energy_id]
        if energy_df.empty:
            continue
        power_df = derive_interval_average_power(energy_df, energy_metric_id=energy_id)
        if not power_df.empty:
            frames.append(power_df)

    if not frames:
        return pd.DataFrame(columns=observed.columns)

    out = pd.concat(frames, ignore_index=True)
    return ensure_point_metadata(mark_as_derived(out))


def synthesize_derived_metrics(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Append attributed energy totals and derived power series to a processed frame."""
    if df_processed.empty:
        return df_processed.copy()

    observed = normalize_observed_rows(df_processed)
    parts = [observed]

    attributed = synthesize_attributed_energy_total(observed)
    if not attributed.empty:
        parts.append(attributed)

    combined = pd.concat(parts, ignore_index=True)
    power = synthesize_derived_power(combined)
    if not power.empty:
        combined = pd.concat([combined, power], ignore_index=True)
    # Rebuild CounterDiff pairs once at the schema boundary. This also assigns
    # globally unique sample ids across original, synthesized, and power rows.
    return expand_counterdiff_rows(combined, already_normalized=True)
