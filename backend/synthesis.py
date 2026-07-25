"""Derived metric synthesis — creates new rows that don't exist in the raw CSV."""

from __future__ import annotations

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


def _sum_observed_by_timestamp_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Sum observed resources without collapsing consumer/attribution identity."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "consumer", "late_attributes", "value"])

    return df.groupby(
        ["timestamp", "consumer", "late_attributes"],
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


def synthesize_attributed_energy_total(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize attributed_energy_total_J metric from attributed_energy_cpu and attributed_energy_gpu metrics.

    Creates two synthetic metrics:
    1. attributed_energy_gpu_total_J: sum of attributed GPU energy across all GPUs per process (pid).
    2. attributed_energy_total_J: CPU + GPU total per process on the union of timestamps.
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

    df_gpu_summed = _sum_observed_by_timestamp_identity(df_gpu)
    df_cpu_summed = _sum_observed_by_timestamp_identity(df_cpu)

    synthetic_parts: list[pd.DataFrame] = []

    identity_columns = ["consumer", "late_attributes"]
    gpu_identities = df_gpu_summed[identity_columns].drop_duplicates()
    for consumer, late_attributes in gpu_identities.itertuples(
        index=False,
        name=None,
    ):
        identity_mask = (df_gpu_summed["consumer"] == consumer) & (df_gpu_summed["late_attributes"] == late_attributes)
        gpu_pid = df_gpu_summed.loc[
            identity_mask,
            ["timestamp", "value"],
        ].copy()
        consumer_kind, consumer_id = _split_kind_id(consumer, "process")
        gpu_pid["metric_id"] = MetricId(
            base_metric="attributed_energy_gpu_total_J",
            resource="gpu_all_",
            consumer=consumer,
            late_attributes=late_attributes,
        ).serialized
        gpu_pid["base_metric"] = "attributed_energy_gpu_total_J"
        gpu_pid["metric"] = "attributed_energy_gpu_total_J"
        gpu_pid["resource_kind"] = "gpu"
        gpu_pid["resource_id"] = "all"
        gpu_pid["consumer_kind"] = consumer_kind
        gpu_pid["consumer_id"] = consumer_id
        gpu_pid["__late_attributes"] = late_attributes
        synthetic_parts.append(gpu_pid)

    cpu_identities = set(df_cpu_summed[identity_columns].itertuples(index=False, name=None))
    gpu_identity_set = set(df_gpu_summed[identity_columns].itertuples(index=False, name=None))
    for consumer, late_attributes in sorted(cpu_identities & gpu_identity_set):
        cpu_mask = (df_cpu_summed["consumer"] == consumer) & (df_cpu_summed["late_attributes"] == late_attributes)
        gpu_mask = (df_gpu_summed["consumer"] == consumer) & (df_gpu_summed["late_attributes"] == late_attributes)
        cpu_pid = df_cpu_summed.loc[cpu_mask, ["timestamp", "value"]].copy()
        gpu_pid = df_gpu_summed.loc[gpu_mask, ["timestamp", "value"]].copy()
        if cpu_pid.empty or gpu_pid.empty:
            continue

        timeline = pd.DatetimeIndex(pd.Index(cpu_pid["timestamp"]).union(pd.Index(gpu_pid["timestamp"])).sort_values())
        if timeline.empty:
            continue

        cpu_aligned = _align_counterdiff_energy_to_timeline(cpu_pid, timeline, "cpu_value")
        gpu_aligned = _align_counterdiff_energy_to_timeline(gpu_pid, timeline, "gpu_value")

        total_pid = pd.DataFrame(
            {
                "timestamp": timeline,
                "cpu_value": cpu_aligned.to_numpy(),
                "gpu_value": gpu_aligned.to_numpy(),
            }
        )
        total_pid.dropna(subset=["cpu_value", "gpu_value"], inplace=True)
        if total_pid.empty:
            continue
        total_pid["value"] = total_pid["cpu_value"] + total_pid["gpu_value"]
        consumer_kind, consumer_id = _split_kind_id(consumer, "process")
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
        synthetic_parts.append(total_pid.drop(columns=["cpu_value", "gpu_value"]))

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
    energy = synthesize_attributed_energy_total(observed)
    if not energy.empty:
        parts.append(energy)

    combined = pd.concat(parts, ignore_index=True)
    power = synthesize_derived_power(combined)
    if not power.empty:
        combined = pd.concat([combined, power], ignore_index=True)
    # Rebuild CounterDiff pairs once at the schema boundary. This also assigns
    # globally unique sample ids across original, synthesized, and power rows.
    return expand_counterdiff_rows(combined, already_normalized=True)
