"""Derived metric synthesis — creates new rows that don't exist in the raw CSV."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from backend.counterdiff import (
    _counterdiff_mask,
    derive_interval_average_power,
    ensure_point_metadata,
    expand_counterdiff_rows,
    normalize_observed_rows,
    observed_only,
)
from backend.metrics import (
    MetricId,
    classification_stem,
    energy_ids_for_derived_power,
    is_running_total_base_metric,
    mark_as_derived,
    running_total_base_metric,
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
    metric_ids = out["metric_id"].astype(str)
    parsed = {metric_id: MetricId.parse(metric_id) for metric_id in metric_ids.unique()}
    consumer_by_id = {
        metric_id: identity.consumer if identity.is_process_consumer else None
        for metric_id, identity in parsed.items()
    }
    late_by_id = {
        metric_id: identity.late_attributes if identity.is_process_consumer else None
        for metric_id, identity in parsed.items()
    }
    out["consumer"] = metric_ids.map(consumer_by_id)
    out["late_attributes"] = metric_ids.map(late_by_id)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out.dropna(subset=["consumer", "late_attributes", "timestamp"])


def _stem_series(base_metrics: pd.Series) -> pd.Series:
    """Map classification stems, computing each unique base metric once."""
    unique = base_metrics.dropna().astype(str).unique()
    mapping = {name: classification_stem(name) for name in unique}
    return base_metrics.map(mapping)


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


def _datetime_ns(timestamps) -> np.ndarray:
    if isinstance(timestamps, pd.DatetimeIndex):
        idx = timestamps
    elif isinstance(timestamps, pd.Series) and pd.api.types.is_datetime64_any_dtype(timestamps.dtype):
        idx = pd.DatetimeIndex(timestamps)
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.as_unit("ns").asi8


def _union_datetime_index(series_list: list) -> pd.DatetimeIndex:
    """Unique sorted timestamps, preserving the timezone of the first series."""
    ns_parts: list[np.ndarray] = []
    tz = None
    for values in series_list:
        if values is None or len(values) == 0:
            continue
        if isinstance(values, pd.DatetimeIndex):
            idx = values
        elif isinstance(values, pd.Series) and pd.api.types.is_datetime64_any_dtype(values.dtype):
            idx = pd.DatetimeIndex(values)
        else:
            idx = pd.DatetimeIndex(pd.to_datetime(values))
        if tz is None:
            tz = idx.tz
        ns_parts.append(_datetime_ns(idx))
    if not ns_parts:
        return pd.DatetimeIndex([])
    ns = np.unique(np.concatenate(ns_parts))
    out = pd.DatetimeIndex(ns.astype("datetime64[ns]"))
    if tz is not None:
        out = out.tz_localize("UTC").tz_convert(tz)
    return out


def _running_total_on_timeline(
    df_metric: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    value_name: str,
) -> pd.Series:
    """
    Forward-fill a CounterDiff series' running total onto a shared timeline.

    Each original sample is counted once: cumsum on the series' own timestamps,
    then hold until the next sample. Before the first sample the total is 0.
    After the last sample the total holds.
    """
    if df_metric.empty:
        return pd.Series(0.0, index=timeline, name=value_name, dtype="float64")

    if df_metric["timestamp"].duplicated().any():
        raise ValueError(
            "CounterDiff alignment requires unique aggregated timestamps; "
            f"found duplicates: {df_metric.loc[df_metric['timestamp'].duplicated(), 'timestamp'].tolist()}"
        )

    ordered = df_metric.sort_values("timestamp", kind="mergesort")
    running = ordered["value"].to_numpy(dtype="float64").cumsum()
    src_ns = _datetime_ns(ordered["timestamp"])
    tgt_ns = _datetime_ns(timeline)
    last_src = np.searchsorted(src_ns, tgt_ns, side="right") - 1
    aligned = np.where(last_src >= 0, running[np.maximum(last_src, 0)], 0.0)
    return pd.Series(aligned, index=timeline, name=value_name, dtype="float64")


def _interval_deltas_from_running_total(running: pd.Series) -> pd.Series:
    """Recover CounterDiff interval deltas from a running-total series."""
    if running.empty:
        return running.copy()
    deltas = running.diff()
    deltas.iloc[0] = running.iloc[0]
    return deltas


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
    package_mask = late.str.contains(PACKAGE_TOTAL_REGEX.pattern, regex=True, na=False)
    package_total = df_cpu.loc[package_mask]
    if not package_total.empty:
        return package_total.copy()

    unique_late = sorted(set(late.tolist()))
    if len(unique_late) == 1:
        return df_cpu.copy()

    kind_mask = late.str.contains(KIND_TOTAL_REGEX.pattern, regex=True, na=False)
    kind_total = df_cpu.loc[kind_mask]
    if not kind_total.empty:
        return kind_total.copy()

    return df_cpu.iloc[0:0].copy()


def _union_timeline(frames: list[pd.DataFrame]) -> pd.DatetimeIndex:
    return _union_datetime_index(
        [frame["timestamp"] for frame in frames if frame is not None and not frame.empty]
    )


def _aligned_counterdiff(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine CounterDiff energy series without double-counting joules.

    Accumulate each series on its own timestamps, forward-fill those running
    totals onto the union timeline, add them, then difference so the result
    is still interval-delta CounterDiff. A device contributes new joules only
    at its own samples.
    """
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=["timestamp", "value"])

    timeline = _union_timeline(nonempty)
    if timeline.empty:
        return pd.DataFrame(columns=["timestamp", "value"])

    total_running = pd.Series(0.0, index=timeline, dtype="float64")
    for frame in nonempty:
        total_running = total_running + _running_total_on_timeline(frame, timeline, "running")
    return pd.DataFrame(
        {
            "timestamp": timeline,
            "value": _interval_deltas_from_running_total(total_running).to_numpy(),
        }
    )


def _build_cpu_gpu_total_rows(
    cpu_pid: pd.DataFrame,
    gpu_pid: pd.DataFrame,
    *,
    consumer: str,
) -> pd.DataFrame:
    """Accumulate CPU/GPU energy, forward-fill running totals, emit interval-delta totals."""
    total_pid = _aligned_counterdiff([cpu_pid, gpu_pid])
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

    CPU+GPU and multi-GPU totals accumulate each series first, forward-fill
    those running totals onto the union of their clocks, add, and difference.
    Per-device power is E/dt on that device's own samples; total power is the
    sum of those stairs, not E_total/dt on the union.
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

    stems = _stem_series(observed["base_metric"])
    cpu_mask = stems.eq("attributed_energy_cpu")
    gpu_mask = stems.eq("attributed_energy_gpu")

    df_cpu = observed.loc[cpu_mask, ["metric_id", "timestamp", "value"]].copy()
    df_gpu = observed.loc[gpu_mask, ["metric_id", "timestamp", "value"]].copy()

    if df_cpu.empty and df_gpu.empty:
        return pd.DataFrame(columns=observed.columns)

    df_cpu = _attach_process_identity(df_cpu)
    df_gpu = _attach_process_identity(df_gpu)

    if df_cpu.empty and df_gpu.empty:
        return pd.DataFrame(columns=observed.columns)

    df_cpu = _select_attributed_cpu_rows(df_cpu)
    df_cpu_summed = _sum_observed_by_timestamp_consumer(df_cpu)

    gpu_frames: dict[str, list[pd.DataFrame]] = {}
    if not df_gpu.empty:
        for (consumer, _metric_id), group in df_gpu.groupby(["consumer", "metric_id"], dropna=False, sort=False):
            if pd.isna(consumer):
                continue
            summed = group.groupby("timestamp", as_index=False, sort=False)["value"].sum()
            gpu_frames.setdefault(str(consumer), []).append(summed[["timestamp", "value"]])

    synthetic_parts: list[pd.DataFrame] = []
    gpu_totals: dict[str, pd.DataFrame] = {}

    for consumer, frames in sorted(gpu_frames.items()):
        gpu_pid = _aligned_counterdiff(frames)
        if gpu_pid.empty:
            continue
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
        gpu_totals[consumer] = gpu_pid[["timestamp", "value"]].copy()

    cpu_consumers = (
        {str(consumer) for consumer in df_cpu_summed["consumer"].unique()} if not df_cpu_summed.empty else set()
    )
    for consumer in sorted(cpu_consumers & set(gpu_totals)):
        total_pid = _build_cpu_gpu_total_rows(
            df_cpu_summed.loc[df_cpu_summed["consumer"].astype(str) == consumer, ["timestamp", "value"]],
            gpu_totals[consumer],
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
    per-device attributed power always; derives NVML/AMD/Grace energy power only
    when a matching measured power series is absent. Synthesized energy totals
    are not converted with `E/Δt` here.
    """
    observed = observed_only(df_processed)
    if observed.empty or "metric_id" not in observed.columns:
        return pd.DataFrame(columns=list(df_processed.columns))

    available_ids = set(observed["metric_id"].astype(str).unique())
    energy_ids = energy_ids_for_derived_power(available_ids)
    if not energy_ids:
        return pd.DataFrame(columns=observed.columns)

    energy_df = observed[observed["metric_id"].astype(str).isin(energy_ids)]
    power_df = derive_interval_average_power(energy_df)
    if power_df.empty:
        return pd.DataFrame(columns=observed.columns)
    return ensure_point_metadata(mark_as_derived(power_df))


def _step_power_at(power: pd.DataFrame, target_ns: np.ndarray) -> np.ndarray:
    """Evaluate a power staircase on (interval_start, timestamp]."""
    out = np.zeros(len(target_ns), dtype="float64")
    if power.empty or "interval_start" not in power.columns or len(target_ns) == 0:
        return out
    starts = _datetime_ns(power["interval_start"])
    ends = _datetime_ns(power["timestamp"])
    vals = power["value"].to_numpy(dtype="float64")
    order = np.argsort(ends, kind="mergesort")
    starts = starts[order]
    ends = ends[order]
    vals = vals[order]
    idx = np.searchsorted(ends, target_ns, side="left")
    valid = idx < len(ends)
    if not np.any(valid):
        return out
    chosen = idx[valid]
    hit = (target_ns[valid] > starts[chosen]) & (target_ns[valid] <= ends[chosen])
    dest = np.flatnonzero(valid)[hit]
    out[dest] = vals[chosen[hit]]
    return out


def _aligned_power_sum(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum power stairs onto the union of their interval boundaries."""
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=["timestamp", "value", "interval_start"])

    timeline = _union_datetime_index(
        [frame["timestamp"] for frame in nonempty] + [frame["interval_start"] for frame in nonempty]
    )
    if len(timeline) < 2:
        return pd.DataFrame(columns=["timestamp", "value", "interval_start"])

    target_ns = _datetime_ns(timeline)
    total = np.zeros(len(timeline), dtype="float64")
    for frame in nonempty:
        total = total + _step_power_at(frame, target_ns)
    return pd.DataFrame(
        {
            "timestamp": timeline[1:],
            "value": total[1:],
            "interval_start": timeline[:-1],
        }
    )


def _stamp_attributed_power_rows(
    frame: pd.DataFrame,
    *,
    base_metric: str,
    resource: str,
    resource_kind: str,
    resource_id: str,
    consumer: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    consumer_kind, consumer_id = _split_kind_id(consumer, "process")
    out["metric_id"] = MetricId(
        base_metric=base_metric,
        resource=resource,
        consumer=consumer,
        late_attributes="",
    ).serialized
    out["base_metric"] = base_metric
    out["metric"] = base_metric
    out["resource_kind"] = resource_kind
    out["resource_id"] = resource_id
    out["consumer_kind"] = consumer_kind
    out["consumer_id"] = consumer_id
    out["__late_attributes"] = ""
    return out


def synthesize_attributed_power_total(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize process-scoped attributed power totals from component stairs.

    `attributed_power_gpu_total_W` and `attributed_power_total_W` are sums of
    per-device interval-average power, evaluated on the union of those devices'
    own averaging windows. They are not `E_total / Δt` on the energy-union
    timestamp since the divisor is skewed, not a measurement interval.
    """
    observed = observed_only(df_processed)
    if observed.empty or "base_metric" not in observed.columns:
        return pd.DataFrame(columns=list(df_processed.columns) if len(df_processed.columns) else [])

    stems = _stem_series(observed["base_metric"])
    keep = ["metric_id", "timestamp", "value", "interval_start"]
    keep = [column for column in keep if column in observed.columns]
    df_cpu = observed.loc[stems.eq("attributed_power_cpu"), keep].copy()
    df_gpu = observed.loc[stems.eq("attributed_power_gpu"), keep].copy()
    if df_cpu.empty and df_gpu.empty:
        return pd.DataFrame(columns=observed.columns)
    if "interval_start" not in keep:
        return pd.DataFrame(columns=observed.columns)

    df_cpu = _attach_process_identity(df_cpu) if not df_cpu.empty else df_cpu
    df_gpu = _attach_process_identity(df_gpu) if not df_gpu.empty else df_gpu
    if not df_cpu.empty:
        df_cpu = _select_attributed_cpu_rows(df_cpu)

    cpu_energy = observed.loc[stems.eq("attributed_energy_cpu"), ["metric_id", "timestamp", "value"]].copy()
    gpu_energy = observed.loc[stems.eq("attributed_energy_gpu"), ["metric_id", "timestamp", "value"]].copy()
    if not cpu_energy.empty:
        cpu_energy = _select_attributed_cpu_rows(_attach_process_identity(cpu_energy))
    if not gpu_energy.empty:
        gpu_energy = _attach_process_identity(gpu_energy)
    process_consumers = (
        set(cpu_energy["consumer"].astype(str)) & set(gpu_energy["consumer"].astype(str))
        if not cpu_energy.empty and not gpu_energy.empty
        else set()
    )

    gpu_frames: dict[str, list[pd.DataFrame]] = {}
    if not df_gpu.empty:
        for (consumer, _metric_id), group in df_gpu.groupby(["consumer", "metric_id"], dropna=False, sort=False):
            if pd.isna(consumer):
                continue
            gpu_frames.setdefault(str(consumer), []).append(
                group[["timestamp", "value", "interval_start"]]
            )

    cpu_frames: dict[str, list[pd.DataFrame]] = {}
    if not df_cpu.empty:
        for (consumer, _metric_id), group in df_cpu.groupby(["consumer", "metric_id"], dropna=False, sort=False):
            if pd.isna(consumer):
                continue
            cpu_frames.setdefault(str(consumer), []).append(
                group[["timestamp", "value", "interval_start"]]
            )

    synthetic_parts: list[pd.DataFrame] = []
    for consumer, frames in sorted(gpu_frames.items()):
        gpu_pid = _aligned_power_sum(frames)
        if gpu_pid.empty:
            continue
        synthetic_parts.append(
            _stamp_attributed_power_rows(
                gpu_pid,
                base_metric="attributed_power_gpu_total_W",
                resource="gpu_all_",
                resource_kind="gpu",
                resource_id="all",
                consumer=consumer,
            )
        )

    for consumer in sorted(process_consumers):
        total_pid = _aligned_power_sum([*cpu_frames.get(consumer, []), *gpu_frames.get(consumer, [])])
        if total_pid.empty:
            continue
        synthetic_parts.append(
            _stamp_attributed_power_rows(
                total_pid,
                base_metric="attributed_power_total_W",
                resource="total_",
                resource_kind="total",
                resource_id="",
                consumer=consumer,
            )
        )

    if not synthetic_parts:
        return pd.DataFrame(columns=observed.columns)
    return ensure_point_metadata(mark_as_derived(pd.concat(synthetic_parts, ignore_index=True)))


def _running_total_id_maps(metric_ids: pd.Series, base_metrics: pd.Series) -> tuple[dict[str, str], dict[str, str]]:
    """Map each CounterDiff id to its running-total sibling id and base name."""
    unique = pd.DataFrame({"metric_id": metric_ids, "base_metric": base_metrics}).drop_duplicates()
    new_base_by_old: dict[str, str] = {}
    new_id_by_old: dict[str, str] = {}
    for old_id, old_base in zip(unique["metric_id"].astype(str), unique["base_metric"].astype(str)):
        new_base = running_total_base_metric(old_base)
        new_base_by_old[old_id] = new_base
        new_id_by_old[old_id] = MetricId.parse(old_id).with_base_metric(new_base).serialized
    return new_id_by_old, new_base_by_old


def synthesize_running_totals(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Per-series running totals of CounterDiff interval deltas.

    Vectorized `groupby(metric_id).cumsum` so preprocessing compute does not walk
    every series with a Python filter. Each observed sample is counted
    once. The result is a gauge (not CounterDiff), so it is not expanded
    into spike pairs.
    """
    observed = observed_only(df_processed)
    if observed.empty or "metric_id" not in observed.columns:
        return pd.DataFrame(columns=list(df_processed.columns))

    keep = _counterdiff_mask(observed["metric_id"])
    origin_col = "base_metric" if "base_metric" in observed.columns else "metric_id"
    already_map = {
        name: is_running_total_base_metric(str(name)) for name in observed[origin_col].dropna().unique()
    }
    already = observed[origin_col].map(already_map).fillna(False)
    subset = observed.loc[keep & ~already]
    if subset.empty:
        return pd.DataFrame(columns=observed.columns)

    subset = subset.copy()
    dup_mask = subset.duplicated(subset=["metric_id", "timestamp"], keep=False)
    if dup_mask.any():
        dups = subset.loc[dup_mask, ["metric_id", "timestamp"]]
        raise ValueError(
            "Duplicate observed timestamps for running-total synthesis; "
            "refusing silent drop_duplicates. "
            f"timestamps={dups.groupby(dups['metric_id'].astype(str))['timestamp'].apply(list).to_dict()}"
        )

    if "base_metric" not in subset.columns:
        subset["base_metric"] = subset["metric_id"].map(lambda metric_id: MetricId.parse(metric_id).base_metric)

    subset = subset.sort_values(["metric_id", "timestamp"], kind="mergesort")
    subset["value"] = subset.groupby("metric_id", sort=False)["value"].cumsum()

    new_id_by_old, new_base_by_old = _running_total_id_maps(subset["metric_id"], subset["base_metric"])
    subset["base_metric"] = subset["metric_id"].map(new_base_by_old)
    if "metric" in subset.columns:
        subset["metric"] = subset["base_metric"]
    subset["metric_id"] = subset["metric_id"].map(new_id_by_old)
    if "interval_start" in subset.columns:
        subset = subset.drop(columns=["interval_start"])

    return ensure_point_metadata(mark_as_derived(subset.reset_index(drop=True)))


def synthesize_derived_metrics(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Append attributed energy totals, derived power, and running totals.

    Energy totals (cumsum_ffill): cumsum each device, ffill C onto the union, add, diff.
    Per-device power: E/t on that device's own poll interval (skip the first sample).
    Total power: sum of those stairs, not E_total/t on the union timeline.
    """
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
    attributed_power = synthesize_attributed_power_total(combined)
    if not attributed_power.empty:
        combined = pd.concat([combined, attributed_power], ignore_index=True)
    running = synthesize_running_totals(combined)
    if not running.empty:
        combined = pd.concat([combined, running], ignore_index=True)
    # Rebuild CounterDiff pairs once at the schema boundary. This also assigns
    # globally unique sample ids across original, synthesized, and power rows.
    # Running-total gauges are not CounterDiff and stay as single observed rows.
    return expand_counterdiff_rows(combined, already_normalized=True)
