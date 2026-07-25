"""
CounterDiff processing for Alumet interval-delta metrics.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Iterable

import numpy as np
import pandas as pd

from backend.metrics import (
    MetricId,
    MetricOrigin,
    derived_power_base_metric,
    is_counterdiff_metric,
)


# ---------------------------------------------------------------------------
# 1. Classes and constants for CounterDiff
# ---------------------------------------------------------------------------

class PointRole(StrEnum):
    """Whether a row is a observed Alumet sample or synthetic zero-padding for CounterDiff."""

    OBSERVED = "observed"
    SYNTHETIC = "synthetic"


class PointOrder(IntEnum):
    """
    Ordering for rows that share one physical timestamp..
    Observed comes first (the spike peak), then the synthetic zero (return to baseline).
    """

    OBSERVED = 0
    POST_SAMPLE_ZERO = 1

# Columns added by CounterDiff zero-padding.
_POINT_METADATA_COLUMNS = ("point_role", "point_order", "sample_id")
# Columns every non-empty processed dataframe must contain.
_REQUIRED_PROCESSED_COLUMNS = ("metric_id", "base_metric", "timestamp", "value")

POINT_ORDER_BY_ROLE = {
    PointRole.OBSERVED: PointOrder.OBSERVED,
    PointRole.SYNTHETIC: PointOrder.POST_SAMPLE_ZERO,
}


# ---------------------------------------------------------------------------
# 2. Helper functions for CounterDiff
# ---------------------------------------------------------------------------

def _counterdiff_mask(metric_ids: pd.Series) -> pd.Series:
    """
    Return a per-row True/False mask that answers whether this row is a CounterDiff metric.
    """
    # Collects the unique metric ids and classifies each unique id once
    classifications = {metric_id: is_counterdiff_metric(str(metric_id)) for metric_id in metric_ids.dropna().unique()}
    # Maps that answer back onto every row to avoid calling `is_counterdiff_metric` on every row. Missing/NaN metric ids become False.
    return metric_ids.map(classifications).fillna(False).astype(bool)


def ensure_point_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing CounterDiff bookkeeping columns with safe defaults to ensure `point_role`, `point_order`, and `sample_id` columns exist.

    If a column is absent, fill it as an observed row (`point_role` = `PointRole.OBSERVED`, `point_order` = `PointOrder.OBSERVED`, `sample_id` = `NA`).
    Existing values are kept; nulls are filled with those same defaults.
    Returns a copy of the dataframe.
    """
    if df.empty:
        out = df.copy()
        for col in _POINT_METADATA_COLUMNS:
            if col not in out.columns:
                dtype = "string" if col == "point_role" else "Int64"
                out[col] = pd.Series(dtype=dtype)
        return out

    out = df.copy()
    if "point_role" not in out.columns:
        out["point_role"] = PointRole.OBSERVED.value
    else:
        out["point_role"] = out["point_role"].fillna(PointRole.OBSERVED.value).astype("string")

    if "point_order" not in out.columns:
        out["point_order"] = int(PointOrder.OBSERVED)
    else:
        out["point_order"] = out["point_order"].fillna(int(PointOrder.OBSERVED)).astype("Int64")

    if "sample_id" not in out.columns:
        out["sample_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["sample_id"] = pd.to_numeric(out["sample_id"], errors="coerce").astype("Int64")
    return out


def require_processed_columns(df: pd.DataFrame) -> None:
    """
    Raise `ValueError` if `metric_id`, `base_metric`, `timestamp`, and `value` columns are missing.
    """
    missing = [column for column in _REQUIRED_PROCESSED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Processed dataframe missing required columns: {missing}")


def observed_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only observed rows; drop synthetic zero-padding.
    If `point_role` is missing, treat every row as observed and return a copy of the whole frame.
    """
    if df.empty:
        return df.copy()
    if "point_role" not in df.columns:
        return df.copy()
    return df[df["point_role"].astype(str) == PointRole.OBSERVED.value].copy()


# ---------------------------------------------------------------------------
# 3. Sanity checks after CounterDiff zero-padding
# ---------------------------------------------------------------------------

def validate_point_metadata(df: pd.DataFrame) -> None:
    """
    Raise if a CounterDiff zero-padded dataframe breaks bookkeeping rules.

    Checks:
    - `point_role` / `point_order` / `sample_id` exist and are not null
    - `point_order` matches `point_role`
    - synthetic rows always have value 0
    - each CounterDiff `(metric_id, sample_id)` is exactly one observed + one synthetic pair at a single timestamp
    - observed CounterDiff timestamps are unique per series
    """
    missing = [column for column in _POINT_METADATA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing point metadata columns: {missing}")
    if df.empty:
        return
    if df[list(_POINT_METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Point metadata columns must not contain null values")

    valid_roles = {role.value for role in PointRole}
    invalid_roles = set(df["point_role"].astype(str)) - valid_roles
    if invalid_roles:
        raise ValueError(f"Unknown point_role values: {sorted(invalid_roles)}")

    expected_orders = {role.value: int(order) for role, order in POINT_ORDER_BY_ROLE.items()}
    actual_orders = df["point_order"].astype(int)
    mismatched = actual_orders != df["point_role"].astype(str).map(expected_orders)
    if mismatched.any():
        raise ValueError(f"point_order does not match point_role for rows {df.index[mismatched].tolist()}")

    synthetic = df["point_role"].astype(str) == PointRole.SYNTHETIC.value
    if synthetic.any() and not np.allclose(
        pd.to_numeric(df.loc[synthetic, "value"], errors="coerce"), 0.0, equal_nan=False
    ):
        raise ValueError("Synthetic CounterDiff padding rows must have value 0")

    observed = df["point_role"].astype(str) == PointRole.OBSERVED.value
    counterdiff = _counterdiff_mask(df["metric_id"])
    duplicate_observed = df.loc[observed & counterdiff].duplicated(subset=["metric_id", "timestamp"], keep=False)
    if duplicate_observed.any():
        rows = df.loc[observed & counterdiff].loc[duplicate_observed, ["metric_id", "timestamp"]]
        raise ValueError(f"Duplicate observed timestamps in CounterDiff series: {rows.to_dict('records')}")
    duplicate_sample_ids = df.loc[observed, "sample_id"].duplicated(keep=False)
    if duplicate_sample_ids.any():
        ids = df.loc[observed, "sample_id"].loc[duplicate_sample_ids].astype(int).tolist()
        raise ValueError(f"Observed sample_id values must be globally unique: {ids}")

    counterdiff_rows = df.loc[counterdiff]
    if not counterdiff_rows.empty:
        pair_key = ["metric_id", "sample_id"]
        pair_sizes = counterdiff_rows.groupby(pair_key, sort=False, observed=True, dropna=False).size()
        duplicate_roles = counterdiff_rows.duplicated(subset=[*pair_key, "point_role"], keep=False)
        invalid_sizes = pair_sizes[pair_sizes != 2]
        if not invalid_sizes.empty or duplicate_roles.any():
            invalid_key = (
                invalid_sizes.index[0]
                if not invalid_sizes.empty
                else tuple(
                    counterdiff_rows.loc[
                        duplicate_roles,
                        pair_key,
                    ].iloc[0]
                )
            )
            raise ValueError(f"CounterDiff sample {invalid_key} is not an observed/synthetic pair")

        timestamp_counts = counterdiff_rows.groupby(pair_key, sort=False, observed=True, dropna=False)[
            "timestamp"
        ].nunique(dropna=False)
        invalid_timestamps = timestamp_counts[timestamp_counts != 1]
        if not invalid_timestamps.empty:
            raise ValueError(f"CounterDiff sample {invalid_timestamps.index[0]} spans timestamps")


# ---------------------------------------------------------------------------
# 4. Core engine: clean observations, then zero-pad CounterDiff pairs
#
# For CounterDiff:
#   normalize_observed_rows  → unique observed samples
#   expand_counterdiff_rows  → add synthetic zeros for CounterDiff metrics
# For Gauge / raw-counter rows stay as a single observed row.
# ---------------------------------------------------------------------------

def _normalize_duplicate_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve duplicate rows that share the same metric id and timestamp.

    Policy:
    - Exact duplicates (same semantic columns): keep one copy for any metric.
    - CounterDiff with one non-zero and one zero at the same time: keep the non-zero.
    - CounterDiff with two different non-zero values: raise ValueError
    - Gauge / non-CounterDiff metrics may keep multiple values at one time
    """
    series_key = ["metric_id", "timestamp"]
    duplicate_keys = df.duplicated(subset=series_key, keep=False)
    if not duplicate_keys.any():
        return df

    semantic_columns = [column for column in df.columns if column not in _POINT_METADATA_COLUMNS]
    deduplicated = df.drop_duplicates(subset=semantic_columns, keep="first")
    conflicting = deduplicated.duplicated(subset=series_key, keep=False)
    if not conflicting.any():
        return deduplicated

    counterdiff_rows = _counterdiff_mask(deduplicated["metric_id"])
    counterdiff_conflicts = conflicting & counterdiff_rows
    if counterdiff_conflicts.any() and "value" in deduplicated.columns:
        conflict_rows = deduplicated.loc[counterdiff_conflicts, [*series_key, "value"]].copy()
        numeric_values = pd.to_numeric(conflict_rows["value"], errors="coerce")
        conflict_rows["_zero"] = numeric_values.eq(0.0)
        conflict_rows["_nonzero"] = numeric_values.notna() & ~conflict_rows["_zero"]
        conflict_rows["_valid"] = numeric_values.notna()

        group_stats = conflict_rows.groupby(series_key, sort=False, observed=True, dropna=False)[
            ["_zero", "_nonzero", "_valid"]
        ].sum()
        group_sizes = conflict_rows.groupby(series_key, sort=False, observed=True, dropna=False).size()
        recoverable = group_stats["_zero"].ge(1) & group_stats["_nonzero"].eq(1) & group_stats["_valid"].eq(group_sizes)
        recoverable_keys = group_stats.index[recoverable]
        if len(recoverable_keys):
            row_keys = pd.MultiIndex.from_frame(deduplicated[series_key])
            all_values = pd.to_numeric(deduplicated["value"], errors="coerce")
            legacy_padding = row_keys.isin(recoverable_keys) & all_values.eq(0.0)
            deduplicated = deduplicated.loc[~legacy_padding]

    remaining_counterdiff_conflicts = deduplicated.duplicated(subset=series_key, keep=False) & _counterdiff_mask(
        deduplicated["metric_id"]
    )
    if remaining_counterdiff_conflicts.any():
        detail_columns = [column for column in ("metric_id", "timestamp", "value") if column in deduplicated.columns]
        conflicts = deduplicated.loc[remaining_counterdiff_conflicts, detail_columns]
        preview = conflicts.head(20).to_dict("records")
        remainder = len(conflicts) - len(preview)
        suffix = f" (and {remainder} more rows)" if remainder else ""
        raise ValueError(
            "CounterDiff expansion found conflicting observed values for the same "
            f"series and timestamp: {preview}{suffix}"
        )
    return deduplicated


def normalize_observed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a clean observed-only dataframe ready for CounterDiff zero-padding."""
    if df.empty:
        return ensure_point_metadata(df)

    require_processed_columns(df)
    base = ensure_point_metadata(df)

    invalid_roles = set(base["point_role"].astype(str)) - {role.value for role in PointRole}
    if invalid_roles:
        raise ValueError(f"Unknown point_role values: {sorted(invalid_roles)}")
    return _normalize_duplicate_observations(observed_only(base))


def _expand_normalized_counterdiff_rows(base: pd.DataFrame) -> pd.DataFrame:
    """
    Turn each CounterDiff observation into (value, zero) at each measured timestamp.
    Caller must pass an already-normalized observed dataframe.
    """
    if base.empty:
        raise ValueError("Cannot expand a dataframe containing only synthetic rows")

    sort_cols = [c for c in ("metric_id", "timestamp", "point_order") if c in base.columns]
    base = base.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # `sample_id` is globally unique within a processed dataframe. A CounterDiff observation and its synthetic zero share the same id.
    base["sample_id"] = pd.array(
        np.arange(len(base), dtype=np.int64),
        dtype="Int64",
    )

    counterdiff_mask = _counterdiff_mask(base["metric_id"])
    gauge_rows = base.loc[~counterdiff_mask].copy()
    counterdiff_rows = base.loc[counterdiff_mask].copy()

    if gauge_rows.empty and counterdiff_rows.empty:
        return base

    if not gauge_rows.empty:
        gauge_rows["point_role"] = PointRole.OBSERVED.value
        gauge_rows["point_order"] = int(PointOrder.OBSERVED)

    if counterdiff_rows.empty:
        result = gauge_rows.reset_index(drop=True)
        validate_point_metadata(result)
        return result

    counterdiff_rows = counterdiff_rows.reset_index(drop=True)

    observed = counterdiff_rows.copy()
    observed["point_role"] = PointRole.OBSERVED.value
    observed["point_order"] = int(PointOrder.OBSERVED)

    synthetic = counterdiff_rows.copy()
    synthetic["point_role"] = PointRole.SYNTHETIC.value
    synthetic["point_order"] = int(PointOrder.POST_SAMPLE_ZERO)
    synthetic["value"] = 0.0

    expanded = pd.concat([observed, synthetic], ignore_index=True)
    if not gauge_rows.empty:
        expanded = pd.concat([expanded, gauge_rows], ignore_index=True)

    sort_cols = [c for c in ("metric_id", "timestamp", "point_order") if c in expanded.columns]
    result = expanded.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    result["point_order"] = result["point_order"].astype("Int64")
    result["sample_id"] = result["sample_id"].astype("Int64")
    validate_point_metadata(result)
    return result


def expand_counterdiff_rows(
    df: pd.DataFrame,
    *,
    already_normalized: bool = False,
) -> pd.DataFrame:
    """
    Expand CounterDiff observations into ordered observed + synthetic-zero rows.
    Non-CounterDiff metrics stay as one observed row. Pass `already_normalized=True` when the caller already ran `normalize_observed_rows` and wants to skip that work.
    """
    if df.empty:
        return ensure_point_metadata(df)

    if already_normalized:
        require_processed_columns(df)
        base = ensure_point_metadata(observed_only(df) if "point_role" in df.columns else df)
    else:
        base = normalize_observed_rows(df)

    return _expand_normalized_counterdiff_rows(base)


# ---------------------------------------------------------------------------
# 5. Using zero-padded CounterDiff values: interpolate and derive power
#
# Interpolation reconstructs the counterdiff value between observed samples.
# Power turns interval energy (Joules) into average watts over each interval.
# ---------------------------------------------------------------------------

def interpolate_counterdiff_at_timeline(
    timestamps: pd.Series | Iterable,
    values: pd.Series | Iterable,
    target_timestamps: pd.Series | Iterable,
) -> pd.Series:
    """
    Estimate CounterDiff values on a different time grid.

    At each observed `t_i` the value is exactly `values[i]`. Right after sample `t_i` the quantity is back at zero, 
    then it accumulates again until `t_{i+1}`. So for a query time `t` where `t_i < t < t_{i+1}`:

        value(t) = values[i+1] * (t - t_i) / (t_{i+1} - t_i)

    Before the first sample the value is 0; after the last sample the value is NaN. 
    """
    src_ts = pd.to_datetime(pd.Series(list(timestamps)), utc=False)
    src_vals = pd.Series(list(values), dtype="float64")
    tgt_ts = pd.to_datetime(pd.Series(list(target_timestamps)), utc=False)

    if src_ts.empty:
        return pd.Series(np.nan, index=range(len(tgt_ts)), dtype="float64")

    order = np.argsort(src_ts.to_numpy(dtype="datetime64[ns]"), kind="mergesort")
    src_ts = src_ts.iloc[order].reset_index(drop=True)
    src_vals = src_vals.iloc[order].reset_index(drop=True)

    if src_ts.duplicated().any():
        raise ValueError(
            "CounterDiff interpolation requires unique source timestamps; "
            f"found duplicates: {src_ts[src_ts.duplicated()].tolist()}"
        )

    src_ns = src_ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    tgt_ns = tgt_ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    vals = src_vals.to_numpy(dtype="float64")

    out = np.full(len(tgt_ns), np.nan, dtype="float64")
    if len(src_ns) == 0:
        return pd.Series(out)

    first_ns = src_ns[0]
    last_ns = src_ns[-1]

    before = tgt_ns < first_ns
    out[before] = 0.0

    # Exact observation hits.
    exact_idx = np.searchsorted(src_ns, tgt_ns, side="left")
    exact_mask = (exact_idx < len(src_ns)) & (src_ns[np.minimum(exact_idx, len(src_ns) - 1)] == tgt_ns)
    out[exact_mask] = vals[exact_idx[exact_mask]]

    # Strict interior of intervals: after t_i, before t_{i+1}.
    interior = (~before) & (tgt_ns < last_ns) & (~exact_mask)
    if interior.any():
        # For t in (t_i, t_{i+1}), searchsorted(..., side="right") - 1 gives i.
        right = np.searchsorted(src_ns, tgt_ns[interior], side="right")
        left_i = right - 1
        right_i = right
        valid = (left_i >= 0) & (right_i < len(src_ns))
        interior_idx = np.flatnonzero(interior)[valid]
        left_i = left_i[valid]
        right_i = right_i[valid]
        dt = (src_ns[right_i] - src_ns[left_i]).astype("float64")
        frac = (tgt_ns[interior_idx] - src_ns[left_i]).astype("float64") / dt
        out[interior_idx] = vals[right_i] * frac

    # After last observation remains NaN (already initialized).
    return pd.Series(out, index=range(len(tgt_ts)), dtype="float64")


def derive_interval_average_power(
    energy_df: pd.DataFrame,
    *,
    energy_metric_id: str | None = None,
    power_base_metric: str | None = None,
) -> pd.DataFrame:
    """
    Derive interval-average power (W) from observed CounterDiff energy (J).

    For sample `i >= 1`:
        power_i = energy_i / (t_i - t_{i-1})
    """
    observed = observed_only(energy_df)
    if observed.empty:
        return observed.iloc[0:0].copy()

    if energy_metric_id is not None:
        observed = observed[observed["metric_id"].astype(str) == str(energy_metric_id)].copy()
    if observed.empty:
        return observed

    sort_cols = [c for c in ("metric_id", "timestamp") if c in observed.columns]
    observed = observed.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for metric_id, group in observed.groupby("metric_id", sort=False):
        group = group.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        if len(group) < 2:
            continue

        ts = pd.to_datetime(group["timestamp"])
        deltas = ts.diff().dt.total_seconds()
        if (deltas.iloc[1:] <= 0).any():
            bad = deltas.iloc[1:][deltas.iloc[1:] <= 0]
            raise ValueError(f"Non-positive timestamp interval while deriving power for {metric_id}: {bad.tolist()}")

        power_vals = group["value"].astype("float64") / deltas
        power_rows = group.iloc[1:].copy()
        power_rows["value"] = power_vals.iloc[1:].to_numpy()

        energy_base = (
            str(group["base_metric"].iloc[0])
            if "base_metric" in group.columns
            else MetricId.parse(str(metric_id)).base_metric
        )
        power_base = power_base_metric or derived_power_base_metric(energy_base)
        power_metric_id = MetricId.parse(str(metric_id)).with_base_metric(power_base).serialized

        power_rows["metric"] = power_base
        power_rows["base_metric"] = power_base
        power_rows["metric_id"] = power_metric_id
        # Left edge of the averaging window (t_{i-1}, t_i]; used only for step plotting.
        power_rows["interval_start"] = ts.iloc[:-1].to_numpy()
        power_rows["point_role"] = PointRole.OBSERVED.value
        power_rows["point_order"] = int(PointOrder.OBSERVED)
        power_rows["metric_origin"] = MetricOrigin.DERIVED.value
        frames.append(power_rows)

    if not frames:
        return observed.iloc[0:0].copy()

    out = pd.concat(frames, ignore_index=True)
    out["sample_id"] = pd.array(
        np.arange(len(out), dtype=np.int64),
        dtype="Int64",
    )
    return out


# ---------------------------------------------------------------------------
# 6. Plot helpers
# Spikes = CounterDiff interval energy. Steps = derived interval-average power.
# ---------------------------------------------------------------------------

def sort_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by metric id, timestamp, then point order for plotting.
    """
    if df.empty:
        return df.copy()
    cols = [c for c in ("metric_id", "timestamp", "point_order") if c in df.columns]
    if not cols:
        return df.copy()
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def build_counterdiff_spike_coordinates(
    timestamps: Iterable,
    values: Iterable,
    point_roles: Iterable | None = None,
    point_orders: Iterable | None = None,
) -> tuple[list, list]:
    """
    Build (x, y) lists for one vertical spike per observed sample. Each spike is four points: [baseline, peak, baseline, gap].

    Specifically, for each observed sample `(t, v)`:
        x: t, t, t, None
        y: 0, v, 0, None
    """
    ts_list = list(timestamps)
    val_list = list(values)
    if point_roles is None:
        roles = [PointRole.OBSERVED.value] * len(ts_list)
    else:
        roles = [str(r) for r in point_roles]
    if point_orders is None:
        orders = [0] * len(ts_list)
    else:
        orders = [int(o) for o in point_orders]

    pairs = sorted(
        zip(ts_list, val_list, roles, orders),
        key=lambda row: (pd.Timestamp(row[0]), row[3]),
    )

    x_out: list = []
    y_out: list = []
    for ts, val, role, _order in pairs:
        if role != PointRole.OBSERVED.value:
            continue
        if pd.isna(val):
            continue
        x_out.extend([ts, ts, ts, None])
        y_out.extend([0.0, float(val), 0.0, None])
    return x_out, y_out


def counterdiff_spike_marker_sizes(
    spike_coordinates: Iterable,
    marker_size: float = 6,
) -> list[float]:
    """Marker sizes aligned with spike coordinates"""
    coordinates = list(spike_coordinates)
    return [
        marker_size if index % 4 == 1 and coordinate is not None else 0 for index, coordinate in enumerate(coordinates)
    ]


def build_step_power_coordinates(
    timestamps: Iterable,
    values: Iterable,
    point_roles: Iterable | None = None,
    interval_starts: Iterable | None = None,
) -> tuple[list, list]:
    """Build (x, y) lists for a staircase of interval-average power."""
    ts_list = list(pd.to_datetime(list(timestamps)))
    val_list = list(values)
    if point_roles is None:
        roles = [PointRole.OBSERVED.value] * len(ts_list)
    else:
        roles = [str(r) for r in point_roles]
    if interval_starts is None:
        starts = [pd.NaT] * len(ts_list)
    else:
        starts = list(pd.to_datetime(list(interval_starts)))

    observed = [
        (start, ts, float(val))
        for start, ts, val, role in zip(starts, ts_list, val_list, roles)
        if role == PointRole.OBSERVED.value and not pd.isna(val)
    ]
    observed.sort(key=lambda row: row[1])

    x_out: list = []
    y_out: list = []
    prev_end = None
    prev_val: float | None = None
    for interval_start, ts, val in observed:
        start = interval_start if not pd.isna(interval_start) else prev_end
        if start is None or not (start < ts):
            prev_end = ts
            prev_val = val
            continue

        if prev_end is None:
            x_out.extend([start, ts])
            y_out.extend([val, val])
        elif start > prev_end:
            x_out.append(None)
            y_out.append(None)
            x_out.extend([start, ts])
            y_out.extend([val, val])
        else:
            # Contiguous/overlapping with the previous interval end: keep the staircase connected.
            if prev_val is None or val != prev_val:
                x_out.append(prev_end)
                y_out.append(val)
            x_out.append(ts)
            y_out.append(val)

        prev_end = ts
        prev_val = val
    return x_out, y_out


# ---------------------------------------------------------------------------
# 7. CSV export helpers
#
# Exports must only contain Alumet measured data without synthetic zeros or internal
# ---------------------------------------------------------------------------

def strip_internal_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that Alumet measurement CSVs never contain.

    Drops, when present:
    - `point_role` / `point_order` / `sample_id` — expansion bookkeeping
    - `metric_origin` — measured vs derived provenance
    - `interval_start` — power-step plotting needed for `derive_interval_average_power`
    """
    out = df.copy()
    drop_cols = [
        column for column in (*_POINT_METADATA_COLUMNS, "metric_origin", "interval_start") if column in out.columns
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def export_observed_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Export-ready dataframe: observed rows only, internal columns removed."""
    return strip_internal_export_columns(observed_only(df))
