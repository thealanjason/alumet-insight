from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.counterdiff import interpolate_counterdiff_at_timeline, observed_only
from backend.metrics import MetricId, derived_power_base_metric
from backend.synthesis import (
    _attach_process_identity,
    _datetime_ns,
    _select_attributed_cpu_rows,
    _step_power_at,
    synthesize_attributed_energy_total,
    synthesize_derived_metrics,
    synthesize_derived_power,
    synthesize_running_totals,
)
from tests.fixtures import attributed_energy_source_rows, topo_uncertain_attributed_energy_excerpt

# t         CPU               GPU
# 0         1                 
# 1                           2
# 2         3
TOY_CUMSUM_FFILL_INTERVAL_J = [1.0, 2.0, 3.0]
TOY_CUMSUM_FFILL_CUMULATIVE_J = [1.0, 3.0, 6.0]
TOY_CUMSUM_INTERP_INTERVAL_J = [1.0, 3.5, 1.5]
TOY_CUMSUM_INTERP_CUMULATIVE_J = [1.0, 4.5, 6.0]
TOY_SAWTOOTH_HEIGHT_J = [1.0, 3.5, 3.0]


def _process_energy(
    cpu_timestamps,
    cpu_values,
    gpu_timestamps,
    gpu_values,
    *,
    pid: str = "7",
    extra_gpu: list[tuple[list, list, str]] | None = None,
) -> pd.DataFrame:
    """Measured attributed CPU/GPU interval energy for one process."""
    rows: list[dict] = []
    for timestamp, value in zip(cpu_timestamps, cpu_values):
        rows.append(
            {
                "metric_id": (
                    f"attributed_energy_cpu_J_R_local_machine__C_process_{pid}_A_"
                    "domain=package_total,kind=total"
                ),
                "base_metric": "attributed_energy_cpu_J",
                "timestamp": pd.Timestamp(timestamp),
                "value": value,
            }
        )
    gpu_series = [(gpu_timestamps, gpu_values, "0")]
    if extra_gpu:
        gpu_series.extend(extra_gpu)
    for timestamps, values, gpu_id in gpu_series:
        for timestamp, value in zip(timestamps, values):
            rows.append(
                {
                    "metric_id": f"attributed_energy_gpu_J_R_gpu_{gpu_id}_C_process_{pid}_A_",
                    "base_metric": "attributed_energy_gpu_J",
                    "timestamp": pd.Timestamp(timestamp),
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def _observed_metric(df: pd.DataFrame, base_metric: str) -> pd.DataFrame:
    return (
        observed_only(df)
        .loc[lambda frame: frame["base_metric"] == base_metric]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _process_consumer(metric_id) -> str | None:
    identity = MetricId.parse(str(metric_id))
    return identity.consumer if identity.is_process_consumer else None


def _for_consumer(df: pd.DataFrame, consumer: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return (
        df.loc[df["metric_id"].map(_process_consumer) == consumer]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _observed_metric_for_consumer(df: pd.DataFrame, base_metric: str, consumer: str) -> pd.DataFrame:
    return _for_consumer(_observed_metric(df, base_metric), consumer)


def _timestamp_ns(timestamp) -> int:
    ts = pd.Timestamp(pd.to_datetime(timestamp, utc=True))
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.value)


def _timestamps_ns(timestamps) -> pd.Series:
    index = timestamps.index if isinstance(timestamps, pd.Series) else None
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.Series(idx.as_unit("ns").asi8, index=index, dtype="int64")


def _posted_at(df: pd.DataFrame) -> dict[int, float]:
    """Sum of interval energy posted at each timestamp (independent of synthesis)."""
    posted: dict[int, float] = {}
    if df.empty:
        return posted
    for timestamp, value in zip(df["timestamp"], df["value"]):
        key = _timestamp_ns(timestamp)
        posted[key] = posted.get(key, 0.0) + float(value)
    return posted


def _interp_at(running: pd.DataFrame, timestamp) -> float:
    """Linear running total at t. 0 before the first sample, hold after the last."""
    if running.empty:
        return 0.0
    target = _timestamp_ns(timestamp)
    ns = _timestamps_ns(running["timestamp"]).to_numpy()
    vals = running["value"].to_numpy(dtype="float64")
    order = np.argsort(ns, kind="mergesort")
    ns = ns[order]
    vals = vals[order]
    if len(ns) == 1:
        return 0.0 if target < int(ns[0]) else float(vals[0])
    origin = int(ns[0])
    return float(
        np.interp(
            float(target - origin),
            (ns - origin).astype("float64"),
            vals,
            left=0.0,
            right=float(vals[-1]),
        )
    )


def _power_covering(power: pd.DataFrame, timestamp) -> float:
    """Stair on (interval_start, timestamp]; 0 if none covers t."""
    if power.empty or "interval_start" not in power.columns:
        return 0.0
    target = _timestamp_ns(timestamp)
    covering = power.loc[
        power["interval_start"].map(_timestamp_ns).lt(target)
        & power["timestamp"].map(_timestamp_ns).ge(target)
    ]
    if covering.empty:
        return 0.0
    return float(covering["value"].iloc[-1])


def _power_metric_id(energy_metric_id: str) -> str:
    identity = MetricId.parse(str(energy_metric_id))
    return identity.with_base_metric(derived_power_base_metric(identity.base_metric)).serialized


def _series_key(metric_id) -> tuple:
    return MetricId.parse(str(metric_id)).series_key


def _cpu_gpu_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cpu = df.loc[df["base_metric"] == "attributed_energy_cpu_J", ["timestamp", "value"]].copy()
    gpu = df.loc[df["base_metric"] == "attributed_energy_gpu_J", ["timestamp", "value"]].copy()
    cpu["timestamp"] = pd.to_datetime(cpu["timestamp"], utc=True)
    gpu["timestamp"] = pd.to_datetime(gpu["timestamp"], utc=True)
    return cpu.sort_values("timestamp").reset_index(drop=True), gpu.sort_values("timestamp").reset_index(drop=True)


def _toy_cpu_gpu() -> tuple[pd.DataFrame, pd.DataFrame]:
    cpu = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:02"]),
            "value": [1.0, 3.0],
        }
    )
    gpu = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00:01"]),
            "value": [2.0],
        }
    )
    return cpu, gpu


def _union_clocks(cpu: pd.DataFrame, gpu: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Index(cpu["timestamp"]).union(pd.Index(gpu["timestamp"]))).sort_values()


def _diff_cumulative(cumulative_J: pd.Series) -> pd.Series:
    interval_J = cumulative_J.diff()
    interval_J.iloc[0] = cumulative_J.iloc[0]
    return interval_J


def _align_frame(timeline: pd.DatetimeIndex, cumulative_J: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timeline,
            "interval_J": _diff_cumulative(cumulative_J).to_numpy(),
            "cumulative_J": cumulative_J.to_numpy(),
        }
    )


def _cumsum_ffill(df: pd.DataFrame, timeline: pd.DatetimeIndex) -> pd.Series:
    """cumsum on the series' own stamps, then ffill onto timeline (0 before first)."""
    running = df.set_index("timestamp")["value"].astype("float64").cumsum()
    return running.reindex(timeline).ffill().fillna(0.0)


def _cumsum_interp(df: pd.DataFrame, timeline: pd.DatetimeIndex) -> pd.Series:
    """cumsum on the series' own stamps, then time-interpolate onto timeline."""
    running = df.set_index("timestamp")["value"].astype("float64").cumsum()
    mixed = running.reindex(running.index.union(timeline)).sort_index()
    mixed = mixed.interpolate(method="time")
    return mixed.reindex(timeline).ffill().fillna(0.0)


def _cumsum_ffill_total(cpu: pd.DataFrame, gpu: pd.DataFrame) -> pd.DataFrame:
    """Per-stamp union: missing device contributes 0. Not the derived total."""
    timeline = _union_clocks(cpu, gpu)
    cumulative_J = _cumsum_ffill(cpu, timeline) + _cumsum_ffill(gpu, timeline)
    return _align_frame(timeline, cumulative_J)


def _cumsum_interp_total(cpu: pd.DataFrame, gpu: pd.DataFrame) -> pd.DataFrame:
    """Derived total: cumsum, time-interpolate, add, diff."""
    timeline = _union_clocks(cpu, gpu)
    cumulative_J = _cumsum_interp(cpu, timeline) + _cumsum_interp(gpu, timeline)
    return _align_frame(timeline, cumulative_J)


def _sawtooth_increments(df: pd.DataFrame, timeline: pd.DatetimeIndex) -> pd.Series:
    """cumsum_interp rewritten as increments of the CounterDiff sawtooth.

    Height at a real sample is E_i. Between samples it ramps from 0 toward the
    next E. Interval energy is the increment of that ramp, not the peak height.
    """
    ordered = df.sort_values("timestamp")
    heights = interpolate_counterdiff_at_timeline(ordered["timestamp"], ordered["value"], timeline)
    observed = {_timestamp_ns(ts) for ts in ordered["timestamp"]}
    interval_J: list[float] = []
    in_progress_J = 0.0
    for timestamp, height in zip(timeline, heights.to_numpy()):
        if pd.isna(height):
            interval_J.append(0.0)
            continue
        interval_J.append(float(height) - in_progress_J)
        in_progress_J = 0.0 if _timestamp_ns(timestamp) in observed else float(height)
    return pd.Series(interval_J, index=timeline, dtype="float64")


def _cumsum_interp_via_sawtooth(cpu: pd.DataFrame, gpu: pd.DataFrame) -> pd.DataFrame:
    timeline = _union_clocks(cpu, gpu)
    interval_J = _sawtooth_increments(cpu, timeline).fillna(0.0) + _sawtooth_increments(gpu, timeline).fillna(0.0)
    return pd.DataFrame(
        {
            "timestamp": timeline,
            "interval_J": interval_J.to_numpy(),
            "cumulative_J": interval_J.cumsum().to_numpy(),
        }
    )


def _sawtooth_height_sum(cpu: pd.DataFrame, gpu: pd.DataFrame) -> pd.Series:
    """Invalid (issue #49): treat interpolant height as a new interval sample."""
    timeline = _union_clocks(cpu, gpu)
    cpu_h = interpolate_counterdiff_at_timeline(cpu["timestamp"], cpu["value"], timeline).fillna(0.0)
    gpu_h = interpolate_counterdiff_at_timeline(gpu["timestamp"], gpu["value"], timeline).fillna(0.0)
    return pd.Series((cpu_h + gpu_h).to_numpy(), index=timeline, dtype="float64")


class _SynthesisAssertions(unittest.TestCase):
    """Shared identities: joule conservation, per-device E=P*dt, stair-sum totals."""

    def _assert_close_floats(self, got, want, *, label: str, places: int = 12) -> None:
        self.assertEqual(len(got), len(want), f"{label}: length {len(got)} != {len(want)}")
        for i, (have, expected) in enumerate(zip(got, want)):
            self.assertAlmostEqual(float(have), float(expected), places=places, msg=f"{label}[{i}]")

    def _assert_same_clocks(self, left, right, *, label: str) -> None:
        self.assertEqual(
            [_timestamp_ns(ts) for ts in left],
            [_timestamp_ns(ts) for ts in right],
            f"{label}: timestamps",
        )

    def _assert_matches_cumsum_interp(
        self, production: pd.DataFrame, cpu: pd.DataFrame, gpu: pd.DataFrame, *, label: str
    ) -> None:
        expected = _cumsum_interp_total(cpu, gpu)
        production = production.sort_values("timestamp").reset_index(drop=True)
        self._assert_same_clocks(production["timestamp"], expected["timestamp"], label=label)
        self._assert_close_floats(
            production["value"].tolist(),
            expected["interval_J"].tolist(),
            label=f"{label} interval_J",
            places=5,
        )

    def _assert_cumulative_is_cumsum(self, energy: pd.DataFrame, cumulative: pd.DataFrame, *, label: str) -> None:
        if energy.empty:
            return
        self.assertFalse(cumulative.empty, f"{label}: missing running-total series")
        energy = energy.sort_values("timestamp").reset_index(drop=True)
        cumulative = cumulative.sort_values("timestamp").reset_index(drop=True)
        self._assert_same_clocks(energy["timestamp"], cumulative["timestamp"], label=f"{label} cumulative")
        expected = energy["value"].astype(float).cumsum().tolist()
        self._assert_close_floats(cumulative["value"].tolist(), expected, label=f"{label} cumulative")

    def _assert_power_reconstructs_energy(self, energy: pd.DataFrame, power: pd.DataFrame, *, label: str) -> None:
        energy = energy.sort_values("timestamp").reset_index(drop=True)
        power = power.sort_values("timestamp").reset_index(drop=True)
        if len(energy) < 2:
            self.assertTrue(power.empty, f"{label}: single energy sample must not yield power")
            return
        self.assertFalse(power.empty, f"{label}: missing power for {len(energy)} energy samples")
        self.assertIn("interval_start", power.columns, f"{label}: power missing interval_start")
        self.assertEqual(len(power), len(energy) - 1, f"{label}: power row count")
        self.assertNotIn(
            _timestamp_ns(energy["timestamp"].iloc[0]),
            {_timestamp_ns(ts) for ts in power["timestamp"]},
            f"{label}: first energy sample must not have a power row",
        )
        for i, power_row in enumerate(power.itertuples(index=False)):
            prev_ts = energy["timestamp"].iloc[i]
            end_ts = energy["timestamp"].iloc[i + 1]
            self.assertEqual(_timestamp_ns(power_row.timestamp), _timestamp_ns(end_ts), f"{label}: power ts")
            self.assertEqual(
                _timestamp_ns(power_row.interval_start), _timestamp_ns(prev_ts), f"{label}: interval_start"
            )
            dt = (_timestamp_ns(end_ts) - _timestamp_ns(prev_ts)) / 1e9
            self.assertGreater(dt, 0.0, f"{label}: non-positive dt at {end_ts}")
            expected_e = float(energy["value"].iloc[i + 1])
            reconstructed = float(power_row.value) * dt
            self.assertAlmostEqual(
                expected_e,
                reconstructed,
                msg=f"{label}: E=P*dt failed at {end_ts}: E={expected_e} P={power_row.value} dt={dt}",
            )

    def _series_running_total(self, running: pd.DataFrame, energy_id: str) -> pd.DataFrame:
        key = _series_key(energy_id)
        return running.loc[running["metric_id"].map(_series_key) == key]

    def _series_power(self, observed: pd.DataFrame, energy_id: str) -> pd.DataFrame:
        return observed.loc[observed["metric_id"].astype(str) == _power_metric_id(str(energy_id))]

    def _assert_series_energy_identities(
        self, energy: pd.DataFrame, running: pd.DataFrame, observed: pd.DataFrame, *, label: str
    ) -> None:
        for energy_id, series in energy.groupby(energy["metric_id"].astype(str), sort=False):
            series = series.sort_values("timestamp")
            self._assert_cumulative_is_cumsum(
                series, self._series_running_total(running, str(energy_id)), label=f"{label} {energy_id}"
            )
            self._assert_power_reconstructs_energy(
                series, self._series_power(observed, str(energy_id)), label=f"{label} {energy_id}"
            )

    def _assert_total_power_is_component_sum(
        self, total_power: pd.DataFrame, components: list[pd.DataFrame], *, label: str
    ) -> None:
        components = [frame for frame in components if frame is not None and not frame.empty]
        if not components:
            self.assertTrue(total_power.empty, f"{label}: no component power but total power exists")
            return
        self.assertFalse(total_power.empty, f"{label}: missing summed power")
        self.assertIn("interval_start", total_power.columns, f"{label}: power missing interval_start")
        max_components = sum(float(frame["value"].max()) for frame in components)
        self.assertLessEqual(
            float(total_power["value"].max()),
            max_components + 1e-6,
            f"{label}: {float(total_power['value'].max())} W exceeds component max sum {max_components} W",
        )
        for row in total_power.sort_values("timestamp").itertuples(index=False):
            expected = sum(_power_covering(frame, row.timestamp) for frame in components)
            self.assertAlmostEqual(
                float(row.value), expected, msg=f"{label}: P={row.value} != stair sum {expected} at {row.timestamp}"
            )

    def _cpu_for_process_total(self, cpu: pd.DataFrame) -> pd.DataFrame:
        if cpu.empty:
            return cpu.copy()
        selected = _select_attributed_cpu_rows(
            _attach_process_identity(cpu[["metric_id", "timestamp", "value"]])
        )
        if selected.empty:
            return selected
        return selected.sort_values("timestamp").reset_index(drop=True)

    def _assert_energy_conservation(self, processed: pd.DataFrame) -> None:
        observed = observed_only(processed)
        consumers = sorted(
            {consumer for consumer in observed["metric_id"].map(_process_consumer) if consumer}
        )
        self.assertTrue(consumers, "expected process-attributed energy rows")
        for consumer in consumers:
            self._assert_energy_conservation_for_consumer(observed, consumer)

    def _assert_energy_conservation_for_consumer(self, observed: pd.DataFrame, consumer: str) -> None:
        cpu = _observed_metric_for_consumer(observed, "attributed_energy_cpu_J", consumer)
        gpu = _observed_metric_for_consumer(observed, "attributed_energy_gpu_J", consumer)
        gpu_total = _observed_metric_for_consumer(observed, "attributed_energy_gpu_total_J", consumer)
        total = _observed_metric_for_consumer(observed, "attributed_energy_total_J", consumer)
        cpu_cum = _observed_metric_for_consumer(observed, "attributed_energy_cpu_cumulative_J", consumer)
        gpu_cum = _observed_metric_for_consumer(observed, "attributed_energy_gpu_cumulative_J", consumer)
        gpu_total_cum = _observed_metric_for_consumer(
            observed, "attributed_energy_gpu_total_cumulative_J", consumer
        )
        total_cum = _observed_metric_for_consumer(observed, "attributed_energy_total_cumulative_J", consumer)
        gpu_total_power = _observed_metric_for_consumer(observed, "attributed_power_gpu_total_W", consumer)
        total_power = _observed_metric_for_consumer(observed, "attributed_power_total_W", consumer)

        gpu_sum = float(gpu["value"].sum()) if not gpu.empty else 0.0
        if not gpu.empty:
            self.assertAlmostEqual(float(gpu_total["value"].sum()), gpu_sum, msg=f"{consumer}: gpu_total sum")
            self._assert_cumulative_is_cumsum(gpu_total, gpu_total_cum, label=f"{consumer} gpu_total")
            self._assert_series_energy_identities(gpu, gpu_cum, observed, label=f"{consumer} gpu")
            self._assert_total_power_is_component_sum(
                gpu_total_power,
                [self._series_power(observed, str(energy_id)) for energy_id in gpu["metric_id"].astype(str).unique()],
                label=f"{consumer} gpu_total",
            )
            if gpu["metric_id"].nunique() == 1:
                gpu_sorted = gpu.sort_values("timestamp").reset_index(drop=True)
                self._assert_same_clocks(
                    gpu_total["timestamp"], gpu_sorted["timestamp"], label=f"{consumer} single-GPU gpu_total"
                )
                self._assert_close_floats(
                    gpu_total["value"].tolist(), gpu_sorted["value"].tolist(), label=f"{consumer} single-GPU gpu_total"
                )
            for row in gpu_total_cum.itertuples(index=False):
                expected = sum(
                    _interp_at(self._series_running_total(gpu_cum, str(energy_id)), row.timestamp)
                    for energy_id in gpu["metric_id"].astype(str).unique()
                )
                self.assertAlmostEqual(
                    float(row.value),
                    expected,
                    msg=f"{consumer}: gpu_total cumulative != interpolated per-GPU running totals at {row.timestamp}",
                )
        if not cpu.empty:
            self._assert_series_energy_identities(cpu, cpu_cum, observed, label=f"{consumer} cpu")

        cpu_for_total = self._cpu_for_process_total(cpu)
        if cpu_for_total.empty or gpu.empty or total.empty:
            return

        cpu_sum = float(cpu_for_total["value"].sum())
        self.assertAlmostEqual(
            float(total["value"].sum()),
            cpu_sum + gpu_sum,
            msg=f"{consumer}: process total {float(total['value'].sum())} J != CPU {cpu_sum} J + GPU {gpu_sum} J",
        )
        self._assert_cumulative_is_cumsum(total, total_cum, label=f"{consumer} process total")
        self._assert_total_power_is_component_sum(
            total_power,
            [self._series_power(observed, str(energy_id)) for energy_id in cpu_for_total["metric_id"].astype(str).unique()]
            + [self._series_power(observed, str(energy_id)) for energy_id in gpu["metric_id"].astype(str).unique()],
            label=f"{consumer} process total",
        )

        cpu_summed = (
            cpu_for_total.groupby("timestamp", as_index=False, sort=False)["value"].sum().sort_values("timestamp")
        )
        cpu_summed_cum = cpu_summed.copy()
        cpu_summed_cum["value"] = cpu_summed["value"].astype(float).cumsum()
        for row in total_cum.itertuples(index=False):
            expected = _interp_at(cpu_summed_cum, row.timestamp) + _interp_at(gpu_total_cum, row.timestamp)
            self.assertAlmostEqual(
                float(row.value),
                expected,
                msg=f"{consumer}: cumulative total != interp(CPU C)+interp(gpu_total C) at {row.timestamp}",
            )


class EnergyTotalTests(_SynthesisAssertions):
    def test_coincident_samples_sum_cpu_and_gpus(self):
        synthetic = observed_only(synthesize_attributed_energy_total(attributed_energy_source_rows()))
        self.assertIn("attributed_energy_gpu_total_J", set(synthetic["base_metric"]))
        self.assertIn("attributed_energy_total_J", set(synthetic["base_metric"]))
        self.assertEqual(
            synthetic.loc[synthetic["base_metric"] == "attributed_energy_total_J", "value"].iloc[0],
            6.0,
        )
        self.assertEqual(set(synthetic["point_role"]), {"observed"})

    def test_empty_input_returns_empty(self):
        empty = synthesize_attributed_energy_total(
            pd.DataFrame(columns=["metric_id", "base_metric", "timestamp", "value"])
        )
        self.assertTrue(empty.empty)

    def test_gpu_only_has_gpu_total_not_process_total(self):
        gpu_only = pd.DataFrame(
            {
                "metric_id": ["attributed_energy_gpu_J_R_gpu_0_C_process_1_A_"],
                "base_metric": ["attributed_energy_gpu_J"],
                "timestamp": [pd.Timestamp("2024-01-01")],
                "value": [4.0],
            }
        )
        synthetic = observed_only(synthesize_attributed_energy_total(gpu_only))
        self.assertEqual(set(synthetic["base_metric"]), {"attributed_energy_gpu_total_J"})

    def test_offset_clocks_are_cumsum_interp_not_cumsum_ffill(self):
        df = _process_energy(
            ["2024-01-01 00:00:00", "2024-01-01 00:00:02"],
            [1.0, 3.0],
            ["2024-01-01 00:00:01"],
            [2.0],
        )
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        totals = _observed_metric(processed, "attributed_energy_total_J")
        cpu, gpu = _cpu_gpu_frames(df)
        self._assert_matches_cumsum_interp(totals, cpu, gpu, label="toy")
        self.assertEqual(totals["value"].tolist(), TOY_CUMSUM_INTERP_INTERVAL_J)
        self.assertNotEqual(totals["value"].tolist(), TOY_CUMSUM_FFILL_INTERVAL_J)
        self.assertEqual(
            _observed_metric(processed, "attributed_energy_total_cumulative_J")["value"].tolist(),
            TOY_CUMSUM_INTERP_CUMULATIVE_J,
        )

    def test_synthetic_counterdiff_zeros_do_not_enter_gpu_total(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_gpu_J_R_gpu_0_C_process_9_A_",
                    "attributed_energy_gpu_J_R_gpu_1_C_process_9_A_",
                ],
                "base_metric": ["attributed_energy_gpu_J", "attributed_energy_gpu_J"],
                "timestamp": [pd.Timestamp("2024-01-01")] * 2,
                "value": [2.0, 3.0],
                "point_role": ["observed", "observed"],
                "point_order": [0, 0],
                "sample_id": [0, 1],
            }
        )
        padded = pd.concat([df, df.assign(point_role="synthetic", point_order=1, value=0.0)], ignore_index=True)
        synthetic = observed_only(synthesize_attributed_energy_total(padded))
        gpu_total = synthetic.loc[synthetic["base_metric"] == "attributed_energy_gpu_total_J", "value"].iloc[0]
        self.assertEqual(gpu_total, 5.0)

    def test_silent_device_adds_zero_on_the_union(self):
        t0, t1, t2, t3 = pd.date_range("2024-01-01", periods=4, freq="s")
        df = _process_energy([t0, t1, t2, t3], [1.0, 3.0, 5.0, 7.0], [t1, t2], [10.0, 20.0], pid="5")
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        totals = _observed_metric(processed, "attributed_energy_total_J")
        self._assert_same_clocks(totals["timestamp"], [t0, t1, t2, t3], label="union")
        self.assertEqual(totals["value"].tolist(), [1.0, 13.0, 25.0, 7.0])
        self.assertEqual(
            _observed_metric(processed, "attributed_energy_total_cumulative_J")["value"].tolist(),
            [1.0, 14.0, 39.0, 46.0],
        )

    def test_gpu_starts_before_cpu(self):
        t0, t1, t2, t3 = pd.date_range("2024-01-01", periods=4, freq="s")
        df = _process_energy([t1, t2, t3], [4.0, 5.0, 6.0], [t0, t1, t2], [10.0, 20.0, 30.0])
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        self.assertEqual(
            _observed_metric(processed, "attributed_energy_total_J")["value"].tolist(),
            [10.0, 24.0, 35.0, 6.0],
        )

    def test_aligned_clocks_sum_per_timestamp(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        df = _process_energy(timestamps, [1.0, 3.0, 9.0], timestamps, [2.0, 4.0, 10.0])
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        self.assertEqual(
            _observed_metric(processed, "attributed_energy_total_J")["value"].tolist(),
            [3.0, 7.0, 19.0],
        )

    def test_processes_do_not_share_joules(self):
        processed = synthesize_derived_metrics(
            pd.concat(
                [
                    _process_energy(
                        ["2024-01-01 00:00:00", "2024-01-01 00:00:02"],
                        [1.0, 3.0],
                        ["2024-01-01 00:00:01"],
                        [10.0],
                        pid="11",
                    ),
                    _process_energy(
                        ["2024-01-01 00:00:00", "2024-01-01 00:00:02"],
                        [100.0, 300.0],
                        ["2024-01-01 00:00:01"],
                        [1000.0],
                        pid="22",
                    ),
                ],
                ignore_index=True,
            )
        )
        self._assert_energy_conservation(processed)
        self.assertEqual(
            _observed_metric_for_consumer(processed, "attributed_energy_total_J", "process_11")["value"].tolist(),
            [1.0, 11.5, 1.5],
        )
        self.assertEqual(
            _observed_metric_for_consumer(processed, "attributed_energy_total_J", "process_22")["value"].tolist(),
            [100.0, 1150.0, 150.0],
        )
        self.assertAlmostEqual(
            float(
                _observed_metric_for_consumer(
                    processed, "attributed_energy_total_cumulative_J", "process_11"
                )["value"].iloc[-1]
            ),
            14.0,
        )
        self.assertAlmostEqual(
            float(
                _observed_metric_for_consumer(
                    processed, "attributed_energy_total_cumulative_J", "process_22"
                )["value"].iloc[-1]
            ),
            1400.0,
        )

    def test_unequal_rapl_nvml_rates_conserve(self):
        start = pd.Timestamp("2024-01-01")
        cpu_ts = pd.date_range(start, periods=81, freq="50ms")
        gpu_ts = pd.date_range(start + pd.Timedelta("2s"), periods=11, freq="200ms")
        df = _process_energy(cpu_ts, [2.0] * len(cpu_ts), gpu_ts, [10.0] * len(gpu_ts), pid="1")
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        self.assertAlmostEqual(
            float(_observed_metric(processed, "attributed_energy_total_cumulative_J")["value"].iloc[-1]),
            272.0,
        )

    def test_offset_gpus_conserve_gpu_total(self):
        start = pd.Timestamp("2024-01-01")
        gpu0 = pd.date_range(start, periods=3, freq="100ms")
        gpu1 = pd.date_range(start + pd.Timedelta("30ms"), periods=3, freq="100ms")
        cpu = pd.date_range(start, periods=3, freq="100ms")
        df = _process_energy(cpu, [3.0] * 3, gpu0, [1.0] * 3, extra_gpu=[(list(gpu1), [2.0] * 3, "1")])
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        self.assertAlmostEqual(float(_observed_metric(processed, "attributed_energy_gpu_total_J")["value"].sum()), 9.0)

    def test_single_gpu_total_copies_that_gpu(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        df = _process_energy(timestamps, [1.0, 2.0, 3.0], timestamps, [4.0, 5.0, 6.0])
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        gpu = _observed_metric(processed, "attributed_energy_gpu_J")
        gpu_total = _observed_metric(processed, "attributed_energy_gpu_total_J")
        self.assertEqual(gpu["value"].tolist(), gpu_total["value"].tolist())
        self._assert_same_clocks(gpu["timestamp"], gpu_total["timestamp"], label="single GPU")

    def test_cpu_only_and_gpu_only_skip_process_total(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        cpu_processed = synthesize_derived_metrics(_process_energy(timestamps, [2.0, 3.0, 5.0], [], []))
        gpu_processed = synthesize_derived_metrics(_process_energy([], [], timestamps, [4.0, 6.0, 8.0]))
        self._assert_energy_conservation(cpu_processed)
        self._assert_energy_conservation(gpu_processed)
        self.assertTrue(_observed_metric(cpu_processed, "attributed_energy_total_J").empty)
        self.assertTrue(_observed_metric(gpu_processed, "attributed_energy_total_J").empty)
        self.assertEqual(
            _observed_metric(gpu_processed, "attributed_energy_gpu_total_J")["value"].tolist(),
            [4.0, 6.0, 8.0],
        )

    def test_resynthesis_does_not_readd_totals_or_cumulatives(self):
        df = _process_energy(
            ["2024-01-01 00:00:00", "2024-01-01 00:00:02"], [1.0, 3.0], ["2024-01-01 00:00:01"], [2.0]
        )
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        first = _observed_metric(processed, "attributed_energy_total_J")
        again = synthesize_attributed_energy_total(observed_only(processed))
        second = _observed_metric(again, "attributed_energy_total_J")
        self.assertEqual(first["value"].tolist(), TOY_CUMSUM_INTERP_INTERVAL_J)
        self.assertEqual(second["value"].tolist(), TOY_CUMSUM_INTERP_INTERVAL_J)
        self.assertAlmostEqual(float(_observed_metric(again, "attributed_energy_gpu_total_J")["value"].sum()), 2.0)

    def test_running_totals_are_per_series_cumsum_gauges(self):
        timestamps = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01", "2024-01-01 00:00:02"])
        df = pd.DataFrame(
            {
                "metric_id": ["attributed_energy_cpu_J_R_cpu_0_C_process_1_A_"] * 3,
                "base_metric": ["attributed_energy_cpu_J"] * 3,
                "metric": ["attributed_energy_cpu_J"] * 3,
                "timestamp": timestamps,
                "value": [2.0, 3.0, 5.0],
                "resource_kind": ["cpu"] * 3,
                "resource_id": ["0"] * 3,
                "consumer_kind": ["process"] * 3,
                "consumer_id": ["1"] * 3,
                "__late_attributes": [""] * 3,
            }
        )
        running = synthesize_running_totals(df)
        self.assertEqual(set(running["base_metric"]), {"attributed_energy_cpu_cumulative_J"})
        self.assertEqual(running["value"].tolist(), [2.0, 5.0, 10.0])
        self.assertTrue((running["metric_origin"] == "derived").all())
        self.assertEqual(running["metric_id"].iloc[0], "attributed_energy_cpu_cumulative_J_R_cpu_0_C_process_1_A_")

        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        cumulative = processed.loc[processed["base_metric"] == "attributed_energy_cpu_cumulative_J"]
        self.assertEqual(len(cumulative), 3)
        self.assertEqual(set(cumulative["point_role"]), {"observed"})
        self.assertEqual(cumulative.sort_values("timestamp")["value"].tolist(), [2.0, 5.0, 10.0])


class PowerTests(_SynthesisAssertions):
    def test_rapl_power_is_energy_over_own_interval(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "rapl_consumed_energy_J_R_package_0_C__A_",
                    "rapl_consumed_energy_J_R_package_0_C__A_",
                ],
                "base_metric": ["rapl_consumed_energy_J", "rapl_consumed_energy_J"],
                "timestamp": [pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:00:02")],
                "value": [10.0, 20.0],
                "point_role": ["observed", "observed"],
                "point_order": [0, 0],
                "sample_id": [0, 1],
            }
        )
        power = synthesize_derived_power(df)
        self.assertEqual(power["base_metric"].iloc[0], "rapl_average_power_W")
        self.assertAlmostEqual(power["value"].iloc[0], 10.0)
        self._assert_power_reconstructs_energy(df, power, label="rapl package")

    def test_step_power_covers_open_close_intervals_and_gaps(self):
        """Stair on (start, end]: exact end is covered, exact start is not, gaps are 0."""
        power = pd.DataFrame(
            {
                "interval_start": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:03"]),
                "timestamp": pd.to_datetime(["2024-01-01 00:00:01", "2024-01-01 00:00:04"]),
                "value": [10.0, 20.0],
            }
        )
        targets = pd.to_datetime(
            [
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:01",
                "2024-01-01 00:00:02",
                "2024-01-01 00:00:03",
                "2024-01-01 00:00:04",
            ]
        )
        got = _step_power_at(power, _datetime_ns(targets))
        self.assertEqual(got.tolist(), [0.0, 10.0, 0.0, 0.0, 20.0])

    def test_nvml_energy_does_not_derive_power_when_gauge_exists(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "nvml_energy_consumption_J_R_gpu_0_C__A_",
                    "nvml_energy_consumption_J_R_gpu_0_C__A_",
                    "nvml_instant_power_W_R_gpu_0_C__A_",
                ],
                "base_metric": ["nvml_energy_consumption_J", "nvml_energy_consumption_J", "nvml_instant_power_W"],
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
                "value": [1.0, 2.0, 50.0],
                "point_role": ["observed"] * 3,
                "point_order": [0, 0, 0],
                "sample_id": [0, 1, 2],
            }
        )
        self.assertTrue(synthesize_derived_power(df).empty)

    def test_measured_power_precedence_is_per_identity(self):
        rows = []
        timestamps = pd.date_range("2024-01-01", periods=2, freq="s")
        for gpu_id in ("0", "1"):
            for timestamp, value in zip(timestamps, (1.0, 2.0)):
                rows.append(
                    {
                        "metric_id": f"nvml_energy_consumption_J_R_gpu_{gpu_id}_C__A_",
                        "base_metric": "nvml_energy_consumption_J",
                        "timestamp": timestamp,
                        "value": value,
                        "point_role": "observed",
                        "point_order": 0,
                        "sample_id": len(rows),
                    }
                )
        rows.append(
            {
                "metric_id": "nvml_instant_power_W_R_gpu_0_C__A_",
                "base_metric": "nvml_instant_power_W",
                "timestamp": timestamps[0],
                "value": 50.0,
                "point_role": "observed",
                "point_order": 0,
                "sample_id": len(rows),
            }
        )
        power = synthesize_derived_power(pd.DataFrame(rows))
        self.assertEqual(power["metric_id"].unique().tolist(), ["nvml_average_power_W_R_gpu_1_C__A_"])

    def test_total_power_is_stair_sum_not_union_e_over_dt(self):
        cpu_ts = pd.to_datetime(
            ["2024-01-01 00:00:00.000000", "2024-01-01 00:00:00.050000", "2024-01-01 00:00:00.100000"]
        )
        gpu_ts = pd.to_datetime(["2024-01-01 00:00:00.000060", "2024-01-01 00:00:00.100060"])
        df = pd.DataFrame(
            {
                "metric_id": (
                    ["attributed_energy_cpu_J_R_local_machine__C_process_9_A_domain=package_total"] * 3
                    + ["attributed_energy_gpu_J_R_gpu_0_C_process_9_A_"] * 2
                ),
                "base_metric": ["attributed_energy_cpu_J"] * 3 + ["attributed_energy_gpu_J"] * 2,
                "timestamp": list(cpu_ts) + list(gpu_ts),
                "value": [0.0, 0.1, 0.1, 0.2, 0.2],
            }
        )
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        observed = observed_only(processed)
        power_rows = observed.loc[observed["base_metric"] == "attributed_power_total_W"]
        cpu_power = observed.loc[observed["base_metric"] == "attributed_power_cpu_W"]
        gpu_power = observed.loc[observed["base_metric"] == "attributed_power_gpu_W"]
        self.assertFalse(power_rows.empty)
        self.assertFalse(cpu_power.empty)

        gpu_first = _timestamp_ns(gpu_ts[0])
        at_gpu_first = power_rows.loc[power_rows["timestamp"].map(_timestamp_ns) == gpu_first]
        self.assertFalse(at_gpu_first.empty)
        union_dt_s = (_timestamp_ns(gpu_ts[0]) - _timestamp_ns(cpu_ts[0])) / 1e9
        self.assertAlmostEqual(union_dt_s, 60e-6)
        energy_at_gpu = _observed_metric(processed, "attributed_energy_total_J")
        energy_at_gpu = energy_at_gpu.loc[energy_at_gpu["timestamp"].map(_timestamp_ns) == gpu_first]
        # GPU's 0.2 J plus the slice of the open CPU interval up to this stamp.
        self.assertAlmostEqual(float(energy_at_gpu["value"].iloc[0]), 0.2 + 0.1 * (60e-6 / 0.05), places=6)
        false_union_power = float(energy_at_gpu["value"].iloc[0]) / union_dt_s
        self.assertGreater(false_union_power, 1000.0)
        self.assertLess(float(power_rows["value"].max()), 10.0)
        self.assertAlmostEqual(float(at_gpu_first["value"].iloc[0]), float(cpu_power["value"].iloc[0]))
        gpu_cap = float(gpu_power["value"].max()) if not gpu_power.empty else 0.0
        self.assertLessEqual(float(power_rows["value"].max()), float(cpu_power["value"].max()) + gpu_cap + 1e-6)

    def test_gpu_total_power_stays_within_per_gpu_stairs(self):
        gpu0 = pd.to_datetime(
            ["2024-01-01 00:00:00.000000", "2024-01-01 00:00:00.200000", "2024-01-01 00:00:00.400000"]
        )
        gpu1 = pd.to_datetime(
            ["2024-01-01 00:00:00.000030", "2024-01-01 00:00:00.200030", "2024-01-01 00:00:00.400030"]
        )
        df = _process_energy(
            gpu0, [0.0, 0.1, 0.1], gpu0, [0.0, 8.0, 8.0], extra_gpu=[(list(gpu1), [0.0, 4.0, 4.0], "1")], pid="5"
        )
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        observed = observed_only(processed)
        gpu_total_rows = observed.loc[observed["base_metric"] == "attributed_power_gpu_total_W"]
        per_gpu = observed.loc[observed["base_metric"] == "attributed_power_gpu_W"]
        self.assertFalse(gpu_total_rows.empty)
        per_gpu_cap = sum(
            float(series["value"].max()) for _, series in per_gpu.groupby(per_gpu["metric_id"].astype(str), sort=False)
        )
        self.assertLessEqual(float(gpu_total_rows["value"].max()), per_gpu_cap + 1e-6)
        self.assertLess(float(gpu_total_rows["value"].max()), 100.0)

    def test_first_energy_sample_has_no_power(self):
        df = _process_energy(["2024-01-01"], [1.0], ["2024-01-01 00:00:01"], [2.0])
        processed = synthesize_derived_metrics(df)
        self._assert_energy_conservation(processed)
        self.assertTrue(_observed_metric(processed, "attributed_power_cpu_W").empty)
        self.assertTrue(_observed_metric(processed, "attributed_power_gpu_W").empty)
        self.assertTrue(_observed_metric(processed, "attributed_power_gpu_total_W").empty)
        self.assertTrue(_observed_metric(processed, "attributed_power_total_W").empty)


class AttributionPolicyTests(_SynthesisAssertions):
    def test_package_total_cpu_ignores_other_domains(self):
        timestamp = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_local_machine__C_process_7_A_domain=package_total,kind=total",
                    "attributed_energy_cpu_J_R_local_machine__C_process_7_A_domain=dram_total,kind=total",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_7_A_",
                ],
                "base_metric": ["attributed_energy_cpu_J", "attributed_energy_cpu_J", "attributed_energy_gpu_J"],
                "timestamp": [timestamp] * 3,
                "value": [1.0, 100.0, 2.0],
            }
        )
        totals = observed_only(synthesize_attributed_energy_total(df))
        self.assertEqual(totals.loc[totals["base_metric"] == "attributed_energy_total_J", "value"].tolist(), [3.0])

    def test_ambiguous_cpu_late_attributes_block_process_total(self):
        timestamp = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_kind=user",
                    "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_kind=system",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_7_A_",
                ],
                "base_metric": ["attributed_energy_cpu_J", "attributed_energy_cpu_J", "attributed_energy_gpu_J"],
                "timestamp": [timestamp] * 3,
                "value": [1.0, 10.0, 2.0],
            }
        )
        totals = observed_only(synthesize_attributed_energy_total(df))
        self.assertNotIn("attributed_energy_total_J", set(totals["base_metric"]))
        self.assertIn("attributed_energy_gpu_total_J", set(totals["base_metric"]))

    def test_rapl_and_gpu_late_attributes_do_not_block_join(self):
        timestamps = pd.date_range("2024-01-01", periods=2, freq="s")
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_local_machine__C_process_42_A_domain=package_total,kind=total",
                    "attributed_energy_cpu_J_R_local_machine__C_process_42_A_domain=package_total,kind=total",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_42_A_",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_42_A_",
                ],
                "base_metric": [
                    "attributed_energy_cpu_J",
                    "attributed_energy_cpu_J",
                    "attributed_energy_gpu_J",
                    "attributed_energy_gpu_J",
                ],
                "timestamp": list(timestamps) * 2,
                "value": [1.0, 3.0, 2.0, 4.0],
            }
        )
        totals = observed_only(synthesize_attributed_energy_total(df))
        combined = totals.loc[totals["base_metric"] == "attributed_energy_total_J"].sort_values("timestamp")
        self.assertEqual(
            combined["metric_id"].tolist(),
            ["attributed_energy_total_J_R_total__C_process_42_A_"] * 2,
        )
        self.assertEqual(combined["__late_attributes"].tolist(), ["", ""])
        self.assertEqual(combined["value"].tolist(), [3.0, 7.0])
        self._assert_energy_conservation(synthesize_derived_metrics(df))

    def test_host_rapl_and_nvml_do_not_create_process_or_machine_totals(self):
        timestamps = pd.date_range("2024-01-01", periods=2, freq="s")
        df = pd.DataFrame(
            {
                "metric_id": (
                    ["rapl_consumed_energy_J_R_package_0_C__A_"] * 2 + ["nvml_energy_consumption_J_R_gpu_0_C__A_"] * 2
                ),
                "base_metric": ["rapl_consumed_energy_J"] * 2 + ["nvml_energy_consumption_J"] * 2,
                "timestamp": list(timestamps) * 2,
                "value": [1.0, 3.0, 2.0, 4.0],
            }
        )
        bases = set(observed_only(synthesize_derived_metrics(df))["base_metric"])
        self.assertNotIn("attributed_energy_total_J", bases)
        self.assertNotIn("attributed_energy_gpu_total_J", bases)
        self.assertNotIn("attributed_power_total_W", bases)
        self.assertNotIn("compute_energy_total_J", bases)
        self.assertNotIn("compute_power_total_W", bases)


class CumsumAlignTests(_SynthesisAssertions):
    """Independent oracles: cumsum_interp (production), cumsum_ffill, sawtooth_height."""

    def test_toy_cumsum_interp_matches_production(self):
        cpu, gpu = _toy_cpu_gpu()
        oracle = _cumsum_interp_total(cpu, gpu)
        rejected = _cumsum_ffill_total(cpu, gpu)
        self.assertEqual(oracle["interval_J"].tolist(), TOY_CUMSUM_INTERP_INTERVAL_J)
        self.assertEqual(oracle["cumulative_J"].tolist(), TOY_CUMSUM_INTERP_CUMULATIVE_J)
        self.assertEqual(rejected["interval_J"].tolist(), TOY_CUMSUM_FFILL_INTERVAL_J)
        production = _observed_metric(
            synthesize_attributed_energy_total(_process_energy(cpu["timestamp"], cpu["value"], gpu["timestamp"], gpu["value"])),
            "attributed_energy_total_J",
        )
        self._assert_matches_cumsum_interp(production, cpu, gpu, label="toy cumsum_interp")
        self.assertNotEqual(production["value"].tolist(), rejected["interval_J"].tolist())

    def test_toy_cumsum_interp_splits_the_open_cpu_interval(self):
        cpu, gpu = _toy_cpu_gpu()
        oracle = _cumsum_interp_total(cpu, gpu)
        via_sawtooth = _cumsum_interp_via_sawtooth(cpu, gpu)
        self.assertEqual(oracle["interval_J"].tolist(), TOY_CUMSUM_INTERP_INTERVAL_J)
        self.assertEqual(oracle["cumulative_J"].tolist(), TOY_CUMSUM_INTERP_CUMULATIVE_J)
        self.assertEqual(via_sawtooth["interval_J"].tolist(), TOY_CUMSUM_INTERP_INTERVAL_J)
        self.assertGreater(float(oracle["interval_J"].iloc[1]), float(gpu["value"].iloc[0]))
        self.assertLess(float(oracle["interval_J"].iloc[2]), float(gpu["value"].iloc[0]))

    def test_toy_sawtooth_height_overcounts(self):
        cpu, gpu = _toy_cpu_gpu()
        naive = _sawtooth_height_sum(cpu, gpu)
        posted_J = float(cpu["value"].sum() + gpu["value"].sum())
        self.assertEqual(naive.tolist(), TOY_SAWTOOTH_HEIGHT_J)
        self.assertAlmostEqual(float(naive.sum()), 7.5)
        self.assertAlmostEqual(posted_J, 6.0)
        self.assertAlmostEqual(float(naive.sum()) - posted_J, 1.5)

    def test_excerpt_union_interleaves_six_cpu_and_two_gpu_samples(self):
        cpu, gpu = _cpu_gpu_frames(topo_uncertain_attributed_energy_excerpt())
        timeline = _union_clocks(cpu, gpu)
        self.assertEqual(len(cpu), 6)
        self.assertEqual(len(gpu), 2)
        self.assertEqual(len(timeline), 8)
        self.assertTrue(cpu["timestamp"].iloc[0] < gpu["timestamp"].iloc[0] < cpu["timestamp"].iloc[1])
        self.assertTrue(cpu["timestamp"].iloc[4] < gpu["timestamp"].iloc[1] < cpu["timestamp"].iloc[5])

    def test_excerpt_production_is_cumsum_interp(self):
        excerpt = topo_uncertain_attributed_energy_excerpt()
        cpu, gpu = _cpu_gpu_frames(excerpt)
        oracle = _cumsum_interp_total(cpu, gpu)
        posted = _cumsum_ffill_total(cpu, gpu)
        production = _observed_metric(synthesize_attributed_energy_total(excerpt), "attributed_energy_total_J")
        self._assert_matches_cumsum_interp(production, cpu, gpu, label="excerpt cumsum_interp")
        self.assertAlmostEqual(float(production["value"].sum()), float(cpu["value"].sum() + gpu["value"].sum()))
        self.assertGreater(
            abs(float(production["value"].iloc[1]) - float(posted["interval_J"].iloc[1])),
            1e-6,
        )
        processed = synthesize_derived_metrics(excerpt)
        self._assert_energy_conservation(processed)
        self.assertAlmostEqual(float(oracle["interval_J"].sum()), float(production["value"].sum()))

    def test_excerpt_cumsum_interp_splits_open_intervals(self):
        cpu, gpu = _cpu_gpu_frames(topo_uncertain_attributed_energy_excerpt())
        oracle = _cumsum_interp_total(cpu, gpu)
        via_sawtooth = _cumsum_interp_via_sawtooth(cpu, gpu)
        self.assertLess(
            float((oracle["interval_J"].astype("float64") - via_sawtooth["interval_J"].astype("float64")).abs().max()),
            5e-6,
        )
        c0, c1 = cpu["timestamp"].iloc[0], cpu["timestamp"].iloc[1]
        g0, g1 = gpu["timestamp"].iloc[0], gpu["timestamp"].iloc[1]
        e_cpu_1 = float(cpu["value"].iloc[1])
        e_gpu_0 = float(gpu["value"].iloc[0])
        e_gpu_1 = float(gpu["value"].iloc[1])
        cpu_slice = e_cpu_1 * float((g0 - c0) / (c1 - c0))
        cpu_remainder = e_cpu_1 - cpu_slice
        gpu_progress_at_c1 = e_gpu_1 * float((c1 - g0) / (g1 - g0))
        self.assertAlmostEqual(float(oracle["interval_J"].iloc[1]), e_gpu_0 + cpu_slice, places=6)
        self.assertAlmostEqual(float(oracle["interval_J"].iloc[2]), cpu_remainder + gpu_progress_at_c1, places=6)
        self.assertNotAlmostEqual(float(oracle["interval_J"].iloc[2]), e_cpu_1)
        self.assertGreater(float(oracle["interval_J"].iloc[1]), e_gpu_0)

    def test_excerpt_cumsum_ffill_and_interp_same_sum_not_same_samples(self):
        cpu, gpu = _cpu_gpu_frames(topo_uncertain_attributed_energy_excerpt())
        ffill = _cumsum_ffill_total(cpu, gpu)
        interp = _cumsum_interp_total(cpu, gpu)
        posted_J = float(cpu["value"].sum() + gpu["value"].sum())
        cpu_posted = _posted_at(cpu)
        gpu_posted = _posted_at(gpu)
        for row in ffill.itertuples(index=False):
            t = _timestamp_ns(row.timestamp)
            self.assertAlmostEqual(float(row.interval_J), cpu_posted.get(t, 0.0) + gpu_posted.get(t, 0.0))
        self.assertNotAlmostEqual(float(interp["interval_J"].iloc[1]), float(ffill["interval_J"].iloc[1]))
        self.assertAlmostEqual(float(ffill["interval_J"].sum()), posted_J)
        self.assertAlmostEqual(float(interp["interval_J"].sum()), posted_J)
        self.assertAlmostEqual(float(ffill["cumulative_J"].iloc[-1]), posted_J)
        self.assertAlmostEqual(float(interp["cumulative_J"].iloc[-1]), posted_J)
        self.assertGreater(abs(float(ffill["interval_J"].iloc[1]) - float(interp["interval_J"].iloc[1])), 1e-6)
        self.assertAlmostEqual(float(ffill["cumulative_J"].iloc[1]), float(cpu["value"].iloc[0] + gpu["value"].iloc[0]))
        self.assertGreater(float(interp["cumulative_J"].iloc[1]), float(ffill["cumulative_J"].iloc[1]))

    def test_excerpt_sawtooth_height_overcounts(self):
        cpu, gpu = _cpu_gpu_frames(topo_uncertain_attributed_energy_excerpt())
        naive = _sawtooth_height_sum(cpu, gpu)
        self.assertGreater(float(naive.sum()), float(cpu["value"].sum() + gpu["value"].sum()))


if __name__ == "__main__":
    unittest.main()
