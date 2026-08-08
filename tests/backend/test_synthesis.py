import unittest

import pandas as pd

from backend.counterdiff import observed_only
from backend.synthesis import (
    synthesize_attributed_energy_total,
    synthesize_derived_metrics,
    synthesize_derived_power,
    synthesize_compute_energy_total,
)
from tests.fixtures import attributed_energy_source_rows


class SynthesisTests(unittest.TestCase):
    def test_synthesize_adds_gpu_total_and_combined_total_rows(self):
        synthetic = synthesize_attributed_energy_total(attributed_energy_source_rows())
        observed = observed_only(synthetic)

        self.assertIn("attributed_energy_gpu_total_J", set(observed["base_metric"]))
        self.assertIn("attributed_energy_total_J", set(observed["base_metric"]))
        total = observed.loc[observed["base_metric"] == "attributed_energy_total_J", "value"].iloc[0]
        self.assertEqual(total, 6.0)
        self.assertEqual(set(synthetic["point_role"]), {"observed"})

    def test_synthesize_returns_empty_for_missing_sources(self):
        empty = synthesize_attributed_energy_total(
            pd.DataFrame(columns=["metric_id", "base_metric", "timestamp", "value"])
        )
        self.assertTrue(empty.empty)

    def test_synthesize_gpu_only_produces_gpu_total_without_combined(self):
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

    def test_synthesize_aligns_cpu_and_gpu_timelines_via_phase_aware_interpolation(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_",
                    "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_7_A_",
                ],
                "base_metric": [
                    "attributed_energy_cpu_J",
                    "attributed_energy_cpu_J",
                    "attributed_energy_gpu_J",
                ],
                "timestamp": [
                    pd.Timestamp("2024-01-01 00:00:00"),
                    pd.Timestamp("2024-01-01 00:00:02"),
                    pd.Timestamp("2024-01-01 00:00:01"),
                ],
                "value": [1.0, 3.0, 2.0],
            }
        )
        synthetic = observed_only(synthesize_attributed_energy_total(df))
        totals = synthetic.loc[synthetic["base_metric"] == "attributed_energy_total_J"].sort_values("timestamp")

        self.assertFalse(totals.empty)
        self.assertGreaterEqual(len(totals), 2)
        # At the GPU midpoint, phase-aware interp gives CPU=1.5 → total=3.5.
        mid = totals.loc[totals["timestamp"] == pd.Timestamp("2024-01-01 00:00:01"), "value"].iloc[0]
        self.assertAlmostEqual(mid, 3.5)

    def test_synthesize_aggregates_multi_gpu_before_padding(self):
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
        # Add synthetic zeros that must not contribute to the sum.
        padded = pd.concat(
            [
                df,
                df.assign(point_role="synthetic", point_order=1, value=0.0),
            ],
            ignore_index=True,
        )
        synthetic = observed_only(synthesize_attributed_energy_total(padded))
        gpu_total = synthetic.loc[synthetic["base_metric"] == "attributed_energy_gpu_total_J", "value"].iloc[0]
        self.assertEqual(gpu_total, 5.0)

    def test_synthesize_derived_power_from_rapl_energy(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "rapl_consumed_energy_J_R_package_0_C__A_",
                    "rapl_consumed_energy_J_R_package_0_C__A_",
                ],
                "base_metric": ["rapl_consumed_energy_J", "rapl_consumed_energy_J"],
                "timestamp": [
                    pd.Timestamp("2024-01-01 00:00:00"),
                    pd.Timestamp("2024-01-01 00:00:02"),
                ],
                "value": [10.0, 20.0],
                "point_role": ["observed", "observed"],
                "point_order": [0, 0],
                "sample_id": [0, 1],
            }
        )
        power = synthesize_derived_power(df)
        self.assertEqual(power["base_metric"].iloc[0], "rapl_average_power_W")
        self.assertAlmostEqual(power["value"].iloc[0], 10.0)  # 20 J / 2 s

    def test_synthesize_derived_power_skips_nvml_when_measured_power_exists(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "nvml_energy_consumption_J_R_gpu_0_C__A_",
                    "nvml_energy_consumption_J_R_gpu_0_C__A_",
                    "nvml_instant_power_W_R_gpu_0_C__A_",
                ],
                "base_metric": [
                    "nvml_energy_consumption_J",
                    "nvml_energy_consumption_J",
                    "nvml_instant_power_W",
                ],
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
                "value": [1.0, 2.0, 50.0],
                "point_role": ["observed"] * 3,
                "point_order": [0, 0, 0],
                "sample_id": [0, 1, 2],
            }
        )
        power = synthesize_derived_power(df)
        self.assertTrue(power.empty)

    def test_synthesize_derived_power_matches_measured_precedence_by_identity(self):
        rows = []
        timestamps = pd.date_range("2024-01-01", periods=2, freq="s")
        for gpu_id in ("0", "1"):
            for timestamp, value in zip(timestamps, (1.0, 2.0)):
                rows.append(
                    {
                        "metric_id": (f"nvml_energy_consumption_J_R_gpu_{gpu_id}_C__A_"),
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

        self.assertEqual(
            power["metric_id"].unique().tolist(),
            ["nvml_average_power_W_R_gpu_1_C__A_"],
        )

    def test_synthesize_total_uses_package_total_cpu_and_ignores_other_domains(self):
        timestamp = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_local_machine__C_process_7_A_domain=package_total,kind=total",
                    "attributed_energy_cpu_J_R_local_machine__C_process_7_A_domain=dram_total,kind=total",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_7_A_",
                ],
                "base_metric": [
                    "attributed_energy_cpu_J",
                    "attributed_energy_cpu_J",
                    "attributed_energy_gpu_J",
                ],
                "timestamp": [timestamp] * 3,
                "value": [1.0, 100.0, 2.0],
            }
        )

        totals = observed_only(synthesize_attributed_energy_total(df))
        combined = totals.loc[totals["base_metric"] == "attributed_energy_total_J", "value"]
        self.assertEqual(combined.tolist(), [3.0])

    def test_synthesize_total_refuses_ambiguous_cpu_late_attributes(self):
        """Do not invent a process total by summing mixed CPU attribution tags."""
        timestamp = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_kind=user",
                    "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_kind=system",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_7_A_",
                ],
                "base_metric": [
                    "attributed_energy_cpu_J",
                    "attributed_energy_cpu_J",
                    "attributed_energy_gpu_J",
                ],
                "timestamp": [timestamp] * 3,
                "value": [1.0, 10.0, 2.0],
            }
        )

        totals = observed_only(synthesize_attributed_energy_total(df))
        self.assertNotIn("attributed_energy_total_J", set(totals["base_metric"]))
        self.assertIn("attributed_energy_gpu_total_J", set(totals["base_metric"]))

    def test_synthesize_total_joins_mismatched_rapl_and_gpu_late_attributes(self):
        """RAPL CPU attribution tags must not block a per-process CPU+GPU total."""
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
        combined = totals.loc[
            totals["base_metric"] == "attributed_energy_total_J"
        ].sort_values("timestamp")

        self.assertEqual(
            combined["metric_id"].tolist(),
            [
                "attributed_energy_total_J_R_total__C_process_42_A_",
                "attributed_energy_total_J_R_total__C_process_42_A_",
            ],
        )
        self.assertEqual(combined["__late_attributes"].tolist(), ["", ""])
        self.assertEqual(combined["value"].tolist(), [3.0, 7.0])

    def test_derived_power_matches_energy_times_interval(self):
        """Interval-average power must satisfy sum(E) == sum(P · Δt) for derived pairs."""
        timestamps = pd.to_datetime(
            [
                "2024-01-01 00:00:00",
                "2024-01-01 00:00:02",
                "2024-01-01 00:00:05",
            ]
        )
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_local_machine__C_process_9_A_domain=package_total,kind=total",
                    "attributed_energy_cpu_J_R_local_machine__C_process_9_A_domain=package_total,kind=total",
                    "attributed_energy_cpu_J_R_local_machine__C_process_9_A_domain=package_total,kind=total",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_9_A_",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_9_A_",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_9_A_",
                ],
                "base_metric": (
                    ["attributed_energy_cpu_J"] * 3 + ["attributed_energy_gpu_J"] * 3
                ),
                "timestamp": list(timestamps) * 2,
                "value": [1.0, 3.0, 9.0, 2.0, 4.0, 10.0],
            }
        )

        processed = observed_only(synthesize_derived_metrics(df))
        energy = processed.loc[
            processed["base_metric"] == "attributed_energy_total_J"
        ].sort_values("timestamp")
        power = processed.loc[
            processed["base_metric"] == "attributed_power_total_W"
        ].sort_values("timestamp")

        self.assertFalse(energy.empty)
        self.assertFalse(power.empty)
        self.assertEqual(len(power), len(energy) - 1)

        energy_after_first = energy["value"].astype(float).iloc[1:].to_numpy()
        dt = (
            pd.to_datetime(power["timestamp"]) - pd.to_datetime(power["interval_start"])
        ).dt.total_seconds().to_numpy()
        reconstructed = power["value"].astype(float).to_numpy() * dt

        self.assertTrue((dt > 0).all())
        self.assertAlmostEqual(float(energy_after_first.sum()), float(reconstructed.sum()))
        for expected_e, got_e in zip(energy_after_first, reconstructed):
            self.assertAlmostEqual(float(expected_e), float(got_e))

    def test_synthesize_compute_energy_and_power_from_rapl_package_total_and_nvml(self):
        timestamps = pd.date_range("2024-01-01", periods=2, freq="s")
        df = pd.DataFrame(
            {
                "metric_id": [
                    "rapl_consumed_energy_J_R_local_machine__C_local_machine__A_domain=package_total",
                    "rapl_consumed_energy_J_R_local_machine__C_local_machine__A_domain=package_total",
                    "rapl_consumed_energy_J_R_local_machine__C_local_machine__A_domain=dram_total",
                    "nvml_energy_consumption_J_R_gpu_0_C_local_machine__A_",
                    "nvml_energy_consumption_J_R_gpu_0_C_local_machine__A_",
                    "nvml_energy_consumption_J_R_gpu_1_C_local_machine__A_",
                    "nvml_energy_consumption_J_R_gpu_1_C_local_machine__A_",
                ],
                "base_metric": (
                    ["rapl_consumed_energy_J"] * 3
                    + ["nvml_energy_consumption_J"] * 4
                ),
                "timestamp": [
                    timestamps[0],
                    timestamps[1],
                    timestamps[0],
                    timestamps[0],
                    timestamps[1],
                    timestamps[0],
                    timestamps[1],
                ],
                "value": [10.0, 20.0, 1000.0, 1.0, 2.0, 3.0, 4.0],
            }
        )

        energy = observed_only(synthesize_compute_energy_total(df)).sort_values("timestamp")
        self.assertEqual(energy["base_metric"].unique().tolist(), ["compute_energy_total_J"])
        # DRAM must not be included: (10+1+3)=14 and (20+2+4)=26
        self.assertEqual(energy["value"].tolist(), [14.0, 26.0])

        processed = observed_only(synthesize_derived_metrics(df))
        power = processed.loc[
            processed["base_metric"] == "compute_power_total_W"
        ].sort_values("timestamp")
        self.assertFalse(power.empty)
        # Second interval: 26 J / 1 s
        self.assertAlmostEqual(float(power["value"].iloc[0]), 26.0)

        dt = (
            pd.to_datetime(power["timestamp"]) - pd.to_datetime(power["interval_start"])
        ).dt.total_seconds()
        reconstructed = float((power["value"].astype(float) * dt).sum())
        self.assertAlmostEqual(reconstructed, 26.0)

    def test_attributed_total_fills_missing_side_with_zero_on_union_timeline(self):
        """Process totals span the union window; absent GPU/CPU side counts as 0 J."""
        t0, t1, t2, t3 = pd.date_range("2024-01-01", periods=4, freq="s")
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_local_machine__C_process_5_A_domain=package_total,kind=total",
                    "attributed_energy_cpu_J_R_local_machine__C_process_5_A_domain=package_total,kind=total",
                    "attributed_energy_cpu_J_R_local_machine__C_process_5_A_domain=package_total,kind=total",
                    "attributed_energy_cpu_J_R_local_machine__C_process_5_A_domain=package_total,kind=total",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_5_A_",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_5_A_",
                ],
                "base_metric": (
                    ["attributed_energy_cpu_J"] * 4 + ["attributed_energy_gpu_J"] * 2
                ),
                # GPU starts later and ends earlier than CPU.
                "timestamp": [t0, t1, t2, t3, t1, t2],
                "value": [1.0, 3.0, 5.0, 7.0, 10.0, 20.0],
            }
        )

        totals = observed_only(synthesize_attributed_energy_total(df))
        combined = totals.loc[
            totals["base_metric"] == "attributed_energy_total_J"
        ].sort_values("timestamp")

        self.assertEqual(list(combined["timestamp"]), [t0, t1, t2, t3])
        # t0: CPU-only (GPU not started → 0), t1/t2: both, t3: CPU-only (GPU ended → 0)
        self.assertEqual(combined["value"].tolist(), [1.0, 13.0, 25.0, 7.0])

    def test_compute_total_uses_overlap_window_only(self):
        """Machine compute totals omit times before both instruments have started."""
        t0, t1, t2, t3 = pd.date_range("2024-01-01", periods=4, freq="s")
        df = pd.DataFrame(
            {
                "metric_id": [
                    "rapl_consumed_energy_J_R_local_machine__C_local_machine__A_domain=package_total",
                    "rapl_consumed_energy_J_R_local_machine__C_local_machine__A_domain=package_total",
                    "rapl_consumed_energy_J_R_local_machine__C_local_machine__A_domain=package_total",
                    "rapl_consumed_energy_J_R_local_machine__C_local_machine__A_domain=package_total",
                    "nvml_energy_consumption_J_R_gpu_0_C_local_machine__A_",
                    "nvml_energy_consumption_J_R_gpu_0_C_local_machine__A_",
                ],
                "base_metric": (
                    ["rapl_consumed_energy_J"] * 4 + ["nvml_energy_consumption_J"] * 2
                ),
                # NVML starts later and ends earlier than RAPL.
                "timestamp": [t0, t1, t2, t3, t1, t2],
                "value": [10.0, 20.0, 30.0, 40.0, 1.0, 2.0],
            }
        )

        energy = observed_only(synthesize_compute_energy_total(df)).sort_values("timestamp")
        # Overlap is [t1, t2] only — not RAPL-only t0/t3 with NVML treated as 0.
        self.assertEqual(list(energy["timestamp"]), [t1, t2])
        self.assertEqual(energy["value"].tolist(), [21.0, 32.0])
        self.assertNotIn(t0, set(energy["timestamp"]))
        self.assertNotIn(t3, set(energy["timestamp"]))


if __name__ == "__main__":
    unittest.main()
