import unittest

import pandas as pd

from backend.counterdiff import observed_only
from backend.synthesis import synthesize_attributed_energy_total, synthesize_derived_power
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

    def test_synthesize_totals_preserve_attribution_identity(self):
        timestamp = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_user",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_7_A_user",
                    "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_system",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_7_A_system",
                ],
                "base_metric": [
                    "attributed_energy_cpu_J",
                    "attributed_energy_gpu_J",
                    "attributed_energy_cpu_J",
                    "attributed_energy_gpu_J",
                ],
                "timestamp": [timestamp] * 4,
                "value": [1.0, 2.0, 10.0, 20.0],
            }
        )

        totals = observed_only(synthesize_attributed_energy_total(df))
        combined = totals[totals["base_metric"] == "attributed_energy_total_J"].set_index("metric_id")["value"]

        self.assertEqual(
            combined["attributed_energy_total_J_R_total__C_process_7_A_user"],
            3.0,
        )
        self.assertEqual(
            combined["attributed_energy_total_J_R_total__C_process_7_A_system"],
            30.0,
        )


if __name__ == "__main__":
    unittest.main()
