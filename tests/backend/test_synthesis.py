import unittest

import pandas as pd

from backend.synthesis import synthesize_attributed_energy_total
from tests.fixtures import attributed_energy_source_rows


class SynthesisTests(unittest.TestCase):
    def test_synthesize_adds_gpu_total_and_combined_total_rows(self):
        synthetic = synthesize_attributed_energy_total(attributed_energy_source_rows())

        self.assertIn("attributed_energy_gpu_total_J", set(synthetic["base_metric"]))
        self.assertIn("attributed_energy_total_J", set(synthetic["base_metric"]))
        total = synthetic.loc[synthetic["base_metric"] == "attributed_energy_total_J", "value"].iloc[0]
        self.assertEqual(total, 6.0)

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
        synthetic = synthesize_attributed_energy_total(gpu_only)
        self.assertEqual(set(synthetic["base_metric"]), {"attributed_energy_gpu_total_J"})

    def test_synthesize_aligns_cpu_and_gpu_timelines_via_interpolation(self):
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
        synthetic = synthesize_attributed_energy_total(df)
        totals = synthetic.loc[synthetic["base_metric"] == "attributed_energy_total_J"]

        self.assertFalse(totals.empty)
        self.assertGreaterEqual(len(totals), 2)
        # CPU is interpolated to the middle timestamp where GPU has a sample.
        self.assertEqual(totals["value"].iloc[-1], 4.0)


if __name__ == "__main__":
    unittest.main()
