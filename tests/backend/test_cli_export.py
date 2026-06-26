import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.cli_export import (
    build_metric_id_listing,
    export_csvs,
    export_figures,
    summary,
)
from tests.fixtures import make_alumetdata_stub


class CliExportTests(unittest.TestCase):
    def test_summary_contains_expected_sections(self):
        data = make_alumetdata_stub()
        result = summary(data)
        self.assertIn("Base metrics", result)
        self.assertIn("Next Steps", result)
        self.assertIn("--list-metric-ids", result)

    def test_summary_not_include_raw_metric_ids(self):
        data = make_alumetdata_stub()
        result = summary(data)
        self.assertNotIn("nvml_instant_power_W_R_gpu_0_C_process_123_A_", result)

    def test_build_metric_id_listing_all(self):
        data = make_alumetdata_stub()
        result = build_metric_id_listing(data)
        self.assertIn("All metric IDs", result)
        self.assertIn("nvml_instant_power_W_R_gpu_0_C_process_123_A_", result)

    def test_build_metric_id_listing_by_category(self):
        data = make_alumetdata_stub()
        result = build_metric_id_listing(data, category="power")
        self.assertIn("Metric IDs in category: power", result)
        self.assertIn("nvml_instant_power_W_R_gpu_0_C_process_123_A_", result)

    def test_build_metric_id_listing_by_metric_name(self):
        data = make_alumetdata_stub()
        result = build_metric_id_listing(data, metric_name="nvml_instant_power_W")
        self.assertIn("Metric IDs for base metric: nvml_instant_power_W", result)

    def test_export_csvs_by_category(self):
        data = make_alumetdata_stub()
        with tempfile.TemporaryDirectory() as tmp:
            created = export_csvs(data, Path(tmp), category="power", process_specific=True)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].exists())

    def test_export_csvs_single_metric_id(self):
        data = make_alumetdata_stub()
        metric_id = "nvml_instant_power_W_R_gpu_0_C_process_123_A_"
        with tempfile.TemporaryDirectory() as tmp:
            created = export_csvs(data, Path(tmp), metric_id=metric_id)
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].parent.name, "csv")
            exported = pd.read_csv(created[0])
            self.assertEqual(exported["metric_id"].unique().tolist(), [metric_id])

    def test_export_csvs_single_metric_id_under_matching_category(self):
        data = make_alumetdata_stub()
        metric_id = "nvml_instant_power_W_R_gpu_0_C_process_123_A_"
        with tempfile.TemporaryDirectory() as tmp:
            created = export_csvs(data, Path(tmp), category="power", metric_id=metric_id)
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].parent.parent.name, "power")

    def test_export_csvs_rejects_metric_id_category_mismatch(self):
        data = make_alumetdata_stub()
        metric_id = "nvml_instant_power_W_R_gpu_0_C_process_123_A_"
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "is not in category 'energy'"):
                export_csvs(data, Path(tmp), category="energy", metric_id=metric_id)

    def test_export_csvs_cpu_core_suffix(self):
        data = make_alumetdata_stub()
        with tempfile.TemporaryDirectory() as tmp:
            created = export_csvs(data, Path(tmp), category=None, cpu_core="0", process_specific=False)
            self.assertGreaterEqual(len(created), 1)
            kernel_paths = [p for p in created if p.parent.parent.name == "kernel_cpu_time"]
            self.assertTrue(kernel_paths)
            self.assertIn("_core_0", kernel_paths[0].stem)

    def test_export_figures_single_metric_id(self):
        data = make_alumetdata_stub()
        metric_id = "nvml_instant_power_W_R_gpu_0_C_process_123_A_"
        with tempfile.TemporaryDirectory() as tmp:
            created = export_figures(data, Path(tmp), metric_id=metric_id)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].exists())


if __name__ == "__main__":
    unittest.main()
