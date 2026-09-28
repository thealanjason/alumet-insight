import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.cli_export import (
    build_metric_id_listing,
    export_comparative_csv,
    export_comparative_figure,
    export_csvs,
    export_figures,
    summary,
)
from backend.counterdiff import expand_counterdiff_rows
from tests.fixtures import (
    CPU_ENERGY_ID,
    DOWNLOAD_CPU_TOTAL,
    DOWNLOAD_GPU_TOTAL,
    GPU_ENERGY_ID,
    NVML_POWER_ID,
    RAPL_ENERGY_ID,
    download_cpu_gpu_energy_rows,
    make_alumetdata_stub,
    rapl_energy_rows,
)


class CliExportTests(unittest.TestCase):
    def test_summary_contains_expected_sections(self):
        data = make_alumetdata_stub()
        result = summary(data)
        self.assertIn("Base metrics", result)
        self.assertIn("Next Steps", result)
        self.assertIn("--list-metric-ids", result)
        self.assertIn("--compare-metric-id", result)

    def test_summary_not_include_raw_metric_ids(self):
        data = make_alumetdata_stub()
        result = summary(data)
        self.assertNotIn(NVML_POWER_ID, result)

    def test_build_metric_id_listing_all(self):
        data = make_alumetdata_stub()
        result = build_metric_id_listing(data)
        self.assertIn("All metric IDs", result)
        self.assertIn(NVML_POWER_ID, result)

    def test_build_metric_id_listing_by_category(self):
        data = make_alumetdata_stub()
        result = build_metric_id_listing(data, category="power")
        self.assertIn("Metric IDs in category: power", result)
        self.assertIn(NVML_POWER_ID, result)

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
        metric_id = NVML_POWER_ID
        with tempfile.TemporaryDirectory() as tmp:
            created = export_csvs(data, Path(tmp), metric_id=metric_id)
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].parent.name, "csv")
            exported = pd.read_csv(created[0])
            self.assertEqual(exported["metric_id"].unique().tolist(), [metric_id])
            self.assertNotIn("unit", exported.columns)

    def test_export_csvs_single_metric_id_under_matching_category(self):
        data = make_alumetdata_stub()
        metric_id = NVML_POWER_ID
        with tempfile.TemporaryDirectory() as tmp:
            created = export_csvs(data, Path(tmp), category="power", metric_id=metric_id)
            self.assertEqual(len(created), 1)
            self.assertEqual(created[0].parent.parent.name, "power")

    def test_export_csvs_counterdiff_is_measurement_faithful(self):
        metric_id = RAPL_ENERGY_ID
        processed = expand_counterdiff_rows(rapl_energy_rows([7.0]))
        data = make_alumetdata_stub(processed_df=processed)

        with tempfile.TemporaryDirectory() as tmp:
            created = export_csvs(data, Path(tmp), metric_id=metric_id)
            exported = pd.read_csv(created[0])

        self.assertEqual(exported["value"].tolist(), [7.0])
        self.assertTrue({"point_role", "point_order", "sample_id"}.isdisjoint(exported.columns))

    def test_export_csvs_rejects_metric_id_category_mismatch(self):
        data = make_alumetdata_stub()
        metric_id = NVML_POWER_ID
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
        metric_id = NVML_POWER_ID
        with tempfile.TemporaryDirectory() as tmp:
            created = export_figures(data, Path(tmp), metric_id=metric_id)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].exists())

    def _energy_pair_stub(self):
        processed = download_cpu_gpu_energy_rows()
        return CPU_ENERGY_ID, GPU_ENERGY_ID, make_alumetdata_stub(
            processed_df=processed, source_df=processed
        )

    def test_export_comparative_csv_matches_dashboard_download(self):
        from backend.transforms import comparative_download_table

        cpu_id, gpu_id, data = self._energy_pair_stub()
        start, end = data.process_time_range
        expected, _ = comparative_download_table(data.processed_df, cpu_id, gpu_id, start, end)
        with tempfile.TemporaryDirectory() as tmp:
            created = export_comparative_csv(data, Path(tmp), cpu_id, gpu_id)
            self.assertEqual(len(created), 1)
            self.assertTrue(str(created[0].parent).endswith("comparative/csv"))
            table = pd.read_csv(created[0])
            self.assertIn(cpu_id, table.columns)
            self.assertIn(gpu_id, table.columns)
            self.assertNotIn("x_unit", table.columns)
            self.assertNotIn("y_unit", table.columns)
            self.assertNotIn(f"{cpu_id}_cumsum", table.columns)
            self.assertAlmostEqual(float(table[cpu_id].iloc[-1]), DOWNLOAD_CPU_TOTAL)
            self.assertAlmostEqual(float(table[gpu_id].iloc[-1]), DOWNLOAD_GPU_TOTAL)
            self.assertEqual(list(table.columns), list(expected.columns))

    def test_export_comparative_figure_and_scatter(self):
        cpu_id, gpu_id, data = self._energy_pair_stub()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cumulative = export_comparative_figure(data, root, cpu_id, gpu_id)
            scatter = export_comparative_figure(data, root, cpu_id, gpu_id, scatter=True)
            self.assertEqual(len(cumulative), 1)
            self.assertEqual(len(scatter), 1)
            self.assertTrue(cumulative[0].exists())
            self.assertTrue(scatter[0].exists())
            self.assertTrue(scatter[0].stem.endswith("_scatter"))
            self.assertEqual(cumulative[0].parent.name, "plots")
            self.assertEqual(cumulative[0].parent.parent.name, "comparative")

    def test_comparative_figure_locks_equal_aspect_for_same_unit_xy(self):
        import matplotlib.pyplot as plt

        from backend.figures import _apply_equal_xy_scale

        fig, ax = plt.subplots()
        _apply_equal_xy_scale(
            ax,
            [0.0, 162.0],
            [0.0, 110.0],
            CPU_ENERGY_ID,
            GPU_ENERGY_ID,
            include_zero=True,
        )
        self.assertEqual(ax.get_xlim(), ax.get_ylim())
        self.assertEqual(ax.get_xlim()[0], 0.0)
        self.assertEqual(ax.get_aspect(), 1.0)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
