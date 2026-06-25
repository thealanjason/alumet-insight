import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.data import (
    AlumetData,
    _read_csv_with_polars,
    load_csv_from_path,
    preprocess_dataframe_for_visualization,
)
from tests.fixtures import (
    TempMeasurementDirectory,
    make_alumetdata_stub,
    processed_rows,
    sample_csv_body,
)


class DataTests(unittest.TestCase):
    def test_preprocess_dataframe_for_visualization_builds_metric_id(self):
        df = pd.DataFrame(
            {
                "metric": ["cpu_percent"],
                "resource_kind": ["local_machine"],
                "resource_id": [""],
                "consumer_kind": ["process"],
                "consumer_id": ["123"],
                "__late_attributes": ["kind_user"],
                "timestamp": [pd.Timestamp("2024-01-01")],
                "value": [50.0],
            }
        )
        out = preprocess_dataframe_for_visualization(df)
        self.assertEqual(out.loc[0, "base_metric"], "cpu_percent")
        self.assertEqual(out.loc[0, "metric_id"], "cpu_percent_R_local_machine__C_process_123_A_kind_user")

    def test_load_csv_from_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "run.csv"
            csv_path.write_text(sample_csv_body(), encoding="utf-8")
            loaded = load_csv_from_path(csv_path)

        self.assertEqual(len(loaded), 2)
        self.assertIn("metric", loaded.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded["timestamp"]))

    def test_load_csv_from_path_missing_and_empty_inputs(self):
        with self.assertRaises(ValueError):
            load_csv_from_path(Path("missing.csv"))

        with tempfile.TemporaryDirectory() as tmp:
            empty_csv = Path(tmp) / "empty.csv"
            empty_csv.write_text(
                "metric;resource_kind;resource_id;consumer_kind;consumer_id;__late_attributes;timestamp;value\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_csv_from_path(empty_csv)

    def test_read_csv_with_polars_uses_parquet_sidecar(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "run.csv"
            csv_path.write_text(sample_csv_body(), encoding="utf-8")

            first = _read_csv_with_polars(csv_path)
            parquet_path = csv_path.with_suffix(".parquet")
            self.assertTrue(parquet_path.exists())

            csv_path.write_text("corrupt;csv;content\n", encoding="utf-8")
            parquet_mtime = parquet_path.stat().st_mtime
            os.utime(csv_path, (parquet_mtime - 10, parquet_mtime - 10))

            second = _read_csv_with_polars(csv_path)

        pd.testing.assert_frame_equal(first.to_pandas(), second.to_pandas())

    def test_alumetdata_loads_measurement_directory(self):
        with TempMeasurementDirectory() as directory:
            data = AlumetData(directory)

        self.assertEqual(data.pid, 42)
        self.assertEqual(data.device, "CPU + GPU")
        self.assertGreater(len(data.metrics), 0)
        self.assertGreater(len(data.metric_ids), 0)

    def test_alumetdata_loads_csv_when_log_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "measurement.csv").write_text(sample_csv_body(), encoding="utf-8")
            data = AlumetData(root)

        self.assertIsNone(data.pid)
        self.assertGreater(len(data.metrics), 0)
        self.assertFalse(data.source_df.empty)
        self.assertFalse(data.processed_df.empty)

    def test_alumetdata_query_and_export_helpers(self):
        data = make_alumetdata_stub()

        self.assertEqual(data.pid, 99)
        self.assertEqual(data.device, "CPU")
        self.assertIn("Number of metrics", data.summary())
        self.assertEqual(data.metrics, sorted(processed_rows()["base_metric"].unique().tolist()))
        self.assertEqual(len(data.filter_by_metric("mem_total_kB")), 1)
        self.assertEqual(len(data.filter_process_metrics()), 3)
        self.assertEqual(len(data.filter_by_category("memory")), 1)

        windowed = data.filter_to_process_time_range(processed_rows())
        self.assertGreaterEqual(len(windowed), 1)

        with tempfile.TemporaryDirectory() as tmp:
            created = data.export_csvs(Path(tmp), category="power", process_specific=True)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].exists())

    def test_alumetdata_device_detection_variants(self):
        self.assertEqual(make_alumetdata_stub(log_content="pid 1\nnvml").device, "GPU")
        self.assertEqual(make_alumetdata_stub(log_content="pid 1\nrapl").device, "CPU")
        self.assertEqual(make_alumetdata_stub(log_content="pid 1").device, "CPU + GPU")

    def test_alumetdata_export_csvs_cpu_core_suffix(self):
        data = make_alumetdata_stub()

        with tempfile.TemporaryDirectory() as tmp:
            created = data.export_csvs(Path(tmp), category=None, cpu_core="0", process_specific=False)
            self.assertGreaterEqual(len(created), 1)

            kernel_paths = [p for p in created if p.parent.parent.name == "kernel_cpu_time"]
            self.assertTrue(kernel_paths)
            self.assertIn("_core_0", kernel_paths[0].stem)


if __name__ == "__main__":
    unittest.main()
