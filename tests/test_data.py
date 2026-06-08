import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.data import (
    AlumetData,
    _read_csv_with_polars,
    extract_pid_from_content,
    filter_to_time_range,
    find_measurement_file_in_directory,
    get_process_time_range_from_df,
    is_cpu_from_content,
    is_gpu_from_content,
    load_csv_from_path,
    preprocess_dataframe_for_visualization,
    read_file_content,
    synthesize_attributed_energy_total,
)
from tests.fixtures import (
    TempMeasurementDirectory,
    attributed_energy_source_rows,
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

    def test_read_csv_with_polars(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "run.csv"
            csv_path.write_text(sample_csv_body(), encoding="utf-8")

            first = _read_csv_with_polars(csv_path)
            parquet_path = csv_path.with_suffix(".parquet")
            self.assertTrue(parquet_path.exists())

            # Sidecar reuse kicks in when parquet is newer than CSV.
            csv_path.write_text("corrupt;csv;content\n", encoding="utf-8")
            parquet_mtime = parquet_path.stat().st_mtime
            os.utime(csv_path, (parquet_mtime - 10, parquet_mtime - 10))

            second = _read_csv_with_polars(csv_path)

        pd.testing.assert_frame_equal(first.to_pandas(), second.to_pandas())

    def test_find_measurement_file_in_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.csv").write_text("x", encoding="utf-8")
            (root / "b.log").write_text("y", encoding="utf-8")

            self.assertEqual(find_measurement_file_in_directory(str(root), [".csv"]).name, "a.csv")
            with self.assertRaises(ValueError):
                find_measurement_file_in_directory(str(root), [".toml"])
            with self.assertRaises(ValueError):
                find_measurement_file_in_directory(str(root / "missing"), [".csv"])

    def test_read_file_content(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write("hello")
            path = Path(handle.name)

        self.assertEqual(read_file_content(path), "hello")
        path.unlink()
        with self.assertRaises(ValueError):
            read_file_content(Path("missing-file.txt"))

    def test_process_time_range_uses_active_process_samples(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="s"),
                "consumer_kind": ["process", "process", "process", ""],
                "value": [0.0, 1.0, 2.0, 10.0],
            }
        )

        start, end = get_process_time_range_from_df(df)

        self.assertEqual(start, pd.Timestamp("2024-01-01 00:00:01"))
        self.assertEqual(end, pd.Timestamp("2024-01-01 00:00:02"))

    def test_get_process_time_range_fallbacks(self):
        empty_start, empty_end = get_process_time_range_from_df(pd.DataFrame())
        self.assertIsNone(empty_start)
        self.assertIsNone(empty_end)

        no_process = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="s"),
                "consumer_kind": ["", ""],
                "value": [1.0, 2.0],
            }
        )
        start, end = get_process_time_range_from_df(no_process)
        self.assertEqual(start, pd.Timestamp("2024-01-01 00:00:00"))
        self.assertEqual(end, pd.Timestamp("2024-01-01 00:00:01"))

    def test_filter_to_time_range_handles_bounds_and_missing_bounds(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="s"), "value": [1, 2, 3]})

        filtered = filter_to_time_range(df, pd.Timestamp("2024-01-01 00:00:01"), pd.Timestamp("2024-01-01 00:00:02"))

        self.assertEqual(filtered["value"].tolist(), [2, 3])
        self.assertTrue(filter_to_time_range(df, None, None).empty)
        self.assertEqual(filter_to_time_range(df, None, None, require_bounds=False)["value"].tolist(), [1, 2, 3])
        self.assertTrue(filter_to_time_range(pd.DataFrame(), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")).empty)
        with self.assertRaises(ValueError):
            filter_to_time_range(df.drop(columns=["timestamp"]), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))

    def test_log_helpers_detect_pid_and_device_plugins(self):
        content = "starting alumet\npid 12345\nloaded nvml and rapl plugins"

        self.assertEqual(extract_pid_from_content(content), 12345)
        self.assertTrue(is_gpu_from_content(content))
        self.assertTrue(is_cpu_from_content(content))

    def test_log_helpers_handle_missing_information(self):
        self.assertIsNone(extract_pid_from_content(""))
        self.assertIsNone(extract_pid_from_content("no pid here"))
        self.assertFalse(is_gpu_from_content(""))
        self.assertFalse(is_cpu_from_content("plain log"))

    def test_synthesize_attributed_energy_total_adds_gpu_and_total_rows(self):
        synthetic = synthesize_attributed_energy_total(attributed_energy_source_rows())

        self.assertIn("attributed_energy_gpu_total_J", set(synthetic["base_metric"]))
        self.assertIn("attributed_energy_total_J", set(synthetic["base_metric"]))
        total = synthetic.loc[synthetic["base_metric"] == "attributed_energy_total_J", "value"].iloc[0]
        self.assertEqual(total, 6.0)

    def test_synthesize_attributed_energy_total_returns_empty_for_missing_sources(self):
        empty = synthesize_attributed_energy_total(pd.DataFrame(columns=["metric_id", "base_metric", "timestamp", "value"]))
        self.assertTrue(empty.empty)

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

    def test_alumetdata_loads_measurement_directory(self):
        with TempMeasurementDirectory() as directory:
            data = AlumetData(directory)

        self.assertEqual(data.pid, 42)
        self.assertEqual(data.device, "CPU + GPU")
        self.assertGreater(len(data.metrics), 0)
        self.assertGreater(len(data.metric_ids), 0)

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

    def test_alumetdata_export_csvs_without_process_filter_and_cpu_core_suffix(self):
        data = make_alumetdata_stub()

        with tempfile.TemporaryDirectory() as tmp:
            created = data.export_csvs(Path(tmp), category=None, cpu_core="0", process_specific=False)
            self.assertGreaterEqual(len(created), 1)

            kernel_paths = [p for p in created if p.parent.parent.name == "kernel_cpu_time"]
            self.assertTrue(kernel_paths)
            self.assertIn("_core_0", kernel_paths[0].stem)

    def test_alumetdata_loads_csv_when_log_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "measurement.csv").write_text(sample_csv_body(), encoding="utf-8")

            data = AlumetData(root)

        self.assertIsNone(data.pid)
        self.assertGreater(len(data.metrics), 0)
        self.assertFalse(data.raw_df.empty)
        self.assertFalse(data.processed_df.empty)

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

    def test_synthesize_attributed_energy_total_aligns_cpu_and_gpu_timelines(self):
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
