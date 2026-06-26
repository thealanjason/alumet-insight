import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.categories import category_for_metric_id, filter_time_series_category
from backend.data import (
    AlumetData,
    _read_csv_with_polars,
    load_csv_from_path,
    preprocess_dataframe_for_visualization,
)
from backend.metrics import filter_by_base_metric, metric_id_is_process_consumer
from backend.transforms import parse_timestamp, validate_time_range
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

    def test_alumetdata_state_properties(self):
        data = make_alumetdata_stub()

        self.assertEqual(data.pid, 99)
        self.assertEqual(data.device, "CPU")
        self.assertEqual(data.metrics, sorted(processed_rows()["base_metric"].unique().tolist()))
        self.assertEqual(len(filter_by_base_metric(data.processed_df, "mem_total_kB")), 1)
        self.assertEqual(data.processed_df["metric_id"].apply(metric_id_is_process_consumer).sum(), 3)
        self.assertEqual(len(filter_time_series_category(data.processed_df, "memory")), 1)

    def test_alumetdata_device_detection_variants(self):
        self.assertEqual(make_alumetdata_stub(log_content="pid 1\nnvml").device, "GPU")
        self.assertEqual(make_alumetdata_stub(log_content="pid 1\nrapl").device, "CPU")
        self.assertEqual(make_alumetdata_stub(log_content="pid 1").device, "CPU + GPU")

    def test_validate_time_range_against_data_bounds(self):
        data = make_alumetdata_stub()

        start, end = validate_time_range("2024-01-01T00:00:01", "2024-01-01T00:00:02", *data.data_time_range)
        self.assertEqual(start, pd.Timestamp("2024-01-01T00:00:01"))
        self.assertEqual(end, pd.Timestamp("2024-01-01T00:00:02"))

        with self.assertRaises(AssertionError):
            validate_time_range("2023-12-31T23:59:59", "2024-01-01T00:00:02", *data.data_time_range)

        with self.assertRaises(AssertionError):
            validate_time_range("2024-01-01T00:00:01", "2024-01-01T00:00:04", *data.data_time_range)

    def test_filter_by_category(self):
        data = make_alumetdata_stub()
        power_df = filter_time_series_category(data.processed_df, "power")
        power_ids = sorted(power_df["metric_id"].astype(str).unique().tolist())
        self.assertEqual(power_ids, ["nvml_instant_power_W_R_gpu_0_C_process_123_A_"])

        energy_df = filter_time_series_category(data.processed_df, "energy")
        energy_ids = energy_df["metric_id"].astype(str).tolist()
        self.assertIn("attributed_energy_J_R_local_machine__C_process_123_A_", energy_ids)

        temp_df = filter_time_series_category(data.processed_df, "temperature")
        self.assertTrue(temp_df.empty)

    def test_filter_by_base_metric(self):
        data = make_alumetdata_stub()
        df = filter_by_base_metric(data.processed_df, "nvml_instant_power_W")
        ids = sorted(df["metric_id"].astype(str).unique().tolist())
        self.assertEqual(ids, ["nvml_instant_power_W_R_gpu_0_C_process_123_A_"])
        self.assertTrue(filter_by_base_metric(data.processed_df, "nonexistent").empty)

    def test_category_for_metric_id(self):
        data = make_alumetdata_stub()
        metric_id = "nvml_instant_power_W_R_gpu_0_C_process_123_A_"
        self.assertEqual(category_for_metric_id(data.processed_df, metric_id), "power")
        self.assertEqual(category_for_metric_id(data.processed_df, metric_id, category="power"), "power")

    def test_parse_timestamp_invalid(self):
        with self.assertRaisesRegex(ValueError, "Invalid --start-time"):
            parse_timestamp("not-a-date", "--start-time")

    def test_parse_timestamp_valid(self):
        ts = parse_timestamp("2024-01-01T00:00:01", "--start-time")
        self.assertEqual(ts, pd.Timestamp("2024-01-01T00:00:01"))


if __name__ == "__main__":
    unittest.main()
