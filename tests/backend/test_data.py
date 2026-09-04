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
    write_measurement_directory,
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

    def test_finalize_processed_dataframe_expands_counterdiff(self):
        from backend.data import finalize_processed_dataframe

        df = pd.DataFrame(
            {
                "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"],
                "base_metric": ["rapl_consumed_energy_J"],
                "timestamp": [pd.Timestamp("2024-01-01")],
                "value": [1.5],
            }
        )
        out = finalize_processed_dataframe(df)
        self.assertIn("point_role", out.columns)
        self.assertEqual(set(out["point_role"]), {"observed", "synthetic"})
        self.assertEqual(out.loc[out["point_role"] == "observed", "value"].iloc[0], 1.5)
        self.assertEqual(out.loc[out["point_role"] == "synthetic", "value"].iloc[0], 0.0)

    def test_preprocess_attaches_measured_semantics_and_identity_columns(self):
        df = pd.DataFrame(
            {
                "metric": ["nvml_instant_power_W", "cpu_percent"],
                "resource_kind": ["gpu", "local_machine"],
                "resource_id": ["0", ""],
                "consumer_kind": ["", "process"],
                "consumer_id": ["", "9"],
                "__late_attributes": ["", ""],
                "timestamp": [
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-01-01"),
                ],
                "value": [11.0, 3.0],
            }
        )
        out = preprocess_dataframe_for_visualization(df)
        self.assertIn("resource_kind", out.columns)
        self.assertIn("consumer_id", out.columns)
        self.assertEqual(set(out["metric_origin"]), {"measured"})
        self.assertNotIn("power_semantics", out.columns)

    def test_finalize_processed_dataframe_end_to_end_provenance(self):
        from backend.counterdiff import observed_only, validate_point_metadata
        from backend.data import finalize_processed_dataframe

        raw = pd.DataFrame(
            {
                "metric_id": [
                    "rapl_consumed_energy_J_R_pkg_0_C__A_",
                    "rapl_consumed_energy_J_R_pkg_0_C__A_",
                    "rapl_consumed_energy_J_R_pkg_0_C__A_",
                    "attributed_energy_cpu_J_R_cpu_0_C_process_1_A_",
                    "attributed_energy_cpu_J_R_cpu_0_C_process_1_A_",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_1_A_",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_1_A_",
                ],
                "base_metric": [
                    "rapl_consumed_energy_J",
                    "rapl_consumed_energy_J",
                    "rapl_consumed_energy_J",
                    "attributed_energy_cpu_J",
                    "attributed_energy_cpu_J",
                    "attributed_energy_gpu_J",
                    "attributed_energy_gpu_J",
                ],
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:01",
                        "2024-01-01 00:00:01",
                        "2024-01-01 00:00:03",
                        "2024-01-01 00:00:01",
                        "2024-01-01 00:00:03",
                        "2024-01-01 00:00:01",
                        "2024-01-01 00:00:03",
                    ]
                ),
                "value": [8.0, 0.0, 4.0, 1.0, 2.0, 3.0, 5.0],
                "metric_origin": ["measured"] * 7,
                "resource_kind": ["pkg", "pkg", "pkg", "cpu", "cpu", "gpu", "gpu"],
                "resource_id": ["0", "0", "0", "0", "0", "0", "0"],
                "consumer_kind": ["", "", "", "process", "process", "process", "process"],
                "consumer_id": ["", "", "", "1", "1", "1", "1"],
                "__late_attributes": [""] * 7,
                "metric": [
                    "rapl_consumed_energy_J",
                    "rapl_consumed_energy_J",
                    "rapl_consumed_energy_J",
                    "attributed_energy_cpu_J",
                    "attributed_energy_cpu_J",
                    "attributed_energy_gpu_J",
                    "attributed_energy_gpu_J",
                ],
            }
        )
        out = finalize_processed_dataframe(raw)
        validate_point_metadata(out)

        observed = observed_only(out)
        self.assertIn("rapl_average_power_W", set(observed["base_metric"]))
        self.assertIn("attributed_energy_total_J", set(observed["base_metric"]))
        self.assertIn("attributed_power_total_W", set(observed["base_metric"]))

        rapl_obs = observed.loc[observed["metric_id"] == "rapl_consumed_energy_J_R_pkg_0_C__A_"]
        self.assertEqual(sorted(rapl_obs["value"].tolist()), [4.0, 8.0])
        self.assertEqual(set(rapl_obs["metric_origin"]), {"measured"})

        power = observed.loc[observed["base_metric"] == "rapl_average_power_W"]
        self.assertTrue((power["metric_origin"] == "derived").all())
        self.assertIn("interval_start", power.columns)

        totals = observed.loc[observed["base_metric"] == "attributed_energy_total_J"]
        self.assertTrue((totals["metric_origin"] == "derived").all())

        cumulative = observed.loc[observed["base_metric"] == "attributed_energy_cpu_cumulative_J"]
        self.assertFalse(cumulative.empty)
        self.assertTrue((cumulative["metric_origin"] == "derived").all())
        self.assertAlmostEqual(float(cumulative["value"].iloc[-1]), float(observed.loc[
            observed["metric_id"] == "attributed_energy_cpu_J_R_cpu_0_C_process_1_A_", "value"
        ].sum()))

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
        self.assertEqual(data.device, "CPU + GPU")
        self.assertGreater(len(data.metrics), 0)
        self.assertFalse(data.source_df.empty)
        self.assertFalse(data.processed_df.empty)

    def test_alumetdata_state_properties(self):
        data = make_alumetdata_stub()

        self.assertEqual(data.pid, 99)
        self.assertEqual(data.device, "CPU")
        self.assertEqual(data.metrics, sorted(processed_rows()["base_metric"].unique().tolist()))
        self.assertEqual(len(filter_by_base_metric(data.processed_df, "mem_total_B")), 1)
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

    def test_alumetdata_canonicalizes_legacy_and_current_memory_names(self):
        csv_body = (
            "metric;resource_kind;resource_id;consumer_kind;consumer_id;__late_attributes;timestamp;value\n"
            "active_kB;local_machine;;;;;2024-01-01T00:00:00;1024.0\n"
            "mem_total_kB;local_machine;;;;;2024-01-01T00:00:00;2048.0\n"
            "cached_B;local_machine;;;;;2024-01-01T00:00:00;512.0\n"
            "memory_usage_B;local_machine;;process;9;;2024-01-01T00:00:00;256.0\n"
            "nvml_gpu_memory_info_B;gpu;0;;;;2024-01-01T00:00:00;4096.0\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_measurement_directory(root, csv_body=csv_body)
            data = AlumetData(root)

        self.assertEqual(
            set(data.source_df["metric"]),
            {"active_B", "mem_total_B", "cached_B", "memory_usage_B", "nvml_gpu_memory_info_B"},
        )
        self.assertEqual(
            data.source_df.loc[data.source_df["metric"] == "active_B", "value"].iloc[0],
            1024.0,
        )
        memory = filter_time_series_category(data.processed_df, "memory")
        self.assertEqual(
            set(memory["base_metric"]),
            {"active_B", "mem_total_B", "cached_B", "memory_usage_B", "nvml_gpu_memory_info_B"},
        )

    def test_parse_timestamp_invalid(self):
        with self.assertRaisesRegex(ValueError, "Invalid --start-time"):
            parse_timestamp("not-a-date", "--start-time")

    def test_parse_timestamp_valid(self):
        ts = parse_timestamp("2024-01-01T00:00:01", "--start-time")
        self.assertEqual(ts, pd.Timestamp("2024-01-01T00:00:01"))


if __name__ == "__main__":
    unittest.main()
