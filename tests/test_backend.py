import tempfile
import unittest
import pandas as pd
from pathlib import Path

from backend.categories import available_category_values, available_cpu_cores, filter_time_series_category
from backend.comparative import (
    align_xy_metrics,
    comparative_metric_ids,
    filter_process_metric_ids,
    pick_xy_values,
    prepare_xy_download,
)
from backend.data import (
    AlumetData,
    extract_pid_from_content,
    filter_to_time_range,
    get_process_time_range_from_df,
    is_cpu_from_content,
    is_gpu_from_content,
    preprocess_dataframe_for_visualization,
    synthesize_attributed_energy_total,
)
from backend.metrics import (
    format_bytes_ticklabel,
    get_bytes_tickvals_ticktext,
    get_metric_unit,
    is_cumulative_metric,
    is_memory_metric,
    metric_id_is_process_consumer,
)
from backend.process_specific import (
    cascade_filter_options,
    filter_single_series,
    normalize_filter_columns,
    prepare_download_df,
    safe_metric_filename,
    unique_nonempty,
)
from backend.timeseries import (
    align_xrange_tz,
    category_yaxis_label,
    compute_yaxis_ranges,
    is_yaxis_shareable,
)
from backend.visualization.interactive import create_all_timeseries_plots, get_color_palette, metric_id_to_plot_label
from backend.visualization.static import export_category_figures


def processed_rows() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=4, freq="s")
    return pd.DataFrame(
        {
            "metric_id": [
                "attributed_energy_J_R_local_machine__C_process_123_A_",
                "nvml_instant_power_mW_R_gpu_0_C_process_123_A_",
                "mem_total_kB_R_local_machine__C__A_",
                "kernel_cpu_time_ms_R_cpu_core_0.0_C_process_123_A_",
            ],
            "base_metric": [
                "attributed_energy_J",
                "nvml_instant_power_mW",
                "mem_total_kB",
                "kernel_cpu_time_ms",
            ],
            "timestamp": ts,
            "value": [10.0, 2000.0, 1024.0, 5.0],
        }
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

    def test_filter_to_time_range_handles_bounds_and_missing_bounds(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="s"), "value": [1, 2, 3]})

        filtered = filter_to_time_range(df, pd.Timestamp("2024-01-01 00:00:01"), pd.Timestamp("2024-01-01 00:00:02"))

        self.assertEqual(filtered["value"].tolist(), [2, 3])
        self.assertTrue(filter_to_time_range(df, None, None).empty)
        self.assertEqual(filter_to_time_range(df, None, None, require_bounds=False)["value"].tolist(), [1, 2, 3])

    def test_log_helpers_detect_pid_and_device_plugins(self):
        content = "starting alumet\npid 12345\nloaded nvml and rapl plugins"

        self.assertEqual(extract_pid_from_content(content), 12345)
        self.assertTrue(is_gpu_from_content(content))
        self.assertTrue(is_cpu_from_content(content))

    def test_synthesize_attributed_energy_total_adds_gpu_and_total_rows(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "attributed_energy_cpu_J_R_cpu_0_C_process_123_A_",
                    "attributed_energy_gpu_J_R_gpu_0_C_process_123_A_",
                    "attributed_energy_gpu_J_R_gpu_1_C_process_123_A_",
                ],
                "base_metric": [
                    "attributed_energy_cpu_J",
                    "attributed_energy_gpu_J",
                    "attributed_energy_gpu_J",
                ],
                "timestamp": [pd.Timestamp("2024-01-01")] * 3,
                "value": [1.0, 2.0, 3.0],
            }
        )

        synthetic = synthesize_attributed_energy_total(df)

        self.assertIn("attributed_energy_gpu_total_J", set(synthetic["base_metric"]))
        self.assertIn("attributed_energy_total_J", set(synthetic["base_metric"]))
        total = synthetic.loc[synthetic["base_metric"] == "attributed_energy_total_J", "value"].iloc[0]
        self.assertEqual(total, 6.0)

    def test_alumetdata_properties_and_csv_export_can_use_in_memory_frames(self):
        data = AlumetData.__new__(AlumetData)
        data.directory = Path("/measurements/run_a")
        data._csv_path = Path("run.csv")
        data._log_path = Path("run.log")
        data._log_content = "pid 99\nrapl"
        data._df_raw = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="s"),
                "consumer_kind": ["process", "process"],
                "value": [1.0, 2.0],
            }
        )
        data._df_processed = processed_rows()

        self.assertEqual(data.pid, 99)
        self.assertEqual(data.device, "CPU")
        self.assertIn("Number of metrics", data.summary())

        with tempfile.TemporaryDirectory() as tmp:
            created = data.export_csvs(Path(tmp), category="power", process_specific=True)
            self.assertEqual(len(created), 1)
            self.assertTrue(created[0].exists())


class CategoryAndMetricTests(unittest.TestCase):
    def test_category_availability_and_filters(self):
        df = processed_rows()

        self.assertEqual(available_category_values(df), ["energy", "power", "memory", "kernel_cpu_time"])
        self.assertEqual(available_cpu_cores(df), ["0"])
        self.assertEqual(filter_time_series_category(df, "power")["value"].tolist(), [2.0])
        self.assertEqual(len(filter_time_series_category(df, "kernel_cpu_time", selected_cpu_core="0")), 1)

    def test_metric_helpers(self):
        self.assertTrue(metric_id_is_process_consumer("metric_R_x_C_process_123_A_"))
        self.assertTrue(is_cumulative_metric("rapl_consumed_energy_J"))
        self.assertFalse(is_cumulative_metric("cpu_percent"))
        self.assertEqual(get_metric_unit("nvml_instant_power_mW"), "mW")
        self.assertTrue(is_memory_metric("mem_total_kB"))
        self.assertEqual(format_bytes_ticklabel(2048), "2.0 KB")
        tickvals, ticktext = get_bytes_tickvals_ticktext(0, 2048, num_ticks=3)
        self.assertEqual(len(tickvals), len(ticktext))


class TimeSeriesTests(unittest.TestCase):
    def test_time_series_axis_helpers(self):
        self.assertTrue(is_yaxis_shareable("energy"))
        self.assertFalse(is_yaxis_shareable("miscellaneous"))
        self.assertEqual(category_yaxis_label("kernel_cpu_time"), "Value (ms)")

        tz = pd.Timestamp("2024-01-01", tz="UTC").tz
        x_min, x_max = align_xrange_tz(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), tz)
        self.assertIsNotNone(x_min.tz)
        self.assertIsNotNone(x_max.tz)

        df = pd.DataFrame(
            {
                "metric_id": ["a", "a", "b", "b"],
                "value": [1.0, 3.0, 2.0, 4.0],
            }
        )
        ranges = compute_yaxis_ranges(df, ["a", "b"], share_yaxis=True, is_memory=False)
        self.assertEqual(set(ranges), {"yaxis", "yaxis2"})
        self.assertEqual(ranges["yaxis"]["range"], ranges["yaxis2"]["range"])


class ProcessSpecificTests(unittest.TestCase):
    def test_filter_options_and_series_filtering(self):
        df = pd.DataFrame(
            {
                "resource_kind": ["cpu", "cpu", "gpu"],
                "resource_id": ["0", "1", "0"],
                "consumer_kind": ["process", "process", ""],
                "consumer_id": ["10", "10", ""],
                "__late_attributes": ["user", "system", ""],
            }
        )
        normed = normalize_filter_columns(df)

        self.assertEqual(unique_nonempty(normed["resource_id"]), ["0", "1"])
        cascade = cascade_filter_options(normed, "cpu", None, "process", "10", None)
        self.assertEqual(cascade["rk"]["effective"], "cpu")
        filtered, _ = filter_single_series(normed, "cpu", "1", "process", "10", "system")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(safe_metric_filename("a/b c"), "a_b_c")

    def test_prepare_download_df_filters_values_and_time_range(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
                "metric": ["m", "m", "m"],
                "value": [1, 2, 3],
                "resource_kind": ["cpu", "cpu", "cpu"],
                "resource_id": ["0", "0", "1"],
                "consumer_kind": ["process", "process", "process"],
                "consumer_id": ["10", "10", "10"],
                "__late_attributes": ["", "", ""],
            }
        )

        out = prepare_download_df(
            df,
            "m",
            "cpu",
            "0",
            "process",
            "10",
            None,
            pd.Timestamp("2024-01-01 00:00:01"),
            pd.Timestamp("2024-01-01 00:00:02"),
        )

        self.assertEqual(out["value"].tolist(), [2])


class ComparativeTests(unittest.TestCase):
    def test_comparative_metric_selection_and_alignment(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="s").tolist() * 2,
                "metric_id": ["x_R_a_C_process_1_A_"] * 3 + ["y_R_a_C_process_1_A_"] * 3,
                "value": [1, 2, 3, 10, 20, 30],
            }
        )
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-01 00:00:02")

        ids = comparative_metric_ids(df, start, end)
        self.assertEqual(ids, ["x_R_a_C_process_1_A_", "y_R_a_C_process_1_A_"])
        self.assertEqual(filter_process_metric_ids(ids, process_only=True), ids)
        self.assertEqual(pick_xy_values(ids, None, None), (ids[0], ids[1]))

        aligned = align_xy_metrics(df, ids[0], ids[1], start, end)
        self.assertEqual(aligned[["x", "y"]].values.tolist(), [[1, 10], [2, 20], [3, 30]])
        exported, filename = prepare_xy_download(aligned, ids[0], ids[1])
        self.assertIn(ids[0], exported.columns)
        self.assertTrue(filename.startswith("xy_"))


class VisualizationTests(unittest.TestCase):
    def test_interactive_visualization_helpers(self):
        self.assertIn("C=process", metric_id_to_plot_label("kernel_cpu_time_ms_R_cpu_core_0_C_process_123_A_"))
        self.assertEqual(len(get_color_palette(12)), 12)

        fig = create_all_timeseries_plots(processed_rows().head(2), category="energy")
        self.assertGreaterEqual(len(fig.data), 1)

    def test_static_export_writes_one_file_per_metric_id(self):
        df = processed_rows().head(2)
        with tempfile.TemporaryDirectory() as tmp:
            created = export_category_figures(df, Path(tmp), "energy", figure_format="png", dpi=50)
            self.assertEqual(len(created), 2)
            self.assertTrue(all(path.exists() for path in created))


if __name__ == "__main__":
    unittest.main()
