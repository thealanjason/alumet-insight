import unittest

import pandas as pd

from backend.metrics import filter_process_metric_ids
from backend.transforms import align_xy_metrics, comparative_metric_ids
from frontend.panes.comparative import (
    comparative_timeseries_trace_config,
    pick_xy_values,
    prepare_xy_download,
    update_process_xy_plot,
)


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
        self.assertIn("x_unit", exported.columns)
        self.assertIn("y_unit", exported.columns)
        self.assertTrue(filename.startswith("xy_"))

    def test_comparative_helpers_handle_empty_and_filtered_inputs(self):
        self.assertEqual(comparative_metric_ids(pd.DataFrame(), None, None), [])
        self.assertEqual(
            comparative_metric_ids(
                pd.DataFrame({"metric_id": ["a"], "timestamp": [pd.Timestamp("2024-01-01")], "value": [1]}),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-02"),
            ),
            [],
        )

        ids = ["host_R_a_C_host_1_A_", "proc_R_a_C_process_1_A_"]
        self.assertEqual(filter_process_metric_ids(ids, process_only=False), ids)
        self.assertEqual(filter_process_metric_ids(ids, process_only=True), [ids[1]])
        self.assertEqual(pick_xy_values(["only"], None, None), ("only", "only"))
        self.assertEqual(pick_xy_values(["a", "b"], "b", "a"), ("b", "a"))

        empty_aligned = align_xy_metrics(
            pd.DataFrame(columns=["metric_id", "timestamp", "value"]),
            "a",
            "b",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02"),
        )
        self.assertTrue(empty_aligned.empty)

    def test_prepare_xy_download_sanitizes_filename(self):
        aligned = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "x": [1.0], "y": [2.0]})
        _, filename = prepare_xy_download(aligned, "bad/id", "also bad")
        self.assertNotIn("/", filename)

    def test_dual_timeseries_counterdiff_uses_spikes_and_observed_markers(self):
        metric_id = "kernel_cpu_time_ms_R_cpu_core_1_C_process_4_A_"
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"]),
                "value": [0.0, 3.0],
            }
        )

        trace = comparative_timeseries_trace_config(
            df,
            metric_id,
            "kernel_cpu_time_ms",
            "blue",
            "y1",
        )

        self.assertEqual(trace["mode"], "lines+markers")
        self.assertEqual(
            trace["y"],
            [0.0, 0.0, 0.0, None, 0.0, 3.0, 0.0, None],
        )
        self.assertEqual(trace["marker"]["size"], [0, 6, 0, 0, 0, 6, 0, 0])
        self.assertFalse(trace["connectgaps"])

    def test_dual_timeseries_derived_power_uses_interval_steps(self):
        metric_id = "rapl_average_power_W_R_pkg_0_C__A_"
        interval_start = pd.Timestamp("2024-01-01")
        interval_end = pd.Timestamp("2024-01-01 00:00:02")
        df = pd.DataFrame(
            {
                "timestamp": [interval_end],
                "interval_start": [interval_start],
                "value": [4.0],
                "point_role": ["observed"],
            }
        )

        trace = comparative_timeseries_trace_config(
            df,
            metric_id,
            "rapl_average_power_W",
            "red",
            "y2",
        )

        self.assertEqual(trace["mode"], "lines")
        self.assertEqual(trace["x"], [interval_start, interval_end])
        self.assertEqual(trace["y"], [4.0, 4.0])
        self.assertFalse(trace["connectgaps"])

    def test_dual_timeseries_gauge_uses_connected_line(self):
        timestamps = pd.date_range("2024-01-01", periods=2, freq="s")
        df = pd.DataFrame({"timestamp": timestamps, "value": [1.0, 2.0]})

        trace = comparative_timeseries_trace_config(
            df,
            "cpu_percent_R_local_machine__C_process_4_A_",
            "cpu_percent",
            "blue",
            "y1",
        )

        self.assertEqual(trace["mode"], "lines")
        self.assertEqual(list(trace["x"]), list(timestamps))
        self.assertEqual(trace["y"].tolist(), [1.0, 2.0])

    def test_dual_timeseries_raw_counter_uses_connected_running_total(self):
        timestamps = pd.date_range("2024-01-01", periods=2, freq="s")
        df = pd.DataFrame({"timestamp": timestamps, "value": [1_000_000, 1_500_000]})

        trace = comparative_timeseries_trace_config(
            df,
            "perf_hardware_INSTRUCTIONS_R_cpu_0_C_process_4_A_",
            "perf_hardware_INSTRUCTIONS",
            "blue",
            "y1",
        )

        self.assertEqual(trace["mode"], "lines")
        self.assertEqual(list(trace["x"]), list(timestamps))
        self.assertEqual(trace["y"].tolist(), [1_000_000, 1_500_000])

    def test_dual_timeseries_keeps_independent_raw_timestamps(self):
        x_metric = "cpu_percent_R_local_machine__C_process_4_A_"
        y_metric = "mem_total_B_R_local_machine__C_process_4_A_"
        x_time = pd.Timestamp("2024-01-01 00:00:00")
        y_time = pd.Timestamp("2024-01-01 00:00:10")
        records = [
            {"timestamp": x_time, "metric_id": x_metric, "value": 1.0},
            {"timestamp": y_time, "metric_id": y_metric, "value": 2.0},
        ]

        figure = update_process_xy_plot(
            x_metric,
            y_metric,
            [],
            False,
            records,
            {
                "start": "2024-01-01 00:00:00",
                "end": "2024-01-01 00:00:10",
            },
        )

        self.assertEqual(list(figure.data[0].x), [x_time])
        self.assertEqual(list(figure.data[1].x), [y_time])


if __name__ == "__main__":
    unittest.main()
