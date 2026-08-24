import unittest

import pandas as pd

from backend.counterdiff import expand_counterdiff_rows
from frontend.figures import create_all_timeseries_plots, build_metric_trace_config


class TimeseriesFigureTests(unittest.TestCase):
    def test_build_metric_trace_config_counterdiff_spikes(self):
        df = expand_counterdiff_rows(
            pd.DataFrame(
                {
                    "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"],
                    "base_metric": ["rapl_consumed_energy_J"],
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "value": [7.0],
                }
            )
        )
        config = build_metric_trace_config(
            df,
            "rapl_consumed_energy_J_R_pkg_0_C__A_",
            color="blue",
            name="energy",
        )
        self.assertEqual(config["y"], [0.0, 7.0, 0.0, None])
        self.assertEqual(config["mode"], "lines+markers")
        self.assertEqual(config["marker"]["size"], [0, 6, 0, 0])
        self.assertFalse(config["connectgaps"])
        self.assertNotIn("fill", config)

    def test_counterdiff_trace_uses_isolated_spikes(self):
        df = expand_counterdiff_rows(
            pd.DataFrame(
                {
                    "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"],
                    "base_metric": ["rapl_consumed_energy_J"],
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "value": [7.0],
                }
            )
        )

        figure = create_all_timeseries_plots(df, category="energy")
        trace = figure.data[0]

        self.assertEqual(list(trace.y), [0.0, 7.0, 0.0, None])
        self.assertIsNone(trace.x[3])
        self.assertEqual(trace.mode, "lines+markers")
        self.assertEqual(list(trace.marker.size), [0, 6, 0, 0])
        self.assertFalse(trace.connectgaps)

    def test_zero_counterdiff_sample_remains_visible_as_observed_marker(self):
        df = expand_counterdiff_rows(
            pd.DataFrame(
                {
                    "metric_id": ["kernel_cpu_time_ms_R_cpu_core_1_C__A_"],
                    "base_metric": ["kernel_cpu_time_ms"],
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "value": [0.0],
                }
            )
        )

        trace = create_all_timeseries_plots(df).data[0]

        self.assertEqual(list(trace.y), [0.0, 0.0, 0.0, None])
        self.assertEqual(list(trace.marker.size), [0, 6, 0, 0])

    def test_derived_power_trace_uses_explicit_interval_start(self):
        interval_start = pd.Timestamp("2024-01-01")
        interval_end = pd.Timestamp("2024-01-01 00:00:02")
        df = pd.DataFrame(
            {
                "metric_id": ["rapl_average_power_W_R_pkg_0_C__A_"],
                "base_metric": ["rapl_average_power_W"],
                "timestamp": [interval_end],
                "interval_start": [interval_start],
                "value": [4.0],
                "point_role": ["observed"],
                "point_order": pd.array([0], dtype="Int64"),
                "sample_id": pd.array([0], dtype="Int64"),
            }
        )

        figure = create_all_timeseries_plots(df, category="power")
        trace = figure.data[0]

        self.assertEqual(list(trace.x), [interval_start, interval_end])
        self.assertEqual(list(trace.y), [4.0, 4.0])


if __name__ == "__main__":
    unittest.main()
