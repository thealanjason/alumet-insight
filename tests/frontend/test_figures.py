import unittest

import pandas as pd

from backend.counterdiff import expand_counterdiff_rows
from frontend.figures import create_all_timeseries_plots, build_metric_trace_configs


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
        stem, peak = build_metric_trace_configs(
            df,
            "rapl_consumed_energy_J_R_pkg_0_C__A_",
            color="blue",
            name="energy",
        )
        self.assertEqual(stem["y"], [0.0, 7.0, 0.0, None])
        self.assertEqual(stem["mode"], "lines")
        self.assertEqual(stem["hoverinfo"], "none")
        self.assertFalse(stem["connectgaps"])
        self.assertFalse(stem["showlegend"])
        self.assertNotIn("fill", stem)
        self.assertEqual(peak["y"], [7.0])
        self.assertEqual(peak["mode"], "markers")
        self.assertEqual(peak["marker"]["size"], 6)
        self.assertNotEqual(peak.get("hoverinfo"), "none")

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
        stem, peak = figure.data[0], figure.data[1]

        self.assertEqual(list(stem.y), [0.0, 7.0, 0.0, None])
        self.assertIsNone(stem.x[3])
        self.assertEqual(stem.mode, "lines")
        self.assertEqual(stem.hoverinfo, "none")
        self.assertFalse(stem.connectgaps)
        self.assertEqual(list(peak.y), [7.0])
        self.assertEqual(peak.mode, "markers")
        self.assertEqual(peak.marker.size, 6)

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

        figure = create_all_timeseries_plots(df)
        stem, peak = figure.data[0], figure.data[1]

        self.assertEqual(list(stem.y), [0.0, 0.0, 0.0, None])
        self.assertEqual(list(peak.y), [0.0])
        self.assertEqual(peak.marker.size, 6)

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

    def test_running_total_plots_as_line_with_derived_title(self):
        timestamps = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"])
        df = pd.DataFrame(
            {
                "metric_id": ["attributed_energy_cpu_cumulative_J_R_cpu_0_C_process_1_A_"] * 2,
                "base_metric": ["attributed_energy_cpu_cumulative_J"] * 2,
                "timestamp": timestamps,
                "value": [2.0, 5.0],
                "metric_origin": ["derived", "derived"],
                "point_role": ["observed", "observed"],
            }
        )
        figure = create_all_timeseries_plots(df, category="energy")
        trace = figure.data[0]
        self.assertEqual(list(trace.y), [2.0, 5.0])
        self.assertIn("lines", trace.mode)
        title = figure.layout.annotations[0].text
        self.assertIn("(derived)", title)
        self.assertIn("color:", title)

    def test_stacked_subplots_keep_a_readable_title_gap(self):
        timestamps = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"])
        df = pd.DataFrame(
            {
                "metric_id": ["a_R_x_C__A_"] * 2 + ["b_R_x_C__A_"] * 2,
                "base_metric": ["a"] * 2 + ["b"] * 2,
                "timestamp": list(timestamps) * 2,
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        figure = create_all_timeseries_plots(df, category="energy")
        # 175px per plot + 64px title/tick gap + 36+36 margins
        self.assertEqual(figure.layout.height, 175 * 2 + 64 + 72)


if __name__ == "__main__":
    unittest.main()
