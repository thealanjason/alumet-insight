import base64
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.counterdiff import expand_counterdiff_rows
from backend.data import finalize_processed_dataframe
from frontend.panes.process_specific import (
    apply_shared_xrange_to_grid_plots,
    cascade_filter_options,
    filter_single_series,
    grid_message_figure,
    normalize_filter_columns,
    prepare_download_df,
    unique_nonempty,
    update_grid_plot_match,
)
from frontend.style import GRID_DATA_MARGIN, GRID_GRAPH_CONFIG, GRID_PLACEHOLDER_MARGIN, GRID_YAXIS_LEFT_MARGIN


class ProcessSpecificTests(unittest.TestCase):
    def test_unique_nonempty_and_normalize_filter_columns(self):
        series = pd.Series(["cpu", "", None, "gpu", "cpu"])
        self.assertEqual(unique_nonempty(series), ["cpu", "gpu"])
        self.assertEqual(unique_nonempty(pd.Series([None, "", " "])), [])

        normed = normalize_filter_columns(
            pd.DataFrame(
                {
                    "resource_kind": ["cpu"],
                    "resource_id": [None],
                    "consumer_kind": ["process"],
                    "consumer_id": ["10"],
                    "__late_attributes": ["user"],
                }
            )
        )
        self.assertEqual(normed.loc[0, "rk"], "cpu")
        self.assertEqual(normed.loc[0, "rid"], "")

    def test_cascade_filter_options_and_series_filtering(self):
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

        cascade = cascade_filter_options(normed, "cpu", None, "process", "10", None)
        self.assertEqual(cascade["rk"]["effective"], "cpu")
        self.assertIn("0", cascade["rid"]["options"])

        reset = cascade_filter_options(
            normed,
            "cpu",
            "1",
            "process",
            "10",
            "system",
            triggered_id="resource-kind-dropdown",
        )
        self.assertIsNone(reset["rid"]["effective"])

        filtered, _ = filter_single_series(normed, "cpu", "1", "process", "10", "system")
        self.assertEqual(len(filtered), 1)

    def test_prepare_download_df_from_processed_schema(self):
        df = expand_counterdiff_rows(
            pd.DataFrame(
                {
                    "metric_id": [
                        "cpu_percent_R_cpu_0_C_process_10_A_",
                        "cpu_percent_R_cpu_0_C_process_10_A_",
                        "cpu_percent_R_cpu_1_C_process_10_A_",
                    ],
                    "base_metric": ["cpu_percent", "cpu_percent", "cpu_percent"],
                    "metric": ["cpu_percent", "cpu_percent", "cpu_percent"],
                    "timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
                    "value": [1, 2, 3],
                    "resource_kind": ["cpu", "cpu", "cpu"],
                    "resource_id": ["0", "0", "1"],
                    "consumer_kind": ["process", "process", "process"],
                    "consumer_id": ["10", "10", "10"],
                    "__late_attributes": ["", "", ""],
                }
            )
        )

        out = prepare_download_df(
            df,
            "cpu_percent",
            "cpu",
            "0",
            "process",
            "10",
            None,
            pd.Timestamp("2024-01-01 00:00:01"),
            pd.Timestamp("2024-01-01 00:00:02"),
        )

        self.assertEqual(out["value"].tolist(), [2])
        self.assertNotIn("point_role", out.columns)
        self.assertNotIn("unit", out.columns)
        self.assertTrue(prepare_download_df(df, "missing", None, None, None, None, None).empty)

    def test_cascade_resets_dependent_filters_on_consumer_kind_change(self):
        df = pd.DataFrame(
            {
                "resource_kind": ["cpu", "cpu"],
                "resource_id": ["0", "1"],
                "consumer_kind": ["process", "host"],
                "consumer_id": ["10", ""],
                "__late_attributes": ["", ""],
            }
        )
        normed = normalize_filter_columns(df)

        reset = cascade_filter_options(
            normed,
            "cpu",
            "0",
            "process",
            "10",
            None,
            triggered_id="consumer-kind-dropdown",
        )
        self.assertIsNone(reset["cid"]["effective"])

    def test_prepare_download_df_includes_late_attribute_filter_and_columns(self):
        df = expand_counterdiff_rows(
            pd.DataFrame(
                {
                    "metric_id": [
                        "cpu_percent_R_cpu_0_C_process_10_A_user",
                        "cpu_percent_R_cpu_0_C_process_10_A_system",
                    ],
                    "base_metric": ["cpu_percent", "cpu_percent"],
                    "metric": ["cpu_percent", "cpu_percent"],
                    "timestamp": pd.date_range("2024-01-01", periods=2, freq="s"),
                    "value": [1, 2],
                    "resource_kind": ["cpu", "cpu"],
                    "resource_id": ["0", "0"],
                    "consumer_kind": ["process", "process"],
                    "consumer_id": ["10", "10"],
                    "__late_attributes": ["user", "system"],
                }
            )
        )

        out = prepare_download_df(df, "cpu_percent", "cpu", "0", "process", "10", "user")
        self.assertEqual(out["value"].tolist(), [1])
        self.assertIn("__late_attributes", out.columns)

    def test_process_graph_uses_unambiguous_double_click_reset(self):
        self.assertEqual(GRID_GRAPH_CONFIG["doubleClick"], "autosize")

    def test_memory_grid_figure_saves_its_original_axis_defaults(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "metric": ["active_kB"] * 3,
                "value": [6.5e9, 6.6e9, 6.7e9],
                "resource_kind": ["local_machine"] * 3,
                "resource_id": ["0"] * 3,
                "consumer_kind": [""] * 3,
                "consumer_id": [""] * 3,
                "__late_attributes": [""] * 3,
            }
        )
        process_range = {
            "start": timestamps[0].isoformat(),
            "end": timestamps[-1].isoformat(),
        }

        figure = update_grid_plot_match(
            "active_kB",
            "local_machine",
            "0",
            None,
            None,
            None,
            False,
            df.to_dict("records"),
            process_range,
            {"index": "0-0"},
        )

        defaults = figure.layout.meta["axis_defaults"]
        self.assertTrue(figure.layout.meta["is_memory"])
        self.assertEqual(list(figure.layout.xaxis.range), defaults["xaxis"]["range"])
        self.assertFalse(figure.layout.xaxis.autorange)
        self.assertEqual(list(figure.layout.yaxis.range), defaults["yaxis"]["range"])
        self.assertFalse(figure.layout.yaxis.autorange)
        self.assertEqual(list(figure.layout.yaxis.ticktext), defaults["yaxis"]["ticktext"])
        self.assertEqual(figure.layout.margin.l, GRID_YAXIS_LEFT_MARGIN)
        self.assertFalse(figure.layout.yaxis.automargin)
        self.assertFalse(figure.layout.xaxis.automargin)

    def test_grid_plots_share_the_same_left_margin(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        process_range = {
            "start": timestamps[0].isoformat(),
            "end": timestamps[-1].isoformat(),
        }
        memory = pd.DataFrame(
            {
                "timestamp": timestamps,
                "metric": ["active_kB"] * 3,
                "value": [6.5e9, 6.6e9, 6.7e9],
                "resource_kind": ["local_machine"] * 3,
                "resource_id": ["0"] * 3,
                "consumer_kind": [""] * 3,
                "consumer_id": [""] * 3,
                "__late_attributes": [""] * 3,
            }
        )
        energy = pd.DataFrame(
            {
                "timestamp": timestamps,
                "metric": ["attributed_energy_J"] * 3,
                "value": [1.0, 2.0, 3.0],
                "resource_kind": ["cpu"] * 3,
                "resource_id": ["0"] * 3,
                "consumer_kind": ["process"] * 3,
                "consumer_id": ["10"] * 3,
                "__late_attributes": [""] * 3,
            }
        )
        memory_fig = update_grid_plot_match(
            "active_kB",
            "local_machine",
            "0",
            None,
            None,
            None,
            False,
            memory.to_dict("records"),
            process_range,
            {"index": "0-0"},
        )
        energy_fig = update_grid_plot_match(
            "attributed_energy_J",
            "cpu",
            "0",
            "process",
            "10",
            None,
            False,
            energy.to_dict("records"),
            process_range,
            {"index": "0-1"},
        )
        empty_fig = grid_message_figure(go.Figure(), "Select a metric", False)

        self.assertEqual(memory_fig.layout.margin.l, energy_fig.layout.margin.l)
        self.assertEqual(memory_fig.layout.margin.l, empty_fig.layout.margin.l)
        self.assertEqual(memory_fig.layout.margin.l, GRID_DATA_MARGIN["l"])
        self.assertFalse(energy_fig.layout.yaxis.automargin)
        self.assertFalse(empty_fig.layout.yaxis.automargin)
        self.assertFalse(bool(energy_fig.layout.yaxis.tickformat))
        self.assertEqual(energy_fig.layout.margin.b, GRID_DATA_MARGIN["b"])

    def test_grid_color_follows_metric(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        df = pd.DataFrame(
            {
                "timestamp": list(timestamps) * 2,
                "metric": ["attributed_energy_J"] * 3 + ["rapl_consumption_J"] * 3,
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "resource_kind": ["cpu"] * 6,
                "resource_id": ["0"] * 6,
                "consumer_kind": ["process"] * 6,
                "consumer_id": ["10"] * 6,
                "__late_attributes": [""] * 6,
            }
        )
        process_range = {
            "start": timestamps[0].isoformat(),
            "end": timestamps[-1].isoformat(),
        }
        kwargs = dict(
            rk="cpu",
            rid="0",
            ck="process",
            cid="10",
            la=None,
            use_light_mode=False,
            processed_df_data=df.to_dict("records"),
            process_time_range=process_range,
        )

        energy_a = update_grid_plot_match(metric="attributed_energy_J", my_id={"index": "0-0"}, **kwargs)
        energy_b = update_grid_plot_match(metric="attributed_energy_J", my_id={"index": "1-1"}, **kwargs)
        rapl = update_grid_plot_match(metric="rapl_consumption_J", my_id={"index": "0-1"}, **kwargs)

        self.assertEqual(energy_a.data[0].line.color, energy_b.data[0].line.color)
        self.assertEqual(energy_a.data[1].marker.color, energy_a.data[0].line.color)
        self.assertNotEqual(energy_a.data[0].line.color, rapl.data[0].line.color)

    def test_grid_reset_restores_each_figure_axis_defaults(self):
        memory_figure = {
            "layout": {
                "xaxis": {"range": ["zoom-start", "zoom-end"], "autorange": False},
                "yaxis": {
                    "range": [0, 7e9],
                    "autorange": True,
                    "tickvals": [0],
                    "ticktext": ["0 B"],
                },
                "meta": {
                    "axis_defaults": {
                        "xaxis": {
                            "range": ["process-start", "process-end"],
                            "autorange": False,
                        },
                        "yaxis": {
                            "range": [6.5e9, 6.8e9],
                            "autorange": False,
                            "tickvals": [6.5e9, 6.8e9],
                            "ticktext": ["6.05 GB", "6.33 GB"],
                        },
                    }
                },
            }
        }
        energy_figure = {
            "data": [
                {
                    "x": ["2024-01-01T00:00:00", "2024-01-01T00:00:01", "2024-01-01T00:00:02"],
                    "y": [10.0, 20.0, 100.0],
                }
            ],
            "layout": {
                "xaxis": {"range": ["zoom-start", "zoom-end"], "autorange": False},
                "yaxis": {"range": [5, 110], "autorange": False},
                "meta": {
                    "is_memory": False,
                    "axis_defaults": {
                        "xaxis": {
                            "range": ["process-start", "process-end"],
                            "autorange": False,
                        },
                        "yaxis": {"range": [1.0, 109.0], "autorange": False},
                    },
                },
            },
        }

        updated = apply_shared_xrange_to_grid_plots(
            {"mode": "reset", "revision": 1},
            [memory_figure, energy_figure],
        )

        for figure in updated:
            self.assertEqual(
                figure["layout"]["xaxis"]["range"],
                ["process-start", "process-end"],
            )
            self.assertFalse(figure["layout"]["xaxis"]["autorange"])

        memory_yaxis = updated[0]["layout"]["yaxis"]
        self.assertEqual(memory_yaxis["range"], [6.5e9, 6.8e9])
        self.assertFalse(memory_yaxis["autorange"])
        self.assertEqual(memory_yaxis["ticktext"], ["6.05 GB", "6.33 GB"])

        energy_yaxis = updated[1]["layout"]["yaxis"]
        self.assertEqual(energy_yaxis["range"], [1.0, 109.0])
        self.assertFalse(energy_yaxis["autorange"])
        self.assertEqual(updated[0]["layout"]["margin"]["l"], GRID_DATA_MARGIN["l"])
        self.assertEqual(updated[0]["layout"]["margin"]["t"], GRID_DATA_MARGIN["t"])

    def test_grid_zoom_keeps_placeholder_top_margin(self):
        placeholder = grid_message_figure(go.Figure(), "Select a metric", False)
        data_figure = {
            "data": [{"x": ["2024-01-01T00:00:00"], "y": [1.0]}],
            "layout": {
                "xaxis": {"range": ["2024-01-01T00:00:00", "2024-01-01T00:00:01"], "autorange": False},
                "yaxis": {"range": [0.0, 2.0], "autorange": False},
                "meta": {
                    "axis_defaults": {
                        "xaxis": {
                            "range": ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
                            "autorange": False,
                        },
                        "yaxis": {"range": [0.0, 2.0], "autorange": False},
                    }
                },
            },
        }
        updated = apply_shared_xrange_to_grid_plots(
            {"mode": "zoom", "x0": "2024-01-01T00:00:00", "x1": "2024-01-01T00:00:01", "revision": 1},
            [placeholder.to_plotly_json(), data_figure],
        )
        self.assertEqual(updated[0]["layout"]["margin"]["t"], GRID_PLACEHOLDER_MARGIN["t"])
        self.assertEqual(updated[0]["layout"]["margin"]["l"], GRID_PLACEHOLDER_MARGIN["l"])
        self.assertEqual(updated[1]["layout"]["margin"]["t"], GRID_DATA_MARGIN["t"])

    def test_grid_zoom_scales_yaxis_to_visible_points(self):
        figure = {
            "data": [
                {
                    "x": ["2024-01-01T00:00:00", "2024-01-01T00:00:01", "2024-01-01T00:00:02"],
                    "y": [10.0, 20.0, 100.0],
                }
            ],
            "layout": {
                "xaxis": {"range": ["2024-01-01T00:00:00", "2024-01-01T00:00:02"], "autorange": False},
                "yaxis": {"range": [1.0, 109.0], "autorange": False},
                "meta": {
                    "is_memory": False,
                    "axis_defaults": {
                        "xaxis": {
                            "range": ["2024-01-01T00:00:00", "2024-01-01T00:00:02"],
                            "autorange": False,
                        },
                        "yaxis": {"range": [1.0, 109.0], "autorange": False},
                    },
                },
            },
        }

        zoomed = apply_shared_xrange_to_grid_plots(
            {"mode": "zoom", "x0": "2024-01-01T00:00:00", "x1": "2024-01-01T00:00:01", "revision": 1},
            [figure],
        )[0]

        self.assertEqual(
            zoomed["layout"]["xaxis"]["range"],
            ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
        )
        y_min, y_max = zoomed["layout"]["yaxis"]["range"]
        self.assertAlmostEqual(y_min, 9.0)
        self.assertAlmostEqual(y_max, 21.0)
        self.assertFalse(zoomed["layout"]["yaxis"]["autorange"])

        reset = apply_shared_xrange_to_grid_plots(
            {"mode": "reset", "revision": 2},
            [zoomed],
        )[0]
        self.assertEqual(reset["layout"]["yaxis"]["range"], [1.0, 109.0])

    def test_grid_zoom_decodes_plotly_bdata_arrays(self):
        y = np.array([10.0, 20.0, 100.0], dtype="f8")
        x = pd.date_range("2024-01-01", periods=3, freq="s")
        figure = {
            "data": [
                {
                    "x": {
                        "dtype": "i8",
                        "bdata": base64.b64encode(x.asi8.tobytes()).decode("ascii"),
                    },
                    "y": {
                        "dtype": "f8",
                        "bdata": base64.b64encode(y.tobytes()).decode("ascii"),
                    },
                }
            ],
            "layout": {
                "xaxis": {"range": ["2024-01-01T00:00:00", "2024-01-01T00:00:02"], "autorange": False},
                "yaxis": {"range": [1.0, 109.0], "autorange": False},
                "meta": {"is_memory": False, "axis_defaults": {"yaxis": {"range": [1.0, 109.0], "autorange": False}}},
            },
        }

        zoomed = apply_shared_xrange_to_grid_plots(
            {"mode": "zoom", "x0": "2024-01-01T00:00:00", "x1": "2024-01-01T00:00:01", "revision": 1},
            [figure],
        )[0]
        y_min, y_max = zoomed["layout"]["yaxis"]["range"]
        self.assertAlmostEqual(y_min, 9.0)
        self.assertAlmostEqual(y_max, 21.0)

    def test_processed_schema_exposes_synthesized_totals_and_power(self):
        processed = finalize_processed_dataframe(
            pd.DataFrame(
                {
                    "metric_id": [
                        "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_",
                        "attributed_energy_cpu_J_R_cpu_0_C_process_7_A_",
                        "attributed_energy_gpu_J_R_gpu_0_C_process_7_A_",
                        "attributed_energy_gpu_J_R_gpu_0_C_process_7_A_",
                    ],
                    "base_metric": [
                        "attributed_energy_cpu_J",
                        "attributed_energy_cpu_J",
                        "attributed_energy_gpu_J",
                        "attributed_energy_gpu_J",
                    ],
                    "metric": [
                        "attributed_energy_cpu_J",
                        "attributed_energy_cpu_J",
                        "attributed_energy_gpu_J",
                        "attributed_energy_gpu_J",
                    ],
                    "timestamp": pd.to_datetime(
                        [
                            "2024-01-01 00:00:01",
                            "2024-01-01 00:00:03",
                            "2024-01-01 00:00:01",
                            "2024-01-01 00:00:03",
                        ]
                    ),
                    "value": [1.0, 2.0, 3.0, 5.0],
                    "resource_kind": ["cpu", "cpu", "gpu", "gpu"],
                    "resource_id": ["0", "0", "0", "0"],
                    "consumer_kind": ["process", "process", "process", "process"],
                    "consumer_id": ["7", "7", "7", "7"],
                    "__late_attributes": ["", "", "", ""],
                    "metric_origin": ["measured"] * 4,
                }
            )
        )
        total = processed[processed["base_metric"] == "attributed_energy_total_J"]
        self.assertFalse(total.empty)
        cascade = cascade_filter_options(
            normalize_filter_columns(total),
            "total",
            "",
            "process",
            "7",
            None,
        )
        self.assertEqual(cascade["ck"]["effective"], "process")

        downloaded = prepare_download_df(
            processed,
            "attributed_energy_total_J",
            "total",
            "",
            "process",
            "7",
            None,
        )
        self.assertFalse(downloaded.empty)
        self.assertNotIn("point_role", downloaded.columns)
        self.assertNotIn("interval_start", downloaded.columns)


if __name__ == "__main__":
    unittest.main()
