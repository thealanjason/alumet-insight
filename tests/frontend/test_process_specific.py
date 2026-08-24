import base64
import unittest

import numpy as np
import pandas as pd

from frontend.panes.process_specific import (
    apply_shared_xrange_to_grid_plots,
    cascade_filter_options,
    filter_single_series,
    normalize_filter_columns,
    prepare_download_df,
    unique_nonempty,
    update_grid_plot_match,
)
from frontend.style import GRID_GRAPH_CONFIG


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

        reset = cascade_filter_options(normed, "cpu", "1", "process", "10", "system", triggered_id="resource-kind-dropdown")
        self.assertIsNone(reset["rid"]["effective"])

        filtered, _ = filter_single_series(normed, "cpu", "1", "process", "10", "system")
        self.assertEqual(len(filtered), 1)

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

        reset = cascade_filter_options(normed, "cpu", "0", "process", "10", None, triggered_id="consumer-kind-dropdown")
        self.assertIsNone(reset["cid"]["effective"])

    def test_prepare_download_df_includes_late_attribute_filter_and_columns(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="s"),
                "metric": ["m", "m"],
                "value": [1, 2],
                "resource_kind": ["cpu", "cpu"],
                "resource_id": ["0", "0"],
                "consumer_kind": ["process", "process"],
                "consumer_id": ["10", "10"],
                "__late_attributes": ["user", "system"],
            }
        )

        out = prepare_download_df(df, "m", "cpu", "0", "process", "10", "user")
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


if __name__ == "__main__":
    unittest.main()
