import json
import math
import unittest

import dash
import pandas as pd
import plotly.io as pio

from backend.transforms import align_xrange_tz, compute_yaxis_ranges, filter_to_time_range
from frontend.cache import cache_dataframe, remember_metric_window_index
from frontend.figures import create_all_timeseries_plots
from frontend.panes.timeseries import (
    _metric_window_index,
    _yaxis_updates_for_window,
    patch_timeseries_on_zoom,
    update_yaxis_on_toggle,
    update_yaxis_on_zoom,
)
from tests.fixtures import CPU_ENERGY_ID, GPU_ENERGY_ID, concat_series, series_rows


class TimeseriesZoomTests(unittest.TestCase):
    def test_zoom_then_reset_restores_all_explicit_defaults(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        df = pd.DataFrame(
            {
                "metric_id": ["metric_a"] * 3 + ["metric_b"] * 3,
                "timestamp": list(timestamps) * 2,
                "value": [0.0, 10.0, 100.0, 50.0, 40.0, 30.0],
            }
        )
        cache_id = cache_dataframe(df, prefix="test_timeseries_reset")
        metric_order = ["metric_a", "metric_b"]
        default_x_range = [
            timestamps[0].isoformat(),
            timestamps[-1].isoformat(),
        ]
        store = {
            "cache_id": cache_id,
            "metric_order": metric_order,
            "default_x_range": default_x_range,
            "is_memory_category": False,
        }
        figure = {
            "data": [
                {
                    "x": [stamp.isoformat() for stamp in timestamps],
                    "y": [0.0, 10.0, 100.0],
                    "yaxis": "y",
                    "name": "metric_a",
                },
                {
                    "x": [stamp.isoformat() for stamp in timestamps],
                    "y": [50.0, 40.0, 30.0],
                    "yaxis": "y2",
                    "name": "metric_b",
                },
            ],
            "layout": {
                "xaxis": {"range": default_x_range, "autorange": False},
                "xaxis2": {"range": default_x_range, "autorange": False},
                "yaxis": {"range": [-10, 110], "autorange": False},
                "yaxis2": {"range": [28, 52], "autorange": False},
                "meta": {
                    "axis_defaults": {
                        "xaxis": {"range": default_x_range, "autorange": False},
                        "xaxis2": {"range": default_x_range, "autorange": False},
                        "yaxis": {"range": [-10, 110], "autorange": False},
                        "yaxis2": {"range": [28, 52], "autorange": False},
                    }
                },
            },
        }

        zoomed = update_yaxis_on_zoom(
            {
                "xaxis.range[0]": timestamps[1].isoformat(),
                "xaxis.range[1]": timestamps[2].isoformat(),
            },
            figure,
            store,
            [],
        )
        zoom_x = [timestamps[1].isoformat(), timestamps[2].isoformat()]
        self.assertEqual(zoomed["layout"]["xaxis"]["range"], zoom_x)
        self.assertEqual(zoomed["layout"]["xaxis2"]["range"], zoom_x)
        self.assertEqual(zoomed["layout"]["yaxis"]["range"][0], 1.0)
        self.assertEqual(zoomed["layout"]["yaxis"]["range"][1], 109.0)
        self.assertEqual(zoomed["layout"]["yaxis2"]["range"][0], 29.0)
        self.assertEqual(zoomed["layout"]["yaxis2"]["range"][1], 41.0)

        reset = update_yaxis_on_zoom(
            {"autosize": True},
            zoomed,
            store,
            [],
        )

        for xaxis_key in ("xaxis", "xaxis2"):
            self.assertEqual(reset["layout"][xaxis_key]["range"], default_x_range)
            self.assertFalse(reset["layout"][xaxis_key]["autorange"])
        self.assertEqual(reset["layout"]["yaxis"]["range"], [-10, 110])
        self.assertEqual(reset["layout"]["yaxis2"]["range"], [28, 52])
        self.assertFalse(reset["layout"]["yaxis"]["autorange"])
        self.assertFalse(reset["layout"]["yaxis2"]["autorange"])

    def test_shared_yaxis_uses_one_scale_for_the_zoomed_window(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        df = pd.DataFrame(
            {
                "metric_id": ["metric_a"] * 3 + ["metric_b"] * 3,
                "timestamp": list(timestamps) * 2,
                "value": [0.0, 10.0, 100.0, 50.0, 40.0, 30.0],
            }
        )
        cache_id = cache_dataframe(df, prefix="test_timeseries_shared")
        store = {
            "cache_id": cache_id,
            "metric_order": ["metric_a", "metric_b"],
            "is_memory_category": False,
        }
        figure = {
            "data": [
                {"x": [stamp.isoformat() for stamp in timestamps], "y": [0.0, 10.0, 100.0], "yaxis": "y"},
                {"x": [stamp.isoformat() for stamp in timestamps], "y": [50.0, 40.0, 30.0], "yaxis": "y2"},
            ],
            "layout": {
                "xaxis": {"range": [timestamps[0].isoformat(), timestamps[-1].isoformat()]},
                "yaxis": {"range": [0, 1]},
                "yaxis2": {"range": [0, 1]},
            },
        }
        zoomed = update_yaxis_on_zoom(
            {
                "xaxis2.range[0]": timestamps[1].isoformat(),
                "xaxis2.range[1]": timestamps[2].isoformat(),
            },
            figure,
            store,
            ["shared"],
        )
        self.assertEqual(zoomed["layout"]["yaxis"]["range"], [1.0, 109.0])
        self.assertEqual(zoomed["layout"]["yaxis2"]["range"], [1.0, 109.0])

    def test_box_zoom_refits_every_subplot_from_spike_traces(self):
        times = pd.date_range("2026-08-12 07:38:39.750", periods=6, freq="50ms")
        df = concat_series(
            series_rows(CPU_ENERGY_ID, [12.0, 0.4, 0.2, 0.3, 0.2, 0.1], timestamps=times),
            series_rows(GPU_ENERGY_ID, [1.0, 9.0, 0.5, 0.4, 0.2, 0.1], timestamps=times),
        )
        figure = json.loads(pio.to_json(create_all_timeseries_plots(df, category="energy")))
        cache_id = cache_dataframe(df, prefix="test_timeseries_spike_zoom")
        store = {
            "cache_id": cache_id,
            "metric_order": [CPU_ENERGY_ID, GPU_ENERGY_ID],
            "is_memory_category": False,
        }
        full_cpu = figure["layout"]["yaxis"]["range"][1]
        zoomed = update_yaxis_on_zoom(
            {
                "xaxis.range[0]": times[2].isoformat(),
                "xaxis.range[1]": times[4].isoformat(),
                "yaxis.range[0]": 0.0,
                "yaxis.range[1]": 1.0,
            },
            figure,
            store,
            [],
        )
        self.assertLess(zoomed["layout"]["yaxis"]["range"][1], full_cpu)
        self.assertLess(zoomed["layout"]["yaxis2"]["range"][1], 2.0)
        self.assertGreater(zoomed["layout"]["yaxis2"]["range"][1], 0.4)

        again = update_yaxis_on_zoom(
            {
                "xaxis.range[0]": times[2].isoformat(),
                "xaxis.range[1]": times[4].isoformat(),
            },
            zoomed,
            store,
            [],
        )
        self.assertIs(again, dash.no_update)

    def test_memory_zoom_rewrites_tick_labels_and_double_click_restores_them(self):
        timestamps = pd.date_range("2024-01-01", periods=4, freq="s")
        values = [1_000_000.0, 2_000_000.0, 8_000_000.0, 9_000_000.0]
        df = pd.DataFrame(
            {
                "metric_id": ["memory_bytes"] * 4,
                "timestamp": list(timestamps),
                "value": values,
            }
        )
        cache_id = cache_dataframe(df, prefix="test_timeseries_memory_ticks")
        default_x_range = [timestamps[0].isoformat(), timestamps[-1].isoformat()]
        original_ticks = ["1 MB", "5 MB", "9 MB"]
        store = {
            "cache_id": cache_id,
            "metric_order": ["memory_bytes"],
            "default_x_range": default_x_range,
            "is_memory_category": True,
        }
        figure = {
            "data": [
                {
                    "x": [stamp.isoformat() for stamp in timestamps],
                    "y": values,
                    "yaxis": "y",
                    "name": "memory_bytes",
                }
            ],
            "layout": {
                "xaxis": {"range": default_x_range, "autorange": False},
                "yaxis": {
                    "range": [0, 10_000_000],
                    "autorange": False,
                    "tickvals": [1_000_000, 5_000_000, 9_000_000],
                    "ticktext": original_ticks,
                },
                "meta": {
                    "axis_defaults": {
                        "xaxis": {"range": default_x_range, "autorange": False},
                        "yaxis": {
                            "range": [0, 10_000_000],
                            "autorange": False,
                            "tickvals": [1_000_000, 5_000_000, 9_000_000],
                            "ticktext": original_ticks,
                        },
                    }
                },
            },
        }

        zoomed = update_yaxis_on_zoom(
            {
                "xaxis.range": [timestamps[1].isoformat(), timestamps[2].isoformat()],
            },
            figure,
            store,
            [],
        )
        self.assertNotEqual(zoomed["layout"]["yaxis"]["ticktext"], original_ticks)
        self.assertLess(zoomed["layout"]["yaxis"]["range"][1], 10_000_000)
        self.assertGreater(zoomed["layout"]["yaxis"]["range"][0], 0)

        reset = update_yaxis_on_zoom({"autosize": True}, zoomed, store, [])
        self.assertEqual(reset["layout"]["yaxis"]["ticktext"], original_ticks)
        self.assertEqual(reset["layout"]["yaxis"]["range"], [0, 10_000_000])
        self.assertEqual(reset["layout"]["xaxis"]["range"], default_x_range)

    def test_zoom_patch_sends_axis_ranges_without_trace_data(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        df = pd.DataFrame(
            {
                "metric_id": ["metric_a"] * 3 + ["metric_b"] * 3,
                "timestamp": list(timestamps) * 2,
                "value": [0.0, 10.0, 100.0, 50.0, 40.0, 30.0],
            }
        )
        cache_id = cache_dataframe(df, prefix="test_timeseries_patch")
        default_x_range = [timestamps[0].isoformat(), timestamps[-1].isoformat()]
        store = {
            "cache_id": cache_id,
            "metric_order": ["metric_a", "metric_b"],
            "default_x_range": default_x_range,
            "is_memory_category": False,
            "axis_defaults": {
                "xaxis": {"range": default_x_range, "autorange": False},
                "xaxis2": {"range": default_x_range, "autorange": False},
                "yaxis": {"range": [-10, 110], "autorange": False},
                "yaxis2": {"range": [28, 52], "autorange": False},
            },
        }
        patch, zoom_state = patch_timeseries_on_zoom(
            {
                "xaxis.range[0]": timestamps[1].isoformat(),
                "xaxis.range[1]": timestamps[2].isoformat(),
            },
            store,
            [],
            None,
        )
        operations = patch.to_plotly_json()["operations"]
        locations = [op["location"] for op in operations]
        self.assertIn(["layout", "yaxis", "range"], locations)
        self.assertIn(["layout", "yaxis2", "range"], locations)
        self.assertTrue(all(op["location"][0] == "layout" for op in operations))
        y_range = next(op["params"]["value"] for op in operations if op["location"] == ["layout", "yaxis", "range"])
        self.assertEqual(y_range, [1.0, 109.0])

        again, again_state = patch_timeseries_on_zoom(
            {
                "xaxis.range[0]": timestamps[1].isoformat(),
                "xaxis.range[1]": timestamps[2].isoformat(),
            },
            store,
            [],
            zoom_state,
        )
        self.assertIs(again, dash.no_update)
        self.assertIs(again_state, dash.no_update)

        restored, reset_state = patch_timeseries_on_zoom({"autosize": True}, store, [], zoom_state)
        restored_ranges = {
            tuple(op["location"]): op["params"]["value"]
            for op in restored.to_plotly_json()["operations"]
            if op["operation"] == "Assign" and op["location"][-1] == "range"
        }
        self.assertEqual(restored_ranges[("layout", "yaxis", "range")], [-10, 110])
        self.assertEqual(restored_ranges[("layout", "yaxis2", "range")], [28, 52])
        self.assertEqual(reset_state["mode"], "reset")

    def test_shared_yaxis_toggle_patches_the_zoomed_window(self):
        timestamps = pd.date_range("2024-01-01", periods=3, freq="s")
        df = pd.DataFrame(
            {
                "metric_id": ["metric_a"] * 3 + ["metric_b"] * 3,
                "timestamp": list(timestamps) * 2,
                "value": [0.0, 10.0, 100.0, 50.0, 40.0, 30.0],
            }
        )
        cache_id = cache_dataframe(df, prefix="test_timeseries_share_patch")
        default_x_range = [timestamps[0].isoformat(), timestamps[-1].isoformat()]
        store = {
            "cache_id": cache_id,
            "metric_order": ["metric_a", "metric_b"],
            "default_x_range": default_x_range,
            "is_memory_category": False,
            "axis_defaults": {
                "xaxis": {"range": default_x_range, "autorange": False},
                "xaxis2": {"range": default_x_range, "autorange": False},
                "yaxis": {"range": [-10, 110], "autorange": False},
                "yaxis2": {"range": [28, 52], "autorange": False},
            },
        }
        zoom_state = {"mode": "zoom", "x0": timestamps[1].isoformat(), "x1": timestamps[2].isoformat()}
        patch = update_yaxis_on_toggle(["shared"], store, zoom_state)
        ranges = {
            tuple(op["location"]): op["params"]["value"]
            for op in patch.to_plotly_json()["operations"]
            if op["operation"] == "Assign" and op["location"][-1] == "range"
        }
        self.assertEqual(ranges[("layout", "yaxis", "range")], [1.0, 109.0])
        self.assertEqual(ranges[("layout", "yaxis2", "range")], [1.0, 109.0])
        self.assertTrue(all(op["location"][0] == "layout" for op in patch.to_plotly_json()["operations"]))

    def test_sorted_window_index_matches_filtered_ranges(self):
        timestamps = pd.date_range("2024-01-01", periods=6, freq="s", tz="UTC")
        df = pd.DataFrame(
            {
                "metric_id": ["metric_a"] * 6 + ["metric_b"] * 6 + ["metric_nan"] * 6,
                "timestamp": list(timestamps) * 3,
                "value": [0.0, 10.0, float("nan"), 100.0, 7.0, 3.0]
                + [50.0, 40.0, 30.0, 0.0, 80.0, 1.0]
                + [float("nan")] * 6,
            }
        )
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        metric_order = ["metric_a", "metric_b", "metric_nan"]
        cache_id = cache_dataframe(df, prefix="test_timeseries_index")
        remember_metric_window_index(cache_id, _metric_window_index(df, metric_order))
        window = [timestamps[1].isoformat(), timestamps[3].isoformat()]
        x_min, x_max = align_xrange_tz(
            pd.to_datetime(window[0]),
            pd.to_datetime(window[1]),
            df["timestamp"].dt.tz,
        )
        visible = filter_to_time_range(df, x_min, x_max)
        self.assertEqual(
            sorted(visible.loc[visible["metric_id"] == "metric_a", "value"].dropna().tolist()),
            [10.0, 100.0],
        )
        for share in (False, True):
            for memory in (False, True):
                fast = _yaxis_updates_for_window(cache_id, metric_order, window, share, memory)
                slow = compute_yaxis_ranges(visible, metric_order, share, memory)
                self.assertTrue(_same_axis_updates(fast, slow), (share, memory, fast, slow))
        self.assertTrue(
            _same_axis_updates(
                _yaxis_updates_for_window(cache_id, metric_order, None, False, False),
                compute_yaxis_ranges(df, metric_order, False, False),
            )
        )
        self.assertIsNone(
            _yaxis_updates_for_window(
                cache_id,
                metric_order,
                ["2020-01-01T00:00:00+00:00", "2020-01-01T00:00:01+00:00"],
                False,
                False,
            )
        )

        store = {
            "cache_id": cache_id,
            "metric_order": ["metric_a", "metric_b"],
            "is_memory_category": False,
        }
        figure = {
            "data": [],
            "layout": {
                "xaxis": {"range": [timestamps[0].isoformat(), timestamps[-1].isoformat()]},
                "yaxis": {"range": [0, 1]},
                "yaxis2": {"range": [0, 1]},
            },
        }
        zoomed = update_yaxis_on_zoom(
            {"xaxis.range[0]": timestamps[1].isoformat(), "xaxis.range[1]": timestamps[3].isoformat()},
            figure,
            store,
            [],
        )
        self.assertEqual(zoomed["layout"]["yaxis"]["range"], [1.0, 109.0])


def _same_axis_updates(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if set(left) != set(right):
        return False
    for key in left:
        if set(left[key]) != set(right[key]):
            return False
        for field, value in left[key].items():
            other = right[key][field]
            if isinstance(value, list):
                if len(value) != len(other):
                    return False
                if not all(_same_number(item, other[index]) for index, item in enumerate(value)):
                    return False
            elif not _same_number(value, other):
                return False
    return True


def _same_number(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if math.isnan(left) and math.isnan(right):
            return True
        return left == right
    return left == right


if __name__ == "__main__":
    unittest.main()
