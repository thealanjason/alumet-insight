import unittest

import pandas as pd

from backend.transforms import compute_yaxis_ranges
from frontend.cache import cache_dataframe
from frontend.panes.timeseries import update_yaxis_on_zoom


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
            "layout": {
                "xaxis": {"range": default_x_range, "autorange": False},
                "xaxis2": {"range": default_x_range, "autorange": False},
                "yaxis": {"range": [-10, 110], "autorange": False},
                "yaxis2": {"range": [28, 52], "autorange": False},
            }
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
        self.assertEqual(
            zoomed["layout"]["xaxis2"]["range"],
            [timestamps[1].isoformat(), timestamps[2].isoformat()],
        )

        reset = update_yaxis_on_zoom(
            {"xaxis.autorange": True, "yaxis.autorange": True},
            zoomed,
            store,
            [],
        )

        expected_yaxes = compute_yaxis_ranges(
            df,
            metric_order,
            share_yaxis=False,
            is_memory=False,
        )
        for xaxis_key in ("xaxis", "xaxis2"):
            self.assertEqual(reset["layout"][xaxis_key]["range"], default_x_range)
            self.assertFalse(reset["layout"][xaxis_key]["autorange"])
        for yaxis_key in ("yaxis", "yaxis2"):
            self.assertEqual(
                reset["layout"][yaxis_key]["range"],
                expected_yaxes[yaxis_key]["range"],
            )
            self.assertFalse(reset["layout"][yaxis_key]["autorange"])


if __name__ == "__main__":
    unittest.main()
