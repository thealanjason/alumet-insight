import unittest

import pandas as pd

from backend.categories import category_yaxis_label, is_yaxis_shareable
from backend.transforms import align_xrange_tz, compute_yaxis_ranges


class TimeSeriesTests(unittest.TestCase):
    def test_is_yaxis_shareable_and_labels(self):
        self.assertTrue(is_yaxis_shareable("energy"))
        self.assertTrue(is_yaxis_shareable("power"))
        self.assertTrue(is_yaxis_shareable("temperature"))
        self.assertFalse(is_yaxis_shareable("miscellaneous"))
        self.assertFalse(is_yaxis_shareable("perf_counters"))
        self.assertEqual(category_yaxis_label("kernel_cpu_time"), "Value (ms)")
        self.assertEqual(category_yaxis_label("temperature"), "Value (°C)")
        self.assertEqual(category_yaxis_label("perf_counters"), "Value (count)")
        self.assertEqual(category_yaxis_label("unknown"), "Value")

    def test_align_xrange_tz_handles_aware_and_naive(self):
        tz = pd.Timestamp("2024-01-01", tz="UTC").tz
        x_min, x_max = align_xrange_tz(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), tz)
        self.assertIsNotNone(x_min.tz)
        self.assertIsNotNone(x_max.tz)

        naive_min, naive_max = align_xrange_tz(
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
            None,
        )
        self.assertIsNone(naive_min.tzinfo)
        self.assertIsNone(naive_max.tzinfo)

    def test_compute_yaxis_ranges_shared_mode(self):
        df = pd.DataFrame(
            {
                "metric_id": ["a", "a", "b", "b"],
                "value": [1.0, 3.0, 2.0, 4.0],
            }
        )
        ranges = compute_yaxis_ranges(df, ["a", "b"], share_yaxis=True, is_memory=False)
        self.assertEqual(set(ranges), {"yaxis", "yaxis2"})
        self.assertEqual(ranges["yaxis"]["range"], ranges["yaxis2"]["range"])

    def test_compute_yaxis_ranges_per_metric_and_memory_mode(self):
        df = pd.DataFrame(
            {
                "metric_id": ["mem_a", "mem_a", "mem_b", "mem_b"],
                "value": [1024.0, 2048.0, 4096.0, 8192.0],
            }
        )
        ranges = compute_yaxis_ranges(df, ["mem_a", "mem_b"], share_yaxis=False, is_memory=True)

        self.assertEqual(set(ranges), {"yaxis", "yaxis2"})
        self.assertNotEqual(ranges["yaxis"]["range"], ranges["yaxis2"]["range"])
        self.assertIn("tickvals", ranges["yaxis"])
        self.assertIn("ticktext", ranges["yaxis"])

    def test_compute_yaxis_ranges_skips_metrics_with_no_visible_data(self):
        df = pd.DataFrame({"metric_id": ["present", "present"], "value": [1.0, 3.0]})
        ranges = compute_yaxis_ranges(df, ["present", "absent"], share_yaxis=False, is_memory=False)
        self.assertEqual(set(ranges), {"yaxis"})


if __name__ == "__main__":
    unittest.main()
