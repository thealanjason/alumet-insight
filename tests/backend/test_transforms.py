import unittest

import pandas as pd

from backend.transforms import (
    align_xy_metrics,
    align_xrange_tz,
    comparative_metric_ids,
    compute_yaxis_ranges,
    filter_to_time_range,
    get_process_time_range_from_df,
    normalize_to_si,
)


class TransformsTests(unittest.TestCase):
    def test_filter_to_time_range_handles_bounds_and_missing_bounds(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="s"), "value": [1, 2, 3]})

        filtered = filter_to_time_range(df, pd.Timestamp("2024-01-01 00:00:01"), pd.Timestamp("2024-01-01 00:00:02"))
        self.assertEqual(filtered["value"].tolist(), [2, 3])
        self.assertTrue(filter_to_time_range(df, None, None).empty)
        self.assertEqual(filter_to_time_range(df, None, None, require_bounds=False)["value"].tolist(), [1, 2, 3])
        self.assertTrue(filter_to_time_range(pd.DataFrame(), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")).empty)
        with self.assertRaises(ValueError):
            filter_to_time_range(df.drop(columns=["timestamp"]), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))

    def test_get_process_time_range_uses_active_process_samples(self):
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

    def test_get_process_time_range_fallbacks(self):
        self.assertEqual(get_process_time_range_from_df(pd.DataFrame()), (None, None))

        no_process = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="s"),
                "consumer_kind": ["", ""],
                "value": [1.0, 2.0],
            }
        )
        start, end = get_process_time_range_from_df(no_process)
        self.assertEqual(start, pd.Timestamp("2024-01-01 00:00:00"))
        self.assertEqual(end, pd.Timestamp("2024-01-01 00:00:01"))

    def test_normalize_to_si_rescales_mw_and_mj(self):
        df = pd.DataFrame(
            {
                "metric": ["nvml_instant_power_mW", "nvml_energy_consumption_mJ", "cpu_percent"],
                "value": [1000.0, 500.0, 50.0],
            }
        )
        out = normalize_to_si(df, col="metric")
        self.assertIn("nvml_instant_power_W", out["metric"].values)
        self.assertIn("nvml_energy_consumption_J", out["metric"].values)
        self.assertAlmostEqual(out.loc[out["metric"] == "nvml_instant_power_W", "value"].iloc[0], 1.0)
        self.assertAlmostEqual(out.loc[out["metric"] == "nvml_energy_consumption_J", "value"].iloc[0], 0.5)
        self.assertIn("cpu_percent", out["metric"].values)

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
        df = pd.DataFrame({"metric_id": ["a", "a", "b", "b"], "value": [1.0, 3.0, 2.0, 4.0]})
        ranges = compute_yaxis_ranges(df, ["a", "b"], share_yaxis=True, is_memory=False)
        self.assertEqual(set(ranges), {"yaxis", "yaxis2"})
        self.assertEqual(ranges["yaxis"]["range"], ranges["yaxis2"]["range"])

    def test_compute_yaxis_ranges_per_metric_and_memory_mode(self):
        df = pd.DataFrame(
            {"metric_id": ["mem_a", "mem_a", "mem_b", "mem_b"], "value": [1024.0, 2048.0, 4096.0, 8192.0]}
        )
        ranges = compute_yaxis_ranges(df, ["mem_a", "mem_b"], share_yaxis=False, is_memory=True)
        self.assertEqual(set(ranges), {"yaxis", "yaxis2"})
        self.assertNotEqual(ranges["yaxis"]["range"], ranges["yaxis2"]["range"])
        self.assertIn("tickvals", ranges["yaxis"])

    def test_compute_yaxis_ranges_skips_metrics_with_no_visible_data(self):
        df = pd.DataFrame({"metric_id": ["present", "present"], "value": [1.0, 3.0]})
        ranges = compute_yaxis_ranges(df, ["present", "absent"], share_yaxis=False, is_memory=False)
        self.assertEqual(set(ranges), {"yaxis"})

    def test_comparative_metric_ids(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="s").tolist() * 2,
                "metric_id": ["x_R_a_C_process_1_A_"] * 3 + ["y_R_a_C_process_1_A_"] * 3,
                "value": [1, 2, 3, 10, 20, 30],
            }
        )
        ids = comparative_metric_ids(df, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 00:00:02"))
        self.assertEqual(ids, ["x_R_a_C_process_1_A_", "y_R_a_C_process_1_A_"])
        self.assertEqual(comparative_metric_ids(pd.DataFrame(), None, None), [])

    def test_align_xy_metrics(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="s").tolist() * 2,
                "metric_id": ["x_R_a_C_process_1_A_"] * 3 + ["y_R_a_C_process_1_A_"] * 3,
                "value": [1, 2, 3, 10, 20, 30],
            }
        )
        start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 00:00:02")
        aligned = align_xy_metrics(df, "x_R_a_C_process_1_A_", "y_R_a_C_process_1_A_", start, end)
        self.assertEqual(aligned[["x", "y"]].values.tolist(), [[1, 10], [2, 20], [3, 30]])

        empty = align_xy_metrics(
            pd.DataFrame(columns=["metric_id", "timestamp", "value"]), "a", "b", start, end
        )
        self.assertTrue(empty.empty)


if __name__ == "__main__":
    unittest.main()
