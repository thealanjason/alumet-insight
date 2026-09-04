import unittest

import pandas as pd

from backend.transforms import (
    align_xrange_tz,
    align_xy_metrics,
    comparative_download_table,
    comparative_metric_ids,
    comparative_xy_frame,
    compute_yaxis_ranges,
    filter_to_time_range,
    get_process_time_range_from_df,
    normalize_to_si,
    align_running_total_xy,
    comparative_cumulative_xy,
    prepare_xy_download,
    xy_running_totals,
)


class TransformsTests(unittest.TestCase):
    def test_filter_to_time_range_handles_bounds_and_missing_bounds(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="s"), "value": [1, 2, 3]})

        filtered = filter_to_time_range(df, pd.Timestamp("2024-01-01 00:00:01"), pd.Timestamp("2024-01-01 00:00:02"))
        self.assertEqual(filtered["value"].tolist(), [2, 3])
        self.assertTrue(filter_to_time_range(df, None, None).empty)
        self.assertEqual(filter_to_time_range(df, None, None, require_bounds=False)["value"].tolist(), [1, 2, 3])
        self.assertTrue(
            filter_to_time_range(pd.DataFrame(), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")).empty
        )
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

    def test_normalize_to_si_relabels_legacy_memory_kb_without_rescaling(self):
        df = pd.DataFrame(
            {
                "metric": ["active_kB", "mem_total_kB", "memory_usage_B", "active_B", "unknown_kB"],
                "value": [1024.0, 2048.0, 512.0, 1024.0, 99.0],
            }
        )
        out = normalize_to_si(df, col="metric")
        self.assertEqual(
            out["metric"].tolist(),
            ["active_B", "mem_total_B", "memory_usage_B", "active_B", "unknown_kB"],
        )
        self.assertEqual(out["value"].tolist(), [1024.0, 2048.0, 512.0, 1024.0, 99.0])

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

        empty = align_xy_metrics(pd.DataFrame(columns=["metric_id", "timestamp", "value"]), "a", "b", start, end)
        self.assertTrue(empty.empty)

    def test_align_xy_metrics_ignores_synthetic_padding(self):
        ts = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "timestamp": [ts, ts, ts, ts],
                "metric_id": [
                    "rapl_consumed_energy_J_R_pkg_0_C__A_",
                    "rapl_consumed_energy_J_R_pkg_0_C__A_",
                    "cpu_percent_R_host__C_process_1_A_",
                    "cpu_percent_R_host__C_process_1_A_",
                ],
                "value": [5.0, 0.0, 40.0, 40.0],
                "point_role": ["observed", "synthetic", "observed", "synthetic"],
                "point_order": [0, 1, 0, 1],
            }
        )
        aligned = align_xy_metrics(
            df,
            "rapl_consumed_energy_J_R_pkg_0_C__A_",
            "cpu_percent_R_host__C_process_1_A_",
            ts,
            ts,
        )
        self.assertEqual(aligned[["x", "y"]].values.tolist(), [[5.0, 40.0]])

    def test_align_xy_metrics_rejects_duplicate_observed_timestamps(self):
        ts = pd.Timestamp("2024-01-01")
        df = pd.DataFrame(
            {
                "timestamp": [ts, ts],
                "metric_id": ["x_R_a_C__A_", "x_R_a_C__A_"],
                "value": [1.0, 2.0],
                "point_role": ["observed", "observed"],
            }
        )
        with self.assertRaises(ValueError):
            align_xy_metrics(df, "x_R_a_C__A_", "x_R_a_C__A_", ts, ts)

    def test_xy_running_totals_preserves_each_series_sum(self):
        """Unequal rates + delayed Y: the 08_topo_uncertain cumulative failure mode.

        CPU-like X at 50 ms, GPU-like Y at 200 ms starting 2 s later.
        ``align_xy_metrics`` then ``cumsum`` drops the unmatched prefix and
        repeats each Y sample onto several X stamps. Running totals must
        still end at ``sum(X)`` and ``sum(Y)``.
        """
        x_id = "attributed_energy_cpu_J_R_pkg_C_process_1_A_"
        y_id = "attributed_energy_gpu_J_R_gpu_C_process_1_A_"
        start = pd.Timestamp("2024-01-01")
        x_times = pd.date_range(start, periods=81, freq="50ms")
        y_times = pd.date_range(start + pd.Timedelta("2s"), periods=11, freq="200ms")
        df = pd.DataFrame(
            {
                "timestamp": list(x_times) + list(y_times),
                "metric_id": [x_id] * len(x_times) + [y_id] * len(y_times),
                "value": [2.0] * len(x_times) + [10.0] * len(y_times),
                "point_role": ["observed"] * (len(x_times) + len(y_times)),
            }
        )
        end = max(x_times[-1], y_times[-1])

        totals = xy_running_totals(df, x_id, y_id, start, end)
        self.assertAlmostEqual(float(totals["x"].iloc[-1]), 162.0)
        self.assertAlmostEqual(float(totals["y"].iloc[-1]), 110.0)
        self.assertTrue((totals.loc[totals["timestamp"] < y_times[0], "y"] == 0.0).all())

        aligned = align_xy_metrics(df, x_id, y_id, start, end)
        self.assertLess(float(aligned["x"].sum()), 162.0)
        self.assertGreater(float(aligned["y"].sum()), 110.0)

    def test_xy_running_totals_ignores_synthetic_padding(self):
        x_id = "attributed_energy_cpu_J_R_pkg_C_process_1_A_"
        y_id = "attributed_energy_gpu_J_R_gpu_C_process_1_A_"
        ts0 = pd.Timestamp("2024-01-01")
        ts1 = pd.Timestamp("2024-01-01 00:00:01")
        df = pd.DataFrame(
            {
                "timestamp": [ts0, ts0, ts1, ts1, ts0, ts1],
                "metric_id": [x_id, x_id, x_id, x_id, y_id, y_id],
                "value": [5.0, 0.0, 7.0, 0.0, 3.0, 4.0],
                "point_role": [
                    "observed",
                    "synthetic",
                    "observed",
                    "synthetic",
                    "observed",
                    "observed",
                ],
            }
        )
        totals = xy_running_totals(df, x_id, y_id, ts0, ts1)
        self.assertEqual(totals[["x", "y"]].values.tolist(), [[5.0, 3.0], [12.0, 7.0]])

    def test_xy_running_totals_empty_when_a_series_is_missing(self):
        start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 00:00:01")
        empty = xy_running_totals(
            pd.DataFrame(columns=["metric_id", "timestamp", "value"]),
            "a",
            "b",
            start,
            end,
        )
        self.assertTrue(empty.empty)

    def test_comparative_cumulative_xy_prefers_precomputed_siblings(self):
        x_id = "attributed_energy_cpu_J_R_pkg_C_process_1_A_"
        y_id = "attributed_energy_gpu_J_R_gpu_C_process_1_A_"
        x_cum = "attributed_energy_cpu_cumulative_J_R_pkg_C_process_1_A_"
        y_cum = "attributed_energy_gpu_cumulative_J_R_gpu_C_process_1_A_"
        start = pd.Timestamp("2024-01-01")
        x_times = pd.date_range(start, periods=3, freq="50ms")
        y_times = pd.date_range(start + pd.Timedelta("100ms"), periods=2, freq="100ms")
        df = pd.DataFrame(
            {
                "timestamp": list(x_times) + list(y_times) + list(x_times) + list(y_times),
                "metric_id": (
                    [x_id] * len(x_times)
                    + [y_id] * len(y_times)
                    + [x_cum] * len(x_times)
                    + [y_cum] * len(y_times)
                ),
                "value": [2.0, 2.0, 2.0, 10.0, 10.0, 2.0, 4.0, 6.0, 10.0, 20.0],
                "point_role": ["observed"] * 10,
            }
        )
        end = y_times[-1]
        from_siblings = align_running_total_xy(df, x_cum, y_cum, start, end)
        from_helper = comparative_cumulative_xy(df, x_id, y_id, start, end)
        self.assertEqual(from_siblings["x"].tolist(), from_helper["x"].tolist())
        self.assertEqual(from_siblings["y"].tolist(), from_helper["y"].tolist())
        self.assertAlmostEqual(float(from_helper["x"].iloc[-1]), 6.0)
        self.assertAlmostEqual(float(from_helper["y"].iloc[-1]), 20.0)

    def test_comparative_download_table_matches_dashboard_columns(self):
        x_id = "attributed_energy_cpu_J_R_pkg_C_process_1_A_"
        y_id = "attributed_energy_gpu_J_R_gpu_C_process_1_A_"
        start = pd.Timestamp("2024-01-01")
        x_times = pd.date_range(start, periods=4, freq="s")
        y_times = pd.date_range(start, periods=2, freq="2s")
        df = pd.DataFrame(
            {
                "timestamp": list(x_times) + list(y_times),
                "metric_id": [x_id] * 4 + [y_id] * 2,
                "value": [1.0, 1.0, 1.0, 1.0, 10.0, 20.0],
            }
        )
        end = x_times[-1]
        frame = comparative_xy_frame(df, x_id, y_id, start, end)
        table, filename = comparative_download_table(df, x_id, y_id, start, end)
        renamed, same_name = prepare_xy_download(frame, x_id, y_id)
        self.assertEqual(filename, same_name)
        self.assertEqual(list(table.columns), list(renamed.columns))
        self.assertIn("x_unit", table.columns)
        self.assertAlmostEqual(float(table[x_id].iloc[-1]), 4.0)
        self.assertAlmostEqual(float(table[y_id].iloc[-1]), 30.0)
        scatter = comparative_xy_frame(df, x_id, y_id, start, end, scatter=True)
        self.assertEqual(list(scatter.columns), ["timestamp", "x", "y"])


if __name__ == "__main__":
    unittest.main()
