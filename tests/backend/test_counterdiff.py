import unittest

import numpy as np
import pandas as pd

from backend.counterdiff import (
    POINT_ORDER_BY_ROLE,
    PointOrder,
    PointRole,
    build_counterdiff_spike_coordinates,
    build_step_power_coordinates,
    counterdiff_spike_marker_sizes,
    derive_interval_average_power,
    expand_counterdiff_rows,
    export_observed_measurements,
    interpolate_counterdiff_at_timeline,
    normalize_observed_rows,
    observed_only,
    require_processed_columns,
    validate_point_metadata,
)
from backend.metrics import (
    MetricType,
    is_counterdiff_metric,
    is_raw_counter_metric,
    metric_type,
    should_derive_power_from_energy,
)


class CounterDiffTests(unittest.TestCase):
    def test_point_role_order_mapping(self):
        self.assertEqual(
            POINT_ORDER_BY_ROLE,
            {
                PointRole.OBSERVED: PointOrder.OBSERVED,
                PointRole.SYNTHETIC: PointOrder.POST_SAMPLE_ZERO,
            },
        )

    def test_metric_type_classification(self):
        self.assertEqual(metric_type("rapl_consumed_energy_J"), MetricType.COUNTER_DIFF)
        self.assertEqual(metric_type("attributed_energy_cpu_J"), MetricType.COUNTER_DIFF)
        self.assertEqual(metric_type("nvml_instant_power_W"), MetricType.GAUGE)
        self.assertEqual(metric_type("rapl_average_power_W"), MetricType.GAUGE)
        self.assertEqual(
            metric_type("perf_hardware_INSTRUCTIONS"),
            MetricType.RAW_COUNTER,
        )
        self.assertTrue(is_raw_counter_metric("perf_hardware_INSTRUCTIONS"))
        self.assertFalse(is_counterdiff_metric("perf_hardware_INSTRUCTIONS"))
        self.assertFalse(is_counterdiff_metric("cpu_percent"))

    def test_expand_counterdiff_rows_for_counterdiff_metric(self):
        df = pd.DataFrame(
            {
                "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"],
                "base_metric": ["rapl_consumed_energy_J"],
                "timestamp": [pd.Timestamp("2024-01-01")],
                "value": [4.0],
            }
        )
        expanded = expand_counterdiff_rows(df)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded["point_role"].tolist(), ["observed", "synthetic"])
        self.assertEqual(expanded["point_order"].tolist(), [0, 1])
        self.assertEqual(expanded["value"].tolist(), [4.0, 0.0])
        self.assertEqual(expanded["timestamp"].nunique(), 1)
        self.assertEqual(expanded["sample_id"].nunique(), 1)

    def test_expand_counterdiff_rows_for_gauge_metric(self):
        df = pd.DataFrame(
            {
                "metric_id": ["nvml_instant_power_W_R_gpu_0_C__A_"],
                "base_metric": ["nvml_instant_power_W"],
                "timestamp": [pd.Timestamp("2024-01-01")],
                "value": [12.0],
            }
        )
        expanded = expand_counterdiff_rows(df)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded.loc[0, "point_role"], "observed")
        self.assertEqual(expanded.loc[0, "point_order"], 0)

    def test_expand_counterdiff_rows_for_raw_counter_metric(self):
        df = pd.DataFrame(
            {
                "metric_id": ["perf_hardware_INSTRUCTIONS_R_cpu_0_C__A_"],
                "base_metric": ["perf_hardware_INSTRUCTIONS"],
                "timestamp": [pd.Timestamp("2024-01-01")],
                "value": [1_000_000],
            }
        )
        expanded = expand_counterdiff_rows(df)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded.loc[0, "point_role"], "observed")
        self.assertEqual(expanded.loc[0, "point_order"], 0)

    def test_expand_counterdiff_rows_is_idempotent_for_mixed_already_padded_and_raw_rows(self):
        raw = pd.DataFrame(
            {
                "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"] * 2,
                "base_metric": ["rapl_consumed_energy_J"] * 2,
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="s"),
                "value": [4.0, 6.0],
            }
        )
        mixed = pd.concat(
            [expand_counterdiff_rows(raw.iloc[:1]), raw.iloc[1:]],
            ignore_index=True,
        )
        expanded = expand_counterdiff_rows(mixed)

        validate_point_metadata(expanded)
        self.assertEqual(
            expanded.groupby("timestamp")["point_role"].count().tolist(),
            [2, 2],
        )
        self.assertEqual(len(observed_only(expanded)), 2)

    def test_expand_counterdiff_rows_for_identical_observations(self):
        timestamp = pd.Timestamp("2024-01-01")
        duplicate = pd.DataFrame(
            {
                "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"] * 2,
                "base_metric": ["rapl_consumed_energy_J"] * 2,
                "timestamp": [timestamp, timestamp],
                "value": [4.0, 4.0],
            }
        )

        expanded = expand_counterdiff_rows(duplicate)

        self.assertEqual(len(observed_only(expanded)), 1)
        self.assertEqual(len(expanded), 2)
        validate_point_metadata(expanded)

    def test_expand_counterdiff_rows_recovers_metadata_for_already_padded_rows(self):
        timestamp = pd.Timestamp("2024-01-01")
        already_padded = pd.DataFrame(
            {
                "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"] * 2,
                "base_metric": ["rapl_consumed_energy_J"] * 2,
                "timestamp": [timestamp, timestamp],
                "value": [8.0, 0.0],
            }
        )

        expanded = expand_counterdiff_rows(already_padded)

        observed = observed_only(expanded)
        self.assertEqual(observed["value"].tolist(), [8.0])
        self.assertEqual(expanded["value"].tolist(), [8.0, 0.0])
        validate_point_metadata(expanded)

    def test_expand_counterdiff_rows_rejects_conflicting_observations_at_same_timestamp(self):
        timestamp = pd.Timestamp("2024-01-01")
        duplicate = pd.DataFrame(
            {
                "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"] * 2,
                "base_metric": ["rapl_consumed_energy_J"] * 2,
                "timestamp": [timestamp, timestamp],
                "value": [4.0, 6.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "conflicting observed values"):
            expand_counterdiff_rows(duplicate)

    def test_expand_counterdiff_rows_for_gauge_metric_keeps_independent_observations_at_same_timestamp(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "cpu_percent_R_local_machine__C_process_1_A_",
                    "cpu_percent_R_local_machine__C_process_1_A_",
                ],
                "base_metric": ["cpu_percent", "cpu_percent"],
                "timestamp": [
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-01-01"),
                ],
                "value": [10.0, 20.0],
            }
        )
        expanded = expand_counterdiff_rows(df)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(sorted(expanded["value"].tolist()), [10.0, 20.0])
        self.assertEqual(expanded["sample_id"].nunique(), 2)

    def test_validate_point_metadata_rejects_nonzero_synthetic_value(self):
        expanded = expand_counterdiff_rows(
            pd.DataFrame(
                {
                    "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"],
                    "base_metric": ["rapl_consumed_energy_J"],
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "value": [4.0],
                }
            )
        )
        expanded.loc[
            expanded["point_role"] == PointRole.SYNTHETIC.value,
            "value",
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "must have value 0"):
            validate_point_metadata(expanded)

    def test_interpolate_counter_diff(self):
        src_ts = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:02"])
        src_vals = [1.0, 3.0]
        targets = pd.to_datetime(
            [
                "2023-12-31 23:59:59",  # before first observation → 0
                "2024-01-01 00:00:00",  # exact observation → 1
                "2024-01-01 00:00:01",  # between observations → 3 * 0.5 = 1.5
                "2024-01-01 00:00:02",  # exact observation → 3
                "2024-01-01 00:00:03",  # after last observation → NaN
            ]
        )
        out = interpolate_counterdiff_at_timeline(src_ts, src_vals, targets)
        self.assertEqual(out.iloc[0], 0.0)
        self.assertEqual(out.iloc[1], 1.0)
        self.assertAlmostEqual(out.iloc[2], 1.5)
        self.assertEqual(out.iloc[3], 3.0)
        self.assertTrue(np.isnan(out.iloc[4]))

    def test_interpolate_rejects_duplicate_source_timestamps(self):
        src_ts = pd.to_datetime(["2024-01-01", "2024-01-01"])
        with self.assertRaises(ValueError):
            interpolate_counterdiff_at_timeline(src_ts, [1.0, 2.0], src_ts)

    def test_interpolate_irregular_time_intervals(self):
        src_ts = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:10"])
        targets = pd.to_datetime(["2024-01-01 00:00:02"])
        out = interpolate_counterdiff_at_timeline(src_ts, [5.0, 20.0], targets)
        self.assertAlmostEqual(out.iloc[0], 20.0 * 0.2)

    def test_derive_interval_average_power_for_irregular_time_intervals(self):
        df = pd.DataFrame(
            {
                "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"] * 3,
                "base_metric": ["rapl_consumed_energy_J"] * 3,
                "timestamp": pd.to_datetime(
                    [
                        "2024-01-01 00:00:00",
                        "2024-01-01 00:00:01",
                        "2024-01-01 00:00:03",
                    ]
                ),
                "value": [2.0, 4.0, 10.0],
                "point_role": ["observed"] * 3,
                "point_order": [0, 0, 0],
                "sample_id": [0, 1, 2],
            }
        )
        power = derive_interval_average_power(df)
        self.assertEqual(len(power), 2)
        self.assertAlmostEqual(power["value"].iloc[0], 4.0)  # 4 J / 1 s
        self.assertAlmostEqual(power["value"].iloc[1], 5.0)  # 10 J / 2 s
        self.assertTrue(power["metric_id"].iloc[0].startswith("rapl_average_power_W"))
        self.assertEqual(
            power["interval_start"].tolist(),
            df["timestamp"].iloc[:-1].tolist(),
        )

    def test_derive_interval_average_power_rejects_non_positive_time_intervals(self):
        df = pd.DataFrame(
            {
                "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"] * 2,
                "base_metric": ["rapl_consumed_energy_J"] * 2,
                "timestamp": pd.to_datetime(["2024-01-01 00:00:01", "2024-01-01 00:00:01"]),
                "value": [1.0, 2.0],
                "point_role": ["observed", "observed"],
                "point_order": [0, 0],
                "sample_id": [0, 1],
            }
        )
        with self.assertRaises(ValueError):
            derive_interval_average_power(df)

    def test_derive_interval_average_power_skips_first_sample(self):
        df = pd.DataFrame(
            {
                "metric_id": ["attributed_energy_cpu_J_R_cpu_0_C_process_1_A_"],
                "base_metric": ["attributed_energy_cpu_J"],
                "timestamp": [pd.Timestamp("2024-01-01")],
                "value": [3.0],
                "point_role": ["observed"],
                "point_order": [0],
                "sample_id": [0],
            }
        )
        self.assertTrue(derive_interval_average_power(df).empty)

    def test_should_derive_power_from_energy_policy(self):
        available = {
            "nvml_energy_consumption_J_R_gpu_0_C__A_",
            "nvml_instant_power_W_R_gpu_0_C__A_",
            "rapl_consumed_energy_J_R_pkg_0_C__A_",
        }
        self.assertFalse(should_derive_power_from_energy("nvml_energy_consumption_J_R_gpu_0_C__A_", available))
        self.assertTrue(should_derive_power_from_energy("rapl_consumed_energy_J_R_pkg_0_C__A_", available))
        self.assertFalse(
            should_derive_power_from_energy(
                "attributed_energy_total_J_R_total__C_process_1_A_",
                available,
            )
        )
        self.assertFalse(
            should_derive_power_from_energy(
                "attributed_energy_gpu_total_J_R_gpu_all__C_process_1_A_",
                available,
            )
        )
        self.assertFalse(
            should_derive_power_from_energy(
                "attributed_energy_cpu_total_J_R_cpu_all__C_process_1_A_",
                available,
            )
        )
        self.assertTrue(
            should_derive_power_from_energy(
                "attributed_energy_cpu_J_R_local_machine__C_process_1_A_domain=package_total",
                available,
            )
        )

    def test_export_observed_measurements_removes_internal_columns(self):
        df = expand_counterdiff_rows(
            pd.DataFrame(
                {
                    "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"],
                    "base_metric": ["rapl_consumed_energy_J"],
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "value": [7.0],
                    "metric_origin": ["measured"],
                }
            )
        )
        power = derive_interval_average_power(
            pd.DataFrame(
                {
                    "metric_id": [
                        "rapl_consumed_energy_J_R_pkg_0_C__A_",
                        "rapl_consumed_energy_J_R_pkg_0_C__A_",
                    ],
                    "base_metric": ["rapl_consumed_energy_J", "rapl_consumed_energy_J"],
                    "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:02"]),
                    "value": [2.0, 8.0],
                }
            )
        )
        exported = export_observed_measurements(pd.concat([df, power], ignore_index=True))
        self.assertEqual(exported.loc[exported["base_metric"] == "rapl_consumed_energy_J", "value"].iloc[0], 7.0)
        self.assertNotIn("point_role", exported.columns)
        self.assertNotIn("sample_id", exported.columns)
        self.assertNotIn("metric_origin", exported.columns)
        self.assertNotIn("interval_start", exported.columns)

    def test_missing_required_columns(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            require_processed_columns(pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")]}))
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            normalize_observed_rows(
                pd.DataFrame(
                    {
                        "metric_id": ["cpu_percent_R_local_machine__C__A_"],
                        "timestamp": [pd.Timestamp("2024-01-01")],
                        "value": [1.0],
                    }
                )
            )

    def test_already_normalized_skips_renormalize(self):
        observed = normalize_observed_rows(
            pd.DataFrame(
                {
                    "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"],
                    "base_metric": ["rapl_consumed_energy_J"],
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "value": [3.0],
                    "metric_origin": ["measured"],
                }
            )
        )
        expanded = expand_counterdiff_rows(observed, already_normalized=True)
        self.assertEqual(len(expanded), 2)
        self.assertEqual(
            expanded.loc[expanded["point_role"] == "synthetic", "metric_origin"].iloc[0],
            "measured",
        )

    def test_spike_coordinates_insert_line_breaks(self):
        ts = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"])
        x, y = build_counterdiff_spike_coordinates(ts, [2.0, 5.0])
        self.assertEqual(y, [0.0, 2.0, 0.0, None, 0.0, 5.0, 0.0, None])
        self.assertIsNone(x[3])

    def test_spike_marker_sizes_show_only_observed_values_including_zero(self):
        ts = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"])
        x, _ = build_counterdiff_spike_coordinates(ts, [0.0, 5.0])

        self.assertEqual(
            counterdiff_spike_marker_sizes(x, marker_size=7),
            [0, 7, 0, 0, 0, 7, 0, 0],
        )

    def test_step_power_coordinates_connect_contiguous_intervals(self):
        timestamps = pd.to_datetime(["2024-01-01 00:00:01", "2024-01-01 00:00:03"])
        starts = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:01"])
        x, y = build_step_power_coordinates(
            timestamps,
            [4.0, 5.0],
            interval_starts=starts,
        )
        self.assertEqual(
            x,
            [starts[0], timestamps[0], timestamps[0], timestamps[1]],
        )
        self.assertEqual(y, [4.0, 4.0, 5.0, 5.0])

    def test_step_power_coordinates_break_on_time_gaps(self):
        timestamps = pd.to_datetime(
            ["2024-01-01 00:00:01", "2024-01-01 00:00:05"]
        )
        starts = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:00:04"])
        x, y = build_step_power_coordinates(
            timestamps,
            [4.0, 5.0],
            interval_starts=starts,
        )
        self.assertEqual(
            x,
            [starts[0], timestamps[0], None, starts[1], timestamps[1]],
        )
        self.assertEqual(y, [4.0, 4.0, None, 5.0, 5.0])

    def test_observed_only_filters_out_synthetic_rows(self):
        df = expand_counterdiff_rows(
            pd.DataFrame(
                {
                    "metric_id": ["rapl_consumed_energy_J_R_pkg_0_C__A_"],
                    "base_metric": ["rapl_consumed_energy_J"],
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "value": [1.0],
                }
            )
        )
        self.assertEqual(len(observed_only(df)), 1)


if __name__ == "__main__":
    unittest.main()
