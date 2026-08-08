import unittest

import pandas as pd

from backend.categories import (
    available_category_values,
    available_cpu_cores,
    category_yaxis_label,
    filter_time_series_category,
    is_yaxis_shareable,
)
from tests.fixtures import processed_rows


class CategoryTests(unittest.TestCase):
    def test_available_category_values_and_cpu_cores(self):
        df = processed_rows()

        self.assertEqual(available_category_values(df), ["energy", "power", "memory", "kernel_cpu_time"])
        self.assertEqual(available_cpu_cores(df), ["0"])
        self.assertEqual(available_cpu_cores(pd.DataFrame(columns=["metric_id", "base_metric"])), [])

    def test_filter_time_series_category_for_each_supported_bucket(self):
        df = processed_rows()

        self.assertEqual(filter_time_series_category(df, "power")["value"].tolist(), [2.0])
        self.assertEqual(len(filter_time_series_category(df, "kernel_cpu_time", selected_cpu_core="0")), 1)
        self.assertEqual(len(filter_time_series_category(df, "memory")), 1)
        self.assertEqual(len(filter_time_series_category(df, None)), len(df))

        energy = filter_time_series_category(df, "energy")
        self.assertTrue((energy["base_metric"] == "attributed_energy_J").any())

    def test_filter_time_series_category_keeps_counterdiff_padding_rows(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "rapl_consumed_energy_J_R_pkg_0_C__A_",
                    "rapl_consumed_energy_J_R_pkg_0_C__A_",
                ],
                "base_metric": ["rapl_consumed_energy_J", "rapl_consumed_energy_J"],
                "timestamp": [pd.Timestamp("2024-01-01")] * 2,
                "value": [3.0, 0.0],
                "point_role": ["observed", "synthetic"],
                "point_order": [0, 1],
            }
        )
        energy = filter_time_series_category(df, "energy")
        self.assertEqual(len(energy), 2)
        self.assertEqual(set(energy["point_role"]), {"observed", "synthetic"})

    def test_filter_time_series_category_includes_derived_power(self):
        df = pd.DataFrame(
            {
                "metric_id": ["rapl_average_power_W_R_pkg_0_C__A_"],
                "base_metric": ["rapl_average_power_W"],
                "timestamp": [pd.Timestamp("2024-01-01")],
                "value": [12.0],
            }
        )
        power = filter_time_series_category(df, "power")
        self.assertEqual(len(power), 1)
        self.assertEqual(available_category_values(df), ["power"])

    def test_filter_time_series_category_miscellaneous_and_utilization(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "cpu_percent_R_local_machine__C_process_1_A_",
                    "custom_counter_R_host__C__A_",
                ],
                "base_metric": ["cpu_percent", "custom_counter"],
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="s"),
                "value": [1.0, 2.0],
            }
        )

        util = filter_time_series_category(df, "utilization")
        misc = filter_time_series_category(df, "miscellaneous")

        self.assertEqual(util["base_metric"].tolist(), ["cpu_percent"])
        self.assertEqual(misc["base_metric"].tolist(), ["custom_counter"])

    def test_filter_time_series_category_unknown_raises(self):
        with self.assertRaises(ValueError):
            filter_time_series_category(processed_rows(), "not-a-category")

    def test_filter_temperature_perf_counters_and_kernel_system(self):
        df = pd.DataFrame(
            {
                "metric_id": [
                    "nvml_temperature_C_R_gpu_0_C_process_1_A_",
                    "perf_hardware_INSTRUCTIONS_R_cpu_0_C_process_1_A_",
                    "perf_cache_LL_READ_MISS_R_cpu_0_C_process_1_A_",
                    "kernel_n_procs_running_R_local__C__A_",
                    "network_rx_bytes_R_eth0__C__A_",
                ],
                "base_metric": [
                    "nvml_temperature_C",
                    "perf_hardware_INSTRUCTIONS",
                    "perf_cache_LL_READ_MISS",
                    "kernel_n_procs_running",
                    "network_rx_bytes",
                ],
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="s"),
                "value": [70.0, 100.0, 10.0, 2.0, 4096.0],
            }
        )

        self.assertEqual(
            filter_time_series_category(df, "temperature")["base_metric"].tolist(),
            ["nvml_temperature_C"],
        )
        self.assertEqual(
            filter_time_series_category(df, "perf_counters")["base_metric"].tolist(),
            ["perf_hardware_INSTRUCTIONS", "perf_cache_LL_READ_MISS"],
        )
        kernel_system = filter_time_series_category(df, "kernel_system")
        self.assertEqual(
            set(kernel_system["base_metric"]),
            {"kernel_n_procs_running", "network_rx_bytes"},
        )

    def test_available_category_values_derives_base_metric_when_missing(self):
        df = pd.DataFrame(
            {
                "metric_id": ["nvml_temperature_C_R_gpu_0_C__A_"],
                "timestamp": [pd.Timestamp("2024-01-01")],
                "value": [65.0],
            }
        )

        self.assertEqual(available_category_values(df), ["temperature"])

    def test_is_yaxis_shareable(self):
        self.assertTrue(is_yaxis_shareable("energy"))
        self.assertTrue(is_yaxis_shareable("power"))
        self.assertTrue(is_yaxis_shareable("temperature"))
        self.assertFalse(is_yaxis_shareable("miscellaneous"))
        self.assertFalse(is_yaxis_shareable("perf_counters"))

    def test_category_yaxis_label(self):
        self.assertEqual(category_yaxis_label("kernel_cpu_time"), "Value (ms)")
        self.assertEqual(category_yaxis_label("temperature"), "Value (°C)")
        self.assertEqual(category_yaxis_label("perf_counters"), "Value (count)")
        self.assertEqual(category_yaxis_label("unknown"), "Value")


if __name__ == "__main__":
    unittest.main()
