import unittest

import pandas as pd

from backend.categories import available_category_values, available_cpu_cores, filter_time_series_category
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
                    "kernel_n_procs_running_R_local__C__A_",
                    "network_rx_bytes_R_eth0__C__A_",
                ],
                "base_metric": [
                    "nvml_temperature_C",
                    "perf_hardware_INSTRUCTIONS",
                    "kernel_n_procs_running",
                    "network_rx_bytes",
                ],
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="s"),
                "value": [70.0, 100.0, 2.0, 4096.0],
            }
        )

        self.assertEqual(
            filter_time_series_category(df, "temperature")["base_metric"].tolist(),
            ["nvml_temperature_C"],
        )
        self.assertEqual(
            filter_time_series_category(df, "perf_counters")["base_metric"].tolist(),
            ["perf_hardware_INSTRUCTIONS"],
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


if __name__ == "__main__":
    unittest.main()
