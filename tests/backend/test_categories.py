import unittest

import pandas as pd

from backend.categories import (
    CATEGORY_VALUES,
    available_category_values,
    available_cpu_cores,
    category_for_metric_id,
    category_yaxis_label,
    classify_metric,
    filter_time_series_category,
    is_yaxis_shareable,
)
from tests.fixtures import (
    NVML_POWER_ID,
    catalog_rows,
    processed_rows,
    rapl_energy_rows,
)


class CategoryTests(unittest.TestCase):
    def test_classify_metric_uses_current_predicates(self):
        self.assertEqual(classify_metric("rapl_average_power_W"), "power")
        self.assertEqual(classify_metric("attributed_energy_cpu_J"), "energy")
        self.assertEqual(classify_metric("attributed_energy_cpu_cumulative_J"), "energy")
        self.assertEqual(classify_metric("nvml_memory_utilization_%"), "utilization")
        self.assertEqual(classify_metric("cpu_percent"), "utilization")
        self.assertEqual(classify_metric("nvml_gpu_memory_info_B"), "memory")
        self.assertEqual(classify_metric("mapped_kB"), "memory")
        self.assertEqual(classify_metric("perf_cache_LL_READ_MISS"), "perf_counters")
        self.assertEqual(classify_metric("kernel_cpu_time_ms"), "kernel_cpu_time")
        self.assertEqual(classify_metric("network_rx_bytes"), "kernel_system")
        self.assertEqual(classify_metric("custom_counter"), "miscellaneous")

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
        df = rapl_energy_rows(
            [3.0, 0.0],
            point_role=["observed", "synthetic"],
            point_order=[0, 1],
        )
        energy = filter_time_series_category(df, "energy")
        self.assertEqual(len(energy), 2)
        self.assertEqual(set(energy["point_role"]), {"observed", "synthetic"})

    def test_filter_time_series_category_unknown_raises(self):
        with self.assertRaises(ValueError):
            filter_time_series_category(processed_rows(), "not-a-category")

    def test_filter_temperature_perf_counters_and_kernel_system(self):
        df = catalog_rows()

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
        self.assertEqual(category_yaxis_label("memory"), "Value (B)")
        self.assertEqual(category_yaxis_label("unknown"), "Value")

    def test_memory_category_includes_generic_and_gpu_byte_metrics(self):
        df = catalog_rows()

        self.assertEqual(available_category_values(df), list(CATEGORY_VALUES))
        self.assertEqual(
            set(filter_time_series_category(df, "memory")["base_metric"]),
            {"mem_total_B", "active_B", "inactive_B", "cached_B", "mapped_kB", "nvml_gpu_memory_info_B"},
        )
        self.assertEqual(
            set(filter_time_series_category(df, "utilization")["base_metric"]),
            {"cpu_percent", "nvml_memory_utilization_%"},
        )
        self.assertEqual(
            filter_time_series_category(df, "miscellaneous")["base_metric"].tolist(),
            ["custom_counter"],
        )

    def test_category_for_metric_id(self):
        self.assertEqual(category_for_metric_id(processed_rows(), NVML_POWER_ID), "power")


if __name__ == "__main__":
    unittest.main()
