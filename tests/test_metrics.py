import unittest

from backend.formatting import format_bytes_ticklabel, get_bytes_tickvals_ticktext
from backend.metrics import (
    get_metric_unit,
    is_cumulative_metric,
    is_memory_metric,
    metric_id_is_process_consumer,
)


class MetricTests(unittest.TestCase):
    def test_metric_id_is_process_consumer(self):
        self.assertTrue(metric_id_is_process_consumer("metric_R_x_C_process_123_A_"))
        self.assertFalse(metric_id_is_process_consumer("metric_R_x_C_host_123_A_"))

    def test_is_cumulative_metric(self):
        self.assertTrue(is_cumulative_metric("rapl_consumed_energy_J"))
        self.assertTrue(is_cumulative_metric("perf_hardware_INSTRUCTIONS"))
        self.assertFalse(is_cumulative_metric("cpu_percent"))
        self.assertFalse(is_cumulative_metric("nvml_instant_power_mW"))
        self.assertFalse(is_cumulative_metric("mem_total_kB"))

    def test_get_metric_unit(self):
        self.assertEqual(get_metric_unit("nvml_instant_power_mW"), "mW")
        self.assertEqual(get_metric_unit("nvml_energy_consumption_mJ"), "mJ")
        self.assertEqual(get_metric_unit("nvml_temperature_C"), "°C")
        self.assertEqual(get_metric_unit("attributed_energy_J"), "J")
        self.assertEqual(get_metric_unit("mem_total_kB"), "B")
        self.assertEqual(get_metric_unit("kernel_cpu_time_ms"), "ms")
        self.assertEqual(get_metric_unit("cpu_percent"), "%")
        self.assertEqual(get_metric_unit("kernel_n_procs_running"), "")

    def test_is_memory_metric(self):
        self.assertTrue(is_memory_metric("mem_total_kB"))
        self.assertFalse(is_memory_metric("nvml_memory_utilization_%"))

    def test_format_bytes_ticklabel(self):
        self.assertEqual(format_bytes_ticklabel(512), "512 B")
        self.assertEqual(format_bytes_ticklabel(2048), "2.0 KB")
        self.assertEqual(format_bytes_ticklabel(2048 ** 2), "4.0 MB")

    def test_get_bytes_tickvals_ticktext(self):
        tickvals, ticktext = get_bytes_tickvals_ticktext(0, 2048, num_ticks=3)
        self.assertEqual(len(tickvals), len(ticktext))
        self.assertTrue(all(val >= 0 for val in tickvals))


if __name__ == "__main__":
    unittest.main()
