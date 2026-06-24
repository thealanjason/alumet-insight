import unittest

from backend.metrics import (
    filter_process_metric_ids,
    get_metric_unit,
    is_cumulative_metric,
    is_memory_metric,
    metric_id_is_process_consumer,
)


class MetricTests(unittest.TestCase):
    def test_metric_id_is_process_consumer(self):
        self.assertTrue(metric_id_is_process_consumer("metric_R_x_C_process_123_A_"))
        self.assertFalse(metric_id_is_process_consumer("metric_R_x_C_host_123_A_"))

    def test_filter_process_metric_ids(self):
        ids = ["host_R_a_C_host_1_A_", "proc_R_a_C_process_1_A_"]
        self.assertEqual(filter_process_metric_ids(ids, process_only=False), ids)
        self.assertEqual(filter_process_metric_ids(ids, process_only=True), [ids[1]])

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


if __name__ == "__main__":
    unittest.main()
