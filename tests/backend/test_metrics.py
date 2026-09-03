import unittest

import pandas as pd

from backend.metrics import (
    MetricId,
    MetricType,
    PowerKind,
    attach_unit_column,
    classification_stem,
    classify_base_metric,
    filter_process_metric_ids,
    get_metric_unit,
    is_counterdiff_base_metric,
    is_counterdiff_metric,
    is_cumulative_metric,
    is_derived_power_metric,
    is_measured_power_metric,
    is_memory_metric,
    is_power_metric,
    is_raw_counter_base_metric,
    is_raw_counter_metric,
    is_running_total_metric,
    is_spike_metric,
    is_step_power_metric,
    memory_kind,
    metric_id_is_process_consumer,
    metric_type,
    power_kind,
    running_total_base_metric,
    running_total_metric_id,
    should_derive_power_from_energy,
)


class MetricIdTests(unittest.TestCase):
    def test_parse_and_serialize_canonical_id(self):
        metric_id = "attributed_energy_cpu_J_R_cpu_0_C_process_123_A_kind_user"
        parsed = MetricId.parse(metric_id)
        self.assertEqual(parsed.base_metric, "attributed_energy_cpu_J")
        self.assertEqual(parsed.resource, "cpu_0")
        self.assertEqual(parsed.consumer, "process_123")
        self.assertEqual(parsed.late_attributes, "kind_user")
        self.assertEqual(parsed.serialized, metric_id)

    def test_metric_id_with_base_only_round_trips(self):
        parsed = MetricId.parse("cpu_percent")
        self.assertEqual(parsed.base_metric, "cpu_percent")
        self.assertIsNone(parsed.resource)
        self.assertEqual(parsed.serialized, "cpu_percent")

    def test_empty_components_are_distinct_from_missing_markers(self):
        metric_id = "kernel_n_procs_running_R_local_machine_C__A_"
        parsed = MetricId.parse(metric_id)
        self.assertEqual(parsed.resource, "local_machine")
        self.assertEqual(parsed.consumer, "")
        self.assertEqual(parsed.late_attributes, "")
        self.assertEqual(parsed.serialized, metric_id)

    def test_replacing_base_preserves_series_identity(self):
        energy = MetricId.parse("nvml_energy_consumption_J_R_gpu_1_C_process_42_A_user")
        power = energy.with_base_metric("nvml_average_power_W")
        self.assertEqual(power.series_key, energy.series_key)
        self.assertEqual(
            power.serialized,
            "nvml_average_power_W_R_gpu_1_C_process_42_A_user",
        )

    def test_known_component_accessors(self):
        process = MetricId.parse("cpu_percent_R_local_machine__C_process_123_A_")
        cpu_core = MetricId.parse("kernel_cpu_time_ms_R_cpu_core_7.0_C_process_123_A_")
        self.assertTrue(process.is_process_consumer)
        self.assertEqual(process.process_id, "123")
        self.assertEqual(process.resource_id("local_machine"), "")
        self.assertEqual(cpu_core.resource_id("cpu_core"), "7.0")

    def test_invalid_component_name_is_rejected(self):
        parsed = MetricId.parse("cpu_percent")
        with self.assertRaises(ValueError):
            parsed.component_id("attribution", "kind")


class MetricKindTests(unittest.TestCase):
    def test_exact_counterdiff_and_unknown_defaults_to_gauge(self):
        self.assertTrue(is_counterdiff_base_metric("rapl_consumed_energy_J"))
        self.assertFalse(is_counterdiff_base_metric("perf_hardware_INSTRUCTIONS"))
        self.assertFalse(is_counterdiff_base_metric("cpu_percent"))
        unknown = classify_base_metric("totally_unknown_metric_xyz")
        self.assertEqual(unknown.metric_type, MetricType.GAUGE)
        self.assertEqual(metric_type("totally_unknown_metric_xyz"), MetricType.GAUGE)

    def test_classification_stem_strips_common_unit_suffixes(self):
        self.assertEqual(classification_stem("cpu_time_delta_ns"), "cpu_time_delta")
        self.assertEqual(
            classification_stem("amd_gpu_energy_consumption_mJ"),
            "amd_gpu_energy_consumption",
        )
        self.assertEqual(
            classification_stem("amd_gpu_energy_consumption_J"),
            "amd_gpu_energy_consumption",
        )
        self.assertEqual(
            classification_stem("perf_hardware_INSTRUCTIONS"),
            "perf_hardware_INSTRUCTIONS",
        )

    def test_interval_energy_and_cpu_time_counterdiff_classification(self):
        counterdiff_metrics = (
            "amd_gpu_energy_consumption_J",
            "amd_gpu_energy_consumption_mJ",
            "grace_energy_consumption_J",
            "grace_energy_consumption_mJ",
            "cpu_time_delta",
            "cpu_time_delta_ns",
        )
        for metric in counterdiff_metrics:
            with self.subTest(metric=metric):
                self.assertTrue(is_counterdiff_base_metric(metric))
                self.assertEqual(metric_type(metric), MetricType.COUNTER_DIFF)

    def test_all_perf_families_are_raw_counters(self):
        perf_metrics = (
            "perf_hardware_INSTRUCTIONS",
            "perf_software_PAGE_FAULTS",
            "perf_cache_LL_READ_MISS",
        )
        for metric in perf_metrics:
            with self.subTest(metric=metric):
                self.assertTrue(is_raw_counter_base_metric(metric))
                self.assertTrue(is_raw_counter_metric(f"{metric}_R_cpu_0_C_process_1_A_"))
                self.assertFalse(is_counterdiff_metric(metric))
                self.assertFalse(is_spike_metric(metric))
                self.assertEqual(metric_type(metric), MetricType.RAW_COUNTER)

    def test_false_positive_substrings_are_not_counterdiff(self):
        self.assertFalse(is_counterdiff_metric("project_energy_budget_note"))
        self.assertFalse(is_counterdiff_metric("memory_page_fault_hint"))
        self.assertFalse(is_cumulative_metric("gpu_power_state"))
        self.assertFalse(is_counterdiff_metric("kernel_n_procs_running"))

    def test_measured_and_derived_power_classification(self):
        measured = classify_base_metric("nvml_instant_power_W")
        self.assertEqual(measured.power_kind, PowerKind.MEASURED)
        self.assertEqual(power_kind("nvml_instant_power_W"), PowerKind.MEASURED)
        self.assertTrue(is_measured_power_metric("nvml_instant_power_W"))

        derived = classify_base_metric("rapl_average_power_W")
        self.assertEqual(derived.power_kind, PowerKind.DERIVED)
        self.assertEqual(power_kind("rapl_average_power_W"), PowerKind.DERIVED)
        self.assertTrue(is_derived_power_metric("rapl_average_power_W"))
        self.assertFalse(is_counterdiff_metric("rapl_average_power_W"))
        self.assertTrue(is_spike_metric("rapl_consumed_energy_J"))
        self.assertFalse(is_spike_metric("rapl_average_power_W"))
        self.assertTrue(is_step_power_metric("rapl_average_power_W"))
        self.assertFalse(is_step_power_metric("nvml_instant_power_W"))

        non_power = classify_base_metric("cpu_percent")
        self.assertEqual(non_power.power_kind, PowerKind.NONE)
        self.assertEqual(power_kind("cpu_percent"), PowerKind.NONE)

    def test_is_cumulative_metric(self):
        self.assertTrue(is_cumulative_metric("rapl_consumed_energy_J"))
        self.assertFalse(is_cumulative_metric("perf_hardware_INSTRUCTIONS"))
        self.assertFalse(is_cumulative_metric("cpu_percent"))
        self.assertFalse(is_cumulative_metric("nvml_instant_power_mW"))
        self.assertFalse(is_cumulative_metric("mem_total_B"))
        self.assertFalse(is_cumulative_metric("mem_total_kB"))
        self.assertFalse(is_cumulative_metric("attributed_energy_cpu_cumulative_J"))

    def test_running_total_naming_and_classification(self):
        self.assertEqual(
            running_total_base_metric("attributed_energy_cpu_J"),
            "attributed_energy_cpu_cumulative_J",
        )
        self.assertEqual(
            running_total_base_metric("kernel_cpu_time_ms"),
            "kernel_cpu_time_cumulative_ms",
        )
        self.assertEqual(
            running_total_metric_id("attributed_energy_cpu_J_R_pkg_C_process_1_A_"),
            "attributed_energy_cpu_cumulative_J_R_pkg_C_process_1_A_",
        )
        self.assertTrue(is_running_total_metric("attributed_energy_cpu_cumulative_J"))
        self.assertFalse(is_counterdiff_metric("attributed_energy_cpu_cumulative_J"))
        self.assertFalse(is_spike_metric("attributed_energy_cpu_cumulative_J"))
        self.assertEqual(metric_type("attributed_energy_cpu_cumulative_J"), MetricType.GAUGE)
        self.assertFalse(
            should_derive_power_from_energy(
                "attributed_energy_cpu_cumulative_J_R_pkg_C_process_1_A_",
                [],
            )
        )

    def test_is_power_metric_only_accepts_supported_power_series(self):
        self.assertTrue(is_power_metric("nvml_instant_power_W"))
        self.assertTrue(is_power_metric("rapl_average_power_W"))
        self.assertTrue(is_power_metric("attributed_power_total_W"))
        self.assertFalse(is_power_metric("nvml_energy_consumption_J"))
        self.assertFalse(is_power_metric("gpu_power_state"))
        self.assertFalse(is_power_metric("gpu_power_cap"))

    def test_metric_type_classification(self):
        self.assertEqual(metric_type("rapl_consumed_energy_J"), MetricType.COUNTER_DIFF)
        self.assertEqual(metric_type("attributed_energy_cpu_J"), MetricType.COUNTER_DIFF)
        self.assertEqual(metric_type("perf_hardware_INSTRUCTIONS"), MetricType.RAW_COUNTER)
        self.assertEqual(metric_type("perf_cache_LL_READ_MISS"), MetricType.RAW_COUNTER)
        self.assertEqual(metric_type("nvml_instant_power_W"), MetricType.GAUGE)
        self.assertEqual(metric_type("rapl_average_power_W"), MetricType.GAUGE)


class MetricSeriesTests(unittest.TestCase):
    def test_metric_id_is_process_consumer(self):
        self.assertTrue(metric_id_is_process_consumer("metric_R_x_C_process_123_A_"))
        self.assertFalse(metric_id_is_process_consumer("metric_R_x_C_host_123_A_"))

    def test_filter_process_metric_ids(self):
        ids = ["host_R_a_C_host_1_A_", "proc_R_a_C_process_1_A_"]
        self.assertEqual(filter_process_metric_ids(ids, process_only=False), ids)
        self.assertEqual(filter_process_metric_ids(ids, process_only=True), [ids[1]])


class MetricUnitTests(unittest.TestCase):
    def test_get_metric_unit(self):
        self.assertEqual(get_metric_unit("nvml_instant_power_mW"), "mW")
        self.assertEqual(get_metric_unit("nvml_instant_power_W"), "W")
        self.assertEqual(get_metric_unit("nvml_energy_consumption_mJ"), "mJ")
        self.assertEqual(get_metric_unit("nvml_temperature_C"), "°C")
        self.assertEqual(get_metric_unit("attributed_energy_J"), "J")
        self.assertEqual(get_metric_unit("rapl_average_power_W"), "W")
        self.assertEqual(get_metric_unit("attributed_power_total_W"), "W")
        self.assertEqual(get_metric_unit("mem_total_kB"), "B")
        self.assertEqual(get_metric_unit("mem_total_B"), "B")
        self.assertEqual(get_metric_unit("active_B"), "B")
        self.assertEqual(get_metric_unit("nvml_gpu_memory_info_B"), "B")
        self.assertEqual(get_metric_unit("kernel_cpu_time_ms"), "ms")
        self.assertEqual(get_metric_unit("cpu_percent"), "%")
        self.assertEqual(get_metric_unit("kernel_n_procs_running"), "")

    def test_units_use_only_the_base_metric(self):
        metric_id = "cpu_percent_R_local_machine__C_process_123_A_"
        self.assertEqual(get_metric_unit(metric_id), "%")

    def test_is_memory_metric(self):
        self.assertTrue(is_memory_metric("mem_total_kB"))
        self.assertTrue(is_memory_metric("mem_total_B"))
        self.assertTrue(is_memory_metric("active_B"))
        self.assertTrue(is_memory_metric("inactive_B"))
        self.assertTrue(is_memory_metric("cached_B"))
        self.assertTrue(is_memory_metric("mapped_B"))
        self.assertTrue(is_memory_metric("swap_cached_B"))
        self.assertTrue(is_memory_metric("memory_usage_B"))
        self.assertTrue(is_memory_metric("nvml_gpu_memory_info_B"))
        self.assertFalse(is_memory_metric("nvml_memory_utilization_%"))
        self.assertFalse(is_memory_metric("cpu_percent"))

    def test_memory_kind_distinguishes_system_process_and_gpu(self):
        self.assertEqual(memory_kind("active_kB"), "system")
        self.assertEqual(memory_kind("active_B"), "system")
        self.assertEqual(memory_kind("memory_usage_B"), "process")
        self.assertEqual(memory_kind("nvml_gpu_memory_info_B"), "gpu")
        self.assertIsNone(memory_kind("nvml_memory_utilization_%"))

    def test_attach_unit_column_inserts_after_value(self):
        df = pd.DataFrame({"metric": ["mem_available_kB", "cpu_percent"], "value": [1024.0, 50.0]})
        out = attach_unit_column(df)
        self.assertEqual(out["unit"].tolist(), ["B", "%"])
        self.assertEqual(list(out.columns), ["metric", "value", "unit"])


if __name__ == "__main__":
    unittest.main()
