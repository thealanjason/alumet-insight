import unittest

from backend.formatting import (
    format_bytes_ticklabel,
    format_metric_choice_label,
    format_metric_title,
    get_bytes_tickvals_ticktext,
    metric_choice_option,
    shared_xy_axis_dtick,
    shared_xy_axis_range,
    split_metric_title,
)


class FormattingTests(unittest.TestCase):
    def test_format_bytes_ticklabel(self):
        self.assertEqual(format_bytes_ticklabel(512), "512 B")
        self.assertEqual(format_bytes_ticklabel(2048), "2.0 KB")
        self.assertEqual(format_bytes_ticklabel(2048 ** 2), "4.0 MB")

    def test_get_bytes_tickvals_ticktext(self):
        tickvals, ticktext = get_bytes_tickvals_ticktext(0, 2048, num_ticks=3)
        self.assertEqual(len(tickvals), len(ticktext))
        self.assertTrue(all(val >= 0 for val in tickvals))

    def test_derived_title_and_choice_labels(self):
        self.assertEqual(format_metric_choice_label("attributed_energy_cpu_J"), "attributed_energy_cpu_J")
        self.assertEqual(
            metric_choice_option("attributed_energy_cpu_J", derived=True),
            {"label": "attributed_energy_cpu_J (derived)", "value": "attributed_energy_cpu_J"},
        )
        self.assertEqual(
            format_metric_choice_label("attributed_energy_cpu_J", derived=True),
            "attributed_energy_cpu_J (derived)",
        )
        title = format_metric_title(
            "attributed_energy_cpu_cumulative_J_R_local_machine__C_process_1_A_",
            derived=True,
        )
        self.assertFalse(title.startswith("CPU "))
        self.assertIn("attributed_energy_cpu_cumulative_J", title)
        self.assertTrue(title.endswith("(derived)"))
        name, meta = split_metric_title(format_metric_title(
            "attributed_energy_cpu_cumulative_J_R_local_machine__C_process_1_A_"
        ))
        self.assertEqual(name, "attributed_energy_cpu_cumulative_J")
        self.assertTrue(meta.startswith("R:"))

    def test_shared_xy_axis_range_starts_at_zero_for_cumulative(self):
        self.assertIsNone(shared_xy_axis_range([], [1.0]))
        lo, hi = shared_xy_axis_range([20.0, 100.0], [10.0, 80.0], include_zero=True)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 105.0)

    def test_shared_xy_axis_range_keeps_tight_scatter_cluster(self):
        lo, hi = shared_xy_axis_range([48.0, 52.0], [49.0, 51.0], include_zero=False)
        self.assertLess(lo, 48.0)
        self.assertGreater(lo, 0.0)
        self.assertGreater(hi, 52.0)
        self.assertAlmostEqual(shared_xy_axis_dtick(0.0, 100.0), 20.0)


if __name__ == "__main__":
    unittest.main()
