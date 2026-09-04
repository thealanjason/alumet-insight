import unittest

from backend.formatting import (
    format_bytes_ticklabel,
    format_metric_choice_label,
    format_metric_title,
    get_bytes_tickvals_ticktext,
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
            format_metric_choice_label("attributed_energy_cpu_J", derived=True),
            "attributed_energy_cpu_J (derived)",
        )
        title = format_metric_title(
            "attributed_energy_cpu_cumulative_J_R_local_machine__C_process_1_A_",
            derived=True,
        )
        self.assertIn("attributed_energy_cpu_cumulative_J", title)
        self.assertTrue(title.endswith("(derived)"))


if __name__ == "__main__":
    unittest.main()
