import unittest

import pandas as pd

from frontend.helpers import available_category_options, normalize_dropdown_value
from tests.fixtures import processed_rows


class HelpersTests(unittest.TestCase):
    def test_normalize_dropdown_value(self):
        self.assertIsNone(normalize_dropdown_value(None))
        self.assertIsNone(normalize_dropdown_value(""))
        self.assertIsNone(normalize_dropdown_value("nan"))
        self.assertIsNone(normalize_dropdown_value("  NaN  "))
        self.assertEqual(normalize_dropdown_value("  cpu  "), "cpu")
        self.assertEqual(normalize_dropdown_value(123), "123")

    def test_available_category_options(self):
        options = available_category_options(processed_rows())

        self.assertEqual(
            [opt["value"] for opt in options],
            ["energy", "power", "memory", "kernel_cpu_time"],
        )
        self.assertEqual(options[0]["label"], "Energy (J)")
        self.assertEqual(options[1]["label"], "Power (W)")

    def test_available_category_options_empty_dataframe(self):
        empty = pd.DataFrame(columns=["metric_id", "base_metric", "timestamp", "value"])
        self.assertEqual(available_category_options(empty), [])


if __name__ == "__main__":
    unittest.main()
