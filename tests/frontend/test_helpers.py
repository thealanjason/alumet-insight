import unittest

import pandas as pd

from frontend.helpers import (
    available_category_options,
    ensure_timestamp_datetime,
    normalize_dropdown_value,
)
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

    def test_ensure_timestamp_datetime(self):
        raw = pd.DataFrame({"timestamp": ["2024-01-01", "2024-01-02"], "value": [1, 2]})
        converted = ensure_timestamp_datetime(raw.copy())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(converted["timestamp"]))

        already_dt = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "value": [1]})
        unchanged = ensure_timestamp_datetime(already_dt)
        pd.testing.assert_frame_equal(unchanged, already_dt)

    def test_ensure_timestamp_datetime_leaves_df_without_timestamp_unchanged(self):
        df = pd.DataFrame({"value": [1, 2]})
        unchanged = ensure_timestamp_datetime(df.copy())
        pd.testing.assert_frame_equal(unchanged, df)


if __name__ == "__main__":
    unittest.main()
