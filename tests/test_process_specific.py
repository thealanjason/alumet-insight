import unittest

import pandas as pd

from backend.process_specific import (
    cascade_filter_options,
    filter_single_series,
    normalize_filter_columns,
    prepare_download_df,
    safe_metric_filename,
    unique_nonempty,
)


class ProcessSpecificTests(unittest.TestCase):
    def test_unique_nonempty_and_normalize_filter_columns(self):
        series = pd.Series(["cpu", "", None, "gpu", "cpu"])
        self.assertEqual(unique_nonempty(series), ["cpu", "gpu"])
        self.assertEqual(unique_nonempty(pd.Series([None, "", " "])), [])

        normed = normalize_filter_columns(
            pd.DataFrame(
                {
                    "resource_kind": ["cpu"],
                    "resource_id": [None],
                    "consumer_kind": ["process"],
                    "consumer_id": ["10"],
                    "__late_attributes": ["user"],
                }
            )
        )
        self.assertEqual(normed.loc[0, "rk"], "cpu")
        self.assertEqual(normed.loc[0, "rid"], "")

    def test_cascade_filter_options_and_series_filtering(self):
        df = pd.DataFrame(
            {
                "resource_kind": ["cpu", "cpu", "gpu"],
                "resource_id": ["0", "1", "0"],
                "consumer_kind": ["process", "process", ""],
                "consumer_id": ["10", "10", ""],
                "__late_attributes": ["user", "system", ""],
            }
        )
        normed = normalize_filter_columns(df)

        cascade = cascade_filter_options(normed, "cpu", None, "process", "10", None)
        self.assertEqual(cascade["rk"]["effective"], "cpu")
        self.assertIn("0", cascade["rid"]["options"])

        reset = cascade_filter_options(normed, "cpu", "1", "process", "10", "system", triggered_id="resource-kind-dropdown")
        self.assertIsNone(reset["rid"]["effective"])

        filtered, _ = filter_single_series(normed, "cpu", "1", "process", "10", "system")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(safe_metric_filename("a/b c"), "a_b_c")

    def test_prepare_download_df_filters_values_and_time_range(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
                "metric": ["m", "m", "m"],
                "value": [1, 2, 3],
                "resource_kind": ["cpu", "cpu", "cpu"],
                "resource_id": ["0", "0", "1"],
                "consumer_kind": ["process", "process", "process"],
                "consumer_id": ["10", "10", "10"],
                "__late_attributes": ["", "", ""],
            }
        )

        out = prepare_download_df(
            df,
            "m",
            "cpu",
            "0",
            "process",
            "10",
            None,
            pd.Timestamp("2024-01-01 00:00:01"),
            pd.Timestamp("2024-01-01 00:00:02"),
        )

        self.assertEqual(out["value"].tolist(), [2])
        self.assertTrue(prepare_download_df(df, "missing", None, None, None, None, None).empty)

    def test_cascade_resets_dependent_filters_on_consumer_kind_change(self):
        df = pd.DataFrame(
            {
                "resource_kind": ["cpu", "cpu"],
                "resource_id": ["0", "1"],
                "consumer_kind": ["process", "host"],
                "consumer_id": ["10", ""],
                "__late_attributes": ["", ""],
            }
        )
        normed = normalize_filter_columns(df)

        reset = cascade_filter_options(normed, "cpu", "0", "process", "10", None, triggered_id="consumer-kind-dropdown")
        self.assertIsNone(reset["cid"]["effective"])

    def test_prepare_download_df_includes_late_attribute_filter_and_columns(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="s"),
                "metric": ["m", "m"],
                "value": [1, 2],
                "resource_kind": ["cpu", "cpu"],
                "resource_id": ["0", "0"],
                "consumer_kind": ["process", "process"],
                "consumer_id": ["10", "10"],
                "__late_attributes": ["user", "system"],
            }
        )

        out = prepare_download_df(df, "m", "cpu", "0", "process", "10", "user")
        self.assertEqual(out["value"].tolist(), [1])
        self.assertIn("__late_attributes", out.columns)


if __name__ == "__main__":
    unittest.main()
