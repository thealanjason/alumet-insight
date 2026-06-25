import unittest

import pandas as pd

from backend.transforms import align_xy_metrics, comparative_metric_ids
from backend.metrics import filter_process_metric_ids
from frontend.panes.comparative import pick_xy_values, prepare_xy_download


class ComparativeTests(unittest.TestCase):
    def test_comparative_metric_selection_and_alignment(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="s").tolist() * 2,
                "metric_id": ["x_R_a_C_process_1_A_"] * 3 + ["y_R_a_C_process_1_A_"] * 3,
                "value": [1, 2, 3, 10, 20, 30],
            }
        )
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-01 00:00:02")

        ids = comparative_metric_ids(df, start, end)
        self.assertEqual(ids, ["x_R_a_C_process_1_A_", "y_R_a_C_process_1_A_"])
        self.assertEqual(filter_process_metric_ids(ids, process_only=True), ids)
        self.assertEqual(pick_xy_values(ids, None, None), (ids[0], ids[1]))

        aligned = align_xy_metrics(df, ids[0], ids[1], start, end)
        self.assertEqual(aligned[["x", "y"]].values.tolist(), [[1, 10], [2, 20], [3, 30]])
        exported, filename = prepare_xy_download(aligned, ids[0], ids[1])
        self.assertIn(ids[0], exported.columns)
        self.assertTrue(filename.startswith("xy_"))

    def test_comparative_helpers_handle_empty_and_filtered_inputs(self):
        self.assertEqual(comparative_metric_ids(pd.DataFrame(), None, None), [])
        self.assertEqual(comparative_metric_ids(pd.DataFrame({"metric_id": ["a"], "timestamp": [pd.Timestamp("2024-01-01")], "value": [1]}), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")), [])

        ids = ["host_R_a_C_host_1_A_", "proc_R_a_C_process_1_A_"]
        self.assertEqual(filter_process_metric_ids(ids, process_only=False), ids)
        self.assertEqual(filter_process_metric_ids(ids, process_only=True), [ids[1]])
        self.assertEqual(pick_xy_values(["only"], None, None), ("only", "only"))
        self.assertEqual(pick_xy_values(["a", "b"], "b", "a"), ("b", "a"))

        empty_aligned = align_xy_metrics(
            pd.DataFrame(columns=["metric_id", "timestamp", "value"]),
            "a",
            "b",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-02"),
        )
        self.assertTrue(empty_aligned.empty)

    def test_prepare_xy_download_sanitizes_filename(self):
        aligned = pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01")], "x": [1.0], "y": [2.0]})
        _, filename = prepare_xy_download(aligned, "bad/id", "also bad")
        self.assertNotIn("/", filename)


if __name__ == "__main__":
    unittest.main()
