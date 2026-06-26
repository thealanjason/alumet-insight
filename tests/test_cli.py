import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.fixtures import write_measurement_directory, sample_csv_body


def _run_cli(argv: list[str]) -> None:
    from cli import main
    main(argv)


class CLIValidationTests(unittest.TestCase):
    """Test CLI argument validation and discovery commands."""

    def _make_dir(self, tmp: str) -> Path:
        root = Path(tmp)
        write_measurement_directory(root, csv_body=sample_csv_body())
        return root

    def test_summary_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with patch("builtins.print") as mock_print:
                _run_cli([str(root), "--summary"])
            output = mock_print.call_args_list[0][0][0]
            self.assertIn("Alumet Measurement Summary", output)
            self.assertIn("Next Steps", output)

    def test_list_metric_ids_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with patch("builtins.print") as mock_print:
                _run_cli([str(root), "--list-metric-ids"])
            output = mock_print.call_args_list[0][0][0]
            self.assertIn("All metric IDs", output)
            self.assertIn("Total:", output)

    def test_list_metric_ids_with_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with patch("builtins.print") as mock_print:
                _run_cli([str(root), "--list-metric-ids", "--category", "utilization"])
            output = mock_print.call_args_list[0][0][0]
            self.assertIn("category: utilization", output)

    def test_list_metric_ids_filtered_by_metric_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with patch("builtins.print") as mock_print:
                _run_cli([str(root), "--list-metric-ids", "--metric-name", "nvml_instant_power_W"])
            output = mock_print.call_args_list[0][0][0]
            self.assertIn("nvml_instant_power_W_R_gpu_0_C_process_123_A_", output)
            self.assertNotIn("cpu_percent", output)

    def test_list_metric_ids_filtered_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with patch("builtins.print") as mock_print:
                _run_cli([str(root), "--list-metric-ids", "--category", "power"])
            output = mock_print.call_args_list[0][0][0]
            self.assertIn("nvml_instant_power_W_R_gpu_0_C_process_123_A_", output)
            self.assertNotIn("mem_total_kB", output)

    def test_list_metric_ids_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with patch("builtins.print") as mock_print:
                _run_cli([str(root), "--list-metric-ids", "--limit", "1"])
            output = mock_print.call_args_list[0][0][0]
            self.assertIn("more", output)

    def test_invalid_metric_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with self.assertRaises(SystemExit):
                _run_cli([str(root), "--export-csv", str(tmp), "--metric-id", "nonexistent_metric"])

    def test_invalid_metric_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with self.assertRaises(SystemExit):
                _run_cli([str(root), "--list-metric-ids", "--metric-name", "nonexistent"])

    def test_metric_id_and_metric_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with self.assertRaises(SystemExit):
                _run_cli([
                    str(root), "--export-csv", str(tmp),
                    "--metric-id", "x", "--metric-name", "y",
                ])

    def test_cpu_core_without_kernel_category_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with self.assertRaises(SystemExit):
                _run_cli([str(root), "--export-csv", str(tmp), "--category", "energy", "--cpu-core", "0"])

    def test_limit_without_list_metric_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with self.assertRaises(SystemExit):
                _run_cli([str(root), "--summary", "--limit", "10"])

    def test_invalid_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with self.assertRaises(SystemExit):
                _run_cli([str(root), "--export-csv", str(tmp), "--start-time", "not-a-date"])

    def test_metric_name_with_export_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with self.assertRaises(SystemExit):
                _run_cli([str(root), "--export-csv", str(tmp), "--metric-name", "cpu_percent"])

    def test_export_figures_without_filter_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            out = Path(tmp) / "figs"
            stderr = io.StringIO()
            with patch("cli.export_figures", return_value=[]):
                with patch("sys.stderr", stderr):
                    _run_cli([str(root), "--export-figures", str(out)])
            self.assertIn("Warning:", stderr.getvalue())
            self.assertIn("metric series", stderr.getvalue())

    def test_export_figures_with_category_does_not_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            out = Path(tmp) / "figs"
            stderr = io.StringIO()
            with patch("cli.export_figures", return_value=[]):
                with patch("sys.stderr", stderr):
                    _run_cli([str(root), "--export-figures", str(out), "--category", "power"])
            self.assertNotIn("Warning:", stderr.getvalue())

    def test_entry_point_forwards_cli_help(self):
        import alumet_insight

        with patch.object(sys, "argv", ["alumet_insight.py", "cli", "-h"]):
            with patch("cli.main") as cli_main:
                alumet_insight.main()
        cli_main.assert_called_once_with(["--help"])

    def test_entry_point_rejects_forwarding_separators(self):
        import alumet_insight

        for separator in ("--", "-"):
            with patch.object(sys, "argv", ["alumet_insight.py", "cli", separator, "/path", "--summary"]):
                with self.assertRaises(SystemExit):
                    alumet_insight.main()


if __name__ == "__main__":
    unittest.main()
