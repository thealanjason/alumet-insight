import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.fixtures import write_measurement_directory, sample_csv_body


def _run_cli(argv: list[str]) -> None:
    from cli import main

    main(argv)


class CLIBasicTests(unittest.TestCase):
    """Smoke tests for existing CLI flags wired through backend/cli_export."""

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
            self.assertIn("Time Ranges", output)

    def test_export_csv_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            out = Path(tmp) / "export"
            with patch("builtins.print"):
                _run_cli([str(root), "--export-csv", str(out), "--category", "power"])
            self.assertTrue(any(out.rglob("*.csv")))

    def test_no_action_prints_help_and_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_dir(tmp)
            with patch("argparse.ArgumentParser.print_help") as mock_help:
                with self.assertRaises(SystemExit) as ctx:
                    _run_cli([str(root)])
            self.assertEqual(ctx.exception.code, 1)
            mock_help.assert_called_once()


if __name__ == "__main__":
    unittest.main()
