import tempfile
import unittest
from pathlib import Path

import pandas as pd

import base64

from backend.utils import (
    extract_pid_from_content,
    find_measurement_file_in_directory,
    is_cpu_from_content,
    is_cpu_from_metrics,
    is_gpu_from_content,
    is_gpu_from_metrics,
    prefer_relative_upload_paths,
    experiment_name_from_upload_filenames,
    save_upload_to_temp_dir,
    read_file_content,
    safe_filename,
)


class UtilsTests(unittest.TestCase):
    def test_find_measurement_file_in_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.csv").write_text("x", encoding="utf-8")
            (root / "b.log").write_text("y", encoding="utf-8")

            self.assertEqual(find_measurement_file_in_directory(str(root), [".csv"]).name, "a.csv")
            with self.assertRaises(ValueError):
                find_measurement_file_in_directory(str(root), [".toml"])
            with self.assertRaises(ValueError):
                find_measurement_file_in_directory(str(root / "missing"), [".csv"])

    def test_read_file_content(self):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as handle:
            handle.write("hello")
            path = Path(handle.name)

        self.assertEqual(read_file_content(path), "hello")
        path.unlink()
        with self.assertRaises(ValueError):
            read_file_content(Path("missing-file.txt"))

    def test_extract_pid_from_content(self):
        self.assertEqual(extract_pid_from_content("starting alumet\npid 12345\nloaded"), 12345)
        self.assertIsNone(extract_pid_from_content(""))
        self.assertIsNone(extract_pid_from_content("no pid here"))

    def test_is_gpu_and_cpu_from_content(self):
        content = "starting alumet\npid 1\nloaded nvml and rapl plugins"
        self.assertTrue(is_gpu_from_content(content))
        self.assertTrue(is_cpu_from_content(content))
        self.assertFalse(is_gpu_from_content(""))
        self.assertFalse(is_cpu_from_content("plain log"))
        self.assertFalse(is_gpu_from_content("rapl only"))
        self.assertFalse(is_cpu_from_content("nvml only"))

    def test_safe_filename(self):
        self.assertEqual(safe_filename("a/b c"), "a_b_c")
        self.assertEqual(safe_filename("valid-name.csv"), "valid-name.csv")
        self.assertEqual(safe_filename("metric:value!"), "metric_value_")
        self.assertNotIn("/", safe_filename("bad/id"))

    def test_is_gpu_and_cpu_from_metrics(self):
        df_gpu = pd.DataFrame({"base_metric": ["nvml_instant_power_W"]})
        df_cpu = pd.DataFrame({"base_metric": ["rapl_consumed_energy_J"]})
        df_both = pd.DataFrame({"base_metric": ["nvml_instant_power_W", "rapl_consumed_energy_J"]})
        df_empty = pd.DataFrame(columns=["base_metric"])

        self.assertTrue(is_gpu_from_metrics(df_gpu))
        self.assertFalse(is_cpu_from_metrics(df_gpu))
        self.assertTrue(is_cpu_from_metrics(df_cpu))
        self.assertFalse(is_gpu_from_metrics(df_cpu))
        self.assertTrue(is_gpu_from_metrics(df_both))
        self.assertTrue(is_cpu_from_metrics(df_both))
        self.assertFalse(is_gpu_from_metrics(df_empty))
        self.assertFalse(is_cpu_from_metrics(df_empty))

    def test_save_upload_to_temp_dir(self):
        def _payload(text: str) -> str:
            return "data:text/plain;base64," + base64.b64encode(text.encode("utf-8")).decode("ascii")

        dir_path, experiment_name = save_upload_to_temp_dir(
            [_payload("csv-body"), _payload("log-body"), _payload("skip-me")],
            ["runA/data.csv", "runA/agent.log", "runA/notes.md"],
        )
        try:
            self.assertEqual(experiment_name, "runA")
            self.assertTrue((dir_path / "data.csv").exists())
            self.assertTrue((dir_path / "agent.log").exists())
            self.assertFalse((dir_path / "notes.md").exists())
            self.assertEqual((dir_path / "data.csv").read_text(encoding="utf-8"), "csv-body")
        finally:
            for path in dir_path.iterdir():
                path.unlink()
            dir_path.rmdir()

        with self.assertRaises(ValueError):
            save_upload_to_temp_dir(None, None)
        with self.assertRaises(ValueError):
            save_upload_to_temp_dir([_payload("x")], ["runA/notes.md"])

    def test_experiment_name_from_upload_filenames(self):
        self.assertEqual(
            experiment_name_from_upload_filenames(["runA/data.csv", "runA/agent.log"]),
            "runA",
        )
        self.assertEqual(
            experiment_name_from_upload_filenames(
                ["runA/data.csv", "alumet-output-runA.csv", "notes.md"]
            ),
            "runA",
        )
        self.assertEqual(experiment_name_from_upload_filenames(None), "N/A")
        self.assertEqual(
            experiment_name_from_upload_filenames(["alumet-output-exp1.csv"]),
            "N/A",
        )

    def test_prefer_relative_upload_paths(self):
        self.assertEqual(
            prefer_relative_upload_paths(
                ["alumet-output-runA.csv"],
                ["runA/alumet-output-runA.csv"],
            ),
            ["runA/alumet-output-runA.csv"],
        )
        self.assertEqual(
            prefer_relative_upload_paths(["alumet-output-runA.csv"], None),
            ["alumet-output-runA.csv"],
        )
        self.assertEqual(
            prefer_relative_upload_paths(
                ["alumet-output-runA.csv", "alumet-agent-runA.log"],
                [
                    "runA/alumet-output-runA.csv",
                    "runA/alumet-agent-runA.log",
                    "runA/notes.md",
                    "runA/readme.txt",
                ],
            ),
            ["runA/alumet-output-runA.csv", "runA/alumet-agent-runA.log"],
        )
        self.assertEqual(
            experiment_name_from_upload_filenames(
                prefer_relative_upload_paths(
                    ["alumet-output-runA.csv", "alumet-agent-runA.log"],
                    ["runA/alumet-output-runA.csv", "runA/alumet-agent-runA.log"],
                )
            ),
            "runA",
        )


if __name__ == "__main__":
    unittest.main()
