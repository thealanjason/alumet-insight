import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.utils import (
    extract_pid_from_content,
    find_measurement_file_in_directory,
    is_cpu_from_content,
    is_cpu_from_metrics,
    is_gpu_from_content,
    is_gpu_from_metrics,
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


if __name__ == "__main__":
    unittest.main()
