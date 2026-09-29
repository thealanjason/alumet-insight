import base64
import shutil
import unittest
from pathlib import Path

from frontend.panes.sidebar import stage_uploaded_folder
from tests.fixtures import sample_csv_body


class StageUploadTests(unittest.TestCase):
    def test_stage_uploaded_folder_writes_directory(self):
        payload = "data:text/csv;base64," + base64.b64encode(sample_csv_body().encode()).decode()
        store, _status = stage_uploaded_folder(
            [payload],
            ["experiment/measurement.csv"],
            None,
            None,
        )
        self.addCleanup(shutil.rmtree, store["path"], True)
        self.assertEqual(store["name"], "experiment")
        self.assertTrue((Path(store["path"]) / "measurement.csv").is_file())
