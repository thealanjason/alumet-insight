import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backend.figures import save_metric_time_series_figure


class BackendFiguresTests(unittest.TestCase):
    def test_counterdiff_export_draws_spike_markers(self):
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-01-01T00:00:01"),
                    pd.Timestamp("2024-01-01T00:00:01"),
                    pd.Timestamp("2024-01-01T00:00:02"),
                    pd.Timestamp("2024-01-01T00:00:02"),
                ],
                "value": [0.0, 2.5, 0.0, 0.0],
                "point_role": ["synthetic", "observed", "synthetic", "observed"],
                "point_order": [0, 1, 0, 1],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "energy.png"
            with patch("backend.figures._plot_counterdiff_spike_markers") as markers:
                save_metric_time_series_figure(
                    df,
                    path,
                    category="energy",
                    metric_id="rapl_consumed_energy_J_R_pkg_0_C__A_",
                )
            markers.assert_called_once()
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
