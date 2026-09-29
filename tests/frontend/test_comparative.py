import unittest

import pandas as pd

from backend.metrics import same_physical_xy_unit
from frontend.panes.comparative import pick_xy_values, update_process_xy_plot
from tests.fixtures import (
    CPU_ENERGY_ID,
    GPU_ENERGY_ID,
    OFFSET_CPU_GPU_X_TOTAL,
    OFFSET_CPU_GPU_Y_TOTAL,
    offset_cpu_gpu_energy_rows,
)


class ComparativeTests(unittest.TestCase):
    def test_pick_xy_values_defaults_and_preserves_selection(self):
        self.assertEqual(pick_xy_values([], None, None), (None, None))
        self.assertEqual(pick_xy_values(["only"], None, None), ("only", "only"))
        self.assertEqual(pick_xy_values(["a", "b"], None, None), ("a", "b"))
        self.assertEqual(pick_xy_values(["a", "b"], "b", "a"), ("b", "a"))

    def test_dual_timeseries_spike_hover_uses_only_measured_peaks(self):
        x_metric = "mem_active_B_R_local_machine__C_process_4_A_"
        y_metric = "attributed_energy_cpu_J_R_pkg_C_process_4_A_"
        t0 = pd.Timestamp("2024-01-01 00:00:00")
        t1 = pd.Timestamp("2024-01-01 00:00:01")
        figure, _title = update_process_xy_plot(
            x_metric,
            y_metric,
            [],
            False,
            [
                {"timestamp": t0, "metric_id": x_metric, "value": 100.0},
                {"timestamp": t1, "metric_id": x_metric, "value": 110.0},
                {"timestamp": t0, "metric_id": y_metric, "value": 7.0},
                {"timestamp": t1, "metric_id": y_metric, "value": 0.0},
            ],
            {"start": "2024-01-01 00:00:00", "end": "2024-01-01 00:00:01"},
        )

        hoverable_energy = [
            float(y)
            for trace in figure.data
            if trace.name == "attributed_energy_cpu_J" and trace.hoverinfo != "none"
            for y in trace.y
        ]
        self.assertEqual(hoverable_energy, [7.0, 0.0])

    def test_cumulative_xy_uses_running_totals_and_shared_scale(self):
        df = offset_cpu_gpu_energy_rows()
        figure, _title = update_process_xy_plot(
            CPU_ENERGY_ID,
            GPU_ENERGY_ID,
            [],
            False,
            df.to_dict("records"),
            {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
            },
        )

        self.assertEqual(len(figure.data), 1)
        self.assertAlmostEqual(float(figure.data[0].x[-1]), OFFSET_CPU_GPU_X_TOTAL)
        self.assertAlmostEqual(float(figure.data[0].y[-1]), OFFSET_CPU_GPU_Y_TOTAL)
        self.assertEqual(list(figure.layout.xaxis.range), list(figure.layout.yaxis.range))
        self.assertEqual(figure.layout.xaxis.range[0], 0)
        self.assertGreaterEqual(figure.layout.xaxis.range[1], OFFSET_CPU_GPU_X_TOTAL)
        self.assertEqual(figure.layout.xaxis.dtick, figure.layout.yaxis.dtick)
        self.assertFalse(figure.layout.xaxis.autorange)
        self.assertIsNone(figure.layout.yaxis.scaleanchor)
        self.assertTrue(figure.layout.meta["equal_xy"])
        self.assertTrue(same_physical_xy_unit(CPU_ENERGY_ID, GPU_ENERGY_ID))

    def test_scatter_xy_locks_equal_scale_only_when_units_match(self):
        df = offset_cpu_gpu_energy_rows()
        matched, _matched_title = update_process_xy_plot(
            CPU_ENERGY_ID,
            GPU_ENERGY_ID,
            ["scatter"],
            False,
            df.to_dict("records"),
            {
                "start": str(df["timestamp"].min()),
                "end": str(df["timestamp"].max()),
            },
        )
        self.assertEqual(list(matched.layout.xaxis.range), list(matched.layout.yaxis.range))
        self.assertGreater(matched.layout.xaxis.range[0], 0)
        self.assertTrue(matched.layout.meta["equal_xy"])

        x_metric = "cpu_percent_R_local_machine__C_process_4_A_"
        y_metric = "mem_total_B_R_local_machine__C_process_4_A_"
        mismatched, _title = update_process_xy_plot(
            x_metric,
            y_metric,
            ["scatter"],
            False,
            [
                {"timestamp": pd.Timestamp("2024-01-01 00:00:00"), "metric_id": x_metric, "value": 1.0},
                {"timestamp": pd.Timestamp("2024-01-01 00:00:00"), "metric_id": y_metric, "value": 2.0},
            ],
            {"start": "2024-01-01 00:00:00", "end": "2024-01-01 00:00:10"},
        )
        self.assertFalse(mismatched.layout.meta["equal_xy"])
        self.assertFalse(same_physical_xy_unit(x_metric, y_metric))


if __name__ == "__main__":
    unittest.main()
