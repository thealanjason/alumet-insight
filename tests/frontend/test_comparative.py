import unittest

import pandas as pd

from backend.metrics import DeviceClass, same_physical_xy_unit
from frontend.panes.comparative import (
    COMPARATIVE_PLOT_AREA_CLASS,
    EQUAL_XY_PLOT_AREA_CLASS,
    comparative_plot_area_class,
    pick_xy_values,
    update_comparative_mode_info,
    update_process_xy_plot,
)
from frontend.style import DEVICE_CLASS_COLORS_DARK, DEVICE_CLASS_COLORS_LIGHT, plot_pair_colors
from tests.fixtures import (
    CPU_ENERGY_ID,
    GPU_ENERGY_ID,
    MEM_TOTAL_ID,
    NETWORK_RX_ID,
    OFFSET_CPU_GPU_X_TOTAL,
    OFFSET_CPU_GPU_Y_TOTAL,
    offset_cpu_gpu_energy_rows,
)


def _heading_text(node) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, (list, tuple)):
        return "".join(_heading_text(child) for child in node)
    return _heading_text(getattr(node, "children", ""))


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
        self.assertTrue(
            any(
                trace.name == "attributed_energy_cpu_J" and trace.hoverinfo == "none"
                for trace in figure.data
            )
        )

    def test_dual_timeseries_keeps_pair_line_colors_and_class_axis_titles(self):
        x_metric = "cpu_percent_R_local_machine__C_process_4_A_"
        y_metric = NETWORK_RX_ID
        x_time = pd.Timestamp("2024-01-01 00:00:00")
        y_time = pd.Timestamp("2024-01-01 00:00:10")
        figure, title = update_process_xy_plot(
            x_metric,
            y_metric,
            [],
            False,
            [
                {"timestamp": x_time, "metric_id": x_metric, "value": 1.0},
                {"timestamp": y_time, "metric_id": y_metric, "value": 2.0},
            ],
            {"start": "2024-01-01 00:00:00", "end": "2024-01-01 00:00:10"},
        )

        accents = plot_pair_colors(False)
        cpu_color = DEVICE_CLASS_COLORS_DARK[DeviceClass.CPU]
        other_color = DEVICE_CLASS_COLORS_DARK[DeviceClass.OTHER]
        self.assertEqual(list(figure.data[0].x), [x_time])
        self.assertEqual(list(figure.data[1].x), [y_time])
        line_by_name = {trace.name: trace.line for trace in figure.data if getattr(trace, "line", None)}
        self.assertEqual(line_by_name["cpu_percent"].color, accents["x"])
        self.assertEqual(line_by_name["network_rx_bytes"].color, accents["y"])
        self.assertNotEqual(line_by_name["network_rx_bytes"].dash, "dash")
        self.assertEqual(figure.layout.yaxis.tickfont.color, accents["x"])
        self.assertEqual(figure.layout.yaxis2.tickfont.color, accents["y"])
        self.assertEqual(figure.layout.yaxis.title.font.color, cpu_color)
        self.assertEqual(figure.layout.yaxis2.title.font.color, other_color)
        self.assertEqual(figure.layout.legend.orientation, "h")
        self.assertEqual(figure.layout.legend.yref, "container")
        self.assertEqual(figure.layout.legend.yanchor, "bottom")
        self.assertEqual(figure.layout.legend.y, 0)
        self.assertGreaterEqual(figure.layout.margin.b, 120)
        self.assertEqual(figure.layout.xaxis.title.standoff, 10)
        self.assertFalse(figure.layout.title.text)
        self.assertFalse(figure.layout.meta["equal_xy"])
        self.assertFalse(same_physical_xy_unit(x_metric, y_metric))
        self.assertEqual(comparative_plot_area_class(False), COMPARATIVE_PLOT_AREA_CLASS)
        self.assertEqual(_heading_text(title), "Time Series:  cpu_percent  vs  network_rx_bytes")

    def test_dual_series_pair_colors_stay_off_the_device_class_swatches(self):
        for use_light_mode, swatches in (
            (False, DEVICE_CLASS_COLORS_DARK),
            (True, DEVICE_CLASS_COLORS_LIGHT),
        ):
            pair = plot_pair_colors(use_light_mode)
            occupied = set(swatches.values())
            self.assertEqual(pair["x"], "#9467bd")
            self.assertEqual(pair["y"], "#d62728")
            self.assertNotIn(pair["x"], occupied)
            self.assertNotIn(pair["y"], occupied)

    def test_visualization_mode_stays_theme_text_and_bold(self):
        info = update_comparative_mode_info(
            "cpu_percent_R_local_machine__C_process_4_A_",
            "mem_total_B_R_local_machine__C_process_4_A_",
        )
        self.assertEqual(info.style["color"], "var(--app-text)")
        self.assertEqual(info.children[0].style["fontWeight"], "600")
        self.assertEqual(info.children[1].style["fontWeight"], "600")
        self.assertEqual(info.children[1].children, "Dual Y-Axis Time Series")
        self.assertNotIn("color", info.children[1].style)

    def test_cumulative_xy_uses_running_totals_and_shared_scale(self):
        df = offset_cpu_gpu_energy_rows()
        figure, title = update_process_xy_plot(
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
        self.assertEqual(comparative_plot_area_class(True), EQUAL_XY_PLOT_AREA_CLASS)
        self.assertFalse(figure.layout.title.text)
        self.assertEqual(
            _heading_text(title),
            "Cumulative:  attributed_energy_cpu_J  vs  attributed_energy_gpu_J",
        )
        cum_color = plot_pair_colors(False)["cumulative"]
        cpu_color = DEVICE_CLASS_COLORS_DARK[DeviceClass.CPU]
        gpu_color = DEVICE_CLASS_COLORS_DARK[DeviceClass.GPU]
        self.assertEqual(figure.data[0].line.color, cum_color)
        self.assertEqual(figure.data[0].marker.color, cum_color)
        self.assertNotEqual(getattr(figure.layout.xaxis.tickfont, "color", None), cpu_color)
        self.assertNotEqual(getattr(figure.layout.yaxis.tickfont, "color", None), gpu_color)
        self.assertEqual(figure.layout.xaxis.title.font.color, cpu_color)
        self.assertEqual(figure.layout.yaxis.title.font.color, gpu_color)

    def test_scatter_xy_locks_equal_scale_only_when_units_match(self):
        df = offset_cpu_gpu_energy_rows()
        matched, matched_title = update_process_xy_plot(
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
        self.assertEqual(
            _heading_text(matched_title),
            "Scatter:  attributed_energy_cpu_J  vs  attributed_energy_gpu_J",
        )
        self.assertEqual(
            matched.layout.xaxis.title.font.color,
            DEVICE_CLASS_COLORS_DARK[DeviceClass.CPU],
        )
        self.assertEqual(
            matched.layout.yaxis.title.font.color,
            DEVICE_CLASS_COLORS_DARK[DeviceClass.GPU],
        )
        self.assertEqual(matched.data[0].marker.color, plot_pair_colors(False)["scatter"])

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

    def test_same_device_class_still_uses_pair_line_colors(self):
        x_metric = "cpu_percent_R_local_machine__C_process_4_A_"
        figure, _title = update_process_xy_plot(
            x_metric,
            MEM_TOTAL_ID,
            [],
            False,
            [
                {"timestamp": pd.Timestamp("2024-01-01 00:00:00"), "metric_id": x_metric, "value": 1.0},
                {"timestamp": pd.Timestamp("2024-01-01 00:00:10"), "metric_id": MEM_TOTAL_ID, "value": 2.0},
            ],
            {"start": "2024-01-01 00:00:00", "end": "2024-01-01 00:00:10"},
        )
        accents = plot_pair_colors(False)
        cpu = DEVICE_CLASS_COLORS_DARK[DeviceClass.CPU]
        line_by_name = {trace.name: trace.line for trace in figure.data if getattr(trace, "line", None)}
        self.assertEqual(line_by_name["cpu_percent"].color, accents["x"])
        self.assertEqual(line_by_name["mem_total_B"].color, accents["y"])
        self.assertNotEqual(line_by_name["cpu_percent"].dash, "dash")
        self.assertNotEqual(line_by_name["mem_total_B"].dash, "dash")
        self.assertEqual(figure.layout.yaxis.title.font.color, cpu)
        self.assertEqual(figure.layout.yaxis2.title.font.color, cpu)


if __name__ == "__main__":
    unittest.main()
