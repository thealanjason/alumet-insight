import unittest
from unittest.mock import patch

import pandas as pd

from backend.counterdiff import expand_counterdiff_rows
from backend.metrics import DeviceClass, MetricOrigin
from frontend.figures import (
    build_metric_trace_configs,
    create_all_timeseries_plots,
    prepare_trace_coordinates,
)
from frontend.style import (
    DEVICE_CLASS_COLORS_DARK,
    device_class_chip,
    device_class_key,
    device_class_selection_caption,
    format_device_class_title_html,
)
from tests.fixtures import (
    CPU_ENERGY_ID,
    GPU_ENERGY_ID,
    NETWORK_RX_ID,
    RAPL_ENERGY_ID,
    RAPL_POWER_ID,
    concat_series,
    rapl_energy_rows,
    series_rows,
)


class TimeseriesFigureTests(unittest.TestCase):
    """Rendering-policy tests. Spike/step geometry is in test_counterdiff."""

    def test_spike_traces_hide_stem_hover_and_keep_peak_markers(self):
        df = expand_counterdiff_rows(rapl_energy_rows([7.0]))
        stem, peak = build_metric_trace_configs(
            df,
            RAPL_ENERGY_ID,
            color="blue",
            name="energy",
        )
        self.assertEqual(stem["mode"], "lines")
        self.assertEqual(stem["hoverinfo"], "none")
        self.assertFalse(stem["connectgaps"])
        self.assertFalse(stem["showlegend"])
        self.assertNotIn("fill", stem)
        self.assertEqual(peak["mode"], "markers")
        self.assertEqual(peak["y"], [7.0])
        self.assertNotEqual(peak.get("hoverinfo"), "none")

    def test_step_power_uses_interval_start_from_the_frame(self):
        interval_start = pd.Timestamp("2024-01-01")
        interval_end = pd.Timestamp("2024-01-01 00:00:02")
        df = series_rows(
            RAPL_POWER_ID,
            [4.0],
            timestamps=[interval_end],
            interval_start=interval_start,
            point_role="observed",
        )
        x_values, y_values = prepare_trace_coordinates(df, RAPL_POWER_ID)
        self.assertEqual(list(x_values), [interval_start, interval_end])
        self.assertEqual(list(y_values), [4.0, 4.0])

    def test_timeseries_uses_webgl_when_expanded_vertices_exceed_threshold(self):
        df = expand_counterdiff_rows(rapl_energy_rows([7.0]))
        self.assertEqual(len(df), 2)
        x_values, _ = prepare_trace_coordinates(df, RAPL_ENERGY_ID)
        self.assertEqual(len(x_values), 4)

        with patch("frontend.figures._WEBGL_COORDINATE_THRESHOLD", 3):
            figure = create_all_timeseries_plots(df, category="energy")
        self.assertEqual(figure.data[0].type, "scattergl")

    def test_subplot_title_uses_device_class_color_for_the_metric_name(self):
        figure = create_all_timeseries_plots(rapl_energy_rows([7.0]), category="energy")
        title = figure.layout.annotations[0].text
        cpu_color = DEVICE_CLASS_COLORS_DARK[DeviceClass.CPU]
        first_line, second_line = title.split("<br>")
        self.assertIn(f'<span style="color:{cpu_color};font-size:14px">rapl_consumed_energy_J</span>', first_line)
        self.assertIn("R:", second_line)
        self.assertNotIn(">CPU</span>", title)

    def test_derived_title_keeps_theme_text_on_the_suffix(self):
        df = series_rows(CPU_ENERGY_ID, [1.0], metric_origin=MetricOrigin.DERIVED.value)
        figure = create_all_timeseries_plots(df, category="energy")
        title = figure.layout.annotations[0].text
        cpu_color = DEVICE_CLASS_COLORS_DARK[DeviceClass.CPU]
        first_line, second_line = title.split("<br>")
        self.assertIn(f'<span style="color:{cpu_color};font-size:14px">attributed_energy_cpu_J</span>', first_line)
        self.assertIn('<span style="font-size:14px">(derived)</span>', first_line)
        self.assertIn("R:", second_line)
        self.assertNotIn("color:#B48EAD", title)
        self.assertNotIn("color:#7E5693", title)

    def test_title_html_and_key_share_class_tokens(self):
        html = format_device_class_title_html(CPU_ENERGY_ID, derived=True)
        cpu_color = DEVICE_CLASS_COLORS_DARK[DeviceClass.CPU]
        first_line, second_line = html.split("<br>")
        self.assertIn(f'<span style="color:{cpu_color};font-size:14px">attributed_energy_cpu_J</span>', first_line)
        self.assertIn('<span style="font-size:14px">(derived)</span>', first_line)
        self.assertIn("R:", second_line)
        self.assertNotIn(">CPU</span>", html)
        key = device_class_key(include_process_active=True)
        process_row, class_row = key
        self.assertEqual(process_row.children[1].children, "Process Active")
        self.assertEqual([child.children for child in class_row.children], ["CPU", "GPU", "Total", "Other"])
        self.assertEqual(
            [child.className.split()[-1] for child in class_row.children],
            ["device-class-cpu", "device-class-gpu", "device-class-total", "device-class-other"],
        )
        self.assertEqual(
            [child.style["color"] for child in class_row.children],
            [DEVICE_CLASS_COLORS_DARK[cls] for cls in DeviceClass],
        )

    def test_three_subplots_keep_the_compact_viewport_height(self):
        df = concat_series(
            series_rows(RAPL_ENERGY_ID, [1.0]),
            series_rows(CPU_ENERGY_ID, [2.0]),
            series_rows(GPU_ENERGY_ID, [3.0]),
        )
        figure = create_all_timeseries_plots(df, category="energy")
        self.assertEqual(figure.layout.height, 145 * 3 + 66 * 2 + 36 + 36)

    def test_process_grid_caption_is_only_the_class_badge(self):
        caption = device_class_selection_caption(CPU_ENERGY_ID)
        self.assertEqual(caption.children, "[CPU]")
        self.assertIn("device-class-cpu", caption.className)
        self.assertEqual(caption.style["color"], DEVICE_CLASS_COLORS_DARK[DeviceClass.CPU])
        self.assertNotIn("attributed_energy_cpu_J", str(caption.children))

    def test_other_chip_uses_the_same_dark_color_as_plotly(self):
        chip = device_class_chip(NETWORK_RX_ID)
        self.assertEqual(chip.children, "Other")
        self.assertEqual(chip.style["color"], DEVICE_CLASS_COLORS_DARK[DeviceClass.OTHER])


if __name__ == "__main__":
    unittest.main()
