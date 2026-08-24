"""Interactive Plotly chart builders for the dashboard."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.categories import category_yaxis_label
from backend.counterdiff import (
    build_counterdiff_spike_coordinates,
    build_step_power_coordinates,
    counterdiff_spike_marker_sizes,
    sort_for_plotting,
)
from backend.formatting import format_metric_title
from backend.metrics import is_spike_metric, is_step_power_metric
from backend.transforms import compute_yaxis_ranges, get_time_range_from_df
from frontend.style import plot_color_palette, process_active_fill, set_plotly_rgba


def get_color_palette(n_colors: int, use_light_mode: bool = False) -> List[str]:
    """Get a theme-aware color palette for n_colors time series."""
    base = list(plot_color_palette(use_light_mode))
    if n_colors <= 0 or not base:
        return []
    colors: List[str] = []
    while len(colors) < n_colors:
        colors.extend(base)
    return colors[:n_colors]


def color_for_metric(
    metric: str,
    use_light_mode: bool = False,
    metric_order: Optional[List[str]] = None,
) -> str:
    """
    Pick a color for a metric name.

    When metric_order is given (typically sorted unique names), colors are
    assigned in that order so the first N metrics are unique for a palette
    of size N. After that the palette wraps. The same metric always maps
    to the same swatch.
    """
    palette = list(plot_color_palette(use_light_mode))
    if not palette:
        return "#636EFA"
    if metric_order:
        try:
            return palette[list(metric_order).index(metric) % len(palette)]
        except ValueError:
            pass
    # Fallback: stable, process-independent index from the metric name.
    idx = sum((i + 1) * ord(ch) for i, ch in enumerate(str(metric)))
    return palette[idx % len(palette)]


def build_metric_trace_config(
    df_series: pd.DataFrame,
    metric_id: str,
    *,
    color: str,
    name: str,
    show_default_markers: bool = True,
    fill_to_zero: bool = False,
    fillcolor: str | None = None,
    step_line_shape: str | None = None,
    marker_outline: bool = False,
    yaxis: str | None = None,
    showlegend: bool | None = None,
) -> dict:
    """
    Build Plotly Scatter settings for one metric time series.

    Pick the drawing style based on the metric type:
    - CounterDiff → spikes with dots on peaks
    - derived power → stepwise lines over each averaging interval
    - otherwise → a normal connected line (optionally with markers)

    Extra keyword args only tweak look-and-feel for each pane 
    (fill, dual-axis, marker outline, etc.).
    """
    roles = df_series["point_role"] if "point_role" in df_series.columns else None
    orders = df_series["point_order"] if "point_order" in df_series.columns else None
    interval_starts = df_series["interval_start"] if "interval_start" in df_series.columns else None

    config: dict = {
        "name": name,
        "line": {"color": color, "width": 2},
        "hovertemplate": (f"<b>{name}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>"),
    }
    if yaxis is not None:
        config["yaxis"] = yaxis
    if showlegend is not None:
        config["showlegend"] = showlegend

    outline = {"width": 1, "color": "rgba(255, 255, 255, 0.5)"} if marker_outline else None

    if is_spike_metric(metric_id):
        x_values, y_values = build_counterdiff_spike_coordinates(
            df_series["timestamp"],
            df_series["value"],
            point_roles=roles,
            point_orders=orders,
        )
        marker: dict = {
            "color": color,
            "size": counterdiff_spike_marker_sizes(x_values),
            "symbol": "circle",
        }
        if outline is not None:
            marker["line"] = outline
        config.update(
            {
                "x": x_values,
                "y": y_values,
                "mode": "lines+markers",
                "marker": marker,
                "connectgaps": False,
            }
        )
        return config

    if is_step_power_metric(metric_id):
        x_values, y_values = build_step_power_coordinates(
            df_series["timestamp"],
            df_series["value"],
            point_roles=roles,
            interval_starts=interval_starts,
        )
        line = {"color": color, "width": 2}
        if step_line_shape is not None:
            line["shape"] = step_line_shape
        config.update(
            {
                "x": x_values,
                "y": y_values,
                "mode": "lines",
                "line": line,
                "connectgaps": False,
            }
        )
        if fill_to_zero:
            config["fill"] = "tozeroy"
            config["fillcolor"] = fillcolor
        return config

    config.update(
        {
            "x": pd.to_datetime(df_series["timestamp"], errors="coerce"),
            "y": df_series["value"],
            "mode": "lines+markers" if show_default_markers else "lines",
        }
    )
    if show_default_markers:
        marker = {"color": color, "size": 6, "symbol": "circle"}
        if outline is not None:
            marker["line"] = outline
        config["marker"] = marker
    if fill_to_zero:
        config["fill"] = "tozeroy"
        config["fillcolor"] = fillcolor
    return config


def create_all_timeseries_plots(
    df_processed: pd.DataFrame,
    proc_start: Optional[pd.Timestamp] = None,
    proc_end: Optional[pd.Timestamp] = None,
    full_time_range: Optional[tuple] = None,
    category: Optional[str] = None,
    share_yaxis: bool = False,
    use_light_mode: bool = False,
) -> go.Figure:
    """Create all time series as scrollable subplots."""
    if df_processed.empty:
        return go.Figure()

    df_processed = df_processed.copy()
    df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"], errors="coerce")

    unique_metrics = df_processed["metric_id"].unique()
    n_metrics = len(unique_metrics)

    if n_metrics == 0:
        return go.Figure()

    if full_time_range:
        x_min, x_max = full_time_range
    else:
        x_min, x_max = get_time_range_from_df(df_processed)
    x_min = pd.Timestamp(x_min)
    x_max = pd.Timestamp(x_max)

    colors = get_color_palette(n_metrics, use_light_mode)
    color_map = {metric: colors[i] for i, metric in enumerate(unique_metrics)}

    MIN_SUBPLOT_HEIGHT = 175
    SUBPLOT_GAP_PX = 40
    total_height = MIN_SUBPLOT_HEIGHT * n_metrics + SUBPLOT_GAP_PX * max(n_metrics - 1, 0)
    vertical_spacing = (SUBPLOT_GAP_PX / total_height) if n_metrics > 1 else 0.05

    formatted_titles = [format_metric_title(metric_id) for metric_id in unique_metrics]
    fig = make_subplots(
        rows=n_metrics,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=[f"<b>{title}</b>" for title in formatted_titles],
    )

    is_memory_category = category == "memory"
    yaxis_ranges = compute_yaxis_ranges(df_processed, list(unique_metrics), share_yaxis, is_memory_category)

    if proc_start and proc_end:
        for idx in range(1, n_metrics + 1):
            yref = f"y{idx} domain" if idx > 1 else "y domain"
            xref = f"x{idx}" if idx > 1 else "x"
            fig.add_shape(
                type="rect",
                x0=proc_start,
                x1=proc_end,
                y0=0,
                y1=1,
                xref=xref,
                yref=yref,
                fillcolor=process_active_fill(use_light_mode),
                line=dict(width=0),
                layer="below",
            )
    df_sorted = sort_for_plotting(df_processed)
    grouped = {mid: grp for mid, grp in df_sorted.groupby("metric_id", observed=True, sort=False)}

    total_points = len(df_processed)
    show_markers_global = total_points < 5000

    for idx, metric_id in enumerate(unique_metrics, start=1):
        metric_data = grouped.get(metric_id, pd.DataFrame())
        if metric_data.empty:
            continue

        n_pts = len(metric_data)
        use_webgl = n_pts > 10000
        show_markers = show_markers_global and n_pts < 5000

        color = color_map[metric_id]
        rgba_fill = set_plotly_rgba(color)

        ScatterClass = go.Scattergl if use_webgl else go.Scatter

        fill_to_zero = (not use_webgl) and (not is_spike_metric(metric_id))
        trace_config = build_metric_trace_config(
            metric_data,
            str(metric_id),
            color=color,
            name=str(metric_id),
            show_default_markers=show_markers,
            fill_to_zero=fill_to_zero,
            fillcolor=rgba_fill if fill_to_zero else None,
            step_line_shape="hv",
            marker_outline=True,
            showlegend=False,
        )

        if show_markers:
            outline_color = "rgba(31, 41, 55, 0.35)" if use_light_mode else "rgba(255, 255, 255, 0.5)"
            marker = dict(trace_config.get("marker") or {})
            marker["color"] = color
            marker["symbol"] = marker.get("symbol", "circle")
            marker["line"] = dict(width=1, color=outline_color)
            if "size" not in marker:
                marker["size"] = 6
            trace_config["marker"] = marker

        fig.add_trace(ScatterClass(**trace_config), row=idx, col=1)

        fig.update_xaxes(
            type="date",
            range=[x_min, x_max],
            gridcolor="rgba(76, 86, 106, 0.2)",
            showticklabels=True,
            showspikes=True,
            spikemode="across",
            spikesnap="data",
            spikethickness=1,
            spikecolor="#1f2937" if use_light_mode else "white",
            spikedash="dot",
            row=idx,
            col=1,
        )

        yaxis_key = "yaxis" if idx == 1 else f"yaxis{idx}"
        yaxis_cfg = yaxis_ranges.get(yaxis_key, {})

        yaxis_config = dict(
            title_text=category_yaxis_label(category),
            fixedrange=False,
            gridcolor="rgba(76, 86, 106, 0.2)",
        )
        if "range" in yaxis_cfg:
            yaxis_config["range"] = yaxis_cfg["range"]
            yaxis_config["autorange"] = False
        else:
            yaxis_config["autorange"] = True
        if "tickvals" in yaxis_cfg:
            yaxis_config["tickvals"] = yaxis_cfg["tickvals"]
            yaxis_config["ticktext"] = yaxis_cfg["ticktext"]

        fig.update_yaxes(**yaxis_config, row=idx, col=1)

    fig.update_xaxes(title_text="Time", row=n_metrics, col=1)

    fig.update_layout(
        height=total_height,
        paper_bgcolor="rgba(46, 52, 64, 0.95)",
        plot_bgcolor="rgba(59, 66, 82, 0.7)",
        font=dict(color="#d8dee9"),
        hovermode="closest",
        margin=dict(l=50, r=20, t=36, b=36),
        autosize=True,
        width=None,
        showlegend=False,
    )
    fig.update_xaxes(type="date", rangeslider=dict(visible=False), row=n_metrics, col=1)

    return fig


def relayout_requests_reset(relayout_data: dict | None) -> bool:
    """Return True when Plotly reports an axis reset or autosize action."""
    if not relayout_data:
        return False
    return any(
        value is True and (key == "autosize" or key.endswith(".autorange"))
        for key, value in relayout_data.items()
    )


def update_xaxis_ranges_in_layout(layout: dict, x_range: list) -> None:
    """Apply one explicit X range to every Cartesian X axis in a layout."""
    for key, axis in layout.items():
        suffix = key.removeprefix("xaxis")
        if key != "xaxis" and (not key.startswith("xaxis") or not suffix.isdigit()):
            continue
        if not isinstance(axis, dict):
            continue
        axis["range"] = list(x_range)
        axis["autorange"] = False


def restore_axis_defaults(axis: dict, defaults: dict) -> None:
    """Restore an axis from defaults saved independently of its live range."""
    if defaults.get("autorange", False):
        axis["autorange"] = True
        axis.pop("range", None)
    else:
        axis["range"] = list(defaults["range"])
        axis["autorange"] = False

    for key in ("tickvals", "ticktext"):
        if key in defaults:
            axis[key] = list(defaults[key])
        else:
            axis.pop(key, None)


def update_yaxis_ranges_in_layout(
    layout: dict,
    yaxis_updates: dict,
    *,
    y_axis_label: Optional[str] = None,
) -> None:
    """Apply compute_yaxis_ranges output to in-place Plotly figure layout dict."""
    for yaxis_key, settings in yaxis_updates.items():
        if yaxis_key not in layout:
            continue
        layout[yaxis_key]["range"] = settings["range"]
        layout[yaxis_key]["autorange"] = settings["autorange"]
        if y_axis_label is not None:
            layout[yaxis_key]["title"] = {"text": y_axis_label}
        if "tickvals" in settings:
            layout[yaxis_key]["tickvals"] = settings["tickvals"]
            layout[yaxis_key]["ticktext"] = settings["ticktext"]
        else:
            layout[yaxis_key].pop("tickvals", None)
            layout[yaxis_key].pop("ticktext", None)
