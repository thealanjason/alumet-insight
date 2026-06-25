"""Interactive Plotly chart builders for the dashboard."""

import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from typing import List, Optional

from backend.categories import category_yaxis_label
from backend.formatting import format_metric_title
from backend.transforms import compute_yaxis_ranges


def get_color_palette(n_colors: int) -> List[str]:
    """Get a color palette for n_colors time series."""
    palettes = [
        pc.qualitative.Plotly,
        pc.qualitative.Set2,
        pc.qualitative.Set3,
        pc.qualitative.Pastel,
        pc.qualitative.Dark2,
        pc.qualitative.Pastel1,
        pc.qualitative.Pastel2,
    ]

    colors = []
    for palette in palettes:
        colors.extend(palette)
        if len(colors) >= n_colors:
            break

    while len(colors) < n_colors:
        colors.extend(colors[: min(len(colors), n_colors - len(colors))])

    return colors[:n_colors]


def create_all_timeseries_plots(
    df_processed: pd.DataFrame,
    proc_start: Optional[pd.Timestamp] = None,
    proc_end: Optional[pd.Timestamp] = None,
    full_time_range: Optional[tuple] = None,
    category: Optional[str] = None,
    share_yaxis: bool = False,
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
        x_min = df_processed["timestamp"].min()
        x_max = df_processed["timestamp"].max()
    x_min = pd.Timestamp(x_min)
    x_max = pd.Timestamp(x_max)

    colors = get_color_palette(n_metrics)
    color_map = {metric: colors[i] for i, metric in enumerate(unique_metrics)}

    MIN_SUBPLOT_HEIGHT = 240
    SUBPLOT_GAP_PX = 52
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
                fillcolor="rgba(136, 192, 208, 0.12)",
                line=dict(width=0),
                layer="below",
            )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="rgba(136, 192, 208, 0.4)", symbol="square"),
                name="Process Active",
                showlegend=True,
                legendgroup="process_active",
            ),
            row=1,
            col=1,
        )

    df_sorted = df_processed.sort_values(["metric_id", "timestamp"])
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

        if color.startswith("#"):
            hex_color = color.lstrip("#")
            rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            rgba_fill = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)"
        elif color.startswith("rgba"):
            rgba_fill = color.rsplit(",", 1)[0] + ", 0.15)"
        elif color.startswith("rgb"):
            rgba_fill = color.replace("rgb", "rgba").replace(")", ", 0.15)")
        else:
            rgba_fill = "rgba(136, 192, 208, 0.15)"

        ScatterClass = go.Scattergl if use_webgl else go.Scatter

        trace_config = dict(
            x=pd.to_datetime(metric_data["timestamp"], errors="coerce"),
            y=metric_data["value"],
            mode="lines+markers" if show_markers else "lines",
            name=metric_id,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{metric_id}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
            showlegend=False,
        )

        if show_markers:
            trace_config["marker"] = dict(
                color=color,
                size=6,
                symbol="circle",
                line=dict(width=1, color="rgba(255, 255, 255, 0.5)"),
            )

        if not use_webgl:
            trace_config["fill"] = "tozeroy"
            trace_config["fillcolor"] = rgba_fill

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
            spikecolor="white",
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
        title=dict(text="<b>📈 Time series of all metrics</b>", x=0.5, font=dict(size=16)),
        paper_bgcolor="rgba(46, 52, 64, 0.95)",
        plot_bgcolor="rgba(59, 66, 82, 0.7)",
        font=dict(color="#d8dee9"),
        hovermode="closest",
        margin=dict(l=50, r=20, t=60, b=40),
        autosize=True,
        width=None,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(46, 52, 64, 0.8)",
            bordercolor="rgba(136, 192, 208, 0.3)",
            borderwidth=1,
            font=dict(color="#d8dee9"),
        ),
    )
    fig.update_xaxes(type="date", rangeslider=dict(visible=False), row=n_metrics, col=1)

    return fig
