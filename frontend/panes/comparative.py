"""Comparative analysis tab: helpers and callbacks for the X-Y metric comparison view."""

from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html

from backend.formatting import format_metric_choice_label, get_bytes_tickvals_ticktext
from backend.metrics import (
    base_metric_from_id,
    derived_metric_ids,
    filter_process_metric_ids,
    get_metric_unit,
    is_cumulative_xy_pair,
    is_memory_metric,
)
from backend.transforms import (
    align_xy_metrics,
    comparative_cumulative_xy,
    comparative_metric_ids,
    filter_to_time_range,
)
from backend.utils import safe_filename
from frontend.app import app
from frontend.cache import df_from_store
from frontend.figures import build_metric_trace_configs
from frontend.helpers import ensure_timestamp_datetime, parse_process_time_range_store
from frontend.layout import empty_comparative_content, is_empty_tab_placeholder
from frontend.style import CARD_STYLE, DROPDOWN_STYLE, apply_figure_theme, plot_pair_colors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_metric_ids(processed_df_data: Any, process_time_range: Any) -> list[str]:
    """Reconstruct dataframe and call backend to get metric IDs."""
    if not processed_df_data or not process_time_range:
        return []
    df_processed = df_from_store(processed_df_data)
    ensure_timestamp_datetime(df_processed)
    proc_start, proc_end = parse_process_time_range_store(process_time_range)
    return comparative_metric_ids(df_processed, proc_start, proc_end)


def pick_xy_values(filtered: list[str], cur_x: Any, cur_y: Any) -> tuple[Any, Any]:
    """Pick valid X/Y dropdown values, preserving current selection when possible."""
    if not filtered:
        return None, None
    if len(filtered) == 1:
        return filtered[0], filtered[0]
    x_val = cur_x if cur_x in filtered else filtered[0]
    others = [m for m in filtered if m != x_val]
    y_val = cur_y if cur_y in others else others[0]
    return x_val, y_val


def prepare_xy_download(
    dfxy: pd.DataFrame,
    x_metric_id: str,
    y_metric_id: str,
) -> tuple[pd.DataFrame, str]:
    """Rename columns for CSV export and compute a safe filename."""
    df_out = dfxy.rename(columns={"x": x_metric_id, "y": y_metric_id})
    df_out["x_unit"] = get_metric_unit(x_metric_id)
    df_out["y_unit"] = get_metric_unit(y_metric_id)
    filename = safe_filename(f"xy_{x_metric_id}_vs_{y_metric_id}.csv")
    return df_out, filename


def comparative_timeseries_trace_configs(
    df_series: pd.DataFrame,
    metric_id: str,
    name: str,
    color: str,
    yaxis: str,
) -> list[dict]:
    """Build dual-axis traces using the shared metric rendering policy."""
    return build_metric_trace_configs(
        df_series,
        metric_id,
        color=color,
        name=name,
        show_default_markers=False,
        fill_to_zero=False,
        yaxis=yaxis,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


# Build comparative tab
@app.callback(
    Output("comparative-content", "children"),
    Input("results-tabs", "value"),
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
    State("comparative-content", "children"),
)
def build_comparative_tab(tab_value, processed_df_data, process_time_range, current_children):
    triggered_id = ctx.triggered_id
    is_data_trigger = triggered_id in ("processed-df-store", "process-time-range-store")

    if is_data_trigger and tab_value != "comparative-tab":
        return empty_comparative_content()

    if triggered_id == "results-tabs":
        if tab_value != "comparative-tab":
            return dash.no_update
        if current_children and not is_empty_tab_placeholder(current_children):
            return dash.no_update

    if not processed_df_data or not process_time_range:
        return empty_comparative_content()

    metric_ids = _resolve_metric_ids(processed_df_data, process_time_range)
    if len(metric_ids) < 2:
        return empty_comparative_content("Need at least 2 metrics inside process window.")

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Metric 1 (X-axis / Left Y-axis):",
                                                style={"color": "var(--app-text)", "fontWeight": "600"},
                                            ),
                                            dcc.Dropdown(
                                                id="ps-xmetric-dropdown",
                                                options=[{"label": m, "value": m} for m in metric_ids],
                                                value=metric_ids[0],
                                                clearable=False,
                                                persistence=True,
                                                className="dark-dropdown",
                                                style=DROPDOWN_STYLE,
                                            ),
                                        ],
                                        width=12,
                                        lg=6,
                                        className="mb-3",
                                    ),
                                    dbc.Col(
                                        [
                                            html.Label(
                                                "Metric 2 (Y-axis / Right Y-axis):",
                                                style={"color": "var(--app-text)", "fontWeight": "600"},
                                            ),
                                            dcc.Dropdown(
                                                id="ps-ymetric-dropdown",
                                                options=[{"label": m, "value": m} for m in metric_ids],
                                                value=metric_ids[1],
                                                clearable=False,
                                                persistence=True,
                                                className="dark-dropdown",
                                                style=DROPDOWN_STYLE,
                                            ),
                                        ],
                                        width=12,
                                        lg=6,
                                        className="mb-3",
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Checklist(
                                                id="comparative-process-only-toggle",
                                                options=[{"label": " Process metrics only", "value": "process_only"}],
                                                value=[],
                                                inline=True,
                                                style={"color": "var(--app-text)", "fontSize": "0.9rem"},
                                                inputStyle={"marginRight": "8px"},
                                            ),
                                        ],
                                        width=12,
                                        className="mb-2",
                                    ),
                                ],
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [html.Div(id="comparative-mode-info", style={"marginBottom": "10px"})],
                                        width=12,
                                        lg=8,
                                        className="mb-2",
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Checklist(
                                                id="scatter-toggle",
                                                options=[{"label": " Show Scatter Plot (X-Y relationship)", "value": "scatter"}],
                                                value=[],
                                                inline=True,
                                                style={"color": "var(--app-text)", "fontSize": "0.9rem"},
                                                inputStyle={"marginRight": "8px"},
                                            ),
                                        ],
                                        width=12,
                                        lg=4,
                                        className="mb-2",
                                        style={"textAlign": "right"},
                                    ),
                                ],
                            ),
                        ],
                        className="comparative-controls",
                    ),
                    html.Div(
                        dcc.Graph(
                            id="ps-xy-graph",
                            style={"height": "100%", "width": "100%"},
                            config={"responsive": True, "displaylogo": False},
                        ),
                        className="comparative-plot-area",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "\U0001f4e5 Download CSV",
                                id="xy-download-btn",
                                n_clicks=0,
                                color="primary",
                                size="sm",
                            ),
                            dcc.Download(id="xy-download"),
                        ],
                        className="comparative-download",
                    ),
                ],
                style={"backgroundColor": "var(--app-card-bg)"},
                className="viewport-card-body comparative-card-body",
            )
        ],
        style=CARD_STYLE,
        className="viewport-card comparative-card",
    )


# Mode info
@app.callback(
    Output("comparative-mode-info", "children"),
    Input("ps-xmetric-dropdown", "value"),
    Input("ps-ymetric-dropdown", "value"),
    prevent_initial_call=True,
)
def update_comparative_mode_info(x_metric_id, y_metric_id):
    """Show info about which visualization mode will be used."""
    if not x_metric_id or not y_metric_id:
        return html.Span("")

    if is_cumulative_xy_pair(x_metric_id, y_metric_id):
        return html.Div(
            [
                html.Span("Visualization Mode: ", style={"fontWeight": "600"}),
                html.Span("Cumulative X-Y Plot", style={"color": "var(--app-success)", "fontWeight": "600"}),
            ],
            style={"color": "var(--app-text)"},
        )
    else:
        return html.Div(
            [
                html.Span("Visualization Mode: ", style={"fontWeight": "600"}),
                html.Span("Dual Y-Axis Time Series", style={"color": "var(--app-warning)", "fontWeight": "600"}),
            ],
            style={"color": "var(--app-text)"},
        )


# Metric dropdowns
@app.callback(
    Output("ps-xmetric-dropdown", "options"),
    Output("ps-xmetric-dropdown", "value"),
    Output("ps-ymetric-dropdown", "options"),
    Output("ps-ymetric-dropdown", "value"),
    Input("comparative-process-only-toggle", "value"),
    Input("results-tabs", "value"),
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
    State("ps-xmetric-dropdown", "value"),
    State("ps-ymetric-dropdown", "value"),
)
def update_comparative_metric_dropdowns(
    process_only_toggle, tab_value, processed_df_data, process_time_range, cur_x, cur_y
):
    """Filter comparative X/Y metric lists to process-attributed series when requested."""
    if tab_value != "comparative-tab":
        raise dash.exceptions.PreventUpdate

    all_ids = _resolve_metric_ids(processed_df_data, process_time_range)
    if len(all_ids) < 2:
        raise dash.exceptions.PreventUpdate

    process_only = bool(process_only_toggle and "process_only" in process_only_toggle)
    filtered = filter_process_metric_ids(all_ids, process_only)

    dfp = df_from_store(processed_df_data)
    derived_ids = derived_metric_ids(dfp)
    opts = [
        {"label": format_metric_choice_label(m, derived=m in derived_ids), "value": m}
        for m in filtered
    ]
    x_val, y_val = pick_xy_values(filtered, cur_x, cur_y)
    return opts, x_val, opts, y_val


# X-Y plot
@app.callback(
    Output("ps-xy-graph", "figure"),
    Input("ps-xmetric-dropdown", "value"),
    Input("ps-ymetric-dropdown", "value"),
    Input("scatter-toggle", "value"),
    Input("theme-switch", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def update_process_xy_plot(
    x_metric_id, y_metric_id, scatter_toggle, use_light_mode, processed_df_data, process_time_range
):
    fig = go.Figure()
    fig.update_layout(margin=dict(l=70, r=70, t=60, b=60), autosize=True)
    apply_figure_theme(fig, use_light_mode)

    if not processed_df_data or not process_time_range or not x_metric_id or not y_metric_id:
        fig.update_layout(title=dict(text="Select both metrics", x=0.5))
        return fig

    dfp = df_from_store(processed_df_data)
    ensure_timestamp_datetime(dfp)

    proc_start, proc_end = parse_process_time_range_store(process_time_range)
    if proc_start is None or proc_end is None:
        fig.update_layout(title=dict(text="Process time range not available", x=0.5))
        return fig

    x_abbrev = base_metric_from_id(x_metric_id)
    y_abbrev = base_metric_from_id(y_metric_id)

    x_unit = get_metric_unit(x_metric_id)
    y_unit = get_metric_unit(y_metric_id)
    x_label = f"{x_abbrev} ({x_unit})" if x_unit else x_abbrev
    y_label = f"{y_abbrev} ({y_unit})" if y_unit else y_abbrev

    both_cumulative = is_cumulative_xy_pair(x_metric_id, y_metric_id)

    show_scatter = scatter_toggle and "scatter" in scatter_toggle

    accents = plot_pair_colors(use_light_mode)
    color_x = accents["x"]
    color_y = accents["y"]

    if show_scatter:
        dfxy = align_xy_metrics(
            dfp,
            x_metric_id,
            y_metric_id,
            proc_start,
            proc_end,
        )
        if dfxy.empty:
            fig.update_layout(
                title=dict(
                    text="Could not align metrics in time (no matches within tolerance)",
                    x=0.5,
                )
            )
            return fig
        hover_times = dfxy["timestamp"].dt.strftime("%H:%M:%S.%f").str[:-3]
        fig.add_trace(
            go.Scatter(
                x=dfxy["x"],
                y=dfxy["y"],
                mode="markers",
                name="Data Points",
                marker=dict(
                    color=accents["scatter"],
                    size=10,
                    opacity=0.85,
                    line=dict(width=1, color=accents["marker_line"]),
                ),
                hovertemplate=(
                    "<b>Time:</b> %{customdata}<br>"
                    f"<b>{x_abbrev}:</b> %{{x:.4f}}<br>"
                    f"<b>{y_abbrev}:</b> %{{y:.4f}}"
                    "<extra></extra>"
                ),
                customdata=hover_times,
            )
        )

        xaxis_config = dict(title=dict(text=x_label, font=dict(size=11)), gridcolor="rgba(76, 86, 106, 0.2)")
        yaxis_config = dict(title=dict(text=y_label, font=dict(size=11)), gridcolor="rgba(76, 86, 106, 0.2)")
        if is_memory_metric(x_metric_id):
            x_tickvals, x_ticktext = get_bytes_tickvals_ticktext(dfxy["x"].min(), dfxy["x"].max(), num_ticks=5)
            xaxis_config["tickvals"] = x_tickvals
            xaxis_config["ticktext"] = x_ticktext
        if is_memory_metric(y_metric_id):
            y_tickvals, y_ticktext = get_bytes_tickvals_ticktext(dfxy["y"].min(), dfxy["y"].max(), num_ticks=5)
            yaxis_config["tickvals"] = y_tickvals
            yaxis_config["ticktext"] = y_ticktext

        fig.update_layout(
            title=dict(text=f"Scatter plot: {y_abbrev} vs {x_abbrev}", x=0.5, font=dict(size=14)),
            xaxis=xaxis_config,
            yaxis=yaxis_config,
            hovermode="closest",
        )

    elif both_cumulative:
        dfxy = comparative_cumulative_xy(dfp, x_metric_id, y_metric_id, proc_start, proc_end)
        if dfxy.empty:
            fig.update_layout(
                title=dict(text="Could not compute running totals (one or both series empty)", x=0.5)
            )
            return fig
        hover_times = dfxy["timestamp"].dt.strftime("%H:%M:%S.%f").str[:-3]

        fig.add_trace(
            go.Scatter(
                x=dfxy["x"],
                y=dfxy["y"],
                mode="lines+markers",
                line=dict(color=accents["cumulative"], width=2),
                marker=dict(color=accents["cumulative"], size=6),
                hovertemplate=(
                    "<b>Time:</b> %{customdata}<br>"
                    f"<b>Cumulative {x_abbrev}:</b> %{{x:.4f}}<br>"
                    f"<b>Cumulative {y_abbrev}:</b> %{{y:.4f}}"
                    "<extra></extra>"
                ),
                customdata=hover_times,
            )
        )

        x_cum_label = f"Cumulative {x_abbrev} ({x_unit})" if x_unit else f"Cumulative {x_abbrev}"
        y_cum_label = f"Cumulative {y_abbrev} ({y_unit})" if y_unit else f"Cumulative {y_abbrev}"

        xaxis_config = dict(title=dict(text=x_cum_label, font=dict(size=11)), gridcolor="rgba(76, 86, 106, 0.2)")
        yaxis_config = dict(title=dict(text=y_cum_label, font=dict(size=11)), gridcolor="rgba(76, 86, 106, 0.2)")

        if is_memory_metric(x_metric_id):
            x_tickvals, x_ticktext = get_bytes_tickvals_ticktext(
                dfxy["x"].min(), dfxy["x"].max(), num_ticks=5
            )
            xaxis_config["tickvals"] = x_tickvals
            xaxis_config["ticktext"] = x_ticktext
        if is_memory_metric(y_metric_id):
            y_tickvals, y_ticktext = get_bytes_tickvals_ticktext(
                dfxy["y"].min(), dfxy["y"].max(), num_ticks=5
            )
            yaxis_config["tickvals"] = y_tickvals
            yaxis_config["ticktext"] = y_ticktext

        fig.update_layout(
            title=dict(text=f"Cumulative {y_abbrev} vs Cumulative {x_abbrev}", x=0.5, font=dict(size=14)),
            xaxis=xaxis_config,
            yaxis=yaxis_config,
            hovermode="closest",
        )

    else:
        df_window = filter_to_time_range(dfp, proc_start, proc_end)
        x_series = df_window[df_window["metric_id"].astype(str) == str(x_metric_id)].copy()
        y_series = df_window[df_window["metric_id"].astype(str) == str(y_metric_id)].copy()

        for trace_config in comparative_timeseries_trace_configs(
            x_series,
            x_metric_id,
            x_abbrev,
            color_x,
            "y1",
        ):
            fig.add_trace(go.Scatter(**trace_config))

        for trace_config in comparative_timeseries_trace_configs(
            y_series,
            y_metric_id,
            y_abbrev,
            color_y,
            "y2",
        ):
            fig.add_trace(go.Scatter(**trace_config))

        yaxis_config = dict(
            title=dict(text=x_label, font=dict(size=11, color=color_x)),
            tickfont=dict(color=color_x),
            gridcolor="rgba(76, 86, 106, 0.2)",
            side="left",
        )
        yaxis2_config = dict(
            title=dict(text=y_label, font=dict(size=11, color=color_y)),
            tickfont=dict(color=color_y),
            overlaying="y",
            side="right",
            showgrid=False,
        )

        if is_memory_metric(x_metric_id):
            x_tickvals, x_ticktext = get_bytes_tickvals_ticktext(
                x_series["value"].min(),
                x_series["value"].max(),
                num_ticks=5,
            )
            yaxis_config["tickvals"] = x_tickvals
            yaxis_config["ticktext"] = x_ticktext
        if is_memory_metric(y_metric_id):
            y_tickvals, y_ticktext = get_bytes_tickvals_ticktext(
                y_series["value"].min(),
                y_series["value"].max(),
                num_ticks=5,
            )
            yaxis2_config["tickvals"] = y_tickvals
            yaxis2_config["ticktext"] = y_ticktext

        fig.update_layout(
            title=dict(text=f"Time Series: {x_abbrev} & {y_abbrev}", x=0.5, font=dict(size=14)),
            xaxis=dict(
                title=dict(text="Time", font=dict(size=12)),
                gridcolor="rgba(76, 86, 106, 0.2)",
                domain=[0.05, 0.95],
                range=[proc_start, proc_end],
            ),
            yaxis=yaxis_config,
            yaxis2=yaxis2_config,
            legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5, bgcolor="rgba(59, 66, 82, 0.8)"),
            margin=dict(b=100),
            hovermode="x unified",
        )

    apply_figure_theme(fig, use_light_mode)
    return fig


# CSV download for X-Y plot
@app.callback(
    Output("xy-download", "data"),
    Input("xy-download-btn", "n_clicks"),
    State("ps-xmetric-dropdown", "value"),
    State("ps-ymetric-dropdown", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def download_xy_csv(n_clicks, x_metric_id, y_metric_id, processed_df_data, process_time_range):
    """Generate and download CSV for the X-Y comparative plot."""
    if not n_clicks or not processed_df_data or not x_metric_id or not y_metric_id:
        return None

    dfp = df_from_store(processed_df_data)
    ensure_timestamp_datetime(dfp)

    proc_start, proc_end = parse_process_time_range_store(process_time_range)
    if proc_start is None or proc_end is None:
        return None

    if is_cumulative_xy_pair(x_metric_id, y_metric_id):
        dfxy = comparative_cumulative_xy(dfp, x_metric_id, y_metric_id, proc_start, proc_end)
    else:
        dfxy = align_xy_metrics(dfp, x_metric_id, y_metric_id, proc_start, proc_end)
    if dfxy.empty:
        return None

    df_out, filename = prepare_xy_download(dfxy, x_metric_id, y_metric_id)
    return dcc.send_data_frame(df_out.to_csv, filename, index=False)
