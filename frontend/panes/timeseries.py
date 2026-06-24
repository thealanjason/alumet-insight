"""Time series tab: callbacks for category filtering, CPU core selector, Y-axis toggle/zoom."""

import copy

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dcc, html

from frontend.app import app
from frontend.cache import cache_dataframe, df_from_store, load_cached_dataframe, ensure_timestamp_datetime
from frontend.style import status_alert_class, apply_figure_theme, DROPDOWN_STYLE
from frontend.layout import empty_time_series_content
from frontend.helpers import available_category_options
from backend.categories import available_cpu_cores, category_yaxis_label, filter_time_series_category, is_yaxis_shareable
from backend.metrics import is_memory_metric
from backend.transforms import align_xrange_tz, compute_yaxis_ranges, filter_to_time_range
from backend.visualization.interactive import create_all_timeseries_plots


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------



@app.callback(
    Output("time-series-content", "children"),
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
)
def build_time_series_tab(processed_df_data, process_time_range):
    if not processed_df_data:
        return empty_time_series_content()

    df_processed = df_from_store(processed_df_data)
    ensure_timestamp_datetime(df_processed)

    available_categories = available_category_options(df_processed)

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Metric Category:",
                                        style={
                                            "color": "var(--app-text)",
                                            "marginRight": "10px",
                                            "fontSize": "1rem",
                                            "fontWeight": "600",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="metric-category-dropdown",
                                        options=available_categories,
                                        placeholder="Select metric category",
                                        style=DROPDOWN_STYLE,
                                        className="dark-dropdown",
                                        clearable=True,
                                    ),
                                ],
                                width=12,
                                lg=4,
                                className="mb-3",
                            ),
                            dbc.Col(
                                [
                                    html.Div(id="cpu-core-selector"),
                                    dcc.Dropdown(
                                        id="cpu-core-dropdown",
                                        options=[],
                                        placeholder="Select CPU core",
                                        style={"display": "none", **DROPDOWN_STYLE},
                                        className="dark-dropdown",
                                        clearable=False,
                                    ),
                                ],
                                width=12,
                                lg=3,
                                className="mb-3",
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Label(
                                            "Y-Axis Options:",
                                            style={
                                                "color": "var(--app-text)",
                                                "marginRight": "10px",
                                                "fontSize": "1rem",
                                                "fontWeight": "600",
                                            },
                                        ),
                                        dcc.Checklist(
                                            id="shared-yaxis-toggle",
                                            options=[{"label": " Share Y-axis range across subplots", "value": "shared"}],
                                            value=[],
                                            style={"color": "var(--app-text)", "fontSize": "0.9rem"},
                                            inputStyle={"marginRight": "8px"},
                                        ),
                                    ],
                                    id="yaxis-options-container",
                                    style={"display": "none"},
                                ),
                                width=12,
                                lg=5,
                                className="mb-3",
                                style={"display": "flex", "flexDirection": "column", "justifyContent": "center"},
                            ),
                        ],
                        className="time-series-controls mb-4",
                    ),
                    html.Div(
                        id="timeseries-plot-container",
                        style={
                            "overflowY": "auto",
                            "overflowX": "hidden",
                            "padding": "15px",
                            "width": "100%",
                        },
                    ),
                ],
                style={"padding": "25px", "backgroundColor": "var(--app-card-bg)"},
                className="viewport-card-body",
            ),
        ],
        style={"backgroundColor": "var(--app-card-bg)", "border": "1px solid var(--app-border)"},
        className="viewport-card timeseries-card",
    )



@app.callback(
    [
        Output("cpu-core-selector", "children"),
        Output("cpu-core-dropdown", "options"),
        Output("cpu-core-dropdown", "style"),
    ],
    Input("metric-category-dropdown", "value"),
    State("processed-df-store", "data"),
)
def update_cpu_core_selector(selected_category, processed_df_data):
    default_style = {"display": "none", "backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"}
    default_options = []
    default_children = html.Div()

    if selected_category != "kernel_cpu_time" or not processed_df_data:
        return default_children, default_options, default_style

    df_processed = df_from_store(processed_df_data)
    cpu_cores = available_cpu_cores(df_processed)

    if not cpu_cores:
        return default_children, default_options, default_style

    options = [{"label": f"Core {core}", "value": core} for core in cpu_cores]
    selector_children = html.Label(
        "CPU Core:",
        style={
            "color": "var(--app-text)",
            "marginRight": "10px",
            "fontSize": "1rem",
            "fontWeight": "600",
        },
    )
    visible_style = {"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"}
    return selector_children, options, visible_style



@app.callback(
    Output("yaxis-options-container", "style"),
    Output("shared-yaxis-toggle", "value"),
    Input("metric-category-dropdown", "value"),
    State("shared-yaxis-toggle", "value"),
)
def update_yaxis_options_visibility(selected_category, current_toggle_value):
    """Show Y-axis options only for categories with same units."""
    if is_yaxis_shareable(selected_category):
        return {"display": "flex", "flexDirection": "column"}, dash.no_update
    else:
        return {"display": "none"}, dash.no_update



@app.callback(
    Output("timeseries-plot-container", "children"),
    Output("timeseries-filtered-df-store", "data"),
    Input("metric-category-dropdown", "value"),
    Input("cpu-core-dropdown", "value"),
    Input("theme-switch", "value"),
    State("shared-yaxis-toggle", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def update_timeseries_plot(selected_category, selected_cpu_core, use_light_mode, shared_yaxis_toggle, processed_df_data, process_time_range):
    if not processed_df_data:
        return dbc.Alert("No data available.", color="warning", className=status_alert_class("warning")), None

    if not selected_category:
        return dbc.Alert("Please select a metric category.", color="warning", className=status_alert_class("warning")), None

    df_processed = df_from_store(processed_df_data)
    ensure_timestamp_datetime(df_processed)

    full_time_min = df_processed["timestamp"].min()
    full_time_max = df_processed["timestamp"].max()
    full_time_range = (full_time_min, full_time_max)

    if selected_category == "kernel_cpu_time":
        if not selected_cpu_core:
            return (
                dbc.Alert(
                    "Please select a CPU core to display kernel CPU time metrics.",
                    color="warning",
                    className=status_alert_class("warning"),
                ),
                None,
            )

    df_filtered = filter_time_series_category(
        df_processed,
        selected_category,
        selected_cpu_core=selected_cpu_core,
    )

    if df_filtered.empty:
        return dbc.Alert("No data available for the selected category.", color="warning", className=status_alert_class("warning")), None

    proc_start = None
    proc_end = None
    if process_time_range and process_time_range.get("start"):
        proc_start = pd.to_datetime(process_time_range["start"])
    if process_time_range and process_time_range.get("end"):
        proc_end = pd.to_datetime(process_time_range["end"])

    metric_order = df_filtered["metric_id"].unique().tolist()

    share_yaxis = (
        is_yaxis_shareable(selected_category)
        and shared_yaxis_toggle
        and "shared" in shared_yaxis_toggle
    )
    fig = create_all_timeseries_plots(df_filtered, proc_start, proc_end, full_time_range, category=selected_category, share_yaxis=share_yaxis)
    apply_figure_theme(fig, use_light_mode)

    df_for_store = df_filtered[["metric_id", "timestamp", "value"]].copy()
    filtered_cache_id = cache_dataframe(df_for_store, prefix="ts_filtered") if not df_for_store.empty else None
    filtered_df_json = {
        "cache_id": filtered_cache_id,
        "metric_order": metric_order,
        "y_axis_label": category_yaxis_label(selected_category),
    }

    graph_component = html.Div(
        dcc.Graph(
            id="timeseries-graph",
            figure=fig,
            style={
                "height": f"{fig.layout.height}px",
                "width": "100%",
                "display": "block",
            },
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "responsive": True,
            },
        ),
        style={
            "width": "100%",
            "height": f"{fig.layout.height}px",
            "minHeight": f"{fig.layout.height}px",
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "flex-start",
        },
    )

    return graph_component, filtered_df_json



@app.callback(
    Output("timeseries-graph", "figure", allow_duplicate=True),
    Input("shared-yaxis-toggle", "value"),
    State("timeseries-graph", "figure"),
    State("timeseries-filtered-df-store", "data"),
    prevent_initial_call=True,
)
def update_yaxis_on_toggle(shared_yaxis_toggle, current_figure, filtered_df_store):
    """Update Y-axis ranges when shared Y-axis toggle is changed."""
    if not current_figure or not filtered_df_store:
        return current_figure if current_figure else dash.no_update

    if not isinstance(filtered_df_store, dict) or "cache_id" not in filtered_df_store:
        return current_figure

    cache_id = filtered_df_store.get("cache_id")
    metric_order = filtered_df_store.get("metric_order", [])
    y_axis_label = filtered_df_store.get("y_axis_label", "Value")

    if not cache_id or not metric_order:
        return current_figure

    df = load_cached_dataframe(cache_id)
    ensure_timestamp_datetime(df)

    if df.empty:
        return current_figure

    updated_figure = copy.deepcopy(current_figure)
    layout = updated_figure.get("layout", {})

    share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle

    xaxis_layout = layout.get("xaxis", {})
    x_range = xaxis_layout.get("range")

    if x_range:
        x_min, x_max = align_xrange_tz(
            pd.to_datetime(x_range[0]),
            pd.to_datetime(x_range[1]),
            df["timestamp"].dt.tz,
        )
        visible_data = filter_to_time_range(df, x_min, x_max)
    else:
        visible_data = df

    if visible_data.empty:
        return current_figure

    is_memory_cat = metric_order and is_memory_metric(metric_order[0])

    yaxis_updates = compute_yaxis_ranges(visible_data, metric_order, share_yaxis, is_memory_cat)
    for yaxis_key, settings in yaxis_updates.items():
        if yaxis_key in layout:
            layout[yaxis_key]["range"] = settings["range"]
            layout[yaxis_key]["autorange"] = settings["autorange"]
            layout[yaxis_key]["title"] = {"text": y_axis_label}
            if "tickvals" in settings:
                layout[yaxis_key]["tickvals"] = settings["tickvals"]
                layout[yaxis_key]["ticktext"] = settings["ticktext"]
            else:
                layout[yaxis_key].pop("tickvals", None)
                layout[yaxis_key].pop("ticktext", None)

    updated_figure["layout"] = layout
    return updated_figure



@app.callback(
    Output("timeseries-graph", "figure", allow_duplicate=True),
    Input("timeseries-graph", "relayoutData"),
    State("timeseries-graph", "figure"),
    State("timeseries-filtered-df-store", "data"),
    State("shared-yaxis-toggle", "value"),
    prevent_initial_call=True,
)
def update_yaxis_on_zoom(relayout_data, current_figure, filtered_df_store, shared_yaxis_toggle):
    """Update Y-axis ranges when X-axis is zoomed to show visible data range."""
    if not relayout_data or not current_figure or not filtered_df_store:
        return current_figure

    if not isinstance(filtered_df_store, dict) or "cache_id" not in filtered_df_store:
        return current_figure

    cache_id = filtered_df_store.get("cache_id")
    metric_order = filtered_df_store.get("metric_order", [])

    if not cache_id or not metric_order:
        return current_figure

    is_reset = any("autorange" in key or "autosize" in key for key in relayout_data)
    if is_reset:
        is_memory_cat = metric_order and is_memory_metric(metric_order[0])

        if not is_memory_cat:
            return current_figure

        df = load_cached_dataframe(cache_id)
        ensure_timestamp_datetime(df)

        updated_figure = copy.deepcopy(current_figure)
        layout = updated_figure.get("layout", {})
        share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle

        yaxis_updates = compute_yaxis_ranges(df, metric_order, share_yaxis, is_memory_cat)
        for yaxis_key, settings in yaxis_updates.items():
            if yaxis_key in layout:
                layout[yaxis_key]["range"] = settings["range"]
                layout[yaxis_key]["autorange"] = settings["autorange"]
                if "tickvals" in settings:
                    layout[yaxis_key]["tickvals"] = settings["tickvals"]
                    layout[yaxis_key]["ticktext"] = settings["ticktext"]

        updated_figure["layout"] = layout
        return updated_figure

    xaxis_changes = {}
    for key in relayout_data:
        if "xaxis" in key and ".range[0]" in key:
            axis_name = key.replace(".range[0]", "")
            range_0_key = f"{axis_name}.range[0]"
            range_1_key = f"{axis_name}.range[1]"
            if range_0_key in relayout_data and range_1_key in relayout_data:
                if axis_name == "xaxis":
                    subplot_idx = 0
                else:
                    try:
                        subplot_idx = int(axis_name.replace("xaxis", "")) - 1
                    except ValueError:
                        continue
                x_min = pd.to_datetime(relayout_data[range_0_key])
                x_max = pd.to_datetime(relayout_data[range_1_key])
                xaxis_changes[subplot_idx] = (x_min, x_max)

    if not xaxis_changes:
        return current_figure

    df = load_cached_dataframe(cache_id)
    ensure_timestamp_datetime(df)

    if df.empty:
        return current_figure

    updated_figure = copy.deepcopy(current_figure)
    layout = updated_figure.get("layout", {})

    share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle

    first_subplot_idx = list(xaxis_changes.keys())[0]
    raw_x_min, raw_x_max = xaxis_changes[first_subplot_idx]
    x_min, x_max = align_xrange_tz(raw_x_min, raw_x_max, df["timestamp"].dt.tz)

    is_memory_cat = metric_order and is_memory_metric(metric_order[0])
    visible_data = filter_to_time_range(df, x_min, x_max)

    if visible_data.empty:
        return current_figure

    yaxis_updates = compute_yaxis_ranges(visible_data, metric_order, share_yaxis, is_memory_cat)
    for yaxis_key, settings in yaxis_updates.items():
        if yaxis_key in layout:
            layout[yaxis_key]["range"] = settings["range"]
            layout[yaxis_key]["autorange"] = settings["autorange"]
            if "tickvals" in settings:
                layout[yaxis_key]["tickvals"] = settings["tickvals"]
                layout[yaxis_key]["ticktext"] = settings["ticktext"]
            else:
                layout[yaxis_key].pop("tickvals", None)
                layout[yaxis_key].pop("ticktext", None)

    updated_figure["layout"] = layout
    return updated_figure
