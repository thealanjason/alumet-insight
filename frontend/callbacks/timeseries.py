"""Time series tab callbacks: build tab, category filtering, CPU core selector, Y-axis toggle/zoom."""

import copy

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dcc, html

from frontend.app import app
from frontend.cache import cache_dataframe, df_from_store, load_cached_dataframe, _ensure_timestamp_datetime
from frontend.theme import status_alert_class, apply_figure_theme
from frontend.layout import empty_time_series_content
from frontend.helpers import available_category_options
from backend.categories import available_cpu_cores, filter_time_series_category
from backend.metrics import is_memory_metric, get_metric_unit, get_bytes_tickvals_ticktext
from visualization.interactive_plotting import create_all_timeseries_plots



@app.callback(
    Output("time-series-content", "children"),
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
)
def build_time_series_tab(processed_df_data, process_time_range):
    if not processed_df_data:
        return empty_time_series_content()

    df_processed = df_from_store(processed_df_data)
    _ensure_timestamp_datetime(df_processed)

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
                                        style={"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"},
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
                                        style={
                                            "display": "none",
                                            "backgroundColor": "var(--app-control-bg)",
                                            "color": "var(--app-text)",
                                        },
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
    valid_categories = ["energy", "power", "utilization", "temperature", "memory", "kernel_cpu_time"]
    if selected_category in valid_categories:
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
    _ensure_timestamp_datetime(df_processed)

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

    yaxis_shareable_categories = {"energy", "power", "utilization", "temperature", "memory", "kernel_cpu_time"}
    share_yaxis = (
        selected_category in yaxis_shareable_categories
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

    if not cache_id or not metric_order:
        return current_figure

    df = load_cached_dataframe(cache_id)
    _ensure_timestamp_datetime(df)

    if df.empty:
        return current_figure

    updated_figure = copy.deepcopy(current_figure)
    layout = updated_figure.get("layout", {})

    share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle

    xaxis_layout = layout.get("xaxis", {})
    x_range = xaxis_layout.get("range")

    if x_range:
        x_min = pd.to_datetime(x_range[0])
        x_max = pd.to_datetime(x_range[1])

        df_tz = df["timestamp"].dt.tz
        if df_tz is not None:
            if x_min.tz is None:
                x_min = x_min.tz_localize(df_tz)
            if x_max.tz is None:
                x_max = x_max.tz_localize(df_tz)
        else:
            if x_min.tz is not None:
                x_min = x_min.tz_convert(None) if hasattr(x_min, "tz_convert") else x_min.replace(tzinfo=None)
            if x_max.tz is not None:
                x_max = x_max.tz_convert(None) if hasattr(x_max, "tz_convert") else x_max.replace(tzinfo=None)

        visible_data = df[(df["timestamp"] >= x_min) & (df["timestamp"] <= x_max)]
    else:
        visible_data = df

    if visible_data.empty:
        return current_figure

    is_memory_category = metric_order and is_memory_metric(metric_order[0])

    if is_memory_category:
        y_axis_label = "Value (B)"
    elif metric_order and get_metric_unit(metric_order[0]) == "J":
        y_axis_label = "Value (J)"
    elif metric_order and get_metric_unit(metric_order[0]) == "ms":
        y_axis_label = "Value (ms)"
    else:
        y_axis_label = "Value"

    if share_yaxis:
        global_y_min = visible_data["value"].min()
        global_y_max = visible_data["value"].max()
        y_range_val = global_y_max - global_y_min if global_y_max != global_y_min else abs(global_y_max) if global_y_max != 0 else 1
        y_padding = 0.1 * y_range_val if y_range_val > 0 else 0.1

        calc_min = global_y_min - y_padding
        calc_max = global_y_max + y_padding
        if calc_min >= calc_max:
            calc_min = global_y_min - 0.1 if global_y_min != 0 else -0.1
            calc_max = global_y_max + 0.1 if global_y_max != 0 else 0.1

        if is_memory_category:
            calc_min = max(0, calc_min)

        shared_tickvals = None
        shared_ticktext = None
        if is_memory_category:
            shared_tickvals, shared_ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)

        for subplot_idx in range(len(metric_order)):
            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            if yaxis_key in layout:
                layout[yaxis_key]["range"] = [calc_min, calc_max]
                layout[yaxis_key]["autorange"] = False
                layout[yaxis_key]["title"] = {"text": y_axis_label}
                if is_memory_category and shared_tickvals is not None:
                    layout[yaxis_key]["tickvals"] = shared_tickvals
                    layout[yaxis_key]["ticktext"] = shared_ticktext
                else:
                    layout[yaxis_key].pop("tickvals", None)
                    layout[yaxis_key].pop("ticktext", None)
    else:
        for subplot_idx in range(len(metric_order)):
            metric_id = metric_order[subplot_idx]
            metric_visible = visible_data[visible_data["metric_id"] == metric_id]

            if metric_visible.empty:
                continue

            y_min_val = metric_visible["value"].min()
            y_max_val = metric_visible["value"].max()
            y_range_val = y_max_val - y_min_val if y_max_val != y_min_val else abs(y_max_val) if y_max_val != 0 else 1
            y_padding = 0.1 * y_range_val if y_range_val > 0 else 0.1

            calc_min = y_min_val - y_padding
            calc_max = y_max_val + y_padding
            if calc_min >= calc_max:
                calc_min = y_min_val - 0.1 if y_min_val != 0 else -0.1
                calc_max = y_max_val + 0.1 if y_max_val != 0 else 0.1

            if is_memory_category:
                calc_min = max(0, calc_min)

            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            if yaxis_key in layout:
                layout[yaxis_key]["range"] = [calc_min, calc_max]
                layout[yaxis_key]["autorange"] = False
                layout[yaxis_key]["title"] = {"text": y_axis_label}
                if is_memory_category:
                    tickvals, ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)
                    layout[yaxis_key]["tickvals"] = tickvals
                    layout[yaxis_key]["ticktext"] = ticktext
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
        is_memory_category = metric_order and is_memory_metric(metric_order[0])

        if not is_memory_category:
            return current_figure

        df = load_cached_dataframe(cache_id)
        _ensure_timestamp_datetime(df)

        updated_figure = copy.deepcopy(current_figure)
        layout = updated_figure.get("layout", {})
        share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle

        if share_yaxis:
            global_y_min = df["value"].min()
            global_y_max = df["value"].max()
            y_range_val = global_y_max - global_y_min if global_y_max != global_y_min else abs(global_y_max) if global_y_max != 0 else 1
            y_padding = 0.1 * y_range_val
            calc_min = max(0, global_y_min - y_padding)
            calc_max = global_y_max + y_padding
            tickvals, ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)

            for subplot_idx in range(len(metric_order)):
                yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
                if yaxis_key in layout:
                    layout[yaxis_key]["range"] = [calc_min, calc_max]
                    layout[yaxis_key]["autorange"] = False
                    layout[yaxis_key]["tickvals"] = tickvals
                    layout[yaxis_key]["ticktext"] = ticktext
        else:
            for subplot_idx, metric_id in enumerate(metric_order):
                metric_data = df[df["metric_id"] == metric_id]
                if metric_data.empty:
                    continue
                y_min_val = metric_data["value"].min()
                y_max_val = metric_data["value"].max()
                y_range_val = y_max_val - y_min_val if y_max_val != y_min_val else abs(y_max_val) if y_max_val != 0 else 1
                y_padding = 0.1 * y_range_val
                calc_min = max(0, y_min_val - y_padding)
                calc_max = y_max_val + y_padding
                tickvals, ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)
                yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
                if yaxis_key in layout:
                    layout[yaxis_key]["range"] = [calc_min, calc_max]
                    layout[yaxis_key]["autorange"] = False
                    layout[yaxis_key]["tickvals"] = tickvals
                    layout[yaxis_key]["ticktext"] = ticktext

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
    _ensure_timestamp_datetime(df)

    if df.empty:
        return current_figure

    df_tz = df["timestamp"].dt.tz

    updated_figure = copy.deepcopy(current_figure)
    layout = updated_figure.get("layout", {})

    share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle

    first_subplot_idx = list(xaxis_changes.keys())[0]
    x_min, x_max = xaxis_changes[first_subplot_idx]

    if df_tz is not None:
        if x_min.tz is None:
            x_min = x_min.tz_localize(df_tz)
        if x_max.tz is None:
            x_max = x_max.tz_localize(df_tz)
    else:
        if x_min.tz is not None:
            x_min = x_min.tz_convert(None) if hasattr(x_min, "tz_convert") else x_min.replace(tzinfo=None)
        if x_max.tz is not None:
            x_max = x_max.tz_convert(None) if hasattr(x_max, "tz_convert") else x_max.replace(tzinfo=None)

    is_memory_category = metric_order and is_memory_metric(metric_order[0])

    if share_yaxis:
        visible_data = df[(df["timestamp"] >= x_min) & (df["timestamp"] <= x_max)]
        if visible_data.empty:
            return current_figure

        global_y_min = visible_data["value"].min()
        global_y_max = visible_data["value"].max()
        y_range_val = global_y_max - global_y_min if global_y_max != global_y_min else abs(global_y_max) if global_y_max != 0 else 1
        y_padding = 0.1 * y_range_val if y_range_val > 0 else 0.1

        calc_min = global_y_min - y_padding
        calc_max = global_y_max + y_padding
        if calc_min >= calc_max:
            calc_min = global_y_min - 0.1 if global_y_min != 0 else -0.1
            calc_max = global_y_max + 0.1 if global_y_max != 0 else 0.1

        if is_memory_category:
            calc_min = max(0, calc_min)

        shared_tickvals = None
        shared_ticktext = None
        if is_memory_category:
            shared_tickvals, shared_ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)

        for subplot_idx in range(len(metric_order)):
            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            if yaxis_key in layout:
                layout[yaxis_key]["range"] = [calc_min, calc_max]
                layout[yaxis_key]["autorange"] = False
                if is_memory_category and shared_tickvals is not None:
                    layout[yaxis_key]["tickvals"] = shared_tickvals
                    layout[yaxis_key]["ticktext"] = shared_ticktext
                else:
                    layout[yaxis_key].pop("tickvals", None)
                    layout[yaxis_key].pop("ticktext", None)
    else:
        for subplot_idx in range(len(metric_order)):
            metric_id = metric_order[subplot_idx]
            metric_data = df[df["metric_id"] == metric_id]
            metric_visible = metric_data[(metric_data["timestamp"] >= x_min) & (metric_data["timestamp"] <= x_max)]
            if metric_visible.empty:
                continue

            y_min_val = metric_visible["value"].min()
            y_max_val = metric_visible["value"].max()
            y_range_val = y_max_val - y_min_val if y_max_val != y_min_val else abs(y_max_val) if y_max_val != 0 else 1
            y_padding = 0.1 * y_range_val if y_range_val > 0 else 0.1

            calc_min = y_min_val - y_padding
            calc_max = y_max_val + y_padding
            if calc_min >= calc_max:
                calc_min = y_min_val - 0.1 if y_min_val != 0 else -0.1
                calc_max = y_max_val + 0.1 if y_max_val != 0 else 0.1

            if is_memory_category:
                calc_min = max(0, calc_min)

            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            if yaxis_key in layout:
                layout[yaxis_key]["range"] = [calc_min, calc_max]
                layout[yaxis_key]["autorange"] = False
                if is_memory_category:
                    tickvals, ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)
                    layout[yaxis_key]["tickvals"] = tickvals
                    layout[yaxis_key]["ticktext"] = ticktext
                else:
                    layout[yaxis_key].pop("tickvals", None)
                    layout[yaxis_key].pop("ticktext", None)

    updated_figure["layout"] = layout
    return updated_figure
