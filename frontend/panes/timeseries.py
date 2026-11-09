"""Time series tab: callbacks for category filtering, CPU core selector, Y-axis toggle/zoom."""

import copy

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, dcc, html

from backend.categories import (
    available_cpu_cores,
    category_yaxis_label,
    filter_time_series_category,
    is_yaxis_shareable,
)
from backend.metrics import is_memory_metric
from backend.transforms import (
    align_xrange_tz,
    compute_yaxis_ranges,
    filter_to_time_range,
    get_time_range_from_df,
    yaxis_ranges_from_extrema,
)
from frontend.app import app
from frontend.cache import (
    cache_dataframe,
    cache_id_from_store,
    delete_cached_dataframe,
    df_from_store,
    load_cached_dataframe,
    metric_window_index,
    remember_metric_window_index,
)
from frontend.figures import (
    cartesian_axis_patch,
    create_all_timeseries_plots,
    relayout_requests_reset,
    relayout_x_windows,
    restore_axis_defaults,
    update_xaxis_ranges_in_layout,
    update_yaxis_ranges_in_layout,
)
from frontend.helpers import available_category_options, ensure_timestamp_datetime, parse_process_time_range_store
from frontend.layout import (
    PLOT_PREPARING_HIDDEN,
    PLOT_PREPARING_VISIBLE,
    empty_time_series_content,
    plot_preparing_overlay,
)
from frontend.style import (
    CARD_STYLE,
    DROPDOWN_STYLE,
    apply_figure_theme,
    device_class_key,
    status_alert_class,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("time-series-content", "children"),
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
    State("theme-switch", "value"),
)
def build_time_series_tab(processed_df_data, process_time_range, use_light_mode):
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
                                            options=[
                                                {"label": " Share Y-axis range across subplots", "value": "shared"}
                                            ],
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
                        className="time-series-controls",
                    ),
                    html.Div(
                        device_class_key(
                            include_process_active=True,
                            use_light_mode=bool(use_light_mode),
                        ),
                        id="timeseries-process-legend",
                        className="timeseries-process-legend",
                        style={"display": "none"},
                    ),
                    html.Div(
                        [
                            html.Div(id="timeseries-plot-container"),
                            plot_preparing_overlay("timeseries-plot-preparing"),
                        ],
                        className="plot-area-with-preparing timeseries-plot-area",
                    ),
                ],
                style={"backgroundColor": "var(--app-card-bg)"},
                className="viewport-card-body",
            ),
        ],
        style=CARD_STYLE,
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
    Output("timeseries-process-legend", "children"),
    Input("theme-switch", "value"),
    prevent_initial_call=True,
)
def update_timeseries_device_key(use_light_mode):
    return device_class_key(include_process_active=True, use_light_mode=bool(use_light_mode))


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
    Output("timeseries-process-legend", "style"),
    Output("timeseries-zoom-store", "data"),
    Input("metric-category-dropdown", "value"),
    Input("cpu-core-dropdown", "value"),
    State("theme-switch", "value"),
    State("shared-yaxis-toggle", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    State("timeseries-filtered-df-store", "data"),
    running=[
        (
            Output("timeseries-plot-preparing", "style"),
            PLOT_PREPARING_VISIBLE,
            PLOT_PREPARING_HIDDEN,
        ),
    ],
    prevent_initial_call=True,
)
def update_timeseries_plot(
    selected_category,
    selected_cpu_core,
    use_light_mode,
    shared_yaxis_toggle,
    processed_df_data,
    process_time_range,
    previous_filtered_store,
):
    legend_hidden = {"display": "none"}
    legend_visible = {"display": "flex"}
    previous_filtered_id = cache_id_from_store(previous_filtered_store)

    if not processed_df_data:
        delete_cached_dataframe(previous_filtered_id)
        return dbc.Alert("No data available.", color="warning", className=status_alert_class("warning")), None, legend_hidden, None

    if not selected_category:
        delete_cached_dataframe(previous_filtered_id)
        return dbc.Alert("Please select a metric category.", color="warning", className=status_alert_class("warning")), None, legend_hidden, None

    df_processed = df_from_store(processed_df_data)
    ensure_timestamp_datetime(df_processed)

    full_time_range = get_time_range_from_df(df_processed)

    if selected_category == "kernel_cpu_time":
        if not selected_cpu_core:
            delete_cached_dataframe(previous_filtered_id)
            return (
                dbc.Alert(
                    "Please select a CPU core to display kernel CPU time metrics.",
                    color="warning",
                    className=status_alert_class("warning"),
                ),
                None,
                legend_hidden,
                None,
            )

    df_filtered = filter_time_series_category(
        df_processed,
        selected_category,
        selected_cpu_core=selected_cpu_core,
    )

    if df_filtered.empty:
        delete_cached_dataframe(previous_filtered_id)
        return dbc.Alert("No data available for the selected category.", color="warning", className=status_alert_class("warning")), None, legend_hidden, None

    proc_start, proc_end = parse_process_time_range_store(process_time_range)

    metric_order = df_filtered["metric_id"].unique().tolist()

    share_yaxis = is_yaxis_shareable(selected_category) and shared_yaxis_toggle and "shared" in shared_yaxis_toggle
    fig = create_all_timeseries_plots(
        df_filtered,
        proc_start,
        proc_end,
        full_time_range,
        category=selected_category,
        share_yaxis=share_yaxis,
        use_light_mode=use_light_mode,
    )
    apply_figure_theme(fig, use_light_mode)

    # Keep CounterDiff / derived-power metadata needed for zoom and y-axis updates.
    df_for_store = df_filtered.copy()
    keep_cols = [
        c
        for c in (
            "metric_id",
            "timestamp",
            "value",
            "interval_start",
            "point_role",
            "point_order",
            "sample_id",
            "base_metric",
            "metric_origin",
        )
        if c in df_for_store.columns
    ]
    df_for_store = df_for_store[keep_cols]
    filtered_cache_id = cache_dataframe(df_for_store, prefix="ts_filtered") if not df_for_store.empty else None
    if filtered_cache_id:
        remember_metric_window_index(filtered_cache_id, _metric_window_index(df_for_store, metric_order))
    if previous_filtered_id and previous_filtered_id != filtered_cache_id:
        delete_cached_dataframe(previous_filtered_id)
    meta = fig.layout.meta
    if isinstance(meta, dict):
        axis_defaults = meta.get("axis_defaults") or {}
    else:
        axis_defaults = getattr(meta, "axis_defaults", None) or {}
    filtered_df_json = {
        "cache_id": filtered_cache_id,
        "metric_order": metric_order,
        "y_axis_label": category_yaxis_label(selected_category),
        "default_x_range": [
            pd.Timestamp(full_time_range[0]).isoformat(),
            pd.Timestamp(full_time_range[1]).isoformat(),
        ],
        "is_memory_category": selected_category == "memory",
        "axis_defaults": axis_defaults,
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
                # Always emit an autorange event.
                # The callback then restores the application's explicit defaults
                # instead of Plotly's mutable internal "initial" ranges.
                # See https://github.com/plotly/plotly.js/issues/6336
                "doubleClick": "autosize",
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

    return graph_component, filtered_df_json, legend_visible, None


def _metric_window_index(df: pd.DataFrame, metric_order: list) -> dict:
    """Sorted timestamps and values for each plotted metric.

    Timestamps stay in the series' own integer unit. Pandas 3 often uses
    microseconds, so these ticks are not ``Timestamp.value`` nanoseconds.
    """
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    ticks = timestamps.astype("int64").to_numpy()
    values = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype="float64")
    metrics = {}
    for metric_id in metric_order:
        mask = (df["metric_id"] == metric_id).to_numpy()
        metric_ticks = ticks[mask]
        metric_values = values[mask]
        order = np.argsort(metric_ticks, kind="mergesort")
        metrics[metric_id] = {
            "t": np.ascontiguousarray(metric_ticks[order]),
            "v": np.ascontiguousarray(metric_values[order]),
        }
    return {"tz": timestamps.dt.tz, "dtype": timestamps.dtype, "metrics": metrics}


def _window_extrema(index: dict, metric_order: list, x_range) -> list[tuple[float, float] | None] | None:
    """Inclusive min/max of each metric inside the time window. None when the window is empty."""
    start_tick = end_tick = None
    if x_range and x_range[0] is not None and x_range[1] is not None:
        x_min, x_max = align_xrange_tz(pd.to_datetime(x_range[0]), pd.to_datetime(x_range[1]), index.get("tz"))
        bounds = pd.Series([x_min, x_max]).astype(index["dtype"]).astype("int64")
        start_tick = int(bounds.iloc[0])
        end_tick = int(bounds.iloc[1])
    metrics = index.get("metrics") or {}
    per_metric: list[tuple[float, float] | None] = []
    saw_row = False
    for metric_id in metric_order:
        entry = metrics.get(metric_id)
        if entry is None:
            entry = metrics.get(str(metric_id))
        if not entry:
            per_metric.append(None)
            continue
        timestamps = entry["t"]
        values = entry["v"]
        if start_tick is not None:
            lo = int(np.searchsorted(timestamps, start_tick, side="left"))
            hi = int(np.searchsorted(timestamps, end_tick, side="right"))
            values = values[lo:hi]
        if values.size == 0:
            per_metric.append(None)
            continue
        saw_row = True
        if np.isnan(values).all():
            per_metric.append((float("nan"), float("nan")))
        else:
            per_metric.append((float(np.nanmin(values)), float(np.nanmax(values))))
    if not saw_row:
        return None
    return per_metric


def _yaxis_updates_for_window(cache_id, metric_order, x_range, share_yaxis, is_memory):
    """Y-axis updates for a time window, using the sorted index when the plot stored one."""
    index = metric_window_index(cache_id)
    if index is not None:
        per_metric = _window_extrema(index, metric_order, x_range)
        if per_metric is None:
            return None
        return yaxis_ranges_from_extrema(per_metric, share_yaxis, is_memory)

    df = load_cached_dataframe(cache_id)
    ensure_timestamp_datetime(df)
    if df.empty:
        return None
    if x_range and x_range[0] is not None and x_range[1] is not None:
        x_min, x_max = align_xrange_tz(
            pd.to_datetime(x_range[0]),
            pd.to_datetime(x_range[1]),
            df["timestamp"].dt.tz,
        )
        visible_data = filter_to_time_range(df, x_min, x_max)
    else:
        visible_data = df
    if visible_data.empty:
        return None
    return compute_yaxis_ranges(visible_data, metric_order, share_yaxis, is_memory)


@app.callback(
    Output("timeseries-graph", "figure", allow_duplicate=True),
    Input("shared-yaxis-toggle", "value"),
    State("timeseries-filtered-df-store", "data"),
    State("timeseries-zoom-store", "data"),
    running=[
        (
            Output("timeseries-plot-preparing", "style", allow_duplicate=True),
            PLOT_PREPARING_VISIBLE,
            PLOT_PREPARING_HIDDEN,
        ),
    ],
    prevent_initial_call=True,
)
def update_yaxis_on_toggle(shared_yaxis_toggle, filtered_df_store, zoom_state):
    """Refit Y ranges for the current time window when sharing is toggled."""
    if not isinstance(filtered_df_store, dict) or "cache_id" not in filtered_df_store:
        return dash.no_update

    cache_id = filtered_df_store.get("cache_id")
    metric_order = filtered_df_store.get("metric_order", [])

    if not cache_id or not metric_order:
        return dash.no_update

    share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle
    zoom_state = zoom_state or {}
    if zoom_state.get("mode") == "zoom" and zoom_state.get("x0") is not None and zoom_state.get("x1") is not None:
        x_range = [zoom_state["x0"], zoom_state["x1"]]
    else:
        x_range = filtered_df_store.get("default_x_range")

    is_memory_cat = filtered_df_store.get(
        "is_memory_category",
        bool(metric_order and is_memory_metric(metric_order[0])),
    )
    yaxis_updates = _yaxis_updates_for_window(
        cache_id,
        metric_order,
        x_range,
        share_yaxis,
        is_memory_cat,
    )
    if yaxis_updates is None:
        return dash.no_update

    layout = _figure_from_axis_defaults(filtered_df_store.get("axis_defaults") or {}, metric_order)["layout"]
    update_yaxis_ranges_in_layout(layout, yaxis_updates)
    _lock_timeseries_y_axes(layout)
    return cartesian_axis_patch(layout)


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

    is_memory_cat = filtered_df_store.get(
        "is_memory_category",
        bool(metric_order and is_memory_metric(metric_order[0])),
    )
    if relayout_requests_reset(relayout_data):
        updated_figure = copy.deepcopy(current_figure)
        layout = updated_figure.get("layout", {})
        defaults = (layout.get("meta") or {}).get("axis_defaults") or {}
        if defaults:
            for key, axis_defaults in defaults.items():
                axis = layout.get(key)
                if isinstance(axis, dict) and isinstance(axis_defaults, dict):
                    restore_axis_defaults(axis, axis_defaults)
        else:
            share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle
            default_x_range = filtered_df_store.get("default_x_range")
            if default_x_range:
                update_xaxis_ranges_in_layout(layout, default_x_range)
            yaxis_updates = _yaxis_updates_for_window(
                cache_id,
                metric_order,
                None,
                share_yaxis,
                is_memory_cat,
            )
            if yaxis_updates is None:
                return dash.no_update
            update_yaxis_ranges_in_layout(layout, yaxis_updates)
        _lock_timeseries_y_axes(layout)

        updated_figure["layout"] = layout
        if _axis_view_matches(current_figure.get("layout", {}), layout):
            return dash.no_update
        return updated_figure

    windows = relayout_x_windows(relayout_data)
    if not windows:
        return dash.no_update

    updated_figure = copy.deepcopy(current_figure)
    layout = updated_figure.get("layout", {})

    share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle

    raw_start, raw_end = next(iter(windows.values()))
    raw_x_min, raw_x_max = pd.to_datetime(raw_start), pd.to_datetime(raw_end)
    yaxis_updates = _yaxis_updates_for_window(
        cache_id,
        metric_order,
        [raw_x_min, raw_x_max],
        share_yaxis,
        is_memory_cat,
    )
    if yaxis_updates is None:
        return dash.no_update

    update_xaxis_ranges_in_layout(layout, [raw_x_min.isoformat(), raw_x_max.isoformat()])
    update_yaxis_ranges_in_layout(layout, yaxis_updates)
    _lock_timeseries_y_axes(layout)

    updated_figure["layout"] = layout
    if _axis_view_matches(current_figure.get("layout", {}), layout):
        return dash.no_update
    return updated_figure


def _lock_timeseries_y_axes(layout: dict) -> None:
    """Keep stacked zoom on the time axis after a Python relayout."""
    for key, axis in layout.items():
        if isinstance(axis, dict) and key.startswith("yaxis"):
            axis["fixedrange"] = True


def _axis_view_matches(current_layout: dict, updated_layout: dict) -> bool:
    """True when X/Y ranges and tick labels are already what the zoom would write."""
    for key, updated in updated_layout.items():
        if not isinstance(key, str) or not (key.startswith("xaxis") or key.startswith("yaxis")):
            continue
        if not isinstance(updated, dict):
            continue
        current = current_layout.get(key) or {}
        if list(current.get("range") or []) != list(updated.get("range") or []):
            return False
        if list(current.get("ticktext") or []) != list(updated.get("ticktext") or []):
            return False
    return True


def _figure_from_axis_defaults(defaults: dict, metric_order: list) -> dict:
    """A layout-only figure so a zoom can refit axes without the plotted points."""
    layout = {}
    for key, axis in (defaults or {}).items():
        if isinstance(axis, dict):
            layout[key] = copy.deepcopy(axis)
    if not layout:
        for index in range(len(metric_order)):
            x_key = "xaxis" if index == 0 else f"xaxis{index + 1}"
            y_key = "yaxis" if index == 0 else f"yaxis{index + 1}"
            layout[x_key] = {"autorange": False}
            layout[y_key] = {"autorange": False}
    layout["meta"] = {"axis_defaults": defaults or {}}
    return {"data": [], "layout": layout}


def _same_time_window(left, right) -> bool:
    try:
        return pd.to_datetime(left[0]) == pd.to_datetime(right[0]) and pd.to_datetime(left[1]) == pd.to_datetime(right[1])
    except (TypeError, ValueError):
        return False


@app.callback(
    Output("timeseries-graph", "figure", allow_duplicate=True),
    Output("timeseries-zoom-store", "data", allow_duplicate=True),
    Input("timeseries-graph", "relayoutData"),
    State("timeseries-filtered-df-store", "data"),
    State("shared-yaxis-toggle", "value"),
    State("timeseries-zoom-store", "data"),
    running=[
        (
            Output("timeseries-plot-preparing", "style", allow_duplicate=True),
            PLOT_PREPARING_VISIBLE,
            PLOT_PREPARING_HIDDEN,
        ),
    ],
    prevent_initial_call=True,
)
def patch_timeseries_on_zoom(relayout_data, filtered_df_store, shared_yaxis_toggle, zoom_state):
    """After a zoom or double-click, send axis ranges only and wait for that layout to apply."""
    if not relayout_data or not isinstance(filtered_df_store, dict):
        return dash.no_update, dash.no_update
    metric_order = filtered_df_store.get("metric_order") or []
    if not filtered_df_store.get("cache_id") or not metric_order:
        return dash.no_update, dash.no_update

    zoom_state = zoom_state or {}
    windows = relayout_x_windows(relayout_data)
    default_x = filtered_df_store.get("default_x_range")
    window = next(iter(windows.values())) if windows else None
    restoring = relayout_requests_reset(relayout_data) or (
        window is not None and default_x is not None and _same_time_window(window, default_x)
    )
    if restoring:
        if zoom_state.get("mode") == "reset":
            return dash.no_update, dash.no_update
        defaults = filtered_df_store.get("axis_defaults") or {}
        if defaults:
            layout = {}
            for key, axis in defaults.items():
                if not isinstance(axis, dict):
                    continue
                layout[key] = copy.deepcopy(axis)
                if str(key).startswith("yaxis"):
                    layout[key]["fixedrange"] = True
            return cartesian_axis_patch(layout), {"mode": "reset"}
        next_state = {"mode": "reset"}
    elif not window:
        return dash.no_update, dash.no_update
    else:
        x0, x1 = window
        if (
            zoom_state.get("mode") == "zoom"
            and _same_time_window((zoom_state.get("x0"), zoom_state.get("x1")), (x0, x1))
        ):
            return dash.no_update, dash.no_update
        next_state = {"mode": "zoom", "x0": x0, "x1": x1}

    current = _figure_from_axis_defaults(filtered_df_store.get("axis_defaults") or {}, metric_order)
    updated = update_yaxis_on_zoom(relayout_data, current, filtered_df_store, shared_yaxis_toggle)
    if not isinstance(updated, dict) or updated is current:
        return dash.no_update, dash.no_update
    return cartesian_axis_patch(updated.get("layout") or {}), next_state
