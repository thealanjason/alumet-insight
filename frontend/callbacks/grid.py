"""Process-specific grid callbacks: build 2x2 grid tab, MATCH filters, plot updates, zoom sync, CSV download."""

import copy
import json

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, MATCH, ALL

from frontend.app import app
from frontend.cache import df_from_store, is_cache_miss, _ensure_timestamp_datetime
from frontend.theme import status_alert_class, apply_figure_theme
from frontend.helpers import normalize_dropdown_value, unique_nonempty
from backend.metrics import get_metric_unit, is_memory_metric, get_bytes_tickvals_ticktext
from visualization.interactive_plotting import get_color_palette



@app.callback(
    Output("process-specific-content", "children"),
    Input("results-tabs", "value"),
    Input("original-df-store", "data"),
    Input("process-time-range-store", "data"),
    State("process-specific-content", "children"),
)
def build_process_specific_tab(tab_value, original_df_data, process_time_range, current_children):
    triggered_id = ctx.triggered_id
    is_data_trigger = triggered_id in ("original-df-store", "process-time-range-store")

    if is_data_trigger and tab_value != "process-specific-tab":
        return []

    if triggered_id == "results-tabs":
        if tab_value != "process-specific-tab":
            return dash.no_update
        if current_children:
            return dash.no_update

    if not original_df_data or not process_time_range:
        return dbc.Alert(
            "No data available. Please load data using the Visualize button.",
            color="warning",
            className=status_alert_class("warning"),
        )

    proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range.get("end") else None

    if proc_start is None or proc_end is None:
        return dbc.Alert(
            "Process time range not available.",
            color="warning",
            className=status_alert_class("warning"),
        )

    df_original = df_from_store(original_df_data)
    if is_cache_miss(df_original):
        return dbc.Alert(
            "Session data expired (server was restarted). Please click Visualize again to reload.",
            color="danger",
            className=status_alert_class("danger"),
        )
    _ensure_timestamp_datetime(df_original)

    unique_metrics = sorted(df_original["metric"].unique().tolist())

    grid_rows = []
    for i in range(2):
        row_children = []
        for j in range(2):
            plot_id = {"type": "grid-plot", "index": f"{i}-{j}"}
            metric_dropdown_id = {"type": "metric-dropdown", "index": f"{i}-{j}"}

            row_children.append(
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Metric:",
                                                    style={
                                                        "color": "var(--app-text)",
                                                        "fontSize": "0.9rem",
                                                        "fontWeight": "500",
                                                        "marginBottom": "4px",
                                                    },
                                                ),
                                                dcc.Dropdown(
                                                    id=metric_dropdown_id,
                                                    options=[{"label": m, "value": m} for m in unique_metrics],
                                                    placeholder="Select metric",
                                                    style={"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"},
                                                    className="dark-dropdown",
                                                    clearable=True,
                                                ),
                                            ],
                                            style={"marginBottom": "8px"},
                                        ),
                                        html.Div(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            html.Div(
                                                                [
                                                                    html.Label("R.Kind", style={"color": "var(--app-accent-soft)", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                    dcc.Dropdown(
                                                                        id={"type": "resource-kind-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="-",
                                                                        style={"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)", "fontSize": "0.75rem"},
                                                                        className="dark-dropdown compact-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "rk-container", "index": f"{i}-{j}"},
                                                                style={"visibility": "hidden"},
                                                            ),
                                                            style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                        ),
                                                        dbc.Col(
                                                            html.Div(
                                                                [
                                                                    html.Label("R.ID", style={"color": "var(--app-accent-soft)", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                    dcc.Dropdown(
                                                                        id={"type": "resource-id-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="-",
                                                                        style={"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)", "fontSize": "0.75rem"},
                                                                        className="dark-dropdown compact-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "rid-container", "index": f"{i}-{j}"},
                                                                style={"visibility": "hidden"},
                                                            ),
                                                            style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                        ),
                                                        dbc.Col(
                                                            html.Div(
                                                                [
                                                                    html.Label("C.Kind", style={"color": "var(--app-accent-soft)", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                    dcc.Dropdown(
                                                                        id={"type": "consumer-kind-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="-",
                                                                        style={"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)", "fontSize": "0.75rem"},
                                                                        className="dark-dropdown compact-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "ck-container", "index": f"{i}-{j}"},
                                                                style={"visibility": "hidden"},
                                                            ),
                                                            style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                        ),
                                                        dbc.Col(
                                                            html.Div(
                                                                [
                                                                    html.Label("C.ID", style={"color": "var(--app-accent-soft)", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                    dcc.Dropdown(
                                                                        id={"type": "consumer-id-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="-",
                                                                        style={"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)", "fontSize": "0.75rem"},
                                                                        className="dark-dropdown compact-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "cid-container", "index": f"{i}-{j}"},
                                                                style={"visibility": "hidden"},
                                                            ),
                                                            style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                        ),
                                                        dbc.Col(
                                                            html.Div(
                                                                [
                                                                    html.Label("Attr", style={"color": "var(--app-accent-soft)", "fontSize": "0.7rem", "marginBottom": "1px", "whiteSpace": "nowrap"}),
                                                                    dcc.Dropdown(
                                                                        id={"type": "late-attr-dropdown", "index": f"{i}-{j}"},
                                                                        options=[],
                                                                        value=None,
                                                                        placeholder="-",
                                                                        style={"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)", "fontSize": "0.75rem"},
                                                                        className="dark-dropdown compact-dropdown",
                                                                        clearable=False,
                                                                    ),
                                                                ],
                                                                id={"type": "la-container", "index": f"{i}-{j}"},
                                                                style={"visibility": "hidden"},
                                                            ),
                                                            style={"paddingRight": "2px", "paddingLeft": "2px"},
                                                        ),
                                                    ],
                                                    className="g-0",
                                                ),
                                            ],
                                            style={"minHeight": "50px", "marginBottom": "8px"},
                                        ),
                                        dcc.Graph(id=plot_id, style={"height": "320px"}),
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "\U0001f4e5 Download CSV",
                                                    id={"type": "grid-download-btn", "index": f"{i}-{j}"},
                                                    n_clicks=0,
                                                    color="primary",
                                                    size="sm",
                                                    style={"fontSize": "0.75rem"},
                                                ),
                                                dcc.Download(id={"type": "grid-download", "index": f"{i}-{j}"}),
                                            ],
                                            style={"textAlign": "right", "marginTop": "25px", "paddingTop": "15px"},
                                        ),
                                    ],
                                    style={"padding": "12px", "backgroundColor": "var(--app-card-bg)"},
                                ),
                            ],
                            style={"height": "100%", "marginBottom": "10px", "backgroundColor": "var(--app-card-bg)", "border": "1px solid var(--app-border)"},
                        ),
                    ],
                    width=12,
                    lg=6,
                    className="mb-2",
                )
            )
        grid_rows.append(dbc.Row(row_children, className="mb-2"))

    return html.Div(grid_rows, className="process-grid-scroll")


# MATCH callback: update filter dropdowns
@app.callback(
    Output({"type": "rk-container", "index": MATCH}, "style"),
    Output({"type": "resource-kind-dropdown", "index": MATCH}, "options"),
    Output({"type": "resource-kind-dropdown", "index": MATCH}, "value"),
    Output({"type": "rid-container", "index": MATCH}, "style"),
    Output({"type": "resource-id-dropdown", "index": MATCH}, "options"),
    Output({"type": "resource-id-dropdown", "index": MATCH}, "value"),
    Output({"type": "ck-container", "index": MATCH}, "style"),
    Output({"type": "consumer-kind-dropdown", "index": MATCH}, "options"),
    Output({"type": "consumer-kind-dropdown", "index": MATCH}, "value"),
    Output({"type": "cid-container", "index": MATCH}, "style"),
    Output({"type": "consumer-id-dropdown", "index": MATCH}, "options"),
    Output({"type": "consumer-id-dropdown", "index": MATCH}, "value"),
    Output({"type": "la-container", "index": MATCH}, "style"),
    Output({"type": "late-attr-dropdown", "index": MATCH}, "options"),
    Output({"type": "late-attr-dropdown", "index": MATCH}, "value"),
    Input({"type": "metric-dropdown", "index": MATCH}, "value"),
    Input({"type": "resource-kind-dropdown", "index": MATCH}, "value"),
    Input({"type": "resource-id-dropdown", "index": MATCH}, "value"),
    Input({"type": "consumer-kind-dropdown", "index": MATCH}, "value"),
    Input({"type": "consumer-id-dropdown", "index": MATCH}, "value"),
    Input({"type": "late-attr-dropdown", "index": MATCH}, "value"),
    State("original-df-store", "data"),
    prevent_initial_call=True,
)
def update_filters_match(metric, rk, rid, ck, cid, la, original_df_data):
    """Update filter dropdowns for a single plot using MATCH."""
    hide = {"visibility": "hidden"}
    show = {"visibility": "visible"}

    if not original_df_data or not metric:
        return (hide, [], None, hide, [], None, hide, [], None, hide, [], None, hide, [], None)

    df = df_from_store(original_df_data)
    dfm = df[df["metric"] == metric].copy()

    dfm["rk"] = dfm["resource_kind"].astype(object).fillna("").astype(str).str.strip()
    dfm["rid"] = dfm["resource_id"].astype(object).fillna("").astype(str).str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(object).fillna("").astype(str).str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(object).fillna("").astype(str).str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(object).fillna("").astype(str).str.strip()

    rk = normalize_dropdown_value(rk)
    rid = normalize_dropdown_value(rid)
    ck = normalize_dropdown_value(ck)
    cid = normalize_dropdown_value(cid)
    la = normalize_dropdown_value(la)

    triggered_id = None
    if ctx.triggered:
        triggered = ctx.triggered[0]
        triggered_prop_id = triggered.get("prop_id", "")
        if ".value" in triggered_prop_id:
            try:
                json_str = triggered_prop_id.split(".value")[0]
                id_dict = json.loads(json_str)
                triggered_id = id_dict.get("type")
            except Exception:
                pass

    rk_opts = unique_nonempty(dfm["rk"])
    rk_eff = rk if rk in rk_opts else (rk_opts[0] if len(rk_opts) == 1 else None)
    df1 = dfm if rk_eff is None else dfm[dfm["rk"] == rk_eff]

    rid_opts = unique_nonempty(df1["rid"])
    if triggered_id == "resource-kind-dropdown":
        rid_eff = None
    else:
        rid_eff = rid if rid in rid_opts else (rid_opts[0] if len(rid_opts) == 1 else None)
    df2 = df1 if rid_eff is None else df1[df1["rid"] == rid_eff]

    ck_opts = unique_nonempty(df2["ck"])
    ck_eff = ck if ck in ck_opts else (ck_opts[0] if len(ck_opts) == 1 else None)
    df3 = df2 if ck_eff is None else df2[df2["ck"] == ck_eff]

    cid_opts = unique_nonempty(df3["cid"])
    if triggered_id == "consumer-kind-dropdown":
        cid_eff = None
    else:
        cid_eff = cid if cid in cid_opts else (cid_opts[0] if len(cid_opts) == 1 else None)
    df4 = df3 if cid_eff is None else df3[df3["cid"] == cid_eff]

    la_opts = unique_nonempty(df4["la"])
    if triggered_id in ["resource-kind-dropdown", "resource-id-dropdown", "consumer-kind-dropdown", "consumer-id-dropdown"]:
        la_eff = None
    else:
        la_eff = la if la in la_opts else None

    rk_style = show if len(rk_opts) > 1 else hide
    rid_style = show if len(rid_opts) > 1 else hide
    ck_style = show if len(ck_opts) > 1 else hide
    cid_style = show if len(cid_opts) > 1 else hide
    la_style = show if len(la_opts) > 1 else hide

    rk_options = [{"label": v, "value": v} for v in rk_opts]
    rid_options = [{"label": v, "value": v} for v in rid_opts]
    ck_options = [{"label": v, "value": v} for v in ck_opts]
    cid_options = [{"label": v, "value": v} for v in cid_opts]
    la_options = [{"label": v, "value": v} for v in la_opts]

    return (
        rk_style, rk_options, rk_eff,
        rid_style, rid_options, rid_eff,
        ck_style, ck_options, ck_eff,
        cid_style, cid_options, cid_eff,
        la_style, la_options, la_eff,
    )


# MATCH callback: update grid plot figure
@app.callback(
    Output({"type": "grid-plot", "index": MATCH}, "figure"),
    Input({"type": "metric-dropdown", "index": MATCH}, "value"),
    Input({"type": "resource-kind-dropdown", "index": MATCH}, "value"),
    Input({"type": "resource-id-dropdown", "index": MATCH}, "value"),
    Input({"type": "consumer-kind-dropdown", "index": MATCH}, "value"),
    Input({"type": "consumer-id-dropdown", "index": MATCH}, "value"),
    Input({"type": "late-attr-dropdown", "index": MATCH}, "value"),
    Input("theme-switch", "value"),
    State("original-df-store", "data"),
    State("process-time-range-store", "data"),
    State({"type": "metric-dropdown", "index": MATCH}, "id"),
)
def update_grid_plot_match(metric, rk, rid, ck, cid, la, use_light_mode, original_df_data, process_time_range, my_id):
    """Update a single grid plot figure using MATCH."""
    fig = go.Figure()
    apply_figure_theme(fig, use_light_mode)

    if not original_df_data or not metric:
        fig.update_layout(title=dict(text="Select a metric", x=0.5))
        return fig

    df = df_from_store(original_df_data)
    _ensure_timestamp_datetime(df)
    dfm = df[df["metric"] == metric].copy()

    dfm["rk"] = dfm["resource_kind"].astype(object).fillna("").astype(str).str.strip()
    dfm["rid"] = dfm["resource_id"].astype(object).fillna("").astype(str).str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(object).fillna("").astype(str).str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(object).fillna("").astype(str).str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(object).fillna("").astype(str).str.strip()

    rk = normalize_dropdown_value(rk)
    rid = normalize_dropdown_value(rid)
    ck = normalize_dropdown_value(ck)
    cid = normalize_dropdown_value(cid)
    la = normalize_dropdown_value(la)

    rk_opts = unique_nonempty(dfm["rk"])
    rk_eff = rk if rk in rk_opts else (rk_opts[0] if len(rk_opts) == 1 else None)
    df1 = dfm if rk_eff is None else dfm[dfm["rk"] == rk_eff]

    rid_opts = unique_nonempty(df1["rid"])
    rid_eff = rid if rid in rid_opts else (rid_opts[0] if len(rid_opts) == 1 else None)
    df2 = df1 if rid_eff is None else df1[df1["rid"] == rid_eff]

    ck_opts = unique_nonempty(df2["ck"])
    ck_eff = ck if ck in ck_opts else (ck_opts[0] if len(ck_opts) == 1 else None)
    df3 = df2 if ck_eff is None else df2[df2["ck"] == ck_eff]

    cid_opts = unique_nonempty(df3["cid"])
    cid_eff = cid if cid in cid_opts else (cid_opts[0] if len(cid_opts) == 1 else None)
    df4 = df3 if cid_eff is None else df3[df3["cid"] == cid_eff]

    la_opts = unique_nonempty(df4["la"])
    la_eff = la if la in la_opts else (la_opts[0] if len(la_opts) == 1 else None)
    dff = df4 if la_eff is None else df4[df4["la"] == la_eff]

    if dff.empty:
        fig.update_layout(title=dict(text="No data available", x=0.5))
        return fig

    combos = dff.groupby(["rk", "rid", "ck", "cid", "la"]).size()
    if len(combos) > 1:
        missing = []
        if len(rk_opts) > 1 and rk_eff is None:
            missing.append("Resource Kind")
        if len(rid_opts) > 1 and rid_eff is None:
            missing.append("Resource ID")
        if len(ck_opts) > 1 and ck_eff is None:
            missing.append("Consumer Kind")
        if len(cid_opts) > 1 and cid_eff is None:
            missing.append("Consumer ID")
        if len(la_opts) > 1 and la_eff is None:
            missing.append("Late Attributes")

        fig.update_layout(
            title=dict(
                text="Please complete selections: " + (", ".join(missing) if missing else "more filters"),
                x=0.5,
                font=dict(size=12),
            )
        )
        return fig

    proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range and process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range and process_time_range.get("end") else None

    dff = dff.sort_values("timestamp")

    if proc_start and proc_end:
        dff = dff[(dff["timestamp"] >= proc_start) & (dff["timestamp"] <= proc_end)]

    if dff.empty:
        fig.update_layout(title=dict(text="No data during process active period", x=0.5))
        return fig

    y_min, y_max = dff["value"].min(), dff["value"].max()
    y_range = (y_max - y_min) if y_max != y_min else (abs(y_max) if y_max != 0 else 1)
    y_pad = 0.1 * y_range
    y_bottom, y_top = y_min - y_pad, y_max + y_pad

    colors = get_color_palette(100)
    idx_str = my_id.get("index", "0-0")
    color = colors[abs(hash(idx_str)) % len(colors)]

    rgba_fill = "rgba(136, 192, 208, 0.15)"
    if isinstance(color, str) and color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = (int(h[k:k+2], 16) for k in (0, 2, 4))
        rgba_fill = f"rgba({r}, {g}, {b}, 0.15)"

    unit = get_metric_unit(metric)
    y_axis_title = f"Value ({unit})" if unit else "Value"

    fig.add_trace(go.Scatter(
        x=dff["timestamp"],
        y=dff["value"],
        mode="lines+markers",
        name=metric,
        line=dict(color=color, width=2),
        marker=dict(color=color, size=6, symbol="circle"),
        fill="tozeroy",
        fillcolor=rgba_fill,
        hovertemplate=f"<b>{metric}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
    ))

    yaxis_config = dict(gridcolor="rgba(76, 86, 106, 0.2)", title=y_axis_title)

    if is_memory_metric(metric):
        tickvals, ticktext = get_bytes_tickvals_ticktext(y_bottom, y_top, num_ticks=5)
        yaxis_config["tickvals"] = tickvals
        yaxis_config["ticktext"] = ticktext
        yaxis_config["range"] = [y_bottom, y_top]
        yaxis_config["autorange"] = False

    fig.update_layout(
        height=350,
        title=dict(text=metric.replace("_", " ") + " (Process Active Period)", x=0.5, font=dict(size=14)),
        hovermode="closest",
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(gridcolor="rgba(76, 86, 106, 0.2)"),
        yaxis=yaxis_config,
        showlegend=False,
    )
    apply_figure_theme(fig, use_light_mode)
    return fig


# Zoom sync: capture relayoutData
@app.callback(
    Output("grid-shared-xrange-store", "data"),
    Input({"type": "grid-plot", "index": "0-0"}, "relayoutData"),
    Input({"type": "grid-plot", "index": "0-1"}, "relayoutData"),
    Input({"type": "grid-plot", "index": "1-0"}, "relayoutData"),
    Input({"type": "grid-plot", "index": "1-1"}, "relayoutData"),
    State("grid-shared-xrange-store", "data"),
    prevent_initial_call=True,
)
def sync_grid_plot_zoom(rd_00, rd_01, rd_10, rd_11, current_shared_range):
    """Sync zoom across all grid plots."""
    triggered = ctx.triggered_id
    if not triggered:
        return dash.no_update

    relayout_map = {"0-0": rd_00, "0-1": rd_01, "1-0": rd_10, "1-1": rd_11}

    if isinstance(triggered, dict):
        triggered_index = triggered.get("index")
    else:
        return dash.no_update

    relayout_data = relayout_map.get(triggered_index)
    if not relayout_data:
        return dash.no_update

    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        new_range = {"x0": relayout_data["xaxis.range[0]"], "x1": relayout_data["xaxis.range[1]"]}
        if (
            current_shared_range
            and not current_shared_range.get("autorange")
            and current_shared_range.get("x0") == new_range["x0"]
            and current_shared_range.get("x1") == new_range["x1"]
        ):
            return dash.no_update
        return new_range

    if "xaxis.autorange" in relayout_data and relayout_data["xaxis.autorange"]:
        if current_shared_range and current_shared_range.get("autorange"):
            return dash.no_update
        return {"autorange": True}

    return dash.no_update


# -- Zoom sync: apply shared x-range --

@app.callback(
    Output({"type": "grid-plot", "index": ALL}, "figure", allow_duplicate=True),
    Input("grid-shared-xrange-store", "data"),
    State({"type": "grid-plot", "index": ALL}, "figure"),
    prevent_initial_call=True,
)
def apply_shared_xrange_to_grid_plots(shared_range, current_figures):
    """Apply shared x-range to all grid plots when zoom/reset occurs."""
    if not shared_range or not current_figures:
        return [dash.no_update] * len(current_figures) if current_figures else dash.no_update

    is_autorange = shared_range.get("autorange", False)
    updated_figures = []

    for fig in current_figures:
        if not fig or not isinstance(fig, dict) or "layout" not in fig:
            updated_figures.append(dash.no_update)
            continue

        new_fig = copy.deepcopy(fig)
        if "xaxis" not in new_fig["layout"]:
            new_fig["layout"]["xaxis"] = {}

        if is_autorange:
            new_fig["layout"]["xaxis"]["autorange"] = True
            new_fig["layout"]["xaxis"].pop("range", None)
        else:
            new_fig["layout"]["xaxis"]["range"] = [shared_range["x0"], shared_range["x1"]]
            new_fig["layout"]["xaxis"]["autorange"] = False

        updated_figures.append(new_fig)

    return updated_figures


# CSV download for grid plots
@app.callback(
    Output({"type": "grid-download", "index": MATCH}, "data"),
    Input({"type": "grid-download-btn", "index": MATCH}, "n_clicks"),
    State({"type": "metric-dropdown", "index": MATCH}, "value"),
    State({"type": "resource-kind-dropdown", "index": MATCH}, "value"),
    State({"type": "resource-id-dropdown", "index": MATCH}, "value"),
    State({"type": "consumer-kind-dropdown", "index": MATCH}, "value"),
    State({"type": "consumer-id-dropdown", "index": MATCH}, "value"),
    State({"type": "late-attr-dropdown", "index": MATCH}, "value"),
    State("original-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def download_grid_csv(n_clicks, metric, rk, rid, ck, cid, la, original_df_data, process_time_range):
    """Generate and download CSV for a specific grid plot."""
    if not n_clicks or not original_df_data or not metric:
        return None

    df_original = df_from_store(original_df_data)
    _ensure_timestamp_datetime(df_original)

    dfm = df_original[df_original["metric"] == metric].copy()

    dfm["rk"] = dfm["resource_kind"].astype(str).replace("nan", "").str.strip()
    dfm["rid"] = dfm["resource_id"].astype(str).replace("nan", "").str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(str).replace("nan", "").str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(str).replace("nan", "").str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(str).replace("nan", "").str.strip()

    def norm_val(v):
        return str(v).strip() if v else ""

    if rk:
        dfm = dfm[dfm["rk"] == norm_val(rk)]
    if rid:
        dfm = dfm[dfm["rid"] == norm_val(rid)]
    if ck:
        dfm = dfm[dfm["ck"] == norm_val(ck)]
    if cid:
        dfm = dfm[dfm["cid"] == norm_val(cid)]
    if la:
        dfm = dfm[dfm["la"] == norm_val(la)]

    if process_time_range:
        proc_start = pd.to_datetime(process_time_range.get("start"))
        proc_end = pd.to_datetime(process_time_range.get("end"))
        if proc_start and proc_end:
            dfm = dfm[(dfm["timestamp"] >= proc_start) & (dfm["timestamp"] <= proc_end)]

    if dfm.empty:
        return None

    dfm = dfm.sort_values("timestamp")

    export_cols = ["timestamp", "metric", "value"]
    for orig_col in ["resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]:
        if orig_col in dfm.columns and dfm[orig_col].notna().any():
            export_cols.append(orig_col)

    df_export = dfm[export_cols].copy()

    safe_metric = "".join(c if c.isalnum() or c in "._-" else "_" for c in metric)
    filename = f"{safe_metric}_process_data.csv"

    return dcc.send_data_frame(df_export.to_csv, filename, index=False)
