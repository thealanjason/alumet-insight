"""Process-specific grid tab: helpers and callbacks for the 2x2 filter/plot grid."""

import copy
import json
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, MATCH, ALL

from frontend.app import app
from frontend.cache import df_from_store, is_cache_miss, ensure_timestamp_datetime
from frontend.theme import status_alert_class, apply_figure_theme
from frontend.layout import empty_process_specific_content, is_empty_tab_placeholder
from frontend.helpers import normalize_dropdown_value
from backend.formatting import get_bytes_tickvals_ticktext
from backend.metrics import get_metric_unit, is_memory_metric
from backend.transforms import filter_to_time_range
from backend.utils import safe_filename as safe_metric_filename
from backend.visualization.interactive import get_color_palette


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unique_nonempty(series: pd.Series) -> list[str]:
    """Get unique non-empty string values from a series, sorted."""
    str_series = series.astype(object).fillna("").astype(str).str.strip()
    mask = str_series != ""
    return sorted(str_series[mask].unique())


def normalize_filter_columns(dfm: pd.DataFrame) -> pd.DataFrame:
    """Add normalised filter columns rk, rid, ck, cid, la."""
    dfm = dfm.copy()
    for col, src in [
        ("rk", "resource_kind"),
        ("rid", "resource_id"),
        ("ck", "consumer_kind"),
        ("cid", "consumer_id"),
        ("la", "__late_attributes"),
    ]:
        dfm[col] = dfm[src].astype(object).fillna("").astype(str).str.strip()
    return dfm


def cascade_filter_options(
    dfm: pd.DataFrame,
    rk: Optional[str],
    rid: Optional[str],
    ck: Optional[str],
    cid: Optional[str],
    la: Optional[str],
    triggered_id: Optional[str] = None,
) -> dict:
    """Compute cascaded filter options from resources/consumers/late attributes."""
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
    if triggered_id in ("resource-kind-dropdown", "resource-id-dropdown",
                        "consumer-kind-dropdown", "consumer-id-dropdown"):
        la_eff = None
    else:
        la_eff = la if la in la_opts else None

    return {
        "rk": {"options": rk_opts, "effective": rk_eff},
        "rid": {"options": rid_opts, "effective": rid_eff},
        "ck": {"options": ck_opts, "effective": ck_eff},
        "cid": {"options": cid_opts, "effective": cid_eff},
        "la": {"options": la_opts, "effective": la_eff},
    }


def filter_single_series(
    dfm: pd.DataFrame,
    rk: Optional[str],
    rid: Optional[str],
    ck: Optional[str],
    cid: Optional[str],
    la: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    """Apply cascading filters and return (filtered_df, cascade_info)."""
    cascade = cascade_filter_options(dfm, rk, rid, ck, cid, la)
    rk_eff = cascade["rk"]["effective"]
    rid_eff = cascade["rid"]["effective"]
    ck_eff = cascade["ck"]["effective"]
    cid_eff = cascade["cid"]["effective"]
    la_eff = cascade["la"]["effective"]

    df = dfm
    if rk_eff is not None:
        df = df[df["rk"] == rk_eff]
    if rid_eff is not None:
        df = df[df["rid"] == rid_eff]
    if ck_eff is not None:
        df = df[df["ck"] == ck_eff]
    if cid_eff is not None:
        df = df[df["cid"] == cid_eff]
    if la_eff is not None:
        df = df[df["la"] == la_eff]
    return df, cascade


def prepare_download_df(
    df_original: pd.DataFrame,
    metric: str,
    rk: Optional[str],
    rid: Optional[str],
    ck: Optional[str],
    cid: Optional[str],
    la: Optional[str],
    proc_start: Optional[pd.Timestamp] = None,
    proc_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Filter and prepare a DataFrame suitable for CSV download."""
    dfm = normalize_filter_columns(df_original[df_original["metric"] == metric])

    if rk:
        dfm = dfm[dfm["rk"] == str(rk).strip()]
    if rid:
        dfm = dfm[dfm["rid"] == str(rid).strip()]
    if ck:
        dfm = dfm[dfm["ck"] == str(ck).strip()]
    if cid:
        dfm = dfm[dfm["cid"] == str(cid).strip()]
    if la:
        dfm = dfm[dfm["la"] == str(la).strip()]

    dfm = filter_to_time_range(dfm, proc_start, proc_end, require_bounds=False)

    if dfm.empty:
        return dfm

    dfm = dfm.sort_values("timestamp")

    export_cols = ["timestamp", "metric", "value"]
    for orig_col in ["resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]:
        if orig_col in dfm.columns and dfm[orig_col].notna().any():
            export_cols.append(orig_col)

    return dfm[export_cols].copy()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------



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
        return empty_process_specific_content()

    if triggered_id == "results-tabs":
        if tab_value != "process-specific-tab":
            return dash.no_update
        if current_children and not is_empty_tab_placeholder(current_children):
            return dash.no_update

    if not original_df_data or not process_time_range:
        return empty_process_specific_content()

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
            "Session data expired (server was restarted). Please load data again.",
            color="danger",
            className=status_alert_class("danger"),
        )
    ensure_timestamp_datetime(df_original)

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
    dfm = normalize_filter_columns(df[df["metric"] == metric])

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

    cascade = cascade_filter_options(dfm, rk, rid, ck, cid, la, triggered_id=triggered_id)

    def _style_and_opts(key):
        opts = cascade[key]["options"]
        eff = cascade[key]["effective"]
        style = show if len(opts) > 1 else hide
        options = [{"label": v, "value": v} for v in opts]
        return style, options, eff

    rk_style, rk_options, rk_eff = _style_and_opts("rk")
    rid_style, rid_options, rid_eff = _style_and_opts("rid")
    ck_style, ck_options, ck_eff = _style_and_opts("ck")
    cid_style, cid_options, cid_eff = _style_and_opts("cid")
    la_style, la_options, la_eff = _style_and_opts("la")

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
    ensure_timestamp_datetime(df)
    dfm = normalize_filter_columns(df[df["metric"] == metric])

    rk = normalize_dropdown_value(rk)
    rid = normalize_dropdown_value(rid)
    ck = normalize_dropdown_value(ck)
    cid = normalize_dropdown_value(cid)
    la = normalize_dropdown_value(la)

    dff, cascade = filter_single_series(dfm, rk, rid, ck, cid, la)

    if dff.empty:
        fig.update_layout(title=dict(text="No data available", x=0.5))
        return fig

    combos = dff.groupby(["rk", "rid", "ck", "cid", "la"]).size()
    if len(combos) > 1:
        label_map = {"rk": "Resource Kind", "rid": "Resource ID",
                     "ck": "Consumer Kind", "cid": "Consumer ID", "la": "Late Attributes"}
        missing = [
            label_map[k] for k in ("rk", "rid", "ck", "cid", "la")
            if len(cascade[k]["options"]) > 1 and cascade[k]["effective"] is None
        ]

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

    dff = filter_to_time_range(dff, proc_start, proc_end, require_bounds=False).sort_values("timestamp")

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
    ensure_timestamp_datetime(df_original)

    proc_start = pd.to_datetime(process_time_range.get("start")) if process_time_range and process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range.get("end")) if process_time_range and process_time_range.get("end") else None

    df_export = prepare_download_df(df_original, metric, rk, rid, ck, cid, la, proc_start, proc_end)

    if df_export.empty:
        return None

    filename = f"{safe_metric_filename(metric)}_process_data.csv"
    return dcc.send_data_frame(df_export.to_csv, filename, index=False)
