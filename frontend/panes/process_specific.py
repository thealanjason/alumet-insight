"""Process-specific grid tab: helpers and callbacks for the 2x2 filter/plot grid."""

import copy
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, MATCH, ALL

from frontend.app import app
from frontend.cache import df_from_store, is_cache_miss
from frontend.helpers import normalize_dropdown_value, triggered_component_type, parse_process_time_range_store, ensure_timestamp_datetime
from frontend.style import (
    CARD_STYLE,
    COMPACT_DROPDOWN_STYLE,
    DROPDOWN_STYLE,
    FILTER_KEYS,
    FILTER_LABEL_MAP,
    FILTER_SPECS,
    GRID_DATA_MARGIN,
    GRID_GRAPH_CONFIG,
    GRID_PLACEHOLDER_MARGIN,
    GRID_SIZE,
    STYLE_FILTER_SLOT_VISIBLE,
    STYLE_HIDDEN,
    STYLE_VISIBLE,
    apply_figure_theme,
    status_alert_class,
)
from frontend.layout import empty_process_specific_content, is_empty_tab_placeholder
from frontend.helpers import normalize_dropdown_value, triggered_component_type
from backend.formatting import get_bytes_tickvals_ticktext
from backend.metrics import get_metric_unit, is_memory_metric
from backend.transforms import filter_to_time_range
from backend.utils import safe_filename
from frontend.figures import get_color_palette


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def empty_filter_callback_response() -> tuple:
    """Default MATCH callback payload when metric data is unavailable."""
    slot_defaults = []
    for _ in FILTER_KEYS:
        slot_defaults.extend([STYLE_HIDDEN, [], None])
    return (STYLE_HIDDEN, *slot_defaults)

def unique_nonempty(series: pd.Series) -> list[str]:
    """Get unique non-empty string values from a series, sorted."""
    str_series = series.astype(object).fillna("").astype(str).str.strip()
    mask = str_series != ""
    return sorted(str_series[mask].unique())


def normalize_filter_columns(dfm: pd.DataFrame) -> pd.DataFrame:
    """Add normalized filter columns rk, rid, ck, cid, la."""
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

def build_filter_callback_response(cascade: dict) -> tuple:
    """Build MATCH callback outputs for one grid cell's filter controls."""
    slot_outputs = []
    for key in FILTER_KEYS:
        opts = cascade[key]["options"]
        eff = cascade[key]["effective"]
        style = STYLE_FILTER_SLOT_VISIBLE if len(opts) > 1 else STYLE_HIDDEN
        options = [{"label": value, "value": value} for value in opts]
        slot_outputs.extend([style, options, eff])

    any_visible = any(len(cascade[key]["options"]) > 1 for key in FILTER_KEYS)
    filters_row_style = STYLE_VISIBLE if any_visible else STYLE_HIDDEN
    return (filters_row_style, *slot_outputs)

def grid_message_figure(fig: go.Figure, title: str, use_light_mode: bool) -> go.Figure:
    """Compact placeholder figure for empty, incomplete, or invalid grid states."""
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=11)),
        margin=GRID_PLACEHOLDER_MARGIN,
        autosize=True,
    )
    apply_figure_theme(fig, use_light_mode)
    return fig

def _filter_slot(cell_index: str, label: str, dropdown_type: str, container_type: str) -> html.Div:
    """One inline filter control in the process-specific toolbar."""
    return html.Div(
        [
            html.Label(label, className="process-grid-filter-label"),
            dcc.Dropdown(
                id={"type": dropdown_type, "index": cell_index},
                options=[],
                value=None,
                placeholder="-",
                style=COMPACT_DROPDOWN_STYLE,
                className="dark-dropdown compact-dropdown",
                clearable=False,
            ),
        ],
        id={"type": container_type, "index": cell_index},
        className="process-grid-filter-slot",
        style=STYLE_HIDDEN,
    )

def _build_grid_cell(i: int, j: int, unique_metrics: list[str]) -> html.Div:
    """Build one viewport-fitted cell for the 2x2 process-specific grid."""
    cell_index = f"{i}-{j}"

    return html.Div(
        dbc.Card(
            [
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Metric:", className="process-grid-metric-label"),
                                        dcc.Dropdown(
                                            id={"type": "metric-dropdown", "index": cell_index},
                                            options=[{"label": metric, "value": metric} for metric in unique_metrics],
                                            placeholder="Select metric",
                                            style=DROPDOWN_STYLE,
                                            className="dark-dropdown process-grid-metric-dropdown",
                                            clearable=True,
                                        ),
                                    ],
                                    className="process-grid-metric-group",
                                ),
                                html.Div(
                                    [
                                        _filter_slot(cell_index, label, dropdown_type, container_type)
                                        for _, label, dropdown_type, container_type in FILTER_SPECS
                                    ],
                                    id={"type": "filters-row", "index": cell_index},
                                    className="process-grid-filters",
                                    style=STYLE_HIDDEN,
                                ),
                            ],
                            className="process-grid-toolbar",
                        ),
                        html.Div(
                            dcc.Graph(
                                id={"type": "grid-plot", "index": cell_index},
                                style={"height": "100%", "width": "100%"},
                                className="grid-plot-graph",
                                config=GRID_GRAPH_CONFIG,
                            ),
                            className="process-grid-plot-area",
                        ),
                        html.Div(
                            [
                                dbc.Button(
                                    "\U0001f4e5 Download CSV",
                                    id={"type": "grid-download-btn", "index": cell_index},
                                    n_clicks=0,
                                    color="primary",
                                    size="sm",
                                    style={"fontSize": "0.75rem"},
                                ),
                                dcc.Download(id={"type": "grid-download", "index": cell_index}),
                            ],
                            className="process-grid-download",
                        ),
                    ],
                    className="process-grid-cell-body",
                    style={"backgroundColor": "var(--app-card-bg)"},
                ),
            ],
            className="process-grid-inner-card",
            style={"height": "100%", **CARD_STYLE},
        ),
        className="process-grid-cell",
    )

def build_process_grid_card(unique_metrics: list[str]) -> dbc.Card:
    """Build the viewport-fitted 2x2 process-specific comparison card."""
    grid_cells = [
        _build_grid_cell(i, j, unique_metrics)
        for i in range(GRID_SIZE)
        for j in range(GRID_SIZE)
    ]
    return dbc.Card(
        [
            dbc.CardBody(
                html.Div(grid_cells, className="process-grid-viewport"),
                className="viewport-card-body process-grid-card-body",
                style={"backgroundColor": "var(--app-card-bg)"},
            ),
        ],
        className="viewport-card process-grid-card",
        style=CARD_STYLE,
    )


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

    proc_start, proc_end = parse_process_time_range_store(process_time_range)

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
    return build_process_grid_card(unique_metrics)


# MATCH callback: update filter dropdowns
@app.callback(
    Output({"type": "filters-row", "index": MATCH}, "style"),
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
    if not original_df_data or not metric:
        return empty_filter_callback_response()

    df = df_from_store(original_df_data)
    dfm = normalize_filter_columns(df[df["metric"] == metric])

    rk = normalize_dropdown_value(rk)
    rid = normalize_dropdown_value(rid)
    ck = normalize_dropdown_value(ck)
    cid = normalize_dropdown_value(cid)
    la = normalize_dropdown_value(la)

    triggered_id = None
    if ctx.triggered:
        triggered_id = triggered_component_type(ctx.triggered[0].get("prop_id", ""))

    cascade = cascade_filter_options(dfm, rk, rid, ck, cid, la, triggered_id=triggered_id)
    return build_filter_callback_response(cascade)


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
        return grid_message_figure(fig, "Select a metric", use_light_mode)

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
        return grid_message_figure(fig, "No data available", use_light_mode)

    combos = dff.groupby(["rk", "rid", "ck", "cid", "la"]).size()
    if len(combos) > 1:
        missing = [
            FILTER_LABEL_MAP[key]
            for key in FILTER_KEYS
            if len(cascade[key]["options"]) > 1 and cascade[key]["effective"] is None
        ]

        message = "Please complete selections: " + (", ".join(missing) if missing else "more filters")
        return grid_message_figure(fig, message, use_light_mode)

    proc_start, proc_end = parse_process_time_range_store(process_time_range)

    dff = filter_to_time_range(dff, proc_start, proc_end, require_bounds=False).sort_values("timestamp")

    if dff.empty:
        return grid_message_figure(fig, "No data during process active period", use_light_mode)

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
        hovermode="closest",
        margin=GRID_DATA_MARGIN,
        xaxis=dict(gridcolor="rgba(76, 86, 106, 0.2)"),
        yaxis=yaxis_config,
        showlegend=False,
        autosize=True,
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

    proc_start, proc_end = parse_process_time_range_store(process_time_range)

    df_export = prepare_download_df(df_original, metric, rk, rid, ck, cid, la, proc_start, proc_end)

    if df_export.empty:
        return None

    filename = f"{safe_filename(metric)}_process_data.csv"
    return dcc.send_data_frame(df_export.to_csv, filename, index=False)
