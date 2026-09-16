"""Process-specific grid tab: helpers and callbacks for the 2x2 filter/plot grid."""

import base64
import copy
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html

from backend.counterdiff import export_observed_measurements
from backend.formatting import format_metric_choice_label, get_bytes_tickvals_ticktext
from backend.metrics import (
    attach_unit_column,
    derived_base_metrics,
    get_metric_unit,
    is_memory_metric,
    is_spike_metric,
)
from backend.transforms import _padded_range, align_xrange_tz, filter_to_time_range
from backend.utils import safe_filename
from frontend.app import app
from frontend.cache import df_from_store, is_cache_miss
from frontend.figures import (
    build_metric_trace_configs,
    color_for_metric,
    relayout_requests_reset,
    restore_axis_defaults,
)
from frontend.helpers import (
    ensure_timestamp_datetime,
    normalize_dropdown_value,
    parse_process_time_range_store,
    triggered_component_type,
)
from frontend.layout import empty_process_specific_content, is_empty_tab_placeholder
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
    set_plotly_rgba,
    status_alert_class,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series_metric_id(df_series: pd.DataFrame, metric: str) -> str:
    """Prefer the resolved full metric_id when a filtered series is unique."""
    if "metric_id" in df_series.columns and not df_series.empty:
        ids = df_series["metric_id"].dropna().astype(str).unique().tolist()
        if len(ids) == 1:
            return ids[0]
    return metric


def _metric_column(df: pd.DataFrame) -> str:
    """Return the column used for process-specific metric selection."""
    if "base_metric" in df.columns:
        return "base_metric"
    return "metric"


def grid_trace_configs(
    metric: str,
    df_series: pd.DataFrame,
    color: str,
    fillcolor: str,
) -> list[dict]:
    """Build process-grid traces using the shared metric rendering policy."""
    series_metric_id = _series_metric_id(df_series, metric)
    return build_metric_trace_configs(
        df_series,
        series_metric_id,
        color=color,
        name=metric,
        show_default_markers=True,
        fill_to_zero=not is_spike_metric(series_metric_id),
        fillcolor=fillcolor,
    )


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


def _pick_filter_value(value: Optional[str], options: list[str], *, reset: bool = False) -> Optional[str]:
    """Keep a user choice if it is still valid; otherwise auto-select a unique option."""
    if reset:
        return None
    if value in options:
        return value
    if len(options) == 1:
        return options[0]
    return None


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


FILTER_DROPDOWN_TO_KEY = {dropdown: key for key, _, dropdown, _ in FILTER_SPECS}
FILTER_KIND_CHILD = {"rk": "rid", "ck": "cid"}
# Option lists stay kind-scoped so sibling IDs remain switchable. Kind and Attr
# lists are global; R.ID follows R.Kind and C.ID follows R.Kind + C.Kind.
FILTER_OPTION_PARENTS = {
    "rk": (),
    "rid": ("rk",),
    "ck": (),
    "cid": ("rk", "ck"),
    "la": (),
}


def cascade_filter_options(
    dfm: pd.DataFrame,
    rk: Optional[str],
    rid: Optional[str],
    ck: Optional[str],
    cid: Optional[str],
    la: Optional[str],
    triggered_id: Optional[str] = None,
) -> dict:
    """Faceted filters: R.Kind, R.ID, C.Kind, C.ID, and Attr are siblings.

    Dropdowns stay visible when that dimension has more than one value on the
    metric. Option lists are not filtered by sibling IDs, so a selected GPU/PID
    pair does not lock the other dropdown. An incompatible sibling is cleared
    (or uniquely auto-filled) instead.
    """
    user = {"rk": rk, "rid": rid, "ck": ck, "cid": cid, "la": la}
    opts_all = {key: unique_nonempty(dfm[key]) for key in FILTER_KEYS}
    triggered_key = FILTER_DROPDOWN_TO_KEY.get(triggered_id) if triggered_id else None
    reset_keys: set[str] = set()
    if triggered_key in FILTER_KIND_CHILD:
        reset_keys.add(FILTER_KIND_CHILD[triggered_key])

    eff = {
        key: _pick_filter_value(user[key], opts_all[key], reset=key in reset_keys)
        for key in FILTER_KEYS
    }

    remaining = dfm
    if triggered_key and eff[triggered_key] is not None:
        remaining = remaining[remaining[triggered_key] == eff[triggered_key]]

    for key in FILTER_KEYS:
        if key == triggered_key or key in reset_keys:
            continue
        value = eff[key]
        if value is not None and bool((remaining[key] == value).any()):
            remaining = remaining[remaining[key] == value]
        elif value is not None:
            eff[key] = None

    for key in FILTER_KEYS:
        if key in reset_keys or eff[key] is not None:
            continue
        choices = unique_nonempty(remaining[key])
        if len(choices) == 1:
            eff[key] = choices[0]
            remaining = remaining[remaining[key] == choices[0]]

    def options_for(key: str) -> list[str]:
        mask = pd.Series(True, index=dfm.index)
        for parent in FILTER_OPTION_PARENTS[key]:
            parent_eff = eff[parent]
            if parent_eff is None:
                continue
            mask &= dfm[parent] == parent_eff
        return unique_nonempty(dfm.loc[mask, key])

    return {
        key: {
            "options": options_for(key),
            "effective": eff[key],
            "show": len(opts_all[key]) > 1,
        }
        for key in FILTER_KEYS
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
    df = dfm
    for key in FILTER_KEYS:
        effective = cascade[key]["effective"]
        if effective is not None:
            df = df[df[key] == effective]
    return df, cascade


def prepare_download_df(
    df_processed: pd.DataFrame,
    metric: str,
    rk: Optional[str],
    rid: Optional[str],
    ck: Optional[str],
    cid: Optional[str],
    la: Optional[str],
    proc_start: Optional[pd.Timestamp] = None,
    proc_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Filter and prepare observed-only CSV rows from processed data."""
    metric_col = _metric_column(df_processed)
    dfm = normalize_filter_columns(df_processed[df_processed[metric_col] == metric])

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

    dfm = export_observed_measurements(dfm.sort_values("timestamp"))
    if dfm.empty:
        return dfm

    if "metric" not in dfm.columns and "base_metric" in dfm.columns:
        dfm = dfm.copy()
        dfm["metric"] = dfm["base_metric"]

    export_cols = [c for c in ("timestamp", "metric", "value") if c in dfm.columns]
    for orig_col in ["resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]:
        if orig_col in dfm.columns and dfm[orig_col].notna().any():
            export_cols.append(orig_col)

    return attach_unit_column(dfm[export_cols].copy())


def build_filter_callback_response(cascade: dict) -> tuple:
    """Build MATCH callback outputs for one grid cell's filter controls."""
    slot_outputs = []
    for key in FILTER_KEYS:
        opts = cascade[key]["options"]
        eff = cascade[key]["effective"]
        show = cascade[key].get("show", len(opts) > 1)
        style = STYLE_FILTER_SLOT_VISIBLE if show else STYLE_HIDDEN
        options = [{"label": value, "value": value} for value in opts]
        slot_outputs.extend([style, options, eff])

    any_visible = any(cascade[key].get("show", len(cascade[key]["options"]) > 1) for key in FILTER_KEYS)
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


def decode_plotly_array(value: Any) -> list:
    """Unpack a Plotly/Dash typed array (`{dtype, bdata}`) or a plain list."""
    if value is None:
        return []
    if isinstance(value, dict):
        bdata = value.get("bdata")
        if not bdata:
            return []
        try:
            raw = base64.b64decode(bdata)
            return np.frombuffer(raw, dtype=str(value.get("dtype") or "f8")).tolist()
        except (ValueError, TypeError):
            return []
    if isinstance(value, (str, bytes)):
        return []
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        return value.tolist()
    try:
        return list(value)
    except TypeError:
        return []


def as_datetime_index(values: list) -> pd.DatetimeIndex:
    """Parse trace X values, including numeric Plotly epoch encodings."""
    if not values:
        return pd.DatetimeIndex([])
    first = values[0]
    if isinstance(first, (int, float, np.integer, np.floating)) and not isinstance(first, bool):
        mag = abs(float(first))
        if mag >= 1e17:
            unit = "ns"
        elif mag >= 1e14:
            unit = "us"
        elif mag >= 1e11:
            unit = "ms"
        elif mag >= 1e9:
            unit = "s"
        else:
            unit = None
        if unit:
            return pd.to_datetime(values, unit=unit, utc=True)
    return pd.to_datetime(values, utc=True)


def visible_trace_y_values(fig: dict, x0, x1) -> list[float]:
    """Return Y values whose timestamps fall inside the shared X window."""
    values: list[float] = []
    x_min = x_max = None
    for trace in fig.get("data") or []:
        xs = decode_plotly_array(trace.get("x"))
        ys = decode_plotly_array(trace.get("y"))
        if not xs or not ys:
            continue
        x_index = as_datetime_index(xs)
        if x_index.empty:
            continue
        if x_min is None:
            x_min, x_max = align_xrange_tz(pd.to_datetime(x0), pd.to_datetime(x1), x_index.tz)
        for x_ts, y_val in zip(x_index, ys):
            if y_val is None or pd.isna(y_val):
                continue
            if x_min <= x_ts <= x_max:
                values.append(float(y_val))
    return values


def apply_visible_yaxis_range(fig: dict, x0, x1) -> None:
    """Fit one grid plot's Y-axis to the points inside the shared X window."""
    layout = fig.get("layout")
    if not isinstance(layout, dict):
        return
    if "yaxis" not in layout:
        layout["yaxis"] = {}

    values = visible_trace_y_values(fig, x0, x1)
    if not values:
        return

    is_memory = bool((layout.get("meta") or {}).get("is_memory"))
    y_bottom, y_top = _padded_range(min(values), max(values), clamp_zero=is_memory)
    layout["yaxis"]["range"] = [y_bottom, y_top]
    layout["yaxis"]["autorange"] = False
    if is_memory:
        tickvals, ticktext = get_bytes_tickvals_ticktext(y_bottom, y_top, num_ticks=5)
        layout["yaxis"]["tickvals"] = list(tickvals)
        layout["yaxis"]["ticktext"] = list(ticktext)
    else:
        layout["yaxis"].pop("tickvals", None)
        layout["yaxis"].pop("ticktext", None)

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


def _build_grid_cell(i: int, j: int, unique_metrics: list[str], derived_metrics: Optional[set[str]] = None) -> html.Div:
    """Build one viewport-fitted cell for the 2x2 process-specific grid."""
    cell_index = f"{i}-{j}"
    derived_metrics = derived_metrics or set()
    metric_options = [
        {"label": format_metric_choice_label(metric, derived=metric in derived_metrics), "value": metric}
        for metric in unique_metrics
    ]

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
                                            options=metric_options,
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


def build_process_grid_card(unique_metrics: list[str], derived_metrics: Optional[set[str]] = None) -> dbc.Card:
    """Build the viewport-fitted 2x2 process-specific comparison card."""
    grid_cells = [
        _build_grid_cell(i, j, unique_metrics, derived_metrics)
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
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
    State("process-specific-content", "children"),
)
def build_process_specific_tab(tab_value, processed_df_data, process_time_range, current_children):
    triggered_id = ctx.triggered_id
    is_data_trigger = triggered_id in ("processed-df-store", "process-time-range-store")

    if is_data_trigger and tab_value != "process-specific-tab":
        return empty_process_specific_content()

    if triggered_id == "results-tabs":
        if tab_value != "process-specific-tab":
            return dash.no_update
        if current_children and not is_empty_tab_placeholder(current_children):
            return dash.no_update

    if not processed_df_data or not process_time_range:
        return empty_process_specific_content()

    proc_start, proc_end = parse_process_time_range_store(process_time_range)

    if proc_start is None or proc_end is None:
        return dbc.Alert(
            "Process time range not available.",
            color="warning",
            className=status_alert_class("warning"),
        )

    df_processed = df_from_store(processed_df_data)
    if is_cache_miss(df_processed):
        return dbc.Alert(
            "Session data expired (server was restarted). Please load data again.",
            color="danger",
            className=status_alert_class("danger"),
        )
    ensure_timestamp_datetime(df_processed)

    metric_col = _metric_column(df_processed)
    unique_metrics = sorted(df_processed[metric_col].dropna().astype(str).unique().tolist())
    return build_process_grid_card(unique_metrics, derived_base_metrics(df_processed))


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
    State("processed-df-store", "data"),
    prevent_initial_call=True,
)
def update_filters_match(metric, rk, rid, ck, cid, la, processed_df_data):
    """Update filter dropdowns for a single plot using MATCH."""
    if not processed_df_data or not metric:
        return empty_filter_callback_response()

    df = df_from_store(processed_df_data)
    metric_col = _metric_column(df)
    dfm = normalize_filter_columns(df[df[metric_col] == metric])

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
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    State({"type": "metric-dropdown", "index": MATCH}, "id"),
)
def update_grid_plot_match(metric, rk, rid, ck, cid, la, use_light_mode, processed_df_data, process_time_range, my_id):
    """Update a single grid plot figure using MATCH."""
    fig = go.Figure()
    apply_figure_theme(fig, use_light_mode)

    if not processed_df_data or not metric:
        return grid_message_figure(fig, "Select a metric", use_light_mode)

    df = df_from_store(processed_df_data)
    ensure_timestamp_datetime(df)
    metric_col = _metric_column(df)
    dfm = normalize_filter_columns(df[df[metric_col] == metric])

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
            if cascade[key].get("show", len(cascade[key]["options"]) > 1)
            and cascade[key]["effective"] is None
        ]

        message = "Please complete selections: " + (", ".join(missing) if missing else "more filters")
        return grid_message_figure(fig, message, use_light_mode)

    proc_start, proc_end = parse_process_time_range_store(process_time_range)

    dff = filter_to_time_range(dff, proc_start, proc_end, require_bounds=False).sort_values("timestamp")

    if dff.empty:
        return grid_message_figure(fig, "No data during process active period", use_light_mode)

    series_metric_id = _series_metric_id(dff, metric)
    y_min, y_max = float(dff["value"].min()), float(dff["value"].max())
    if is_spike_metric(series_metric_id):
        y_min = min(0.0, y_min)
        y_max = max(0.0, y_max)
    is_memory = is_memory_metric(metric)
    y_bottom, y_top = _padded_range(y_min, y_max, clamp_zero=is_memory)

    metric_col = _metric_column(df)
    metric_order = sorted(df[metric_col].dropna().astype(str).unique().tolist()) if metric_col in df.columns else []
    color = color_for_metric(str(metric), use_light_mode, metric_order)
    rgba_fill = set_plotly_rgba(color)

    unit = get_metric_unit(metric)
    y_axis_title = f"Value ({unit})" if unit else "Value"

    for trace_config in grid_trace_configs(metric, dff, color, rgba_fill):
        fig.add_trace(go.Scatter(**trace_config))

    yaxis_config = dict(
        gridcolor="rgba(76, 86, 106, 0.2)",
        title=y_axis_title,
        range=[y_bottom, y_top],
        autorange=False,
    )
    yaxis_defaults = {
        "range": [float(y_bottom), float(y_top)],
        "autorange": False,
    }

    if is_memory:
        tickvals, ticktext = get_bytes_tickvals_ticktext(y_bottom, y_top, num_ticks=5)
        yaxis_config["tickvals"] = tickvals
        yaxis_config["ticktext"] = ticktext
        yaxis_defaults["tickvals"] = list(tickvals)
        yaxis_defaults["ticktext"] = list(ticktext)

    default_x_start = proc_start if proc_start is not None else dff["timestamp"].min()
    default_x_end = proc_end if proc_end is not None else dff["timestamp"].max()
    xaxis_defaults = {
        "range": [
            pd.Timestamp(default_x_start).isoformat(),
            pd.Timestamp(default_x_end).isoformat(),
        ],
        "autorange": False,
    }

    fig.update_layout(
        hovermode="closest",
        margin=GRID_DATA_MARGIN,
        xaxis=dict(
            gridcolor="rgba(76, 86, 106, 0.2)",
            range=xaxis_defaults["range"],
            autorange=False,
        ),
        yaxis=yaxis_config,
        meta={"axis_defaults": {"xaxis": xaxis_defaults, "yaxis": yaxis_defaults}, "is_memory": is_memory},
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

    revision = int((current_shared_range or {}).get("revision", 0)) + 1

    if relayout_requests_reset(relayout_data):
        return {"mode": "reset", "revision": revision}

    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        new_range = {
            "mode": "zoom",
            "x0": relayout_data["xaxis.range[0]"],
            "x1": relayout_data["xaxis.range[1]"],
            "revision": revision,
        }
        if (
            current_shared_range
            and current_shared_range.get("mode") == "zoom"
            and current_shared_range.get("x0") == new_range["x0"]
            and current_shared_range.get("x1") == new_range["x1"]
        ):
            return dash.no_update
        return new_range

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

    is_reset = shared_range.get("mode") == "reset" or shared_range.get("autorange", False)
    updated_figures = []

    for fig in current_figures:
        if not fig or not isinstance(fig, dict) or "layout" not in fig:
            updated_figures.append(dash.no_update)
            continue

        new_fig = copy.deepcopy(fig)
        if "xaxis" not in new_fig["layout"]:
            new_fig["layout"]["xaxis"] = {}

        if is_reset:
            defaults = new_fig["layout"].get("meta", {}).get("axis_defaults", {})
            xaxis_defaults = defaults.get("xaxis")
            if xaxis_defaults:
                restore_axis_defaults(new_fig["layout"]["xaxis"], xaxis_defaults)
            else:
                new_fig["layout"]["xaxis"]["autorange"] = True
                new_fig["layout"]["xaxis"].pop("range", None)

            yaxis_defaults = defaults.get("yaxis")
            if yaxis_defaults and "yaxis" in new_fig["layout"]:
                restore_axis_defaults(new_fig["layout"]["yaxis"], yaxis_defaults)
        else:
            new_fig["layout"]["xaxis"]["range"] = [shared_range["x0"], shared_range["x1"]]
            new_fig["layout"]["xaxis"]["autorange"] = False
            apply_visible_yaxis_range(new_fig, shared_range["x0"], shared_range["x1"])

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
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def download_grid_csv(n_clicks, metric, rk, rid, ck, cid, la, processed_df_data, process_time_range):
    """Generate and download CSV for a specific grid plot."""
    if not n_clicks or not processed_df_data or not metric:
        return None

    df_processed = df_from_store(processed_df_data)
    ensure_timestamp_datetime(df_processed)

    proc_start, proc_end = parse_process_time_range_store(process_time_range)

    df_export = prepare_download_df(df_processed, metric, rk, rid, ck, cid, la, proc_start, proc_end)

    if df_export.empty:
        return None

    filename = f"{safe_filename(metric)}_process_data.csv"
    return dcc.send_data_frame(df_export.to_csv, filename, index=False)
