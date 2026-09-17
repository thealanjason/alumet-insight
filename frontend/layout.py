"""Dash application layout helpers."""

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from frontend.style import status_alert_class, COLOR_PRIMARY, COLOR_DANGER, COLOR_LOADING, empty_theme_figure

LOAD_SOURCE_UPLOAD = "upload"
LOAD_SOURCE_PATH = "path"
SOURCE_FILE_HINT = ".csv, .log, and .toml"
TAB_PANEL_ACTIVE = "tab-panel-scroll tab-panel-active"
TAB_PANEL_IDLE = "tab-panel-scroll tab-panel-idle"
PLOT_PREPARING_HIDDEN = {"display": "none"}
PLOT_PREPARING_VISIBLE = {"display": "flex"}


def plot_preparing_overlay(overlay_id, message: str = "Updating this plot…"):
    """Cover a plot area while its rebuild callback is in flight.

    The live figure stays mounted underneath so zoom/sync state is not replaced
    by a dummy chart.
    """
    return html.Div(
        id=overlay_id,
        className="plot-preparing-overlay",
        style=PLOT_PREPARING_HIDDEN,
        children=html.Div(
            [
                html.Div(className="tab-preparing-spinner"),
                html.Span(message, className="tab-preparing-label"),
            ],
            className="tab-preparing-inner",
        ),
    )


def upload_prompt_children():
    """Compact drop-zone contents to keep the same height as the server-path input."""
    return html.Div(
        [
            html.Div(html.I(className="bi bi-folder2-open"), className="sidebar-upload-icon"),
            html.Div(
                [
                    "Drop / Browse folder with ",
                    html.Span(SOURCE_FILE_HINT),
                ],
                className="sidebar-upload-text",
            ),
        ],
        className="sidebar-upload-inner",
    )


def upload_selected_children(experiment_name: str):
    """Replace the drop zone with the uploaded folder name only."""
    return html.Div(
        [
            html.Div(html.I(className="bi bi-folder2-open"), className="sidebar-upload-icon"),
            html.Div(experiment_name, className="sidebar-upload-name"),
        ],
        className="sidebar-upload-inner sidebar-upload-inner-selected",
        title="Click to replace folder",
    )


def empty_time_series_content():
    """Keep time-series callback targets mounted before data is loaded."""
    return html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(id="metric-category-dropdown", options=[], value=None),
                    html.Div(id="cpu-core-selector"),
                    dcc.Dropdown(id="cpu-core-dropdown", options=[], value=None),
                    html.Div(
                        dcc.Checklist(
                            id="shared-yaxis-toggle",
                            options=[{"label": " Share Y-axis range across subplots", "value": "shared"}],
                            value=[],
                        ),
                        id="yaxis-options-container",
                    ),
                    html.Div(
                        [
                            html.Div(id="timeseries-plot-container"),
                            plot_preparing_overlay("timeseries-plot-preparing"),
                        ],
                        className="plot-area-with-preparing",
                    ),
                    html.Div(id="timeseries-process-legend", style={"display": "none"}),
                ],
                style={"display": "none"},
            ),
            dbc.Alert(
                "No data available. Please load data first.",
                color="warning",
                className=status_alert_class("warning"),
            ),
        ],
        className="empty-time-series-content",
    )


def empty_process_specific_content(message: str = "No data available. Please load data first."):
    """Keep process-specific callback targets mounted before data is loaded."""
    return html.Div(
        dbc.Alert(
            message,
            color="warning",
            className=status_alert_class("warning"),
        ),
        className="empty-process-specific-content",
    )


def empty_comparative_content(message: str = "No data available. Please load data first."):
    """Keep comparative callback targets mounted before data is loaded."""
    return html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(id="ps-xmetric-dropdown", options=[], value=None),
                    dcc.Dropdown(id="ps-ymetric-dropdown", options=[], value=None),
                    dbc.Checklist(id="comparative-process-only-toggle", options=[], value=[]),
                    html.Div(id="comparative-mode-info"),
                    dbc.Checklist(id="scatter-toggle", options=[], value=[]),
                    html.Div(
                        [
                            dcc.Graph(
                                id="ps-xy-graph",
                                figure=empty_theme_figure(),
                                config={"responsive": False},
                            ),
                            plot_preparing_overlay("comparative-plot-preparing"),
                        ],
                        className="plot-area-with-preparing",
                    ),
                ],
                style={"display": "none"},
            ),
            dbc.Alert(
                message,
                color="warning",
                className=status_alert_class("warning"),
            ),
        ],
        className="empty-comparative-content",
    )


def is_empty_tab_placeholder(current_children: Any) -> bool:
    """Detect hidden placeholder content."""
    names = {
        "empty-time-series-content",
        "empty-process-specific-content",
        "empty-comparative-content",
    }
    if isinstance(current_children, dict):
        return current_children.get("props", {}).get("className") in names
    class_name = getattr(current_children, "className", None)
    if class_name in names:
        return True
    props = getattr(current_children, "props", None)
    if isinstance(props, dict) and props.get("className") in names:
        return True
    return False


def tab_body_action(
    triggered_id: str | None,
    active_tab: str | None,
    this_tab: str,
    current_children: Any,
) -> str:
    """Lazy-build a hidden tab, then keep it until the loaded data changes.

    Returns:
        ``keep``: leave the current children (``dash.no_update``)
        ``empty``: drop a stale built body so the next visit rebuilds
        ``build``: construct the tab from the current stores
    """
    placeholder = is_empty_tab_placeholder(current_children)
    is_data = triggered_id in ("processed-df-store", "process-time-range-store")
    on_this_tab = active_tab == this_tab

    if is_data and not on_this_tab:
        if current_children and not placeholder:
            return "empty"
        return "keep"

    if triggered_id == "tab-prefetch-store":
        if placeholder or not current_children:
            return "build"
        return "keep"

    if triggered_id == "results-tabs":
        if not on_this_tab:
            return "keep"
        if current_children and not placeholder:
            return "keep"
        return "build"

    if not on_this_tab:
        return "keep"
    return "build"


def create_layout(app):
    """Build the full application layout tree."""
    return html.Div(
        id="main-container",
        className="app-shell theme-dark dbc",
        children=[
            dbc.Row(
                [
                    # Left sidebar: configuration and run controls
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Img(
                                                src=app.get_asset_url("logo.png"),
                                                className="sidebar-logo",
                                            ),
                                            dbc.Button(
                                                html.I(id="theme-toggle-icon", className="bi bi-sun-fill"),
                                                id="theme-toggle-btn",
                                                n_clicks=0,
                                                color="link",
                                                className="theme-toggle-btn",
                                                title="Toggle light / dark mode",
                                            ),
                                            dbc.Switch(
                                                id="theme-switch",
                                                value=False,
                                                persistence=True,
                                                className="theme-switch-hidden",
                                            ),
                                        ],
                                        className="sidebar-brand-row",
                                    ),
                                    html.H1(
                                        "Alumet-Insight",
                                        className="sidebar-title",
                                    ),
                                    html.P(
                                        [
                                            "Gain insight into process-specific energy and resource usage from ",
                                            html.A(
                                                "Alumet",
                                                href="https://alumet.dev",
                                                target="_blank",
                                                rel="noopener noreferrer",
                                            ),
                                            " measurements.",
                                        ],
                                        className="sidebar-description",
                                    ),
                                ],
                                className="sidebar-intro",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Load Experiment"),
                                    dbc.CardBody(
                                        [
                                            dbc.RadioItems(
                                                id="load-source-mode",
                                                options=[
                                                    {
                                                        "label": "Upload folder",
                                                        "value": LOAD_SOURCE_UPLOAD,
                                                    },
                                                    {
                                                        "label": "Enter path",
                                                        "value": LOAD_SOURCE_PATH,
                                                    },
                                                ],
                                                # Default Enter path; remember the user's last choice.
                                                # No hostname guessing — local vs SSH-tunneled remote
                                                # both look like localhost and cannot be distinguished.
                                                value=LOAD_SOURCE_PATH,
                                                persistence=True,
                                                persistence_type="local",
                                                inline=True,
                                                className="sidebar-source-toggle",
                                            ),
                                            html.Div(
                                                id="upload-source-panel",
                                                className="sidebar-source-panel",
                                                style={"display": "none"},
                                                children=[
                                                    dcc.Upload(
                                                        id="directory-upload",
                                                        children=upload_prompt_children(),
                                                        multiple=True,
                                                        enable_folder_selection=True,
                                                        accept=".csv,.log,.txt,.toml",
                                                        className="sidebar-upload",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="path-source-panel",
                                                className="sidebar-source-panel",
                                                style={"display": "block"},
                                                children=[
                                                    dcc.Input(
                                                        id="directory-path-input",
                                                        type="text",
                                                        placeholder=f"Path with {SOURCE_FILE_HINT}",
                                                        debounce=True,
                                                        className="sidebar-input",
                                                    ),
                                                ],
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dbc.Button(
                                                            "Visualize",
                                                            id="visualize-button",
                                                            n_clicks=0,
                                                            color="primary",
                                                            size="lg",
                                                            className="sidebar-action-btn",
                                                            style={
                                                                "fontSize": "1rem",
                                                                "fontWeight": "600",
                                                                "padding": "clamp(8px, 1.3vh, 11px) 12px",
                                                                "width": "100%",
                                                                "backgroundColor": COLOR_PRIMARY,
                                                                "borderColor": COLOR_PRIMARY,
                                                                "color": "#ffffff",
                                                            },
                                                        ),
                                                        xs=6,
                                                    ),
                                                    dbc.Col(
                                                        dbc.Button(
                                                            "Reset",
                                                            id="reset-button",
                                                            n_clicks=0,
                                                            color="secondary",
                                                            size="lg",
                                                            className="sidebar-action-btn",
                                                            style={
                                                                "fontSize": "1rem",
                                                                "fontWeight": "600",
                                                                "padding": "clamp(8px, 1.3vh, 11px) 12px",
                                                                "width": "100%",
                                                                "backgroundColor": COLOR_DANGER,
                                                                "borderColor": COLOR_DANGER,
                                                                "color": "#ffffff",
                                                            },
                                                        ),
                                                        xs=6,
                                                    ),
                                                ],
                                                className="g-2 sidebar-action-row",
                                            ),
                                            html.Div("Status", className="sidebar-section-label"),
                                            dcc.Loading(
                                                id="loading-status",
                                                type="circle",
                                                color=COLOR_LOADING,
                                                children=html.Div(id="status-message"),
                                            ),
                                        ],
                                    ),
                                ],
                                className="sidebar-card",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div("Experiment Summary", className="sidebar-process-label"),
                                            html.Div(
                                                id="process-info",
                                                children=[
                                                    html.Span(id="experiment-name-display", className="sidebar-info-value"),
                                                    html.Span(id="pid-display", className="sidebar-info-value"),
                                                    html.Span(id="device-display", className="sidebar-info-value", style={"marginBottom": "0"}),
                                                ],
                                            ),
                                        ],
                                        style={"padding": "14px"},
                                    ),
                                ],
                                className="process-summary-card",
                                style={"borderRadius": "8px", "boxShadow": "0 2px 8px var(--app-shadow)"},
                            ),
                        ],
                        xs=12,
                        md=4,
                        lg=3,
                        className="sidebar-col",
                        style={
                            "backgroundColor": "var(--app-sidebar-bg)",
                            "borderRight": "1px solid var(--app-border)",
                            "padding": "clamp(14px, 2vh, 24px) 22px",
                        },
                    ),
                    # Main area: all visualizations and analysis tabs
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="results-tabs",
                                value="time-series-tab",
                                children=[
                                    dcc.Tab(label="\U0001f4c8 Time Series", value="time-series-tab"),
                                    dcc.Tab(label="\U0001f50e Process-Specific Analysis", value="process-specific-tab"),
                                    dcc.Tab(label="\u2696\ufe0f Comparative Analysis", value="comparative-tab"),
                                ],
                                className="results-tabs",
                            ),
                            html.Div(
                                id="tab-content-area",
                                children=[
                                    html.Div(
                                        id="time-series-content",
                                        children=empty_time_series_content(),
                                        className=TAB_PANEL_ACTIVE,
                                    ),
                                    html.Div(
                                        id="process-specific-content",
                                        children=empty_process_specific_content(),
                                        className=TAB_PANEL_IDLE,
                                    ),
                                    html.Div(
                                        id="comparative-content",
                                        children=empty_comparative_content(),
                                        className=TAB_PANEL_IDLE,
                                    ),
                                    html.Div(
                                        id="tab-preparing-overlay",
                                        className="tab-preparing-overlay",
                                        style={"display": "none"},
                                        children=html.Div(
                                            [
                                                html.Div(className="tab-preparing-spinner"),
                                                html.Span("Preparing this view…", className="tab-preparing-label"),
                                            ],
                                            className="tab-preparing-inner",
                                        ),
                                    ),
                                ],
                            ),
                        ],
                        xs=12,
                        md=8,
                        lg=9,
                        className="main-col",
                        style={
                            "backgroundColor": "var(--app-main-bg)",
                            "padding": "28px 28px 40px",
                        },
                    ),
                ],
                className="g-0",
            ),
            # Hidden stores for data
            dcc.Store(id="upload-relative-paths", data=None),
            dcc.Store(id="processed-df-store", data=None),
            dcc.Store(id="process-time-range-store", data=None),
            dcc.Store(id="timeseries-filtered-df-store", data=None),
            dcc.Store(id="grid-shared-xrange-store", data=None),
            # Written when Time Series is idle or another tab is hovered; hidden tabs build from this.
            dcc.Store(id="tab-prefetch-store", data=None),
            # Required Output for the idle-timer clientside callback. Nothing reads this value.
            dcc.Store(id="tab-prefetch-timer-output", data=None),
            # Dummy store: clientside afterTabBuild writes here to re-sync panel height after tab build.
            dcc.Store(id="tab-panel-layout-ts", data=None),
        ],
        style={
            "backgroundColor": "var(--app-main-bg)",
            "width": "100%",
            "maxWidth": "none",
            "margin": "0",
        },
    )
