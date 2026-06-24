"""Dash application layout helpers."""

from typing import Any

import dash_bootstrap_components as dbc
from dash import dcc, html

from frontend.style import status_alert_class, COLOR_PRIMARY, COLOR_DANGER, COLOR_LOADING


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
                    html.Div(id="timeseries-plot-container"),
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
                    dcc.Graph(id="ps-xy-graph"),
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
    if isinstance(current_children, dict):
        return current_children.get("props", {}).get("className") in {
            "empty-time-series-content",
            "empty-process-specific-content",
            "empty-comparative-content",
        }
    return False


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
                                        "Alumet Energy Visualization",
                                        className="sidebar-title",
                                    ),
                                    html.P(
                                        "Visualization dashboard to monitor process-specific compute resource usage measured by Alumet.",
                                        className="sidebar-description",
                                    ),
                                ],
                                className="sidebar-intro",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Configuration Setup"),
                                    dbc.CardBody(
                                        [
                                            html.Label(
                                                [
                                                    "Directory Path ",
                                                    html.Span("(Required)", className="sidebar-label-required"),
                                                ],
                                                className="sidebar-label",
                                            ),
                                            dcc.Input(
                                                id="directory-path-input",
                                                type="text",
                                                placeholder="Path containing .csv, .log/.txt, and .toml files",
                                                debounce=True,
                                                className="sidebar-input",
                                            ),
                                            html.Hr(className="sidebar-hr"),
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
                                            html.Hr(className="sidebar-hr"),
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
                                style={"marginBottom": "25px"},
                            ),
                            html.Div(
                                id="tab-content-area",
                                children=dcc.Loading(
                                    id="loading-tab-content",
                                    type="circle",
                                    color=COLOR_LOADING,
                                    children=[
                                        html.Div(
                                            id="time-series-content",
                                            children=empty_time_series_content(),
                                            className="tab-panel-scroll",
                                            style={"display": "flex", "flexDirection": "column", "marginTop": "10px", "minHeight": 0},
                                        ),
                                        html.Div(
                                            id="process-specific-content",
                                            children=empty_process_specific_content(),
                                            className="tab-panel-scroll",
                                            style={"display": "none", "marginTop": "10px"},
                                        ),
                                        html.Div(
                                            id="comparative-content",
                                            children=empty_comparative_content(),
                                            className="tab-panel-scroll",
                                            style={"display": "none", "marginTop": "10px"},
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "minHeight": "100%",
                                        "overflow": "visible",
                                    },
                                ),
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
            dcc.Store(id="processed-df-store", data=None),
            dcc.Store(id="original-df-store", data=None),
            dcc.Store(id="process-time-range-store", data=None),
            dcc.Store(id="timeseries-filtered-df-store", data=None),
            dcc.Store(id="grid-shared-xrange-store", data=None),
        ],
        style={
            "backgroundColor": "var(--app-main-bg)",
            "width": "100%",
            "maxWidth": "none",
            "margin": "0",
        },
    )
