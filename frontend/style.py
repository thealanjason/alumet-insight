"""Frontend style constants and component theming helpers."""

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html


# ---------------------------------------------------------------------------
# Color constants (Nord palette)
# ---------------------------------------------------------------------------

COLOR_PRIMARY = "#5E81AC"   # primary action button (Visualize)
COLOR_DANGER  = "#BF616A"   # destructive action button (Reset)
COLOR_LOADING = "#88C0D0"   # loading spinner

# Same hue order in both themes so a series keeps its identity when the mode switches. 
# PLOT_COLORS_DARK keeps the current Plotly qualitative look
# PLOT_COLORS_LIGHT uses darker counterparts that stay readable on white.
PLOT_COLORS_DARK = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
    "#88C0D0",
    "#EBCB8B",
    "#D08770",
    "#B48EAD",
)

PLOT_COLORS_LIGHT = (
    "#3D4ED8",
    "#C73E2A",
    "#0A8F6C",
    "#7B3FD4",
    "#D97706",
    "#0E8AAA",
    "#C43D6E",
    "#4F7D3B",
    "#A21CAF",
    "#B45309",
    "#3E6B8F",
    "#A36A00",
    "#B85C38",
    "#7E5693",
)


# ---------------------------------------------------------------------------
# Component style dicts
# ---------------------------------------------------------------------------

DROPDOWN_STYLE = {"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"}
COMPACT_DROPDOWN_STYLE = {**DROPDOWN_STYLE, "fontSize": "0.75rem"}
CARD_STYLE = {"backgroundColor": "var(--app-card-bg)", "border": "1px solid var(--app-border)"}


# ---------------------------------------------------------------------------
# Process-specific 2x2 grid layout constants
# ---------------------------------------------------------------------------

GRID_SIZE = 2

FILTER_KEYS = ("rk", "rid", "ck", "cid", "la")

FILTER_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("rk", "R.Kind", "resource-kind-dropdown", "rk-container"),
    ("rid", "R.ID", "resource-id-dropdown", "rid-container"),
    ("ck", "C.Kind", "consumer-kind-dropdown", "ck-container"),
    ("cid", "C.ID", "consumer-id-dropdown", "cid-container"),
    ("la", "Attr", "late-attr-dropdown", "la-container"),
)

FILTER_LABEL_MAP = {
    "rk": "Resource Kind",
    "rid": "Resource ID",
    "ck": "Consumer Kind",
    "cid": "Consumer ID",
    "la": "Late Attributes",
}

STYLE_HIDDEN = {"display": "none"}
STYLE_VISIBLE = {"display": "flex"}
STYLE_FILTER_SLOT_VISIBLE = {
    "display": "flex",
    "flexDirection": "column",
    "flex": "1 1 0",
    "minWidth": 0,
}

GRID_GRAPH_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "responsive": True,
    "doubleClick": "autosize",
}

GRID_PLACEHOLDER_MARGIN = dict(l=40, r=12, t=28, b=22)
GRID_DATA_MARGIN = dict(l=40, r=12, t=8, b=22)

# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

def status_alert_class(color: str) -> str:
    """Return CSS classes for the muted sidebar status panel."""
    return f"status-alert status-alert-{color}"


def status_alert(
    color: str,
    title,
    detail=None,
    *,
    icon: str | None = None,
):
    """Build a sidebar status alert."""
    line = []
    if icon:
        line.append(icon)
    if isinstance(title, str):
        line.append(html.Strong(title))
    else:
        line.extend(title)
    if detail is not None:
        if line:
            line.append(" ")
        line.append(html.Span(detail, className="status-alert-detail"))
    return dbc.Alert(
        html.Div(line, className="status-alert-line"),
        color=color,
        className=status_alert_class(color),
    )


# ---------------------------------------------------------------------------
# Figure theming
# ---------------------------------------------------------------------------

def plot_color_palette(use_light_mode: bool = False) -> tuple[str, ...]:
    """Return the qualitative series colors for the active theme."""
    return PLOT_COLORS_LIGHT if use_light_mode else PLOT_COLORS_DARK


def plot_pair_colors(use_light_mode: bool = False) -> dict[str, str]:
    """Accent colors for comparative dual-axis, scatter, and cumulative plots."""
    if use_light_mode:
        return {
            "x": "#3E6B8F",
            "y": "#C73E2A",
            "scatter": "#D97706",
            "cumulative": "#4F7D3B",
            "marker_line": "#1F2937",
        }
    return {
        "x": "#88C0D0",
        "y": "#FF6B6B",
        "scatter": "#FF8C42",
        "cumulative": "#A3BE8C",
        "marker_line": "#FFFFFF",
    }


def process_active_fill(use_light_mode: bool = False) -> str:
    """Shaded process-window fill that stays visible on both backgrounds."""
    return "rgba(62, 107, 143, 0.16)" if use_light_mode else "rgba(136, 192, 208, 0.12)"


def apply_figure_theme(fig: go.Figure, use_light_mode: bool = False) -> go.Figure:
    """Apply the dashboard theme colors to Plotly figures."""
    theme = {
        "paper": "#ffffff",
        "plot": "#f7f8fa",
        "font": "#1f2937",
        "grid": "rgba(31, 41, 55, 0.12)",
        "legend": "rgba(255, 255, 255, 0.92)",
        "legend_font": "#000000",
    } if use_light_mode else {
        "paper": "#252c3e",
        "plot": "#1e2433",
        "font": "#ECEFF4",
        "grid": "rgba(216, 222, 233, 0.15)",
        "legend": "rgba(37, 44, 62, 0.88)",
        "legend_font": "#ffffff",
    }
    fig.update_layout(
        paper_bgcolor=theme["paper"],
        plot_bgcolor=theme["plot"],
        font=dict(color=theme["font"]),
    )
    fig.update_xaxes(gridcolor=theme["grid"], zerolinecolor=theme["grid"], tickfont=dict(color=theme["font"]))
    fig.update_yaxes(gridcolor=theme["grid"], zerolinecolor=theme["grid"], tickfont=dict(color=theme["font"]))
    if fig.layout.legend:
        fig.update_layout(legend=dict(bgcolor=theme["legend"], font=dict(color=theme["legend_font"])))
    return fig

def set_plotly_rgba(color: str, alpha: float = 0.15) -> str:
    """Convert a Plotly color string to an rgba fill with the given alpha."""
    if color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = (int(h[k : k + 2], 16) for k in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
    if color.startswith("rgba"):
        return color.rsplit(",", 1)[0] + f", {alpha})"
    if color.startswith("rgb"):
        return color.replace("rgb", "rgba").replace(")", f", {alpha})")
    return f"rgba(136, 192, 208, {alpha})"
