import plotly.graph_objects as go


def status_alert_class(color: str) -> str:
    """Return CSS classes for the muted sidebar status panel."""
    return f"status-alert status-alert-{color}"


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
