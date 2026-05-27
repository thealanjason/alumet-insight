from typing import Any, Optional
import copy
import uuid
import time
import tempfile
import atexit
import shutil
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from pathlib import Path
import dash
from dash import Dash, html, dcc, callback, Input, Output, State, ctx, MATCH, ALL

from utils import (
    load_csv_from_path,
    preprocess_dataframe_for_visualization, 
    metric_id_is_process_consumer,
    get_process_time_range_from_df, 
    read_file_content,
    extract_pid_from_content, 
    is_gpu_from_content,
    is_cpu_from_content,
    get_color_palette,
    create_all_timeseries_plots,
    norm,
    uniq_str,
    find_files_in_directory,
    is_cumulative_metric,
    get_metric_unit,
    is_memory_metric,
    get_bytes_tickvals_ticktext,
    metric_id_is_process_consumer,
    synthesize_attributed_energy_total,
)

# Get base directory
BASE_DIR = Path(__file__).parent.parent
BOOTSTRAP_ICONS = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"

# ============================================================
# Server-side DataFrame Cache
# Use in-memory cache (instant) + Parquet on disk (persistent)
# dcc.Store only holds a reference ID, not the actual data
# ============================================================
CACHE_DIR = Path(tempfile.mkdtemp(prefix="dash_df_cache_"))
CACHE_DIR.chmod(0o700)

# In-memory cache: eliminates repeated Parquet disk reads (11+ per session)
_MEMORY_CACHE: dict[str, pd.DataFrame] = {}

# Clean up cache on exit
def _cleanup_cache():
    _MEMORY_CACHE.clear()
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

atexit.register(_cleanup_cache)


def cache_dataframe(df: pd.DataFrame, prefix: str = "df") -> str:
    """Cache DataFrame to disk and in-memory, return a reference ID.
    
    Uses Parquet format on disk and in-memory dict for fast access.
    
    Args:
        df: DataFrame to cache
        prefix: Prefix for the cache file
    
    Returns:
        Cache ID string to store in dcc.Store
    """
    if df is None or df.empty:
        return None
    
    cache_id = f"{prefix}_{uuid.uuid4().hex[:12]}"
    cache_path = CACHE_DIR / f"{cache_id}.parquet"
    
    # Persist to disk (backup) and keep in memory (fast access)
    df.to_parquet(cache_path, engine="pyarrow", index=False)
    _MEMORY_CACHE[cache_id] = df
    
    return cache_id


def load_cached_dataframe(cache_id: Optional[str]) -> pd.DataFrame:
    """Load DataFrame from cache by ID.
    
    Checks in-memory cache first (instant), falls back to disk Parquet.
    
    Args:
        cache_id: Cache ID returned by cache_dataframe()
    
    Returns:
        Cached DataFrame or empty DataFrame if not found
    """
    if not cache_id:
        return pd.DataFrame()
    
    # Check in-memory cache first (instant, no disk I/O)
    if cache_id in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_id]
    
    # Fall back to disk Parquet and promote to memory
    cache_path = CACHE_DIR / f"{cache_id}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path, engine="pyarrow")
        _MEMORY_CACHE[cache_id] = df  # Promote to memory for next access
        return df
    
    # Cache miss
    return pd.DataFrame({"__cache_miss__": [True]})


def is_cache_miss(df: pd.DataFrame) -> bool:
    """True when df_from_store returned a sentinel for a stale server-side cache."""
    return not df.empty and "__cache_miss__" in df.columns


def df_from_store(store_data: Any) -> pd.DataFrame:
    """Reconstruct DataFrame from dcc.Store data.
    
    Handles:
    - Cache ID string (new optimized approach for large data)
    - 'split' format dict (medium datasets)
    - 'records' format (legacy)
    """
    if store_data is None:
        return pd.DataFrame()
    
    # Check if it's a cache ID (string starting with prefix)
    if isinstance(store_data, str):
        return load_cached_dataframe(store_data)
    
    # Check if it's split format (has 'columns', 'data' keys)
    if isinstance(store_data, dict) and "columns" in store_data and "data" in store_data:
        return pd.DataFrame.from_dict(store_data, orient="split")
    else:
        # Legacy 'records' format or direct dict
        return pd.DataFrame(store_data)


def _ensure_timestamp_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure timestamp column is datetime64 dtype.
    
    Args:
        df: DataFrame to ensure timestamp column is datetime64 dtype
    
    Returns:
        DataFrame with timestamp column as datetime64 dtype
    """
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, BOOTSTRAP_ICONS])
app.config.suppress_callback_exceptions=True

def status_alert_class(color: str) -> str:
    """Return CSS classes for the muted sidebar status panel."""
    return f"status-alert status-alert-{color}"


def status_alert(
    color: str,
    title,
    detail=None,
    *,
    icon: str | None = None,
    detail_style: dict | None = None,
):
    """
    Build a sidebar status alert.
    """
    title_row = []
    if icon:
        title_row.append(icon)
    if isinstance(title, str):
        title_row.append(html.Strong(title))
    else:
        title_row.extend(title)

    children = [html.Div(title_row, className="status-alert-title")]
    if detail is not None:
        detail_content = detail if isinstance(detail, (list, tuple)) else detail
        children.append(
            html.Div(
                detail_content,
                className="status-alert-detail",
                style=detail_style or {},
            )
        )
    return dbc.Alert(children, color=color, className=status_alert_class(color))


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
            "empty-comparative-content",
        }
    return False


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

# App Layout
app.layout = html.Div(
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
                                                            "backgroundColor": "#5E81AC",
                                                            "borderColor": "#5E81AC",
                                                            "color": "#ffffff",
                                                        },
                                                    ),
                                                    xs=7,
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
                                                            "backgroundColor": "#BF616A",
                                                            "borderColor": "#BF616A",
                                                            "color": "#ffffff",
                                                        },
                                                    ),
                                                    xs=5,
                                                ),
                                            ],
                                            className="g-2 sidebar-action-row",
                                        ),
                                        html.Hr(className="sidebar-hr"),
                                        html.Div("Status", className="sidebar-section-label"),
                                        dcc.Loading(
                                            id="loading-status",
                                            type="circle",
                                            color="#88C0D0",
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
                                dcc.Tab(label="📈 Time Series", value="time-series-tab"),
                                dcc.Tab(label="🔎 Process-Specific Analysis", value="process-specific-tab"),
                                dcc.Tab(label="⚖️ Comparative Analysis", value="comparative-tab"),
                            ],
                            style={"marginBottom": "25px"},
                        ),
                        dcc.Loading(
                            id="loading-tab-content",
                            type="circle",
                            color="#88C0D0",
                            children=[
                                html.Div(
                                    id="time-series-content",
                                    children=empty_time_series_content(),
                                    style={"display": "block", "marginTop": "10px"},
                                ),
                                html.Div(
                                    id="process-specific-content",
                                    style={"display": "none", "marginTop": "10px"},
                                ),
                                html.Div(
                                    id="comparative-content",
                                    children=empty_comparative_content(),
                                    style={"display": "none", "marginTop": "10px"},
                                ),
                            ],
                            style={"minHeight": "200px"},
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
        dcc.Store(id="processed-df-store", data=None),  # Store processed dataframe
        dcc.Store(id="original-df-store", data=None),  # Store original dataframe
        dcc.Store(id="process-time-range-store", data=None),  # Store process time range
        dcc.Store(id="timeseries-filtered-df-store", data=None),  # Store filtered dataframe for Y-axis rescaling
        dcc.Store(id="grid-shared-xrange-store", data=None),  # Shared x-range for synchronized grid plot zooming
    ],
    style={
        "backgroundColor": "var(--app-main-bg)",
        "width": "100%",
        "maxWidth": "none",
        "margin": "0",
    },
)

app.clientside_callback(
    """
    function(useLightMode) {
        var theme = useLightMode ? "light" : "dark";
        document.documentElement.setAttribute("data-bs-theme", theme);
        document.body.setAttribute("data-bs-theme", theme);
        return "app-shell theme-" + theme + " dbc";
    }
    """,
    Output("main-container", "className"),
    Input("theme-switch", "value"),
)

@app.callback(
    Output("theme-switch", "value", allow_duplicate=True),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-switch", "value"),
    prevent_initial_call=True,
)
def toggle_theme_switch(n_clicks, current):
    return not current

@app.callback(
    Output("theme-toggle-icon", "className"),
    Input("theme-switch", "value"),
)
def update_theme_icon(use_light_mode):
    return "bi bi-moon-stars-fill" if use_light_mode else "bi bi-sun-fill"

# Callback for Reset button - clears all data and resets input
@app.callback(
    Output("directory-path-input", "value", allow_duplicate=True),
    Output("processed-df-store", "data", allow_duplicate=True),
    Output("original-df-store", "data", allow_duplicate=True),
    Output("process-time-range-store", "data", allow_duplicate=True),
    Output("timeseries-filtered-df-store", "data", allow_duplicate=True),
    Output("experiment-name-display", "children", allow_duplicate=True),
    Output("pid-display", "children", allow_duplicate=True),
    Output("device-display", "children", allow_duplicate=True),
    Output("status-message", "children", allow_duplicate=True),
    Input("reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_app(n_clicks):
    """Reset the application to its initial state."""
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    
    # Return empty/initial values for all outputs
    return (
        "",  # Clear directory path input
        None,  # Clear processed-df-store
        None,  # Clear original-df-store
        None,  # Clear process-time-range-store
        None,  # Clear timeseries-filtered-df-store
        "Name: N/A",  # Reset experiment name display
        "Process ID: N/A",  # Reset pid display
        "Device: N/A",  # Reset device display
        status_alert(
            "warning",
            "Ready to load",
            [
                "Enter a directory path above, then click ",
                html.Strong("Visualize"),
                " or press Enter/Tab to load and visualize data.",
            ],
        ),  # Reset status message
    )

@app.callback(
    Output("experiment-name-display", "children"),
    Output("pid-display", "children"),
    Output("device-display", "children"),
    Input("visualize-button", "n_clicks"),
    Input("directory-path-input", "n_submit"),
    Input("directory-path-input", "n_blur"),
    State("directory-path-input", "value"),
)
def update_process_info(n_clicks, n_submit, n_blur, directory_path):
    if not any([n_clicks, n_submit, n_blur]) or not directory_path or not directory_path.strip():
        return "Name: N/A", "Process ID: N/A", "Device: N/A"
    
    try:
        experiment_name = Path(directory_path.strip()).name or "N/A"
        # Find and read csv and log file
        log_file = find_files_in_directory(directory_path, ['.log', '.txt'])
        log_content = read_file_content(log_file)
        pid = extract_pid_from_content(log_content)
        csv_file = find_files_in_directory(directory_path, ['.csv'])
        csv_content = read_file_content(csv_file)
        if is_gpu_from_content(csv_content) and not is_cpu_from_content(csv_content):
            device = "GPU"
        elif not is_gpu_from_content(csv_content) and is_cpu_from_content(csv_content):
            device = "CPU"
        else:
            device = "CPU + GPU"
    except Exception:
        return "Name: N/A", "Process ID: N/A", "Device: N/A"

    return (
        f"Name: {experiment_name}",
        f"Process ID: {pid or 'N/A'}",
        f"Device: {device}",
    )

@app.callback(
    Output("status-message", "children"),
    Output("processed-df-store", "data"),
    Output("original-df-store", "data"),
    Output("process-time-range-store", "data"),
    Input("visualize-button", "n_clicks"),
    Input("directory-path-input", "n_submit"),
    Input("directory-path-input", "n_blur"),
    State("directory-path-input", "value"),
)
def load_and_visualize(n_clicks, n_submit, n_blur, directory_path):
    if not any([n_clicks, n_submit, n_blur]):
        return (
            status_alert(
                "warning",
                "Ready to load",
                [
                    "Enter a directory path above, then click ",
                    html.Strong("Visualize"),
                    " or press Enter/Tab to load and visualize data.",
                ],
            ),
            None,
            None,
            None,
        )
    
    # Validate directory path
    if not directory_path or not directory_path.strip():
        status_msg = status_alert(
            "danger",
            "Error:",
            "Directory path is required. Please enter a directory path.",
        )
        return status_msg, None, None, None
    
    try:        
        # Find required files
        csv_file = find_files_in_directory(directory_path, ['.csv'])
        log_file = find_files_in_directory(directory_path, ['.log', '.txt'])
        
        # Load all data from CSV file (timed for performance tracking)
        t0 = time.perf_counter()
        df_all = load_csv_from_path(csv_file)
        t_csv = time.perf_counter()
        
        # Preprocess dataframe for all time series visualization
        df_processed = preprocess_dataframe_for_visualization(df_all)
        t_preprocess = time.perf_counter()
        
        # Use server-side cache for large DataFrames (much faster than JSON in dcc.Store)
        # Only store cache IDs in dcc.Store, not the actual data
        processed_cache_id = cache_dataframe(df_processed, prefix="processed")
        original_cache_id = cache_dataframe(df_all, prefix="original")
        t_cache = time.perf_counter()
        
        # Get process time range from the dataframe
        proc_start, proc_end = get_process_time_range_from_df(df_all)
        
        # Compute timing breakdown
        csv_time = t_csv - t0
        preprocess_time = t_preprocess - t_csv
        cache_time = t_cache - t_preprocess
        total_time = t_cache - t0
        n_rows = len(df_all)
        
        status_msg = status_alert(
            "success",
            "Data loaded successfully",
            f"CSV read: {csv_time:.2f}s, preprocess: {preprocess_time:.2f}s, cache: {cache_time:.2f}s",
            icon="✅ ",
            detail_style={"fontSize": "0.85rem", "color": "var(--app-text-muted)"},
        )
        
        process_time_range = {"start": proc_start.isoformat() if proc_start else None, 
                             "end": proc_end.isoformat() if proc_end else None}
        
        return status_msg, processed_cache_id, original_cache_id, process_time_range
        
    except Exception as e:
        status_msg = status_alert(
            "danger",
            "Error loading data:",
            str(e),
            icon="❌ ",
        )
        return status_msg, None, None, None

# Callback to toggle tab visibility
@app.callback(
    Output("time-series-content", "style"),
    Output("process-specific-content", "style"),
    Output("comparative-content", "style"),
    Input("results-tabs", "value"),
)
def toggle_tab_visibility(tab_value):
    """Toggle tab panel visibility. No content is re-created."""
    hidden = {"display": "none", "marginTop": "10px"}
    visible = {"display": "block", "marginTop": "10px"}
    if tab_value == "time-series-tab":
        return visible, hidden, hidden
    elif tab_value == "process-specific-tab":
        return hidden, visible, hidden
    else:
        return hidden, hidden, visible


# Callback for time series tab content
@app.callback(
    Output("time-series-content", "children"),
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
)
def build_time_series_tab(processed_df_data, process_time_range):
    if not processed_df_data:
        return empty_time_series_content()
    
    # Convert stored data back to dataframe
    df_processed = df_from_store(processed_df_data)
    _ensure_timestamp_datetime(df_processed)
    
    # Use pre-computed base_metric if available, otherwise compute it
    if "base_metric" not in df_processed.columns:
        df_processed["base_metric"] = df_processed["metric_id"].str.split("_R_").str[0]
    
    # Get unique base metrics to determine available categories
    base_metrics = df_processed["base_metric"].unique()
    
    # Determine available categories — grouped by *what you're analyzing*,
    # not by where the metric comes from (CPU vs GPU).
    available_categories = []
    
    # Define which metrics belong to which category
    energy_metrics = set()       # Energy (J): RAPL + NVML energy + attributed
    power_metrics = set()        # Power (W): NVML instant power
    utilization_metrics = set()  # Utilization: cpu_percent + nvml utilization metrics together
    temperature_metrics = set()  # Temperature: nvml_temperature (+ CPU temp if available)
    memory_metrics = set()       # Memory: system memory in bytes (NOT nvml_memory_utilization)
    perf_counter_metrics = set() # Perf Counters: perf_hardware_* + perf_software_*
    kernel_cpu_time_metrics = set()  # Kernel CPU Time: kernel_cpu_time_ms
    kernel_system_metrics = set()    # Kernel/System: kernel_* (non-cpu-time) + network_*
    all_categorized = set()
    
    for m in base_metrics:
        m_lower = m.lower()
        # Power (W) — match before generic "energy" / "rapl" buckets
        if "nvml_instant_power" in m_lower:
            power_metrics.add(m)
            all_categorized.add(m)
        # Energy (J): RAPL + NVML energy counters + attributed energy (not instant power)
        elif "energy" in m_lower or "rapl" in m_lower or "attributed" in m_lower:
            energy_metrics.add(m)
            all_categorized.add(m)
        # Utilization: cpu_percent + all nvml utilization/SM metrics
        elif ("cpu_percent" in m_lower
              or "nvml_gpu_utilization" in m_lower
              or "nvml_sm_utilization" in m_lower
              or "nvml_encoder_utilization" in m_lower
              or "nvml_decoder_utilization" in m_lower
              or "nvml_memory_utilization" in m_lower):
            utilization_metrics.add(m)
            all_categorized.add(m)
        # Temperature
        elif "temperature" in m_lower:
            temperature_metrics.add(m)
            all_categorized.add(m)
        # Memory (bytes only — not nvml_memory_utilization which is already in utilization)
        elif ("mem" in m_lower or "memory" in m_lower or "kb" in m_lower) and "nvml" not in m_lower:
            memory_metrics.add(m)
            all_categorized.add(m)
        # Perf counters
        elif m_lower.startswith("perf_hardware") or m_lower.startswith("perf_software"):
            perf_counter_metrics.add(m)
            all_categorized.add(m)
        # Kernel CPU time
        elif "kernel_cpu_time" in m:
            kernel_cpu_time_metrics.add(m)
            all_categorized.add(m)
        # Kernel/System: remaining kernel_* and network_*
        elif m_lower.startswith("kernel_") or m_lower.startswith("network_"):
            kernel_system_metrics.add(m)
            all_categorized.add(m)
    
    # Add categories if they have metrics
    if energy_metrics:
        available_categories.append({"label": "Energy (J)", "value": "energy"})
    if power_metrics:
        available_categories.append({"label": "Power (W)", "value": "power"})
    if utilization_metrics:
        available_categories.append({"label": "Utilization", "value": "utilization"})
    if temperature_metrics:
        available_categories.append({"label": "Temperature", "value": "temperature"})
    if memory_metrics:
        available_categories.append({"label": "Memory", "value": "memory"})
    if perf_counter_metrics:
        available_categories.append({"label": "Perf Counters", "value": "perf_counters"})
    if kernel_cpu_time_metrics:
        available_categories.append({"label": "Kernel CPU Time", "value": "kernel_cpu_time"})
    if kernel_system_metrics:
        available_categories.append({"label": "Kernel/System", "value": "kernel_system"})
    
    # Add miscellaneous category for metrics not in other categories
    miscellaneous_metrics = set(base_metrics) - all_categorized
    if miscellaneous_metrics:
        available_categories.append({"label": "Miscellaneous", "value": "miscellaneous"})
    
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
                                        }
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
                                        style={"display": "none", "backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"},
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
                                            }
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
                                    style={"display": "none"},  # Hidden by default, shown when valid category selected
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


# Callback for process-specific tab content (lazy loading)
@app.callback(
    Output("process-specific-content", "children"),
    Input("results-tabs", "value"),
    Input("original-df-store", "data"),
    Input("process-time-range-store", "data"),
    State("process-specific-content", "children"),
)
def build_process_specific_tab(tab_value, original_df_data, process_time_range, current_children):
    # Defer building until the tab is actually viewed to speed up initial load.
    # Only build when: (a) user switches to this tab for first time, or (b) data reloaded while viewing this tab.
    triggered_id = ctx.triggered_id
    is_data_trigger = triggered_id in ("original-df-store", "process-time-range-store")

    if is_data_trigger and tab_value != "process-specific-tab":
        # Data reloaded but tab not active — clear stale content so it rebuilds on next view
        return []

    if triggered_id == "results-tabs":
        # Tab switch: only build if no content yet (first view after data loaded)
        if tab_value != "process-specific-tab":
            return dash.no_update
        # Preserve existing content (grid or alert) to keep dropdown selections intact.
        # current_children may be a dict (single component like html.Div) or a list.
        if current_children:
            return dash.no_update

    if not original_df_data or not process_time_range:
        return dbc.Alert(
            "No data available. Please load data first.",
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

    # Convert stored data back to dataframe
    df_original = df_from_store(original_df_data)
    if is_cache_miss(df_original):
        return dbc.Alert(
            "Session data expired (server was restarted). Please click Visualize again to reload.",
            color="danger",
            className=status_alert_class("danger"),
        )
    _ensure_timestamp_datetime(df_original)

    # Get unique metrics
    unique_metrics = sorted(df_original["metric"].unique().tolist())
    
    # Create 2x2 grid layout using dbc.Row and dbc.Col
    # Use column-based layout for filters to keep plot positions constant
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
                                        # Metric dropdown - full width
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Metric:",
                                                    style={
                                                        "color": "var(--app-text)",
                                                        "fontSize": "0.9rem",
                                                        "fontWeight": "500",
                                                        "marginBottom": "4px",
                                                    }
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
                                            style={"marginBottom": "8px"}
                                        ),
                                        # Filter dropdowns in a single compact row
                                        # Fixed height container to keep plot position constant
                                        html.Div(
                                            [
                                                dbc.Row(
                                                    [
                                                        # Resource Kind
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
                                                        # Resource ID
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
                                                        # Consumer Kind
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
                                                        # Consumer ID
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
                                                        # Late Attributes
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
                                            style={"minHeight": "50px", "marginBottom": "8px"},  # Fixed height for filter area
                                        ),
                                        dcc.Graph(id=plot_id, style={"height": "320px"}),
                                        # Download CSV button
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "📥 Download CSV",
                                                    id={"type": "grid-download-btn", "index": f"{i}-{j}"},
                                                    n_clicks=0,
                                                    color="primary",
                                                    size="sm",
                                                    style={
                                                        "fontSize": "0.75rem",
                                                    },
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

def _comparative_metric_ids(processed_df_data: Any, process_time_range: Any) -> list:
    """Metric IDs that have samples inside the process active window (for X-Y comparative tab)."""
    if not processed_df_data or not process_time_range:
        return []
    df_processed = df_from_store(processed_df_data)
    _ensure_timestamp_datetime(df_processed)
    proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range.get("end") else None
    if proc_start is None or proc_end is None:
        return []
    df_process_level = df_processed[
        (df_processed["timestamp"] >= proc_start) & (df_processed["timestamp"] <= proc_end)
    ].copy()
    if df_process_level.empty:
        return []
    return sorted(df_process_level["metric_id"].dropna().astype(str).unique().tolist())


def _filter_comparative_metric_ids(metric_ids: list, process_only: bool) -> list:
    """Restrict to process-attributed series (consumer_kind encoded as ``_C_process_`` in metric_id)."""
    if not process_only:
        return list(metric_ids)
    return [m for m in metric_ids if metric_id_is_process_consumer(m)]


def _pick_xy_metric_values(filtered: list, cur_x: Any, cur_y: Any) -> tuple:
    """Pick valid X/Y dropdown values from filtered list, preserving selection when possible."""
    if not filtered:
        return None, None
    if len(filtered) == 1:
        return filtered[0], filtered[0]
    x_val = cur_x if cur_x in filtered else filtered[0]
    others = [m for m in filtered if m != x_val]
    y_val = cur_y if cur_y in others else others[0]
    return x_val, y_val


# Callback for comparative analysis tab content (lazy loading)
@app.callback(
    Output("comparative-content", "children"),
    Input("results-tabs", "value"),
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
    State("comparative-content", "children"),
)
def build_comparative_tab(tab_value, processed_df_data, process_time_range, current_children):
    # Defer building until the tab is actually viewed to speed up initial load.
    # Only build when: (a) user switches to this tab for first time, or (b) data reloaded while viewing this tab.
    triggered_id = ctx.triggered_id
    is_data_trigger = triggered_id in ("processed-df-store", "process-time-range-store")
    
    if is_data_trigger and tab_value != "comparative-tab":
        # Data reloaded but tab not active — clear stale content so it rebuilds on next view
        return empty_comparative_content()
    
    if triggered_id == "results-tabs":
        # Tab switch: only build if no content yet (first view after data loaded)
        if tab_value != "comparative-tab":
            return dash.no_update
        if current_children and not is_empty_tab_placeholder(current_children):
            return dash.no_update

    if not processed_df_data or not process_time_range:
        return empty_comparative_content()

    # Only metrics that have samples inside the process window
    metric_ids = _comparative_metric_ids(processed_df_data, process_time_range)
    if len(metric_ids) < 2:
        return empty_comparative_content("Need at least 2 metrics inside process window.")

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Metric 1 (X-axis / Left Y-axis):",
                                        style={"color": "var(--app-text)", "fontWeight": "600"}
                                    ),
                                    dcc.Dropdown(
                                        id="ps-xmetric-dropdown",
                                        options=[{"label": m, "value": m} for m in metric_ids],
                                        value=metric_ids[0],
                                        clearable=False,
                                        persistence=True,
                                        className="dark-dropdown",
                                        style={"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"},
                                    ),
                                ],
                                width=12, lg=6, className="mb-3",
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        "Metric 2 (Y-axis / Right Y-axis):",
                                        style={"color": "var(--app-text)", "fontWeight": "600"}
                                    ),
                                    dcc.Dropdown(
                                        id="ps-ymetric-dropdown",
                                        options=[{"label": m, "value": m} for m in metric_ids],
                                        value=metric_ids[1],
                                        clearable=False,
                                        persistence=True,
                                        className="dark-dropdown",
                                        style={"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"},
                                    ),
                                ],
                                width=12, lg=6, className="mb-3",
                            ),
                        ]
                    ),
                    # Optional filter: only series attributed to the measured process (consumer_kind=process)
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Checklist(
                                        id="comparative-process-only-toggle",
                                        options=[
                                            {
                                                "label": " Process metrics only",
                                                "value": "process_only",
                                            }
                                        ],
                                        value=[],
                                        inline=True,
                                        style={"color": "var(--app-text)", "fontSize": "0.9rem"},
                                        inputStyle={"marginRight": "8px"},
                                    ),
                                ],
                                width=12,
                                className="mb-2",
                            ),
                        ],
                    ),
                    # Visualization mode info and scatter toggle
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        id="comparative-mode-info",
                                        style={"marginBottom": "10px"},
                                    ),
                                ],
                                width=12, lg=8, className="mb-2",
                            ),
                            dbc.Col(
                                [
                                    dbc.Checklist(
                                        id="scatter-toggle",
                                        options=[{"label": " Show Scatter Plot (X-Y relationship)", "value": "scatter"}],
                                        value=[],
                                        inline=True,
                                        style={"color": "var(--app-text)", "fontSize": "0.9rem"},
                                        inputStyle={"marginRight": "8px"},
                                    ),
                                ],
                                width=12, lg=4, className="mb-2",
                                style={"textAlign": "right"},
                            ),
                        ],
                        className="mb-2",
                    ),
                    dcc.Graph(id="ps-xy-graph", style={"height": "100%"}),
                    # Download CSV button
                    html.Div(
                        [
                            dbc.Button(
                                "📥 Download CSV",
                                id="xy-download-btn",
                                n_clicks=0,
                                color="primary",
                                size="sm",
                                style={"marginTop": "10px"},
                            ),
                            dcc.Download(id="xy-download"),
                        ],
                        style={"textAlign": "right", "marginTop": "5px"},
                    ),
                ],
                style={"padding": "25px", "backgroundColor": "var(--app-card-bg)"},
                className="viewport-card-body comparative-card-body",
            )
        ],
        style={"backgroundColor": "var(--app-card-bg)", "border": "1px solid var(--app-border)"},
        className="viewport-card comparative-card",
    )

# Callback to update the comparative mode info based on selected metrics
@app.callback(
    Output("comparative-mode-info", "children"),
    Input("ps-xmetric-dropdown", "value"),
    Input("ps-ymetric-dropdown", "value"),
    prevent_initial_call=True,
)
def update_comparative_mode_info(x_metric_id, y_metric_id):
    """Show info about which visualization mode will be used."""
    if not x_metric_id or not y_metric_id:
        return html.Span("")
    
    x_cumulative = is_cumulative_metric(x_metric_id)
    y_cumulative = is_cumulative_metric(y_metric_id)
    
    if x_cumulative and y_cumulative:
        return html.Div(
            [
                html.Span("Visualization Mode: ", style={"fontWeight": "600"}),
                html.Span("Cumulative X-Y Plot", style={"color": "var(--app-success)", "fontWeight": "600"}),
            ],
            style={"color": "var(--app-text)"},
        )
    else:
        non_cumulative = []
        if not x_cumulative:
            non_cumulative.append("Metric 1")
        if not y_cumulative:
            non_cumulative.append("Metric 2")
        return html.Div(
            [
                html.Span("Visualization Mode: ", style={"fontWeight": "600"}),
                html.Span("Dual Y-Axis Time Series", style={"color": "var(--app-warning)", "fontWeight": "600"}),
            ],
            style={"color": "var(--app-text)"},
        )

@app.callback(
    Output("ps-xmetric-dropdown", "options"),
    Output("ps-xmetric-dropdown", "value"),
    Output("ps-ymetric-dropdown", "options"),
    Output("ps-ymetric-dropdown", "value"),
    Input("comparative-process-only-toggle", "value"),
    Input("results-tabs", "value"),
    Input("processed-df-store", "data"),
    Input("process-time-range-store", "data"),
    State("ps-xmetric-dropdown", "value"),
    State("ps-ymetric-dropdown", "value"),
)
def update_comparative_metric_dropdowns(
    process_only_toggle, tab_value, processed_df_data, process_time_range, cur_x, cur_y
):
    """Filter comparative X/Y metric lists to process-attributed series when requested."""
    if tab_value != "comparative-tab":
        raise dash.exceptions.PreventUpdate

    all_ids = _comparative_metric_ids(processed_df_data, process_time_range)
    if len(all_ids) < 2:
        raise dash.exceptions.PreventUpdate

    process_only = bool(process_only_toggle and "process_only" in process_only_toggle)
    filtered = _filter_comparative_metric_ids(all_ids, process_only)

    opts = [{"label": m, "value": m} for m in filtered]
    x_val, y_val = _pick_xy_metric_values(filtered, cur_x, cur_y)
    return opts, x_val, opts, y_val


# Callback to show/hide CPU core selector for kernel_cpu_time and update dropdown options
@app.callback(
    [Output("cpu-core-selector", "children"),
     Output("cpu-core-dropdown", "options"),
     Output("cpu-core-dropdown", "style")],
    Input("metric-category-dropdown", "value"),
    State("processed-df-store", "data"),
)
def update_cpu_core_selector(selected_category, processed_df_data):
    # Default: hide the selector
    default_style = {"display": "none", "backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"}
    default_options = []
    default_children = html.Div()
    
    if selected_category != "kernel_cpu_time" or not processed_df_data:
        return default_children, default_options, default_style
    
    df_processed = df_from_store(processed_df_data)
    # Use pre-computed base_metric if available, otherwise compute it
    if "base_metric" not in df_processed.columns:
        df_processed["base_metric"] = df_processed["metric_id"].str.split("_R_").str[0]
    kernel_metrics = df_processed[df_processed["base_metric"] == "kernel_cpu_time_ms"]
    
    if kernel_metrics.empty:
        return default_children, default_options, default_style
    
    # Extract CPU cores (handle both "cpu_core_X.0" and "cpu_core_X" formats)
    cpu_cores = set()
    for mid in kernel_metrics["metric_id"]:
        if "_R_cpu_core_" in mid:
            try:
                core_part = mid.split("_R_cpu_core_")[1].split("_")[0]
                # Remove .0 if present
                core = core_part.replace(".0", "")
                if core:  # Only add if not empty
                    cpu_cores.add(core)
            except (IndexError, AttributeError):
                continue
    cpu_cores = sorted(cpu_cores)
    
    if not cpu_cores:
        return default_children, default_options, default_style
    
    # Create options for individual cores only (no "All Cores" option)
    options = [{"label": f"Core {core}", "value": core} for core in cpu_cores]
    
    # Show the selector with label (the dropdown itself is in the layout, we just show the label)
    selector_children = html.Label(
        "CPU Core:",
        style={
            "color": "var(--app-text)",
            "marginRight": "10px",
            "fontSize": "1rem",
            "fontWeight": "600",
        }
    )
    
    visible_style = {"backgroundColor": "var(--app-control-bg)", "color": "var(--app-text)"}
    
    return selector_children, options, visible_style


# Callback to show/hide Y-axis options based on metric category
# Only show for categories with same units (energy J, power W, memory, …), hide for miscellaneous
@app.callback(
    Output("yaxis-options-container", "style"),
    Output("shared-yaxis-toggle", "value"),
    Input("metric-category-dropdown", "value"),
    State("shared-yaxis-toggle", "value"),
)
def update_yaxis_options_visibility(selected_category, current_toggle_value):
    """Show Y-axis options only for categories with same units, hide for miscellaneous."""
    # Categories where sharing Y-axis makes sense (same units within category)
    valid_categories = ["energy", "power", "utilization", "temperature", "memory", "kernel_cpu_time"]
    
    if selected_category in valid_categories:
        # Show the Y-axis options but do NOT re-emit the toggle value
        # Using dash.no_update avoids triggering update_yaxis_on_toggle during category switches,
        # which would cause a race condition where the toggle callback reads stale figure/store data
        return {"display": "flex", "flexDirection": "column"}, dash.no_update
    else:
        # Hide the options without clearing the toggle value to avoid a stale-figure race.
        return {"display": "none"}, dash.no_update


# MATCH callback to update filter dropdown options and show/hide containers
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
    """Update filter dropdowns for a single plot using MATCH"""
    # Use visibility instead of display to maintain fixed layout space
    hide = {"visibility": "hidden"}
    show = {"visibility": "visible"}

    if not original_df_data or not metric:
        return (hide, [], None, hide, [], None, hide, [], None, hide, [], None, hide, [], None)

    df = df_from_store(original_df_data)
    dfm = df[df["metric"] == metric].copy()

    # Normalize to strings for stable matching.
    # fillna("") before astype(str): category dtype NaN survives as float NaN in newer pandas.
    dfm["rk"] = dfm["resource_kind"].astype(object).fillna("").astype(str).str.strip()
    dfm["rid"] = dfm["resource_id"].astype(object).fillna("").astype(str).str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(object).fillna("").astype(str).str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(object).fillna("").astype(str).str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(object).fillna("").astype(str).str.strip()

    rk = norm(rk)
    rid = norm(rid)
    ck = norm(ck)
    cid = norm(cid)
    la = norm(la)

    # Check which input triggered the callback to reset children appropriately
    triggered_id = None
    if ctx.triggered:
        triggered = ctx.triggered[0]
        triggered_prop_id = triggered.get("prop_id", "")
        # Extract the component type from prop_id (e.g., "resource-kind-dropdown")
        if '.value' in triggered_prop_id:
            try:
                import json
                json_str = triggered_prop_id.split('.value')[0]
                id_dict = json.loads(json_str)
                triggered_id = id_dict.get("type")
            except:
                pass

    # Resource kind options
    rk_opts = uniq_str(dfm["rk"])
    rk_eff = rk if rk in rk_opts else (rk_opts[0] if len(rk_opts) == 1 else None)
    df1 = dfm if rk_eff is None else dfm[dfm["rk"] == rk_eff]

    # Resource id options - reset if resource_kind changed
    rid_opts = uniq_str(df1["rid"])
    if triggered_id == "resource-kind-dropdown":
        # Parent changed - reset child
        rid_eff = None
    else:
        rid_eff = rid if rid in rid_opts else (rid_opts[0] if len(rid_opts) == 1 else None)
    df2 = df1 if rid_eff is None else df1[df1["rid"] == rid_eff]

    # Consumer kind options
    ck_opts = uniq_str(df2["ck"])
    ck_eff = ck if ck in ck_opts else (ck_opts[0] if len(ck_opts) == 1 else None)
    df3 = df2 if ck_eff is None else df2[df2["ck"] == ck_eff]

    # Consumer id options - reset if consumer_kind changed
    cid_opts = uniq_str(df3["cid"])
    if triggered_id == "consumer-kind-dropdown":
        # Parent changed - reset child
        cid_eff = None
    else:
        cid_eff = cid if cid in cid_opts else (cid_opts[0] if len(cid_opts) == 1 else None)
    df4 = df3 if cid_eff is None else df3[df3["cid"] == cid_eff]

    # Late attributes options - reset if any parent changed
    la_opts = uniq_str(df4["la"])
    if triggered_id in ["resource-kind-dropdown", "resource-id-dropdown", "consumer-kind-dropdown", "consumer-id-dropdown"]:
        # Parent changed - reset late_attr
        la_eff = None
    else:
        la_eff = la if la in la_opts else None  # Don't auto-select late_attr

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

    # Return values: use effective values (auto-selected if single option, otherwise use input)
    # Reset children when parents change
    return (
        rk_style, rk_options, rk_eff,
        rid_style, rid_options, rid_eff,
        ck_style, ck_options, ck_eff,
        cid_style, cid_options, cid_eff,
        la_style, la_options, la_eff
    )

# MATCH callback to update grid plot figure
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
    """Update a single grid plot figure using MATCH"""
    fig = go.Figure()
    apply_figure_theme(fig, use_light_mode)

    if not original_df_data or not metric:
        fig.update_layout(title=dict(text="Select a metric", x=0.5))
        return fig

    df = df_from_store(original_df_data)
    _ensure_timestamp_datetime(df)
    dfm = df[df["metric"] == metric].copy()

    # Same string normalization as options callback.
    # fillna("") before astype(str): category dtype NaN survives as float NaN in newer pandas.
    dfm["rk"] = dfm["resource_kind"].astype(object).fillna("").astype(str).str.strip()
    dfm["rid"] = dfm["resource_id"].astype(object).fillna("").astype(str).str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(object).fillna("").astype(str).str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(object).fillna("").astype(str).str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(object).fillna("").astype(str).str.strip()

    rk = norm(rk)
    rid = norm(rid)
    ck = norm(ck)
    cid = norm(cid)
    la = norm(la)

    # Progressively filter with "effective" auto-selection for single-option dims
    rk_opts = uniq_str(dfm["rk"])
    rk_eff = rk if rk in rk_opts else (rk_opts[0] if len(rk_opts) == 1 else None)
    df1 = dfm if rk_eff is None else dfm[dfm["rk"] == rk_eff]

    rid_opts = uniq_str(df1["rid"])
    rid_eff = rid if rid in rid_opts else (rid_opts[0] if len(rid_opts) == 1 else None)
    df2 = df1 if rid_eff is None else df1[df1["rid"] == rid_eff]

    ck_opts = uniq_str(df2["ck"])
    ck_eff = ck if ck in ck_opts else (ck_opts[0] if len(ck_opts) == 1 else None)
    df3 = df2 if ck_eff is None else df2[df2["ck"] == ck_eff]

    cid_opts = uniq_str(df3["cid"])
    cid_eff = cid if cid in cid_opts else (cid_opts[0] if len(cid_opts) == 1 else None)
    df4 = df3 if cid_eff is None else df3[df3["cid"] == cid_eff]

    la_opts = uniq_str(df4["la"])
    la_eff = la if la in la_opts else (la_opts[0] if len(la_opts) == 1 else None)
    dff = df4 if la_eff is None else df4[df4["la"] == la_eff]

    if dff.empty:
        fig.update_layout(title=dict(text="No data available", x=0.5))
        return fig

    # Uniqueness check
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
                font=dict(size=12)
            )
        )
        return fig

    # Get process time range and truncate data to process active period only
    proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range and process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range and process_time_range.get("end") else None

    dff = dff.sort_values("timestamp")
    
    # Truncate data to process active period for process-specific view
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

    # Fill rgba
    rgba_fill = "rgba(136, 192, 208, 0.15)"
    if isinstance(color, str) and color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        rgba_fill = f"rgba({r}, {g}, {b}, 0.15)"

    # Get unit for y-axis label
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

    # Build yaxis config
    yaxis_config = dict(gridcolor="rgba(76, 86, 106, 0.2)", title=y_axis_title)
    
    # For memory metrics, add custom byte tick formattin
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


# Callback to capture zoom events from grid plots and update shared x-range
# Uses ctx.triggered_id to identify which specific plot the user interacted with
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
    """Sync zoom across all grid plots by capturing relayoutData and updating shared x-range."""
    # Identify which plot triggered this callback
    triggered = ctx.triggered_id
    if not triggered:
        return dash.no_update
    
    # Map triggered id to the corresponding relayoutData
    relayout_map = {
        "0-0": rd_00,
        "0-1": rd_01,
        "1-0": rd_10,
        "1-1": rd_11,
    }
    
    # Get the index from the triggered component
    if isinstance(triggered, dict):
        triggered_index = triggered.get("index")
    else:
        return dash.no_update
    
    relayout_data = relayout_map.get(triggered_index)
    if not relayout_data:
        return dash.no_update
    
    # Check for zoom event (xaxis.range)
    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        new_range = {
            "x0": relayout_data["xaxis.range[0]"],
            "x1": relayout_data["xaxis.range[1]"],
        }
        # Avoid re-triggering if the range hasn't changed (prevents infinite loop)
        if (current_shared_range and 
            not current_shared_range.get("autorange") and
            current_shared_range.get("x0") == new_range["x0"] and 
            current_shared_range.get("x1") == new_range["x1"]):
            return dash.no_update
        return new_range
    
    # Check for autorange/reset event (double-click to reset)
    if "xaxis.autorange" in relayout_data and relayout_data["xaxis.autorange"]:
        # Avoid re-triggering if already in autorange state
        if current_shared_range and current_shared_range.get("autorange"):
            return dash.no_update
        return {"autorange": True}
    
    return dash.no_update


# Callback to apply shared x-range to all grid plots
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
        
        # Deep copy the figure to avoid mutating the original
        new_fig = copy.deepcopy(fig)
        
        if "xaxis" not in new_fig["layout"]:
            new_fig["layout"]["xaxis"] = {}
        
        if is_autorange:
            # Reset to autorange (double-click reset)
            new_fig["layout"]["xaxis"]["autorange"] = True
            new_fig["layout"]["xaxis"].pop("range", None)
        else:
            # Apply the shared x-range (zoom)
            new_fig["layout"]["xaxis"]["range"] = [shared_range["x0"], shared_range["x1"]]
            new_fig["layout"]["xaxis"]["autorange"] = False
        
        updated_figures.append(new_fig)
    
    return updated_figures


# Callback to update time series plot based on category and CPU core selection
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
    
    # Convert stored data back to dataframe
    df_processed = df_from_store(processed_df_data)
    _ensure_timestamp_datetime(df_processed)
    
    # Get full time range from ALL data (before filtering) to fix x-axis
    full_time_min = df_processed["timestamp"].min()
    full_time_max = df_processed["timestamp"].max()
    full_time_range = (full_time_min, full_time_max)
    
    # Use pre-computed base_metric if available, otherwise compute it
    if "base_metric" not in df_processed.columns:
        df_processed["base_metric"] = df_processed["metric_id"].str.split("_R_").str[0]
    
    # Filter based on category
    if selected_category == "energy":
        # Energy (J): RAPL/NVML energy + attributed energy
        em = df_processed["base_metric"].str.contains("energy|rapl|attributed", case=False, na=False)
        not_power = ~df_processed["base_metric"].str.contains("nvml_instant_power", case=False, na=False)
        df_filtered = df_processed[em & not_power].copy()
        # Normalize nvml_energy_consumption from milliJoules to Joules
        nvml_energy_mask = df_filtered["base_metric"].str.contains("nvml_energy_consumption", case=False, na=False)
        if nvml_energy_mask.any():
            df_filtered.loc[nvml_energy_mask, "value"] = df_filtered.loc[nvml_energy_mask, "value"] / 1000.0
        # Synthesize attributed_energy_total = attributed_energy_cpu + attributed_energy_gpu
        df_total = synthesize_attributed_energy_total(df_filtered)
        if not df_total.empty:
            df_filtered = pd.concat([df_filtered, df_total], ignore_index=True)
    elif selected_category == "power":
        # Power (W): NVML instant power
        df_filtered = df_processed[
            df_processed["base_metric"].str.contains("nvml_instant_power", case=False, na=False)
        ].copy()
        nvml_power_mask = df_filtered["base_metric"].str.contains("nvml_instant_power", case=False, na=False)
        # Normalize nvml_instant_power from milliWatts to Watts
        if nvml_power_mask.any():
            df_filtered.loc[nvml_power_mask, "value"] = df_filtered.loc[nvml_power_mask, "value"] / 1000.0
    elif selected_category == "utilization":
        # Utilization: cpu_percent + nvml utilization/SM metrics together
        df_filtered = df_processed[
            df_processed["base_metric"].str.contains(
                "cpu_percent|nvml_gpu_utilization|nvml_sm_utilization|nvml_encoder_utilization|nvml_decoder_utilization|nvml_memory_utilization",
                case=False, na=False
            )
        ]
    elif selected_category == "temperature":
        # Temperature metrics
        df_filtered = df_processed[
            df_processed["base_metric"].str.contains("temperature", case=False, na=False)
        ]
    elif selected_category == "memory":
        # Memory (bytes only) — exclude nvml_memory_utilization (it's in utilization)
        mem_mask = df_processed["base_metric"].str.contains("mem|memory|kb", case=False, na=False)
        nvml_mask = df_processed["base_metric"].str.contains("nvml", case=False, na=False)
        df_filtered = df_processed[mem_mask & ~nvml_mask]
    elif selected_category == "perf_counters":
        # Perf hardware and software counters
        df_filtered = df_processed[
            df_processed["base_metric"].str.contains("^perf_hardware|^perf_software", case=False, na=False, regex=True)
        ]
    elif selected_category == "kernel_cpu_time":
        # Require CPU core selection for kernel_cpu_time (too many cores to show all)
        if not selected_cpu_core:
            return dbc.Alert("Please select a CPU core to display kernel CPU time metrics.", color="warning", className=status_alert_class("warning")), None
        
        # Filter for kernel_cpu_time_ms
        df_filtered = df_processed[df_processed["base_metric"] == "kernel_cpu_time_ms"]
        
        # Filter by selected CPU core
        # Match both "cpu_core_X.0" and "cpu_core_X" patterns
        core_patterns = [
            f"_R_cpu_core_{selected_cpu_core}.0_",
            f"_R_cpu_core_{selected_cpu_core}_",
            f"_R_cpu_core_{selected_cpu_core}.",
        ]
        mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
        for pattern in core_patterns:
            mask |= df_filtered["metric_id"].str.contains(pattern, na=False, regex=False)
        df_filtered = df_filtered[mask]
    elif selected_category == "kernel_system":
        # Kernel stats (non-cpu-time) + network metrics
        kernel_mask = df_processed["base_metric"].str.startswith("kernel_")
        kernel_cpu_time_mask = df_processed["base_metric"].str.contains("kernel_cpu_time", na=False)
        network_mask = df_processed["base_metric"].str.startswith("network_")
        df_filtered = df_processed[(kernel_mask & ~kernel_cpu_time_mask) | network_mask]
    elif selected_category == "miscellaneous":
        # Everything not in the above categories
        energy_pat = "energy|rapl|attributed"
        power_pat = "nvml_instant_power"
        util_pat = "cpu_percent|nvml_gpu_utilization|nvml_sm_utilization|nvml_encoder_utilization|nvml_decoder_utilization|nvml_memory_utilization"
        temp_pat = "temperature"
        mem_pat = "mem|memory|kb"
        perf_pat = "^perf_hardware|^perf_software"
        kernel_pat = "^kernel_"
        network_pat = "^network_"
        
        is_energy = df_processed["base_metric"].str.contains(energy_pat, case=False, na=False) & ~df_processed[
            "base_metric"
        ].str.contains(power_pat, case=False, na=False)
        is_power = df_processed["base_metric"].str.contains(power_pat, case=False, na=False)
        is_util = df_processed["base_metric"].str.contains(util_pat, case=False, na=False)
        is_temp = df_processed["base_metric"].str.contains(temp_pat, case=False, na=False)
        is_mem = df_processed["base_metric"].str.contains(mem_pat, case=False, na=False) & ~df_processed["base_metric"].str.contains("nvml", case=False, na=False)
        is_perf = df_processed["base_metric"].str.contains(perf_pat, case=False, na=False, regex=True)
        is_kernel = df_processed["base_metric"].str.contains(kernel_pat, case=False, na=False, regex=True)
        is_network = df_processed["base_metric"].str.contains(network_pat, case=False, na=False, regex=True)
        
        df_filtered = df_processed[~(is_energy | is_power | is_util | is_temp | is_mem | is_perf | is_kernel | is_network)]
    else:
        # Unknown category - return empty
        df_filtered = pd.DataFrame()
    
    if df_filtered.empty:
        return dbc.Alert("No data available for the selected category.", color="warning", className=status_alert_class("warning")), None
    
    # Get process time range for gray highlight
    proc_start = None
    proc_end = None
    if process_time_range and process_time_range.get("start"):
        proc_start = pd.to_datetime(process_time_range["start"])
    if process_time_range and process_time_range.get("end"):
        proc_end = pd.to_datetime(process_time_range["end"])
    
    # Get the metric order BEFORE creating the figure (this order determines subplot assignment)
    metric_order = df_filtered["metric_id"].unique().tolist()
    
    # Create figure with filtered time series, but with full time range for x-axis
    # Pass category to set appropriate Y-axis label
    yaxis_shareable_categories = {"energy", "power", "utilization", "temperature", "memory", "kernel_cpu_time"}
    share_yaxis = (
        selected_category in yaxis_shareable_categories
        and shared_yaxis_toggle
        and "shared" in shared_yaxis_toggle
    )
    fig = create_all_timeseries_plots(df_filtered, proc_start, proc_end, full_time_range, category=selected_category, share_yaxis=share_yaxis)
    apply_figure_theme(fig, use_light_mode)
    
    # Store filtered data for Y-axis rescaling callback using Parquet cache
    # (much faster than JSON records for large datasets)
    df_for_store = df_filtered[["metric_id", "timestamp", "value"]].copy()
    filtered_cache_id = cache_dataframe(df_for_store, prefix="ts_filtered") if not df_for_store.empty else None
    filtered_df_json = {
        "cache_id": filtered_cache_id,
        "metric_order": metric_order  # Preserve the exact order used when creating figure
    }
    
    # Wrap the graph in a div to ensure proper scrolling and full width
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


# Callback to update Y-axis ranges when shared-yaxis toggle is changed
# This preserves the current X-axis zoom and only updates Y-axis ranges
@app.callback(
    Output("timeseries-graph", "figure", allow_duplicate=True),
    Input("shared-yaxis-toggle", "value"),
    State("timeseries-graph", "figure"),
    State("timeseries-filtered-df-store", "data"),
    prevent_initial_call=True,
)
def update_yaxis_on_toggle(shared_yaxis_toggle, current_figure, filtered_df_store):
    """Update Y-axis ranges when shared Y-axis toggle is changed.
    
    Preserves the current X-axis zoom state and only updates Y-axis ranges.
    """
    if not current_figure or not filtered_df_store:
        return current_figure if current_figure else dash.no_update
    
    # Extract data and metric order from the store
    if not isinstance(filtered_df_store, dict) or "cache_id" not in filtered_df_store:
        return current_figure
    
    cache_id = filtered_df_store.get("cache_id")
    metric_order = filtered_df_store.get("metric_order", [])
    
    if not cache_id or not metric_order:
        return current_figure
    
    # Load filtered data from Parquet cache (instant from in-memory cache)
    df = load_cached_dataframe(cache_id)
    _ensure_timestamp_datetime(df)
    
    if df.empty:
        return current_figure
    
    # Deep copy the figure to avoid modifying shared nested dicts
    updated_figure = copy.deepcopy(current_figure)
    layout = updated_figure.get("layout", {})
    
    # Check if shared Y-axis is enabled
    share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle
    
    # Get current X-axis range from the figure (preserves zoom state)
    # Use first xaxis since they're shared
    xaxis_layout = layout.get("xaxis", {})
    x_range = xaxis_layout.get("range")
    
    if x_range:
        # Use the current zoomed X-range
        x_min = pd.to_datetime(x_range[0])
        x_max = pd.to_datetime(x_range[1])
        
        # Ensure timezone compatibility
        df_tz = df["timestamp"].dt.tz
        if df_tz is not None:
            if x_min.tz is None:
                x_min = x_min.tz_localize(df_tz)
            if x_max.tz is None:
                x_max = x_max.tz_localize(df_tz)
        else:
            if x_min.tz is not None:
                x_min = x_min.tz_convert(None) if hasattr(x_min, 'tz_convert') else x_min.replace(tzinfo=None)
            if x_max.tz is not None:
                x_max = x_max.tz_convert(None) if hasattr(x_max, 'tz_convert') else x_max.replace(tzinfo=None)
        
        visible_data = df[(df["timestamp"] >= x_min) & (df["timestamp"] <= x_max)]
    else:
        # No zoom applied, use all data
        visible_data = df
    
    if visible_data.empty:
        return current_figure
    
    # Determine the correct category from the actual metric data
    is_memory_category = metric_order and is_memory_metric(metric_order[0])
    
    # Determine the correct Y-axis label from the actual data
    if is_memory_category:
        y_axis_label = "Value (B)"
    elif metric_order and get_metric_unit(metric_order[0]) == "J":
        y_axis_label = "Value (J)"
    elif metric_order and get_metric_unit(metric_order[0]) == "ms":
        y_axis_label = "Value (ms)"
    else:
        y_axis_label = "Value"
    
    if share_yaxis:
        # Calculate global Y-range across ALL visible data
        global_y_min = visible_data["value"].min()
        global_y_max = visible_data["value"].max()
        y_range_val = global_y_max - global_y_min if global_y_max != global_y_min else abs(global_y_max) if global_y_max != 0 else 1
        y_padding = 0.1 * y_range_val if y_range_val > 0 else 0.1
        
        calc_min = global_y_min - y_padding
        calc_max = global_y_max + y_padding
        if calc_min >= calc_max:
            calc_min = global_y_min - 0.1 if global_y_min != 0 else -0.1
            calc_max = global_y_max + 0.1 if global_y_max != 0 else 0.1
        
        # For memory metrics, ensure minimum is not negative
        if is_memory_category:
            calc_min = max(0, calc_min)
        
        # Calculate shared tick values for memory metrics
        shared_tickvals = None
        shared_ticktext = None
        if is_memory_category:
            shared_tickvals, shared_ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)
        
        # Apply shared Y-range to ALL subplots
        for subplot_idx in range(len(metric_order)):
            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            if yaxis_key in layout:
                layout[yaxis_key]["range"] = [calc_min, calc_max]
                layout[yaxis_key]["autorange"] = False
                # Ensure correct Y-axis title
                layout[yaxis_key]["title"] = {"text": y_axis_label}
                # Apply consistent tick formatting for memory metrics
                if is_memory_category and shared_tickvals is not None:
                    layout[yaxis_key]["tickvals"] = shared_tickvals
                    layout[yaxis_key]["ticktext"] = shared_ticktext
                else:
                    # Clear custom tick values for non-memory metrics
                    layout[yaxis_key].pop("tickvals", None)
                    layout[yaxis_key].pop("ticktext", None)
    else:
        # Calculate Y-range independently for each subplot
        for subplot_idx in range(len(metric_order)):
            metric_id = metric_order[subplot_idx]
            
            # Filter visible data for this specific metric
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
            
            # For memory metrics, ensure minimum is not negative
            if is_memory_category:
                calc_min = max(0, calc_min)
            
            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            if yaxis_key in layout:
                layout[yaxis_key]["range"] = [calc_min, calc_max]
                layout[yaxis_key]["autorange"] = False
                # Ensure correct Y-axis title
                layout[yaxis_key]["title"] = {"text": y_axis_label}
                
                # Update tick formatting for memory metrics, clear for non-memory
                if is_memory_category:
                    tickvals, ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)
                    layout[yaxis_key]["tickvals"] = tickvals
                    layout[yaxis_key]["ticktext"] = ticktext
                else:
                    layout[yaxis_key].pop("tickvals", None)
                    layout[yaxis_key].pop("ticktext", None)
    
    updated_figure["layout"] = layout
    return updated_figure


# Callback to update Y-axis ranges when zooming on X-axis
# This rescales each subplot's Y-axis based on visible data (shared or independent)
@app.callback(
    Output("timeseries-graph", "figure", allow_duplicate=True),
    Input("timeseries-graph", "relayoutData"),
    State("timeseries-graph", "figure"),
    State("timeseries-filtered-df-store", "data"),
    State("shared-yaxis-toggle", "value"),
    prevent_initial_call=True,
)
def update_yaxis_on_zoom(relayout_data, current_figure, filtered_df_store, shared_yaxis_toggle):
    """Update Y-axis ranges when X-axis is zoomed to show visible data range.
    
    If shared Y-axis is enabled, all subplots use the same Y-range based on global min/max.
    Otherwise, each subplot is updated independently based on its own visible data.
    """
    if not relayout_data or not current_figure or not filtered_df_store:
        return current_figure
    
    # Extract data and metric order from the store
    if not isinstance(filtered_df_store, dict) or "cache_id" not in filtered_df_store:
        return current_figure
    
    cache_id = filtered_df_store.get("cache_id")
    metric_order = filtered_df_store.get("metric_order", [])
    
    if not cache_id or not metric_order:
        return current_figure
    
    # Check for autorange resets (double-click to reset)
    is_reset = any('autorange' in key or 'autosize' in key for key in relayout_data)
    if is_reset:
        # Determine the category from the actual metric data
        is_memory_category = metric_order and is_memory_metric(metric_order[0])
        
        if not is_memory_category:
            # For non-memory metrics, Plotly's built-in autorange handles reset fine
            return current_figure
        
        # For memory metrics, we must actively reset Y-axis ranges and tick labels
        # because custom tickvals/ticktext prevent Plotly from auto-resetting properly
        df = load_cached_dataframe(cache_id)
        _ensure_timestamp_datetime(df)
        
        updated_figure = copy.deepcopy(current_figure)
        layout = updated_figure.get("layout", {})
        
        share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle
        
        if share_yaxis:
            # Shared: compute global range across all metrics
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
            # Independent: compute range per metric
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
    
    # Find all X-axis range changes in relayoutData
    # With independent X-axes, each subplot has its own xaxis:
    # - xaxis (row 1, subplot index 0)
    # - xaxis2 (row 2, subplot index 1)
    # - xaxis3 (row 3, subplot index 2)
    # etc.
    
    # Collect all changed X-axis ranges and their corresponding subplot indices
    xaxis_changes = {}  # Maps subplot_index -> (x_min, x_max)
    
    for key in relayout_data:
        if 'xaxis' in key and '.range[0]' in key:
            # Extract axis name (e.g., "xaxis", "xaxis2", "xaxis3")
            axis_name = key.replace('.range[0]', '')
            range_0_key = f"{axis_name}.range[0]"
            range_1_key = f"{axis_name}.range[1]"
            
            if range_0_key in relayout_data and range_1_key in relayout_data:
                # Determine subplot index from axis name
                # xaxis -> index 0, xaxis2 -> index 1, xaxis3 -> index 2, etc.
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
    
    # Load filtered data from Parquet cache
    df = load_cached_dataframe(cache_id)
    _ensure_timestamp_datetime(df)
    
    if df.empty:
        return current_figure
    
    df_tz = df["timestamp"].dt.tz
    
    # Deep copy the figure to avoid modifying shared nested dicts
    updated_figure = copy.deepcopy(current_figure)
    layout = updated_figure.get("layout", {})
    
    # Check if shared Y-axis is enabled
    share_yaxis = shared_yaxis_toggle and "shared" in shared_yaxis_toggle
    
    # Get the zoomed X range (use first changed axis, since X-axes are shared)
    first_subplot_idx = list(xaxis_changes.keys())[0]
    x_min, x_max = xaxis_changes[first_subplot_idx]
    
    # Ensure timezone compatibility for the X range
    if df_tz is not None:
        if x_min.tz is None:
            x_min = x_min.tz_localize(df_tz)
        if x_max.tz is None:
            x_max = x_max.tz_localize(df_tz)
    else:
        if x_min.tz is not None:
            x_min = x_min.tz_convert(None) if hasattr(x_min, 'tz_convert') else x_min.replace(tzinfo=None)
        if x_max.tz is not None:
            x_max = x_max.tz_convert(None) if hasattr(x_max, 'tz_convert') else x_max.replace(tzinfo=None)
    
    # Determine the category from the actual metric data
    is_memory_category = metric_order and is_memory_metric(metric_order[0])
    
    if share_yaxis:
        # Calculate global Y-range across ALL visible data from ALL metrics
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
        
        # For memory metrics, ensure minimum is not negative
        if is_memory_category:
            calc_min = max(0, calc_min)
        
        # Calculate shared tick values for memory metrics
        shared_tickvals = None
        shared_ticktext = None
        if is_memory_category:
            shared_tickvals, shared_ticktext = get_bytes_tickvals_ticktext(calc_min, calc_max, num_ticks=5)
        
        # Apply shared Y-range to ALL subplots
        for subplot_idx in range(len(metric_order)):
            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            if yaxis_key in layout:
                layout[yaxis_key]["range"] = [calc_min, calc_max]
                layout[yaxis_key]["autorange"] = False
                # Apply consistent tick formatting for memory metrics, clear for non-memory
                if is_memory_category and shared_tickvals is not None:
                    layout[yaxis_key]["tickvals"] = shared_tickvals
                    layout[yaxis_key]["ticktext"] = shared_ticktext
                else:
                    layout[yaxis_key].pop("tickvals", None)
                    layout[yaxis_key].pop("ticktext", None)
    else:
        # Update Y-axis independently for each subplot based on its own visible data
        for subplot_idx in range(len(metric_order)):
            metric_id = metric_order[subplot_idx]
            
            # Filter data for this specific metric and visible X range
            metric_data = df[df["metric_id"] == metric_id]
            metric_visible = metric_data[(metric_data["timestamp"] >= x_min) & (metric_data["timestamp"] <= x_max)]
            
            if metric_visible.empty:
                continue
            
            # Calculate Y range for this metric based on visible data
            y_min_val = metric_visible["value"].min()
            y_max_val = metric_visible["value"].max()
            y_range_val = y_max_val - y_min_val if y_max_val != y_min_val else abs(y_max_val) if y_max_val != 0 else 1
            y_padding = 0.1 * y_range_val if y_range_val > 0 else 0.1
            
            # Ensure min < max
            calc_min = y_min_val - y_padding
            calc_max = y_max_val + y_padding
            if calc_min >= calc_max:
                calc_min = y_min_val - 0.1 if y_min_val != 0 else -0.1
                calc_max = y_max_val + 0.1 if y_max_val != 0 else 0.1
            
            # For memory metrics, ensure minimum is not negative
            if is_memory_category:
                calc_min = max(0, calc_min)
            
            # Update this subplot's y-axis
            yaxis_key = "yaxis" if subplot_idx == 0 else f"yaxis{subplot_idx + 1}"
            
            if yaxis_key in layout:
                layout[yaxis_key]["range"] = [calc_min, calc_max]
                layout[yaxis_key]["autorange"] = False
                
                # Update tick formatting for memory metrics, clear for non-memory
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
    Output("ps-xy-graph", "figure"),
    Input("ps-xmetric-dropdown", "value"),
    Input("ps-ymetric-dropdown", "value"),
    Input("scatter-toggle", "value"),
    Input("theme-switch", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def update_process_xy_plot(x_metric_id, y_metric_id, scatter_toggle, use_light_mode, processed_df_data, process_time_range):
    fig = go.Figure()
    fig.update_layout(
        margin=dict(l=70, r=70, t=60, b=60),
    )
    apply_figure_theme(fig, use_light_mode)

    if not processed_df_data or not process_time_range or not x_metric_id or not y_metric_id:
        fig.update_layout(title=dict(text="Select both metrics", x=0.5))
        return fig

    dfp = df_from_store(processed_df_data)
    _ensure_timestamp_datetime(dfp)

    proc_start = pd.to_datetime(process_time_range["start"]) if process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range["end"]) if process_time_range.get("end") else None
    if proc_start is None or proc_end is None:
        fig.update_layout(title=dict(text="Process time range not available", x=0.5))
        return fig

    # Truncate to process window
    dfw = dfp[(dfp["timestamp"] >= proc_start) & (dfp["timestamp"] <= proc_end)].copy()

    dfx = dfw[dfw["metric_id"].astype(str) == str(x_metric_id)][["timestamp", "value"]].rename(columns={"value": "x"})
    dfy = dfw[dfw["metric_id"].astype(str) == str(y_metric_id)][["timestamp", "value"]].rename(columns={"value": "y"})

    if dfx.empty or dfy.empty:
        fig.update_layout(title=dict(text="No samples for one (or both) metrics in process window", x=0.5))
        return fig

    # Remove duplicate timestamps and sort
    dfx = dfx.drop_duplicates(subset=["timestamp"], keep="first")
    dfy = dfy.drop_duplicates(subset=["timestamp"], keep="first")
    dfx = dfx.sort_values("timestamp", ascending=True, ignore_index=True)
    dfy = dfy.sort_values("timestamp", ascending=True, ignore_index=True)

    # Auto tolerance for asof matching: ~2x the larger median sampling interval
    dx = dfx["timestamp"].diff().median()
    dy = dfy["timestamp"].diff().median()
    tol = None
    if pd.notna(dx) or pd.notna(dy):
        base = max([v for v in [dx, dy] if pd.notna(v)], default=pd.Timedelta(milliseconds=0))
        tol = base * 2 if base > pd.Timedelta(0) else pd.Timedelta(seconds=1)

    # Align: for each x timestamp, grab the nearest y (within tolerance)
    dfxy = pd.merge_asof(
        dfx,
        dfy,
        on="timestamp",
        direction="nearest",
        tolerance=tol,
    ).dropna(subset=["y"])
    
    # Sort by timestamp
    dfxy = dfxy.sort_values("timestamp", ascending=True, ignore_index=True)

    if dfxy.empty:
        fig.update_layout(title=dict(text="Could not align metrics in time (no matches within tolerance)", x=0.5))
        return fig

    # Metric names for display (remove resource/consumer suffixes)
    x_abbrev = x_metric_id.split("_R_")[0] if "_R_" in str(x_metric_id) else str(x_metric_id)
    y_abbrev = y_metric_id.split("_R_")[0] if "_R_" in str(y_metric_id) else str(y_metric_id)
    
    # Get units for axis labels
    x_unit = get_metric_unit(x_metric_id)
    y_unit = get_metric_unit(y_metric_id)
    x_label = f"{x_abbrev} ({x_unit})" if x_unit else x_abbrev
    y_label = f"{y_abbrev} ({y_unit})" if y_unit else y_abbrev
    
    # Check if both metrics are cumulative
    x_cumulative = is_cumulative_metric(x_metric_id)
    y_cumulative = is_cumulative_metric(y_metric_id)
    both_cumulative = x_cumulative and y_cumulative
    
    # Check if scatter plot mode is enabled
    show_scatter = scatter_toggle and "scatter" in scatter_toggle
    
    # Colors for the two metrics
    color_x = "#88C0D0"  # Nord cyan - for metric 1
    color_y = "#FF6B6B"  # Bright coral - for metric 2
    
    # Format timestamps for hover display
    hover_times = dfxy["timestamp"].dt.strftime("%H:%M:%S.%f").str[:-3]
    
    if show_scatter:
        # Scatter plot mode - use raw values to show relationship between metrics
        fig.add_trace(
            go.Scatter(
                x=dfxy["x"],
                y=dfxy["y"],
                mode="markers",
                name="Data Points",
                marker=dict(
                    color="#FF8C42",  # Bright orange
                    size=10,
                    opacity=0.85,
                    line=dict(width=1, color="#FFFFFF"),  # White border for better visibility
                ),
                hovertemplate=(
                    "<b>Time:</b> %{customdata}<br>"
                    f"<b>{x_abbrev}:</b> %{{x:.4f}}<br>"
                    f"<b>{y_abbrev}:</b> %{{y:.4f}}"
                    "<extra></extra>"
                ),
                customdata=hover_times,
            )
        )
        
        # Build axis configs with optional byte tick formatting
        xaxis_config = dict(title=dict(text=x_label, font=dict(size=11)), gridcolor="rgba(76, 86, 106, 0.2)")
        yaxis_config = dict(title=dict(text=y_label, font=dict(size=11)), gridcolor="rgba(76, 86, 106, 0.2)")
        x_is_memory = is_memory_metric(x_metric_id)
        y_is_memory = is_memory_metric(y_metric_id)
        if x_is_memory:
            x_tickvals, x_ticktext = get_bytes_tickvals_ticktext(dfxy["x"].min(), dfxy["x"].max(), num_ticks=5)
            xaxis_config["tickvals"] = x_tickvals
            xaxis_config["ticktext"] = x_ticktext
        if y_is_memory:
            y_tickvals, y_ticktext = get_bytes_tickvals_ticktext(dfxy["y"].min(), dfxy["y"].max(), num_ticks=5)
            yaxis_config["tickvals"] = y_tickvals
            yaxis_config["ticktext"] = y_ticktext
        
        fig.update_layout(
            title=dict(text=f"Scatter plot: {y_abbrev} vs {x_abbrev}", x=0.5, font=dict(size=14)),
            xaxis=xaxis_config,
            yaxis=yaxis_config,
            hovermode="closest",
        )
        
    elif both_cumulative:
        # Cumulative X-Y line plot mode
        # Compute cumsum and plot as time-ordered line plot
        
        dfxy["x_cumsum"] = dfxy["x"].cumsum()
        dfxy["y_cumsum"] = dfxy["y"].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=dfxy["x_cumsum"],
                y=dfxy["y_cumsum"],
                mode="lines+markers",
                line=dict(color="#A3BE8C", width=2),
                marker=dict(color="#A3BE8C", size=6),
                hovertemplate=(
                    "<b>Time:</b> %{customdata}<br>"
                    f"<b>Cumulative {x_abbrev}:</b> %{{x:.4f}}<br>"
                    f"<b>Cumulative {y_abbrev}:</b> %{{y:.4f}}"
                    "<extra></extra>"
                ),
                customdata=hover_times,
            )
        )
        
        # Cumulative axis labels with units (memory metrics show units in tick labels)
        x_cum_label = f"Cumulative {x_abbrev} ({x_unit})" if x_unit else f"Cumulative {x_abbrev}"
        y_cum_label = f"Cumulative {y_abbrev} ({y_unit})" if y_unit else f"Cumulative {y_abbrev}"
        
        # Build axis configs with optional byte tick formatting
        xaxis_config = dict(title=dict(text=x_cum_label, font=dict(size=11)), gridcolor="rgba(76, 86, 106, 0.2)")
        yaxis_config = dict(title=dict(text=y_cum_label, font=dict(size=11)), gridcolor="rgba(76, 86, 106, 0.2)")
        
        x_is_memory = is_memory_metric(x_metric_id)
        y_is_memory = is_memory_metric(y_metric_id)
        if x_is_memory:
            x_tickvals, x_ticktext = get_bytes_tickvals_ticktext(dfxy["x_cumsum"].min(), dfxy["x_cumsum"].max(), num_ticks=5)
            xaxis_config["tickvals"] = x_tickvals
            xaxis_config["ticktext"] = x_ticktext
        if y_is_memory:
            y_tickvals, y_ticktext = get_bytes_tickvals_ticktext(dfxy["y_cumsum"].min(), dfxy["y_cumsum"].max(), num_ticks=5)
            yaxis_config["tickvals"] = y_tickvals
            yaxis_config["ticktext"] = y_ticktext
        
        fig.update_layout(
            title=dict(text=f"Cumulative {y_abbrev} vs Cumulative {x_abbrev}", x=0.5, font=dict(size=14)),
            xaxis=xaxis_config,
            yaxis=yaxis_config,
            hovermode="closest",
        )
        
    else:
        # Dual Y-axis time series mode        
        # Plot first metric on left Y-axis
        fig.add_trace(
            go.Scatter(
                x=dfxy["timestamp"],
                y=dfxy["x"],
                mode="lines+markers",
                name=x_abbrev,
                line=dict(color=color_x, width=2),
                marker=dict(color=color_x, size=6),
                yaxis="y1",
                hovertemplate=f"<b>{x_abbrev}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )
        
        # Plot second metric on right Y-axis
        fig.add_trace(
            go.Scatter(
                x=dfxy["timestamp"],
                y=dfxy["y"],
                mode="lines+markers",
                name=y_abbrev,
                line=dict(color=color_y, width=2),
                marker=dict(color=color_y, size=6),
                yaxis="y2",
                hovertemplate=f"<b>{y_abbrev}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )
        
        # Build y axis configs with optional byte tick formatting
        yaxis_config = dict(
            title=dict(text=x_label, font=dict(size=11, color=color_x)),
            tickfont=dict(color=color_x),
            gridcolor="rgba(76, 86, 106, 0.2)",
            side="left",
        )
        yaxis2_config = dict(
            title=dict(text=y_label, font=dict(size=11, color=color_y)),
            tickfont=dict(color=color_y),
            overlaying="y",
            side="right",
            showgrid=False,
        )
        
        x_is_memory = is_memory_metric(x_metric_id)
        y_is_memory = is_memory_metric(y_metric_id)
        if x_is_memory:
            x_tickvals, x_ticktext = get_bytes_tickvals_ticktext(dfxy["x"].min(), dfxy["x"].max(), num_ticks=5)
            yaxis_config["tickvals"] = x_tickvals
            yaxis_config["ticktext"] = x_ticktext
        if y_is_memory:
            y_tickvals, y_ticktext = get_bytes_tickvals_ticktext(dfxy["y"].min(), dfxy["y"].max(), num_ticks=5)
            yaxis2_config["tickvals"] = y_tickvals
            yaxis2_config["ticktext"] = y_ticktext
        
        fig.update_layout(
            title=dict(text=f"Time Series: {x_abbrev} & {y_abbrev}", x=0.5, font=dict(size=14)),
            xaxis=dict(title=dict(text="Time", font=dict(size=12)), gridcolor="rgba(76, 86, 106, 0.2)", domain=[0.05, 0.95]),
            yaxis=yaxis_config,
            yaxis2=yaxis2_config,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,  # Position below x-axis
                xanchor="center",
                x=0.5,
                bgcolor="rgba(59, 66, 82, 0.8)",
            ),
            margin=dict(b=80),  # Add bottom margin to accommodate legend
            hovermode="x unified",
        )
    
    apply_figure_theme(fig, use_light_mode)
    return fig


# Callback to download CSV for grid plots (Process-specific panel)
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
    
    # Filter by metric (original df uses "metric" column, not "metric_id")
    dfm = df_original[df_original["metric"] == metric].copy()
    
    # Create normalized columns for filtering (same as update_filters_match)
    # Convert to string first (handles categorical columns), then replace "nan" with empty string
    dfm["rk"] = dfm["resource_kind"].astype(str).replace("nan", "").str.strip()
    dfm["rid"] = dfm["resource_id"].astype(str).replace("nan", "").str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(str).replace("nan", "").str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(str).replace("nan", "").str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(str).replace("nan", "").str.strip()
    
    # Normalize filter values
    def norm_val(v):
        return str(v).strip() if v else ""
    
    # Apply the same filters as the plot
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
    
    # Truncate to process window
    if process_time_range:
        proc_start = pd.to_datetime(process_time_range.get("start"))
        proc_end = pd.to_datetime(process_time_range.get("end"))
        if proc_start and proc_end:
            dfm = dfm[(dfm["timestamp"] >= proc_start) & (dfm["timestamp"] <= proc_end)]
    
    if dfm.empty:
        return None
    
    # Sort by timestamp
    dfm = dfm.sort_values("timestamp")
    
    # Select relevant columns for export (use original column names)
    export_cols = ["timestamp", "metric", "value"]
    # Add filter columns if they exist and have meaningful data
    for orig_col in ["resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]:
        if orig_col in dfm.columns and dfm[orig_col].notna().any():
            export_cols.append(orig_col)
    
    df_export = dfm[export_cols].copy()
    
    # Generate filename (sanitize metric name)
    safe_metric = "".join(c if c.isalnum() or c in "._-" else "_" for c in metric)
    filename = f"{safe_metric}_process_data.csv"
    
    return dcc.send_data_frame(df_export.to_csv, filename, index=False)


# Callback to download CSV for X-Y plot (Comparative panel)
@app.callback(
    Output("xy-download", "data"),
    Input("xy-download-btn", "n_clicks"),
    State("ps-xmetric-dropdown", "value"),
    State("ps-ymetric-dropdown", "value"),
    State("processed-df-store", "data"),
    State("process-time-range-store", "data"),
    prevent_initial_call=True,
)
def download_xy_csv(n_clicks, x_metric_id, y_metric_id, processed_df_data, process_time_range):
    """Generate and download CSV for the X-Y comparative plot."""
    if not n_clicks or not processed_df_data or not x_metric_id or not y_metric_id:
        return None
    
    dfp = df_from_store(processed_df_data)
    _ensure_timestamp_datetime(dfp)
    
    proc_start = pd.to_datetime(process_time_range.get("start")) if process_time_range and process_time_range.get("start") else None
    proc_end = pd.to_datetime(process_time_range.get("end")) if process_time_range and process_time_range.get("end") else None
    
    if proc_start is None or proc_end is None:
        return None
    
    # Truncate to process window
    dfw = dfp[(dfp["timestamp"] >= proc_start) & (dfp["timestamp"] <= proc_end)].copy()
    
    dfx = dfw[dfw["metric_id"].astype(str) == str(x_metric_id)][["timestamp", "value"]].rename(columns={"value": "x"})
    dfy = dfw[dfw["metric_id"].astype(str) == str(y_metric_id)][["timestamp", "value"]].rename(columns={"value": "y"})
    
    if dfx.empty or dfy.empty:
        return None
    
    # Remove duplicate timestamps
    dfx = dfx.drop_duplicates(subset=["timestamp"], keep="first")
    dfy = dfy.drop_duplicates(subset=["timestamp"], keep="first")
    dfx = dfx.sort_values("timestamp", ascending=True, ignore_index=True)
    dfy = dfy.sort_values("timestamp", ascending=True, ignore_index=True)
    
    # Auto tolerance for asof matching
    dx = dfx["timestamp"].diff().median()
    dy = dfy["timestamp"].diff().median()
    tol = None
    if pd.notna(dx) or pd.notna(dy):
        base = max([v for v in [dx, dy] if pd.notna(v)], default=pd.Timedelta(milliseconds=0))
        tol = base * 2 if base > pd.Timedelta(0) else pd.Timedelta(seconds=1)
    
    # Align metrics in time
    dfxy = pd.merge_asof(
        dfx,
        dfy,
        on="timestamp",
        direction="nearest",
        tolerance=tol,
    ).dropna(subset=["y"])
    
    # Sort by timestamp 
    dfxy = dfxy.sort_values("timestamp", ascending=True, ignore_index=True)
    
    if dfxy.empty:
        return None
    
    # Rename columns for clarity
    dfxy = dfxy.rename(columns={"x": x_metric_id, "y": y_metric_id})
    
    # Generate filename
    filename = f"xy_{x_metric_id}_vs_{y_metric_id}.csv"
    # Sanitize filename (remove special characters)
    filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    
    return dcc.send_data_frame(dfxy.to_csv, filename, index=False)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)