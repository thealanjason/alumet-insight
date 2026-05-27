"""Lifecycle callbacks: data loading, reset, process info, tab toggle, theme."""

import time

import dash
import pandas as pd
from dash import Input, Output, State, html
from pathlib import Path

from frontend.app import app
from frontend.cache import cache_dataframe
from frontend.theme import status_alert
from backend.data import (
    AlumetData,
    find_measurement_file_in_directory,
)

# Theme callbacks
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


# Reset
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

    return (
        "",
        None,
        None,
        None,
        None,
        "Name: N/A",
        "Process ID: N/A",
        "Device: N/A",
        status_alert(
            "warning",
            "Ready to load",
            [
                "Enter a directory path above, then click ",
                html.Strong("Visualize"),
                " or press Enter/Tab to load and visualize data.",
            ],
        ),
    )


# Process info
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
        data = AlumetData(directory_path.strip())
        pid = data.pid
        device = data.device
    except Exception:
        return "Name: N/A", "Process ID: N/A", "Device: N/A"

    return (
        f"Name: {experiment_name}",
        f"Process ID: {pid or 'N/A'}",
        f"Device: {device}",
    )


# Load and visualize
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

    if not directory_path or not directory_path.strip():
        status_msg = status_alert(
            "danger",
            "Error:",
            "Directory path is required. Please enter a directory path.",
        )
        return status_msg, None, None, None

    try:
        dir_path = Path(directory_path.strip())
        if not dir_path.exists():
            status_msg = status_alert(
                "danger",
                "Error:",
                f"Directory does not exist: {directory_path}",
            )
            return status_msg, None, None, None

        if not dir_path.is_dir():
            status_msg = status_alert(
                "danger",
                "Error:",
                f"Path is not a directory: {directory_path}",
            )
            return status_msg, None, None, None

        try:
            csv_file = find_measurement_file_in_directory(directory_path, [".csv"])
        except ValueError:
            csv_file = None
        try:
            log_file = find_measurement_file_in_directory(directory_path, [".log", ".txt"])
        except ValueError:
            log_file = None

        if not csv_file:
            status_msg = status_alert(
                "danger",
                "Error:",
                "CSV file is required. Please ensure the directory contains a .csv file.",
            )
            return status_msg, None, None, None

        if not log_file:
            status_msg = status_alert(
                "danger",
                "Error:",
                "Log file is required. Please ensure the directory contains a .log or .txt file.",
            )
            return status_msg, None, None, None

        t0 = time.perf_counter()
        data = AlumetData(directory_path.strip())
        t_load = time.perf_counter()

        processed_cache_id = cache_dataframe(data.processed_df, prefix="processed")
        original_cache_id = cache_dataframe(data.raw_df, prefix="original")
        t_cache = time.perf_counter()

        proc_start, proc_end = data.process_time_range

        load_time = t_load - t0
        cache_time = t_cache - t_load

        status_msg = status_alert(
            "success",
            "Data loaded successfully",
            f"load and preprocess: {load_time:.2f}s, cache: {cache_time:.2f}s",
            icon="\u2705 ",
            detail_style={"fontSize": "0.85rem", "color": "var(--app-text-muted)"},
        )

        process_time_range = {
            "start": proc_start.isoformat() if proc_start else None,
            "end": proc_end.isoformat() if proc_end else None,
        }

        return status_msg, processed_cache_id, original_cache_id, process_time_range

    except Exception as e:
        status_msg = status_alert(
            "danger",
            "Error loading data:",
            str(e),
            icon="\u274c ",
        )
        return status_msg, None, None, None


# Tab visibility toggle
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
