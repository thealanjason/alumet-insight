"""Lifecycle tab: callbacks for data loading, reset, process info, tab toggle, theme."""

import time

import dash
from dash import ClientsideFunction, Input, Output, State, html
from pathlib import Path

from frontend.app import app
from frontend.cache import cache_dataframe
from frontend.style import status_alert
from backend.data import AlumetData
from backend.utils import find_measurement_file_in_directory, save_upload_to_temp_dir


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

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


@app.callback(
    Output("upload-status", "children"),
    Input("directory-upload", "filename"),
)
def update_upload_status(filenames):
    """Show how many files were staged in the upload control (load still needs Visualize)."""
    if not filenames:
        return ""
    names = [filenames] if isinstance(filenames, str) else list(filenames)
    count = len(names)
    return f"Selected: {count} file{'s' if count != 1 else ''}"


# Reset
@app.callback(
    Output("directory-path-input", "value", allow_duplicate=True),
    Output("directory-upload", "contents"),
    Output("directory-upload", "filename"),
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
        None,
        None,
        "Name: N/A",
        "Process ID: N/A",
        "Device: N/A",
        _ready_status(),
    )


def _ready_status():
    return status_alert(
        "warning",
        "Ready to load",
        [
            "Upload an experiment folder or enter a server directory path, then click ",
            html.Strong("Visualize"),
            " or press Enter to load and visualize data.",
        ],
    )


# Load, visualize, and update process info.
# Do not listen to directory-path-input n_blur: leaving the field to click a
# results tab (or any other control) would re-run a full AlumetData load.
@app.callback(
    Output("status-message", "children"),
    Output("processed-df-store", "data"),
    Output("original-df-store", "data"),
    Output("process-time-range-store", "data"),
    Output("experiment-name-display", "children"),
    Output("pid-display", "children"),
    Output("device-display", "children"),
    Input("visualize-button", "n_clicks"),
    Input("directory-path-input", "n_submit"),
    State("directory-path-input", "value"),
    State("directory-upload", "contents"),
    State("directory-upload", "filename"),
)
def load_and_visualize(n_clicks, n_submit, directory_path, upload_contents, upload_filenames):
    _no_info = ("Name: N/A", "Process ID: N/A", "Device: N/A")

    if not any([n_clicks, n_submit]):
        return (_ready_status(), None, None, None, *_no_info)

    has_upload = bool(upload_contents) and bool(upload_filenames)
    has_path = bool(directory_path and directory_path.strip())

    if not has_upload and not has_path:
        status_msg = status_alert(
            "danger",
            "Error:",
            "Provide an uploaded folder or a server directory path.",
        )
        return status_msg, None, None, None, *_no_info

    try:
        if has_upload:
            dir_path, experiment_name = save_upload_to_temp_dir(upload_contents, upload_filenames)
        else:
            dir_path = Path(directory_path.strip())
            if not dir_path.exists():
                status_msg = status_alert(
                    "danger",
                    "Error:",
                    f"Directory does not exist: {directory_path}",
                )
                return status_msg, None, None, None, *_no_info

            if not dir_path.is_dir():
                status_msg = status_alert(
                    "danger",
                    "Error:",
                    f"Path is not a directory: {directory_path}",
                )
                return status_msg, None, None, None, *_no_info
            experiment_name = dir_path.name or "N/A"

        try:
            csv_file = find_measurement_file_in_directory(str(dir_path), [".csv"])
        except ValueError:
            csv_file = None
        if not csv_file:
            status_msg = status_alert(
                "danger",
                "Error:",
                "CSV file is required. Please ensure the folder contains a .csv file.",
            )
            return status_msg, None, None, None, *_no_info

        t0 = time.perf_counter()
        data = AlumetData(str(dir_path))
        t_load = time.perf_counter()

        processed_cache_id = cache_dataframe(data.processed_df, prefix="processed")
        original_cache_id = cache_dataframe(data.source_df, prefix="original")
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

        pid = data.pid
        device = data.device

        return (
            status_msg,
            processed_cache_id,
            original_cache_id,
            process_time_range,
            f"Name: {experiment_name}",
            f"Process ID: {pid or 'N/A'}",
            f"Device: {device}",
        )

    except Exception as e:
        status_msg = status_alert(
            "danger",
            "Error loading data:",
            str(e),
            icon="\u274c ",
        )
        return status_msg, None, None, None, *_no_info


# Tab visibility and process-specific viewport sizing (see assets/process_grid_layout.js)
app.clientside_callback(
    ClientsideFunction(namespace="process_grid", function_name="toggleTabPanels"),
    Output("time-series-content", "style"),
    Output("process-specific-content", "style"),
    Output("comparative-content", "style"),
    Input("results-tabs", "value"),
)

app.clientside_callback(
    ClientsideFunction(namespace="process_grid", function_name="afterGridBuild"),
    Output("process-grid-layout-ts", "data"),
    Input("process-specific-content", "children"),
)
