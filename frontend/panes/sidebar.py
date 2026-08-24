"""Lifecycle tab: callbacks for data loading, reset, process info, tab toggle, theme."""

import dash
from dash import ClientsideFunction, Input, Output, State
from pathlib import Path

from frontend.app import app
from frontend.cache import cache_dataframe
from frontend.layout import (
    LOAD_SOURCE_PATH,
    LOAD_SOURCE_UPLOAD,
    upload_prompt_children,
    upload_selected_children,
)
from frontend.style import status_alert
from backend.data import AlumetData
from backend.utils import (
    prefer_relative_upload_paths,
    experiment_name_from_upload_filenames,
    find_measurement_file_in_directory,
    save_upload_to_temp_dir,
)


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
    Output("upload-source-panel", "style"),
    Output("path-source-panel", "style"),
    Input("load-source-mode", "value"),
)
def toggle_load_source_panels(mode):
    """Show only the active load method to keep the sidebar compact."""
    show = {"display": "block"}
    hide = {"display": "none"}
    if mode == LOAD_SOURCE_UPLOAD:
        return show, hide
    return hide, show


@app.callback(
    Output("directory-upload", "children"),
    Input("directory-upload", "filename"),
    Input("upload-relative-paths", "data"),
)
def update_upload_control(filenames, relative_paths):
    """Swap the drop zone for the folder name once files are staged."""
    names = prefer_relative_upload_paths(filenames, relative_paths)
    if not names:
        return upload_prompt_children()
    return upload_selected_children(experiment_name_from_upload_filenames(names))


def _ready_status(load_mode=None):
    if load_mode == LOAD_SOURCE_UPLOAD:
        return status_alert("warning", "Ready to load", "upload a folder, then Visualize")
    if load_mode == LOAD_SOURCE_PATH:
        return status_alert("warning", "Ready to load", "enter a path, then Visualize")
    return status_alert("warning", "Ready to load")


@app.callback(
    Output("status-message", "children", allow_duplicate=True),
    Input("load-source-mode", "value"),
    State("processed-df-store", "data"),
    prevent_initial_call=True,
)
def update_ready_hint_on_mode_switch(load_mode, processed_df):
    """Refresh the ready hint when switching source; never clear loaded data."""
    if processed_df:
        raise dash.exceptions.PreventUpdate
    return _ready_status(load_mode)


# Reset
@app.callback(
    Output("directory-path-input", "value", allow_duplicate=True),
    Output("directory-upload", "contents"),
    Output("directory-upload", "filename"),
    Output("upload-relative-paths", "data"),
    Output("processed-df-store", "data", allow_duplicate=True),
    Output("original-df-store", "data", allow_duplicate=True),
    Output("process-time-range-store", "data", allow_duplicate=True),
    Output("timeseries-filtered-df-store", "data", allow_duplicate=True),
    Output("experiment-name-display", "children", allow_duplicate=True),
    Output("pid-display", "children", allow_duplicate=True),
    Output("device-display", "children", allow_duplicate=True),
    Output("status-message", "children", allow_duplicate=True),
    Input("reset-button", "n_clicks"),
    State("load-source-mode", "value"),
    prevent_initial_call=True,
)
def reset_app(n_clicks, load_mode):
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
        None,
        "Name: N/A",
        "Process ID: N/A",
        "Device: N/A",
        _ready_status(load_mode),
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
    State("load-source-mode", "value"),
    State("directory-path-input", "value"),
    State("directory-upload", "contents"),
    State("directory-upload", "filename"),
    State("upload-relative-paths", "data"),
)
def load_and_visualize(
    n_clicks,
    n_submit,
    load_mode,
    directory_path,
    upload_contents,
    upload_filenames,
    upload_relative_paths,
):
    _no_info = ("Name: N/A", "Process ID: N/A", "Device: N/A")
    triggered = dash.callback_context.triggered_id

    if triggered is None or not any([n_clicks, n_submit]):
        return (_ready_status(load_mode), None, None, None, *_no_info)

    # Enter in the path field should only load while Server path is active.
    if triggered == "directory-path-input" and load_mode != LOAD_SOURCE_PATH:
        raise dash.exceptions.PreventUpdate

    use_upload = load_mode == LOAD_SOURCE_UPLOAD
    upload_names = prefer_relative_upload_paths(upload_filenames, upload_relative_paths)
    has_upload = bool(upload_contents) and bool(upload_names)
    has_path = bool(directory_path and directory_path.strip())

    if use_upload and not has_upload:
        status_msg = status_alert("danger", "Error:", "upload a folder, then Visualize")
        return status_msg, None, None, None, *_no_info

    if not use_upload and not has_path:
        status_msg = status_alert("danger", "Error:", "enter a path, then Visualize")
        return status_msg, None, None, None, *_no_info

    try:
        if use_upload:
            dir_path, experiment_name = save_upload_to_temp_dir(upload_contents, upload_names)
        else:
            dir_path = Path(directory_path.strip())
            if not dir_path.exists():
                status_msg = status_alert("danger", "Error:", "directory does not exist")
                return status_msg, None, None, None, *_no_info

            if not dir_path.is_dir():
                status_msg = status_alert("danger", "Error:", "path is not a directory")
                return status_msg, None, None, None, *_no_info
            experiment_name = dir_path.name or "N/A"

        try:
            csv_file = find_measurement_file_in_directory(str(dir_path), [".csv"])
        except ValueError:
            csv_file = None
        if not csv_file:
            status_msg = status_alert("danger", "Error:", "folder must contain a .csv file")
            return status_msg, None, None, None, *_no_info

        data = AlumetData(str(dir_path))

        processed_cache_id = cache_dataframe(data.processed_df, prefix="processed")
        original_cache_id = cache_dataframe(data.source_df, prefix="original")

        proc_start, proc_end = data.process_time_range

        status_msg = status_alert("success", "Data loaded successfully")

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
        status_msg = status_alert("danger", "Error:", str(e))
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
