"""Lifecycle tab: callbacks for data loading, reset, process info, tab toggle, theme."""

import shutil

import dash
from dash import ClientsideFunction, Input, Output, State
from pathlib import Path

from frontend.app import app
from frontend.cache import cache_dataframe, cache_id_from_store, delete_cached_dataframe
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
        if (window.restylePlotlyTheme) {
            window.restylePlotlyTheme(!!useLightMode);
        }
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
    Output("visualize-button", "disabled"),
    Output("visualize-button", "children"),
    Input("directory-upload", "filename"),
    Input("upload-relative-paths", "data"),
    Input("upload-temp-dir-store", "data"),
    Input("load-source-mode", "value"),
)
def update_upload_control(filenames, relative_paths, staged, load_mode):
    """Show the chosen folder, and keep Visualize locked until the server has it."""
    names = prefer_relative_upload_paths(filenames, relative_paths)
    ready = isinstance(staged, dict) and bool(staged.get("path"))
    failed = isinstance(staged, dict) and bool(staged.get("error")) and not ready
    waiting = load_mode == LOAD_SOURCE_UPLOAD and bool(names) and not ready and not failed
    children = (
        upload_prompt_children()
        if not names
        else upload_selected_children(experiment_name_from_upload_filenames(names))
    )
    return children, waiting, "Receiving..." if waiting else "Visualize"


def _delete_store_caches(*store_payloads) -> None:
    """Drop server-side cache entries referenced by dcc.Store payloads."""
    for payload in store_payloads:
        delete_cached_dataframe(cache_id_from_store(payload))


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
    Output("process-time-range-store", "data", allow_duplicate=True),
    Output("timeseries-filtered-df-store", "data", allow_duplicate=True),
    Output("experiment-name-display", "children", allow_duplicate=True),
    Output("pid-display", "children", allow_duplicate=True),
    Output("device-display", "children", allow_duplicate=True),
    Output("status-message", "children", allow_duplicate=True),
    Input("reset-button", "n_clicks"),
    State("load-source-mode", "value"),
    State("processed-df-store", "data"),
    State("timeseries-filtered-df-store", "data"),
    prevent_initial_call=True,
)
def reset_app(n_clicks, load_mode, processed_df_data, filtered_df_data):
    """Reset the application to its initial state."""
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    _delete_store_caches(processed_df_data, filtered_df_data)

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
        _ready_status(load_mode),
    )


def _discard_upload_dir(store_data) -> None:
    path = store_data.get("path") if isinstance(store_data, dict) else None
    if path:
        shutil.rmtree(path, ignore_errors=True)


@app.callback(
    Output("upload-temp-dir-store", "data"),
    Output("status-message", "children", allow_duplicate=True),
    Input("directory-upload", "contents"),
    State("directory-upload", "filename"),
    State("upload-relative-paths", "data"),
    State("upload-temp-dir-store", "data"),
    prevent_initial_call=True,
)
def stage_uploaded_folder(contents, filenames, relative_paths, previous):
    """Write the folder once, when it is chosen, so Visualize does not receive it again."""
    _discard_upload_dir(previous)
    if not contents:
        return None, dash.no_update
    names = prefer_relative_upload_paths(filenames, relative_paths)
    try:
        dir_path, experiment_name = save_upload_to_temp_dir(contents, names)
    except (ValueError, OSError) as exc:
        return {"error": str(exc)}, status_alert("danger", "Error:", str(exc))
    return (
        {"path": str(dir_path), "name": experiment_name},
        status_alert("success", "Folder received"),
    )


@app.callback(
    Output("upload-temp-dir-store", "data", allow_duplicate=True),
    Input("upload-relative-paths", "data"),
    State("directory-upload", "filename"),
    State("upload-temp-dir-store", "data"),
    prevent_initial_call=True,
)
def name_staged_upload(relative_paths, filenames, previous):
    """Fill the experiment name once folder prefixes arrive, without resending the files."""
    if not relative_paths or not isinstance(previous, dict) or not previous.get("path"):
        raise dash.exceptions.PreventUpdate
    name = experiment_name_from_upload_filenames(
        prefer_relative_upload_paths(filenames, relative_paths)
    )
    if not name or name == previous.get("name"):
        raise dash.exceptions.PreventUpdate
    return {**previous, "name": name}


# Load, visualize, and update process info.
# Do not listen to directory-path-input n_blur: leaving the field to click a
# results tab (or any other control) would re-run a full AlumetData load.
@app.callback(
    Output("status-message", "children"),
    Output("processed-df-store", "data"),
    Output("process-time-range-store", "data"),
    Output("experiment-name-display", "children"),
    Output("pid-display", "children"),
    Output("device-display", "children"),
    Input("visualize-button", "n_clicks"),
    Input("directory-path-input", "n_submit"),
    State("load-source-mode", "value"),
    State("directory-path-input", "value"),
    State("directory-upload", "filename"),
    State("upload-temp-dir-store", "data"),
    State("processed-df-store", "data"),
    State("timeseries-filtered-df-store", "data"),
)
def load_and_visualize(
    n_clicks,
    n_submit,
    load_mode,
    directory_path,
    upload_filenames,
    upload_temp_dir,
    previous_processed,
    previous_filtered,
):
    _no_info = ("Name: N/A", "Process ID: N/A", "Device: N/A")
    previous_stores = (previous_processed, previous_filtered)
    triggered = dash.callback_context.triggered_id

    def _cleared(status_msg):
        _delete_store_caches(*previous_stores)
        return status_msg, None, None, *_no_info

    if triggered is None or not any([n_clicks, n_submit]):
        return (_ready_status(load_mode), None, None, *_no_info)

    # Enter in the path field should only load while Server path is active.
    if triggered == "directory-path-input" and load_mode != LOAD_SOURCE_PATH:
        raise dash.exceptions.PreventUpdate

    use_upload = load_mode == LOAD_SOURCE_UPLOAD
    staged = upload_temp_dir if isinstance(upload_temp_dir, dict) else {}
    has_upload = bool(staged.get("path"))
    has_path = bool(directory_path and directory_path.strip())

    if use_upload and staged.get("error"):
        return _cleared(status_alert("danger", "Error:", staged["error"]))

    if use_upload and not has_upload:
        if upload_filenames:
            raise dash.exceptions.PreventUpdate
        return _cleared(status_alert("danger", "Error:", "upload a folder, then Visualize"))

    if not use_upload and not has_path:
        return _cleared(status_alert("danger", "Error:", "enter a path, then Visualize"))

    try:
        if use_upload:
            dir_path = Path(staged["path"])
            experiment_name = staged.get("name") or "N/A"
            if not dir_path.is_dir():
                return _cleared(status_alert("danger", "Error:", "directory does not exist"))
        else:
            dir_path = Path(directory_path.strip())
            if not dir_path.exists():
                return _cleared(status_alert("danger", "Error:", "directory does not exist"))

            if not dir_path.is_dir():
                return _cleared(status_alert("danger", "Error:", "path is not a directory"))
            experiment_name = dir_path.name or "N/A"

        try:
            csv_file = find_measurement_file_in_directory(str(dir_path), [".csv"])
        except ValueError:
            csv_file = None
        if not csv_file:
            return _cleared(status_alert("danger", "Error:", "folder must contain a .csv file"))

        data = AlumetData(str(dir_path))

        # Replace prior dataset caches only after a successful load.
        _delete_store_caches(*previous_stores)

        proc_start, proc_end = data.process_time_range
        pid = data.pid
        device = data.device
        processed_cache_id = cache_dataframe(
            data.handoff_processed_df(), prefix="processed", copy=False
        )

        status_msg = status_alert("success", "Data loaded successfully")

        process_time_range = {
            "start": proc_start.isoformat() if proc_start else None,
            "end": proc_end.isoformat() if proc_end else None,
        }

        return (
            status_msg,
            processed_cache_id,
            process_time_range,
            f"Name: {experiment_name}",
            f"Process ID: {pid or 'N/A'}",
            f"Device: {device}",
        )

    except Exception as e:
        return _cleared(status_alert("danger", "Error:", str(e)))


@app.callback(
    Output("timeseries-filtered-df-store", "data", allow_duplicate=True),
    Input("processed-df-store", "data"),
    State("timeseries-filtered-df-store", "data"),
    prevent_initial_call=True,
)
def clear_filtered_on_dataset_change(_processed_df_data, previous_filtered):
    """Drop the filtered-store reference whenever the loaded dataset changes."""
    _delete_store_caches(previous_filtered)


# Tab visibility and viewport sizing (see assets/tab_panel_layout.js)
app.clientside_callback(
    ClientsideFunction(namespace="tab_panel", function_name="toggleTabPanels"),
    Output("time-series-content", "className"),
    Output("process-specific-content", "className"),
    Output("comparative-content", "className"),
    Output("tab-preparing-overlay", "style"),
    Input("results-tabs", "value"),
)

app.clientside_callback(
    ClientsideFunction(namespace="tab_panel", function_name="afterTabBuild"),
    Output("tab-panel-layout-ts", "data"),
    Output("tab-preparing-overlay", "style", allow_duplicate=True),
    Input("time-series-content", "children"),
    Input("process-specific-content", "children"),
    Input("comparative-content", "children"),
    State("results-tabs", "value"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(_tsChildren, processed) {
        if (window.bindTabHoverPrefetch) {
            window.bindTabHoverPrefetch();
        }
        if (window._tabPrefetchTimer) {
            clearTimeout(window._tabPrefetchTimer);
            window._tabPrefetchTimer = null;
        }
        if (!processed) {
            return window.dash_clientside.no_update;
        }
        window._tabPrefetchTimer = setTimeout(function () {
            if (window.plotUpdateInProgress && window.plotUpdateInProgress()) {
                return;
            }
            if (window.dash_clientside && window.dash_clientside.set_props) {
                window.dash_clientside.set_props("tab-prefetch-store", {data: Date.now()});
            }
        }, 1000);
        return window.dash_clientside.no_update;
    }
    """,
    Output("tab-prefetch-timer-output", "data"),
    Input("time-series-content", "children"),
    State("processed-df-store", "data"),
    prevent_initial_call=True,
)
