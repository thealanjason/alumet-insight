"""Filesystem and log-parsing utilities."""

from __future__ import annotations

import base64
import re
import tempfile
import warnings
import pandas as pd

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ALLOWED_UPLOAD_SUFFIXES = {".csv", ".log", ".txt", ".toml"}


def normalize_upload_filenames(
    filenames: str | Sequence[str] | None,
) -> list[str]:
    """Normalize upload filenames to a list of strings."""
    if not filenames:
        return []
    if isinstance(filenames, str):
        return [filenames]
    return [name for name in filenames if name]


def folder_from_upload_paths(
    filenames: str | Sequence[str] | None,
) -> str | None:
    """Return the top-level folder from relative paths, or None if only basenames."""
    for name in normalize_upload_filenames(filenames):
        parts = Path(name).parts
        if len(parts) >= 2 and parts[0] not in {".", ".."}:
            return parts[0]
    return None


def experiment_name_from_upload_filenames(
    filenames: str | Sequence[str] | None,
) -> str:
    """Same rule as server path: folder name, or N/A when it is unknown."""
    return folder_from_upload_paths(filenames) or "N/A"


def prefer_relative_upload_paths(
    filenames: str | Sequence[str] | None,
    relative_paths: str | Sequence[str] | None,
) -> list[str]:
    """
    Attach folder prefixes from webkitRelativePath onto Dash's accepted files.

    ``filenames`` and Dash's ``contents`` are paired positionally by Dash itself
    (both reflect FileReader completion order), so this must preserve that order
    and only annotate each name with its relative-path prefix. ``relative_paths``
    comes from a separate JS listener in FileList enumeration order, which need
    not match — never substitute it in wholesale, even when the lengths happen
    to be equal, or contents and names silently pair up with the wrong files.
    """
    dash_names = normalize_upload_filenames(filenames)
    rel_names = normalize_upload_filenames(relative_paths)
    if not dash_names:
        return []
    if not folder_from_upload_paths(rel_names):
        return dash_names

    rel_by_basename: dict[str, str] = {}
    for rel in rel_names:
        rel_by_basename[Path(rel).name] = rel
    return [rel_by_basename.get(Path(name).name, name) for name in dash_names]


def save_upload_to_temp_dir(
    contents: str | Sequence[str] | None,
    filenames: str | Sequence[str] | None,
) -> Tuple[Path, str]:
    """
    Decode browser-uploaded folder contents into a temporary directory.

    The experiment display name is taken from the shared top-level folder name when present.

    Returns:
        (directory_path, experiment_name)
    """
    if not contents or not filenames:
        raise ValueError("No files were uploaded.")

    content_list = [contents] if isinstance(contents, str) else list(contents)
    name_list = normalize_upload_filenames(filenames)
    if len(content_list) != len(name_list):
        raise ValueError("Uploaded contents and filenames are out of sync.")

    experiment_name = experiment_name_from_upload_filenames(name_list)

    target_dir = Path(tempfile.mkdtemp(prefix="alumet_upload_"))
    written = 0
    for content, filename in zip(content_list, name_list):
        if not content or not filename:
            continue
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            continue
        if "," not in content:
            raise ValueError(f"Unexpected upload payload for {filename}")
        _, content_string = content.split(",", 1)
        dest = target_dir / Path(filename).name
        dest.write_bytes(base64.b64decode(content_string))
        written += 1

    if written == 0:
        raise ValueError(
            "No supported files found in the upload. "
            "Expected .csv, .log/.txt, and optionally .toml."
        )
    return target_dir, experiment_name


def find_measurement_file_in_directory(directory_path: str, extensions: List[str]) -> Path:
    """
    Find measurement file with specified extensions in a directory.

    Args:
        directory_path: Path to the directory
        extensions: List of file extensions to search for

    Returns:
        Path object for matching file
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    found_files = []
    for ext in extensions:
        found_files.extend(dir_path.glob(f"*{ext}"))
    if not found_files:
        raise ValueError(f"No files found with extensions: {extensions} in directory: {directory_path}")
    if len(found_files) > 1:
        warnings.warn(
            f"Multiple files found with extensions: {extensions} in directory: {directory_path}. Returning the first one."
        )
    return sorted(found_files)[0]


def read_file_content(file_path: Path) -> str:
    """Read file content as string."""
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")
    return file_path.read_text(encoding="utf-8")


def extract_pid_from_content(log_content: str) -> Optional[int]:
    """Extract process ID from Alumet log file content."""
    if not log_content:
        return None
    for line in log_content.split("\n"):
        if "pid" in line:
            match = re.search(r"pid (\d+)", line)
            if match:
                return int(match.group(1))
    return None


def is_gpu_from_content(log_content: str) -> bool:
    """Detect whether the run used GPU-related Alumet plugins (NVML) from agent log text."""
    if not log_content:
        return False
    for line in log_content.split("\n"):
        if re.search(r"nvml", line, re.IGNORECASE):
            return True
    return False


def is_cpu_from_content(log_content: str) -> bool:
    """Detect whether the run used CPU-related Alumet plugins from agent log text."""
    if not log_content:
        return False
    for line in log_content.split("\n"):
        if re.search(r"rapl", line, re.IGNORECASE):
            return True
    return False


_GPU_METRIC_PATTERN = re.compile(r"nvml", re.IGNORECASE)
_CPU_METRIC_PATTERN = re.compile(r"rapl|cpu|kernel|perf|mem", re.IGNORECASE)


def is_gpu_from_metrics(df: pd.DataFrame) -> bool:
    """Detect GPU presence from metric names in the processed dataframe."""
    if df.empty or "base_metric" not in df.columns:
        return False
    return df["base_metric"].str.contains(_GPU_METRIC_PATTERN).any()


def is_cpu_from_metrics(df: pd.DataFrame) -> bool:
    """Detect CPU presence from metric names in the processed dataframe."""
    if df.empty or "base_metric" not in df.columns:
        return False
    return df["base_metric"].str.contains(_CPU_METRIC_PATTERN).any()


def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
