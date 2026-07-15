"""Filesystem and log-parsing utilities."""

from __future__ import annotations

import base64
import re
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


ALLOWED_UPLOAD_SUFFIXES = {".csv", ".log", ".txt", ".toml"}


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
    name_list = [filenames] if isinstance(filenames, str) else list(filenames)
    if len(content_list) != len(name_list):
        raise ValueError("Uploaded contents and filenames are out of sync.")

    first_components = {
        Path(name).parts[0]
        for name in name_list
        if name and Path(name).parts
    }
    experiment_name = next(iter(first_components)) if len(first_components) == 1 else "uploaded"

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


def is_gpu_from_metrics(df: "pd.DataFrame") -> bool:
    """Detect GPU presence from metric names in the processed dataframe."""
    if df.empty or "base_metric" not in df.columns:
        return False
    return df["base_metric"].str.contains(_GPU_METRIC_PATTERN).any()


def is_cpu_from_metrics(df: "pd.DataFrame") -> bool:
    """Detect CPU presence from metric names in the processed dataframe."""
    if df.empty or "base_metric" not in df.columns:
        return False
    return df["base_metric"].str.contains(_CPU_METRIC_PATTERN).any()


def safe_filename(value: str) -> str:
    """Return a filesystem-safe filename stem."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
