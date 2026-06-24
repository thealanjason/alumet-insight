"""Filesystem and log-parsing utilities."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import List, Optional


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
