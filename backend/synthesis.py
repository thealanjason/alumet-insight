"""Derived metric synthesis — creates new rows that don't exist in the raw CSV."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


def _extract_process_pid(metric_id: str) -> Optional[str]:
    pid_pattern = re.compile(r"_C_process_(\d+)")
    m = pid_pattern.search(str(metric_id))
    return m.group(1) if m else None


def _align_cumulative_energy_to_timeline(
    df_metric: pd.DataFrame,
    timeline: pd.DatetimeIndex,
    value_name: str,
) -> pd.Series:
    """
    Align a cumulative metric to a shared timeline.

    Before the first observed sample the value is treated as 0.
    Between observed samples, values are linearly interpolated on timestamp.
    After the last observed sample, values remain N/A so downstream synthesis can dropna to truncate the timeline.
    """
    if df_metric.empty:
        return pd.Series(0.0, index=timeline, name=value_name)

    aligned = (
        df_metric.groupby("timestamp", as_index=True)["value"]
        .sum()
        .sort_index()
        .reindex(timeline)
        .astype(float)
    )

    first_valid = aligned.first_valid_index()
    if first_valid is None:
        return pd.Series(0.0, index=timeline, name=value_name)

    aligned.loc[aligned.index < first_valid] = 0.0
    aligned = aligned.interpolate(method="time", limit_area="inside")
    aligned = aligned.fillna(0.0)
    last_valid = df_metric["timestamp"].max()
    aligned.loc[aligned.index > last_valid] = np.nan
    aligned.name = value_name
    return aligned


def synthesize_attributed_energy_total(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesize attributed_energy_total_J metric from attributed_energy_cpu and attributed_energy_gpu metrics.

    Creates two synthetic metrics:
    1. attributed_energy_gpu_total_J: sum of attributed GPU energy across all GPUs per process (pid).
    2. attributed_energy_total_J: CPU + GPU total per process on the union of timestamps.

    Args:
        df_processed: Preprocessed DataFrame with metric_id, base_metric, timestamp, value columns.

    Returns:
        DataFrame of synthetic rows or empty DataFrame when source data is missing.
    """
    cpu_mask = df_processed["base_metric"].str.contains("attributed_energy_cpu", case=False, na=False)
    gpu_mask = df_processed["base_metric"].str.contains("attributed_energy_gpu", case=False, na=False)

    df_cpu = df_processed.loc[cpu_mask, ["metric_id", "timestamp", "value"]].copy()
    df_gpu = df_processed.loc[gpu_mask, ["metric_id", "timestamp", "value"]].copy()

    if df_cpu.empty and df_gpu.empty:
        return pd.DataFrame(columns=df_processed.columns)

    if not df_cpu.empty:
        df_cpu["pid"] = df_cpu["metric_id"].map(_extract_process_pid)
        df_cpu["timestamp"] = pd.to_datetime(df_cpu["timestamp"], errors="coerce")
        df_cpu.dropna(subset=["pid", "timestamp"], inplace=True)

    if not df_gpu.empty:
        df_gpu["pid"] = df_gpu["metric_id"].map(_extract_process_pid)
        df_gpu["timestamp"] = pd.to_datetime(df_gpu["timestamp"], errors="coerce")
        df_gpu.dropna(subset=["pid", "timestamp"], inplace=True)

    if df_cpu.empty and df_gpu.empty:
        return pd.DataFrame(columns=df_processed.columns)

    df_gpu_summed = (
        df_gpu.groupby(["timestamp", "pid"], as_index=False)["value"].sum()
        if not df_gpu.empty
        else pd.DataFrame(columns=["timestamp", "pid", "value"])
    )
    df_cpu_summed = (
        df_cpu.groupby(["timestamp", "pid"], as_index=False)["value"].sum()
        if not df_cpu.empty
        else pd.DataFrame(columns=["timestamp", "pid", "value"])
    )

    synthetic_parts: list[pd.DataFrame] = []

    for pid in df_gpu_summed["pid"].unique():
        gpu_pid = df_gpu_summed.loc[df_gpu_summed["pid"] == pid, ["timestamp", "value"]].copy()
        gpu_pid["metric_id"] = f"attributed_energy_gpu_total_J_R_gpu_all__C_process_{pid}_A_"
        gpu_pid["base_metric"] = "attributed_energy_gpu_total_J"
        synthetic_parts.append(gpu_pid[["metric_id", "base_metric", "timestamp", "value"]])

    shared_pids = sorted(set(df_cpu_summed["pid"]) & set(df_gpu_summed["pid"]))
    for pid in shared_pids:
        cpu_pid = df_cpu_summed.loc[df_cpu_summed["pid"] == pid, ["timestamp", "value"]].copy()
        gpu_pid = df_gpu_summed.loc[df_gpu_summed["pid"] == pid, ["timestamp", "value"]].copy()
        if cpu_pid.empty or gpu_pid.empty:
            continue

        timeline = pd.DatetimeIndex(
            pd.Index(cpu_pid["timestamp"]).union(pd.Index(gpu_pid["timestamp"])).sort_values()
        )
        if timeline.empty:
            continue

        cpu_aligned = _align_cumulative_energy_to_timeline(cpu_pid, timeline, "cpu_value")
        gpu_aligned = _align_cumulative_energy_to_timeline(gpu_pid, timeline, "gpu_value")

        total_pid = pd.DataFrame({
            "timestamp": timeline,
            "cpu_value": cpu_aligned.to_numpy(),
            "gpu_value": gpu_aligned.to_numpy(),
        })
        total_pid.dropna(subset=["cpu_value", "gpu_value"], inplace=True)
        if total_pid.empty:
            continue
        total_pid["value"] = total_pid["cpu_value"] + total_pid["gpu_value"]
        total_pid["metric_id"] = f"attributed_energy_total_J_R_total__C_process_{pid}_A_"
        total_pid["base_metric"] = "attributed_energy_total_J"
        synthetic_parts.append(total_pid[["metric_id", "base_metric", "timestamp", "value"]])

    if not synthetic_parts:
        return pd.DataFrame(columns=df_processed.columns)

    return pd.concat(synthetic_parts, ignore_index=True)
