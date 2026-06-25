"""Shared sample dataframes and filesystem helpers for unit tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from backend.data import AlumetData


def processed_rows() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=4, freq="s")
    return pd.DataFrame(
        {
            "metric_id": [
                "attributed_energy_J_R_local_machine__C_process_123_A_",
                "nvml_instant_power_W_R_gpu_0_C_process_123_A_",
                "mem_total_kB_R_local_machine__C__A_",
                "kernel_cpu_time_ms_R_cpu_core_0.0_C_process_123_A_",
            ],
            "base_metric": [
                "attributed_energy_J",
                "nvml_instant_power_W",
                "mem_total_kB",
                "kernel_cpu_time_ms",
            ],
            "timestamp": ts,
            "value": [10.0, 2.0, 1024.0, 5.0],
        }
    )


def attributed_energy_source_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric_id": [
                "attributed_energy_cpu_J_R_cpu_0_C_process_123_A_",
                "attributed_energy_gpu_J_R_gpu_0_C_process_123_A_",
                "attributed_energy_gpu_J_R_gpu_1_C_process_123_A_",
            ],
            "base_metric": [
                "attributed_energy_cpu_J",
                "attributed_energy_gpu_J",
                "attributed_energy_gpu_J",
            ],
            "timestamp": [pd.Timestamp("2024-01-01")] * 3,
            "value": [1.0, 2.0, 3.0],
        }
    )


def make_alumetdata_stub(
    *,
    log_content: str = "pid 99\nrapl",
    source_df: pd.DataFrame | None = None,
    processed_df: pd.DataFrame | None = None,
    directory: str | Path = "/measurements/run_a",
) -> AlumetData:
    data = AlumetData.__new__(AlumetData)
    data.directory = Path(directory)
    data._csv_path = Path("run.csv")
    data._log_path = Path("run.log")
    data._log_content = log_content
    data._df_source = source_df if source_df is not None else pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="s"),
            "consumer_kind": ["process", "process"],
            "value": [1.0, 2.0],
        }
    )
    data._df_processed = processed_df if processed_df is not None else processed_rows()
    return data


def write_measurement_directory(
    directory: Path,
    *,
    csv_body: str,
    log_body: str = "pid 42\nloaded nvml and rapl plugins\n",
) -> None:
    (directory / "measurement.csv").write_text(csv_body, encoding="utf-8")
    (directory / "agent.log").write_text(log_body, encoding="utf-8")


def sample_csv_body() -> str:
    return (
        "metric;resource_kind;resource_id;consumer_kind;consumer_id;__late_attributes;timestamp;value\n"
        "cpu_percent;local_machine;;process;123;;2024-01-01T00:00:00;50.0\n"
        "nvml_instant_power_mW;gpu;0;process;123;;2024-01-01T00:00:01;2000.0\n"
    )


class TempMeasurementDirectory:
    """Context manager that creates a temporary measurement directory."""

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name)
        write_measurement_directory(self.path, csv_body=sample_csv_body())
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        self._tmp.cleanup()
