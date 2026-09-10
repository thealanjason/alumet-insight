"""Shared sample dataframes and filesystem helpers for unit tests.

Named frames cover shapes reused across modules. Case-specific edges
(conflicting timestamps, late-attribute ambiguity, offset clocks) stay
inline in the test that owns them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from backend.data import AlumetData
from backend.metrics import MetricId


DEFAULT_START = pd.Timestamp("2024-01-01")

# IDs that appear in CLI, export, category, figure, and comparative tests.
NVML_POWER_ID = "nvml_instant_power_W_R_gpu_0_C_process_123_A_"
CPU_PERCENT_ID = "cpu_percent_R_local_machine__C_process_123_A_"
RAPL_ENERGY_ID = "rapl_consumed_energy_J_R_pkg_0_C__A_"
RAPL_POWER_ID = "rapl_average_power_W_R_pkg_0_C__A_"
KERNEL_CPU_TIME_ID = "kernel_cpu_time_ms_R_cpu_core_0.0_C_process_123_A_"
ATTRIBUTED_ENERGY_ID = "attributed_energy_J_R_local_machine__C_process_123_A_"
MEM_TOTAL_ID = "mem_total_B_R_local_machine__C__A_"
CPU_ENERGY_ID = "attributed_energy_cpu_J_R_pkg_C_process_1_A_"
GPU_ENERGY_ID = "attributed_energy_gpu_J_R_gpu_C_process_1_A_"
CPU_ENERGY_CUM_ID = "attributed_energy_cpu_cumulative_J_R_pkg_C_process_1_A_"
GPU_ENERGY_CUM_ID = "attributed_energy_gpu_cumulative_J_R_gpu_C_process_1_A_"
ENERGY_TOTAL_ID = "attributed_energy_total_J_R_local_machine__C_process_1_A_"
GPU_ENERGY_TOTAL_ID = "attributed_energy_gpu_total_J_R_local_machine__C_process_1_A_"
NETWORK_RX_ID = "network_rx_bytes_R_eth0__C__A_"
CUSTOM_COUNTER_ID = "custom_counter_R_host__C__A_"

OFFSET_CPU_GPU_X_TOTAL = 162.0
OFFSET_CPU_GPU_Y_TOTAL = 110.0
DOWNLOAD_CPU_TOTAL = 4.0
DOWNLOAD_GPU_TOTAL = 30.0


def _repeat(value: Any, n: int) -> list[Any]:
    if isinstance(value, (str, bytes, pd.Timestamp)) or not isinstance(value, Iterable):
        return [value] * n
    values = list(value)
    if len(values) != n:
        raise ValueError(f"column length {len(values)} does not match {n} rows")
    return values


def series_rows(
    metric_id: str,
    values: Iterable[Any],
    *,
    timestamps: Any | None = None,
    start: Any = DEFAULT_START,
    freq: str = "s",
    **extra: Any,
) -> pd.DataFrame:
    """Build one processed-schema series. ``base_metric`` is parsed from the id."""
    values = list(values)
    n = len(values)
    if timestamps is None:
        timestamps = pd.date_range(pd.Timestamp(start), periods=n, freq=freq)
    elif isinstance(timestamps, (str, pd.Timestamp)):
        timestamps = [pd.Timestamp(timestamps)] * n
    parsed = MetricId.parse(metric_id)
    data: dict[str, Any] = {
        "metric_id": [metric_id] * n,
        "base_metric": [parsed.base_metric] * n,
        "timestamp": list(timestamps),
        "value": values,
    }
    for key, value in extra.items():
        data[key] = _repeat(value, n)
    return pd.DataFrame(data)


def concat_series(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def processed_rows() -> pd.DataFrame:
    """Small AlumetData stub: energy, power, memory, kernel_cpu_time."""
    ts = pd.date_range(DEFAULT_START, periods=4, freq="s")
    return concat_series(
        series_rows(ATTRIBUTED_ENERGY_ID, [10.0], timestamps=[ts[0]]),
        series_rows(NVML_POWER_ID, [2.0], timestamps=[ts[1]]),
        series_rows(MEM_TOTAL_ID, [1024.0], timestamps=[ts[2]]),
        series_rows(KERNEL_CPU_TIME_ID, [5.0], timestamps=[ts[3]]),
    )


def catalog_rows() -> pd.DataFrame:
    """One representative series for every time-series category, plus memory variants."""
    ts = pd.date_range(DEFAULT_START, periods=17, freq="s")
    rows = [
        (ATTRIBUTED_ENERGY_ID, 10.0),
        (NVML_POWER_ID, 2.0),
        (CPU_PERCENT_ID, 50.0),
        ("nvml_memory_utilization_%_R_gpu_0_C__A_", 40.0),
        ("nvml_temperature_C_R_gpu_0_C_process_1_A_", 70.0),
        (MEM_TOTAL_ID, 1024.0),
        ("active_B_R_local_machine__C__A_", 1.0),
        ("inactive_B_R_local_machine__C__A_", 1.0),
        ("cached_B_R_local_machine__C__A_", 1.0),
        ("mapped_kB_R_local_machine__C__A_", 1.0),
        ("nvml_gpu_memory_info_B_R_gpu_0_C__A_", 1.0),
        ("perf_hardware_INSTRUCTIONS_R_cpu_0_C_process_1_A_", 100.0),
        ("perf_cache_LL_READ_MISS_R_cpu_0_C_process_1_A_", 10.0),
        (KERNEL_CPU_TIME_ID, 5.0),
        ("kernel_n_procs_running_R_local__C__A_", 2.0),
        (NETWORK_RX_ID, 4096.0),
        (CUSTOM_COUNTER_ID, 1.0),
    ]
    return concat_series(
        *(series_rows(metric_id, [value], timestamps=[ts[i]]) for i, (metric_id, value) in enumerate(rows))
    )


def attributed_energy_source_rows() -> pd.DataFrame:
    """CPU + two GPUs at one timestamp; combined observed total is 6 J."""
    return concat_series(
        series_rows("attributed_energy_cpu_J_R_cpu_0_C_process_123_A_", [1.0]),
        series_rows("attributed_energy_gpu_J_R_gpu_0_C_process_123_A_", [2.0]),
        series_rows("attributed_energy_gpu_J_R_gpu_1_C_process_123_A_", [3.0]),
    )


def rapl_energy_rows(values: Iterable[Any], **extra: Any) -> pd.DataFrame:
    return series_rows(RAPL_ENERGY_ID, values, **extra)


def offset_cpu_gpu_energy_rows() -> pd.DataFrame:
    """Unequal rates + delayed GPU: CPU 50 ms × 81 @ 2 J, GPU 200 ms × 11 @ 10 J.

    Running totals end at ``OFFSET_CPU_GPU_X_TOTAL`` / ``OFFSET_CPU_GPU_Y_TOTAL``.
    Nearest-align then cumsum does not.
    """
    x_times = pd.date_range(DEFAULT_START, periods=81, freq="50ms")
    y_times = pd.date_range(DEFAULT_START + pd.Timedelta("2s"), periods=11, freq="200ms")
    return concat_series(
        series_rows(CPU_ENERGY_ID, [2.0] * len(x_times), timestamps=x_times, point_role="observed"),
        series_rows(GPU_ENERGY_ID, [10.0] * len(y_times), timestamps=y_times, point_role="observed"),
    )


def download_cpu_gpu_energy_rows() -> pd.DataFrame:
    """CPU 1 J × 4 at 1 s, GPU 10/20 J at 2 s. Cumulative ends 4 J / 30 J."""
    cpu_times = pd.date_range(DEFAULT_START, periods=4, freq="s")
    gpu_times = pd.date_range(DEFAULT_START, periods=2, freq="2s")
    return concat_series(
        series_rows(CPU_ENERGY_ID, [1.0] * 4, timestamps=cpu_times, consumer_kind="process"),
        series_rows(GPU_ENERGY_ID, [10.0, 20.0], timestamps=gpu_times, consumer_kind="process"),
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
            "timestamp": pd.date_range(DEFAULT_START, periods=2, freq="s"),
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
