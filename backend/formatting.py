"""Human-readable formatting for metric identifiers and axis labels.

No renderer dependency — safe to import from both the CLI and the dashboard.
"""

from __future__ import annotations

import re

import numpy as np


# ---------------------------------------------------------------------------
# Metric ID → display string
# ---------------------------------------------------------------------------

def _split_kind_id(part: str) -> tuple[str, str]:
    if not part:
        return "", ""
    if "_" in part:
        parts_list = part.rsplit("_", 1)
        if len(parts_list) == 2:
            potential_id = parts_list[1]
            if (
                potential_id.replace(".", "").replace("-", "").isdigit()
                or potential_id in ["total", "0", "1", ""]
                or (len(potential_id) <= 15 and "_" not in potential_id)
            ):
                kind = parts_list[0].replace("_", " ") if parts_list[0] else ""
                return kind, potential_id
    return part.replace("_", " "), ""


def _format_id(id_str: str) -> str:
    if not id_str:
        return ""
    try:
        float_val = float(id_str)
        return str(int(float_val)) if float_val.is_integer() else id_str
    except (ValueError, TypeError):
        return id_str


def format_metric_title(metric_id: str) -> str:
    """Format a metric_id into a human-readable plot title.

    Parses: {base_metric}_R_{resource_kind}_{resource_id}_C_{consumer_kind}_{consumer_id}_A_{late_attributes}
    Returns: "{base_metric} R: {resource_kind} {resource_id} C: {consumer_kind} {consumer_id} A: {late_attributes}"
    """
    try:
        if "_R_" not in metric_id:
            return metric_id

        parts = metric_id.split("_R_", 1)
        base_metric = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if "_C_" not in rest:
            resource_part = rest.split("_A_")[0] if "_A_" in rest else rest
            consumer_part = ""
            late_attr = rest.split("_A_", 1)[1] if "_A_" in rest else ""
        else:
            resource_consumer = rest.split("_C_", 1)
            resource_part = resource_consumer[0]
            rest = resource_consumer[1] if len(resource_consumer) > 1 else ""
            if "_A_" not in rest:
                consumer_part = rest
                late_attr = ""
            else:
                consumer_late = rest.split("_A_", 1)
                consumer_part = consumer_late[0]
                late_attr = consumer_late[1] if len(consumer_late) > 1 else ""

        resource_kind, resource_id = _split_kind_id(resource_part)
        consumer_kind, consumer_id = _split_kind_id(consumer_part)
        resource_id = _format_id(resource_id)
        consumer_id = _format_id(consumer_id)

        title_parts = [base_metric]
        if resource_kind or resource_id:
            resource_str = f"R: {resource_kind}"
            if resource_id:
                resource_str += f" {resource_id}"
            title_parts.append(resource_str)
        if consumer_kind or consumer_id:
            consumer_str = f"C: {consumer_kind}"
            if consumer_id:
                consumer_str += f" {consumer_id}"
            title_parts.append(consumer_str)
        if late_attr:
            title_parts.append(f"A: {late_attr.replace('_', ' ')}")

        return " ".join(title_parts)
    except Exception:
        return metric_id.replace("_", " ")


def metric_id_to_plot_label(metric_id: str, max_len: int = 60) -> str:
    """Shorten a metric_id to a compact display label."""
    if not metric_id:
        return ""
    s = str(metric_id)

    s = s.replace("_R_", " | R=")
    s = s.replace("_C_", " | C=")
    s = s.replace("_A_", " | ")
    s = s.replace("__", "_")

    s = s.replace("local_machine", "local")
    s = s.replace("cpu_percent_%", "cpu%")
    s = s.replace("kernel_cpu_time_ms", "kernel_cpu_ms")

    s = re.sub(r"\| C=process_\d+(\.\d+)?", "| C=process", s)

    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


# ---------------------------------------------------------------------------
# Byte-value axis formatting
# ---------------------------------------------------------------------------

def format_bytes_ticklabel(value: float, decimals: int = 1) -> str:
    """Format a byte value with appropriate unit (B, KB, MB, GB, TB)."""
    d = max(0, min(6, int(decimals)))
    if abs(value) < 1024:
        return f"{value:.0f} B"
    elif abs(value) < 1024 ** 2:
        return f"{value / 1024:.{d}f} KB"
    elif abs(value) < 1024 ** 3:
        return f"{value / (1024 ** 2):.{d}f} MB"
    elif abs(value) < 1024 ** 4:
        return f"{value / (1024 ** 3):.{d}f} GB"
    else:
        return f"{value / (1024 ** 4):.{d}f} TB"


def get_bytes_tickvals_ticktext(y_min: float, y_max: float, num_ticks: int = 5) -> tuple:
    """Generate tick values and formatted tick labels for byte-valued axes.

    Uses adaptive precision and label deduplication so narrow ranges
    don't produce multiple overlapping labels.
    """
    y_min = max(0, float(y_min))
    y_max = float(y_max)

    if y_max <= y_min:
        y_max = y_min + 1

    span = y_max - y_min
    span_ratio = span / max(abs(y_max), abs(y_min), 1e-12)

    nt_candidates = [num_ticks, max(3, num_ticks - 1), 3]
    if span_ratio < 0.001:
        nt_candidates = [4, 3]

    for nt in nt_candidates:
        tickvals = np.linspace(y_min, y_max, nt)
        for dec in (1, 2, 3, 4):
            ticktext = [format_bytes_ticklabel(float(v), decimals=dec) for v in tickvals]
            if len(set(ticktext)) == len(ticktext):
                return list(tickvals), ticktext
        ticktext = [format_bytes_ticklabel(float(v), decimals=4) for v in tickvals]
        out_vals: list = []
        out_text: list = []
        seen: set = set()
        for v, t in zip(tickvals, ticktext):
            if t not in seen:
                seen.add(t)
                out_vals.append(float(v))
                out_text.append(t)
        if len(out_vals) >= 2:
            return out_vals, out_text

    return (
        [y_min, y_max],
        [
            format_bytes_ticklabel(y_min, decimals=4),
            format_bytes_ticklabel(y_max, decimals=4),
        ],
    )
