import re
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from typing import List, Optional

from backend.metrics import get_bytes_tickvals_ticktext


def _split_kind_id(part: str) -> tuple:
    if not part:
        return "", ""
    # Split on last underscore if the last part appears to be an ID
    if "_" in part:
        parts_list = part.rsplit("_", 1)
        if len(parts_list) == 2:
            potential_id = parts_list[1]
            # Check if it appears to be an ID (number or decimal number value)
            if (potential_id.replace(".", "").replace("-", "").isdigit() or 
                potential_id in ["total", "0", "1", ""] or
                (len(potential_id) <= 15 and "_" not in potential_id)):
                kind = parts_list[0].replace("_", " ") if parts_list[0] else ""
                return kind, potential_id
    # If we can't determine, show the whole string with underscores replaced by spaces for readability
    return part.replace("_", " "), ""

def _format_id(id_str: str) -> str:
    # Convert ID to int if it's a whole number
    if not id_str:
        return ""
    try:
        float_val = float(id_str)
        if float_val.is_integer():
            return str(int(float_val))
        else:
            return id_str  
    except (ValueError, TypeError):
        return id_str

def _format_metric_title(metric_id: str) -> str:
    """Format metric_id into more well-structured plot title.
    
    Parses metric_id format: {base_metric}_R_{resource_kind}_{resource_id}_C_{consumer_kind}_{consumer_id}_A_{late_attributes}
    Returns formatted string: "{base_metric} R: {resource_kind} {resource_id} C: {consumer_kind} {consumer_id} A: {__late_attributes}"
    """
    try:
        # Split by the delimiter "_R_" to get the base metric
        if "_R_" not in metric_id:
            return metric_id  # Return as-is if format doesn't match
        
        parts = metric_id.split("_R_", 1)
        base_metric = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        
        
        if "_C_" not in rest:
            # No consumer part, just resource part
            # Split by the delimiter "_A_" to get the late attributes if present
            resource_part = rest.split("_A_")[0] if "_A_" in rest else rest
            consumer_part = ""
            late_attr = rest.split("_A_", 1)[1] if "_A_" in rest else ""
        else:
            # Split by the delimiter "_C_" to get the resource part, consumer part and rest
            resource_consumer = rest.split("_C_", 1)
            resource_part = resource_consumer[0]
            rest = resource_consumer[1] if len(resource_consumer) > 1 else ""
            # Split by the delimiter "_A_" to get the late attributes if present
            if "_A_" not in rest:
                consumer_part = rest
                late_attr = ""
            else:
                consumer_late = rest.split("_A_", 1)
                consumer_part = consumer_late[0]
                late_attr = consumer_late[1] if len(consumer_late) > 1 else ""
        
        resource_kind, resource_id = _split_kind_id(resource_part)
        consumer_kind, consumer_id = _split_kind_id(consumer_part)
        
        # Format IDs as integers if they're numeric
        resource_id_formatted = _format_id(resource_id)
        consumer_id_formatted = _format_id(consumer_id)
        
        # Build formatted title
        title_parts = [base_metric]
        
        # Add resource info if present
        if resource_kind or resource_id_formatted:
            resource_str = f"R: {resource_kind}"
            if resource_id_formatted:
                resource_str += f" {resource_id_formatted}"
            title_parts.append(resource_str)
        
        # Add consumer info if present
        if consumer_kind or consumer_id_formatted:
            consumer_str = f"C: {consumer_kind}"
            if consumer_id_formatted:
                consumer_str += f" {consumer_id_formatted}"
            title_parts.append(consumer_str)
        
        # Add late attributes if present with underscores replaced by spaces for readability
        if late_attr:
            late_attr_formatted = late_attr.replace("_", " ")
            title_parts.append(f"A: {late_attr_formatted}")
        
        return " ".join(title_parts)
    except Exception:
        # If parsing fails, return the original metric_id with underscores replaced by spaces for readability
        return metric_id.replace("_", " ")

def metric_id_to_plot_label(metric_id: str, max_len: int = 60) -> str:
    if not metric_id:
        return ""
    s = str(metric_id)

    # normalizations
    s = s.replace("_R_", " | R=")
    s = s.replace("_C_", " | C=")
    s = s.replace("_A_", " | ")
    s = s.replace("__", "_")

    # shorten some metric names
    s = s.replace("local_machine", "local")
    s = s.replace("cpu_percent_%", "cpu%")
    s = s.replace("kernel_cpu_time_ms", "kernel_cpu_ms")

    # remove process IDs
    s = re.sub(r"\| C=process_\d+(\.\d+)?", "| C=process", s)

    # truncate for display
    if len(s) > max_len:
        s = s[: max_len - 1] + "\u2026"
    return s

def get_color_palette(n_colors: int) -> List[str]:
    """Get a color palette for n_colors time series.
    Uses Plotly's qualitative color palettes and cycles through them if needed.
    """
    # Use Plotly's qualitative palettes
    # Combine multiple palettes for more colors
    palettes = [
        pc.qualitative.Plotly,      # 10 colors
        pc.qualitative.Set2,        # 8 colors
        pc.qualitative.Set3,        # 12 colors
        pc.qualitative.Pastel,      # 10 colors
        pc.qualitative.Dark2,       # 8 colors
        pc.qualitative.Pastel1,     # 9 colors
        pc.qualitative.Pastel2,     # 8 colors
    ]
    
    colors = []
    for palette in palettes:
        colors.extend(palette)
        if len(colors) >= n_colors:
            break
    
    # If still not enough, cycle through the colors
    while len(colors) < n_colors:
        colors.extend(colors[:min(len(colors), n_colors - len(colors))])
    
    return colors[:n_colors]

def create_all_timeseries_plots(df_processed: pd.DataFrame, proc_start: Optional[pd.Timestamp] = None, proc_end: Optional[pd.Timestamp] = None, full_time_range: Optional[tuple] = None, category: Optional[str] = None, share_yaxis: bool = False) -> go.Figure:
    """Create all time series as scrollable subplots.
    
    Args:
        df_processed: Filtered dataframe with metric_id, timestamp, and value
        proc_start: Process start time for gray highlight
        proc_end: Process end time for gray highlight
        full_time_range: Tuple of (min_time, max_time) for full measurement range to fix x-axis
        category: Metric category ("energy", "utilization", "temperature", "memory", "perf_counters", "kernel_cpu_time", "kernel_system", "miscellaneous") to set appropriate Y-axis label
        share_yaxis: If True, all subplots share the same Y-axis range for easier comparison
    """
    if df_processed.empty:
        return go.Figure()
    
    # Coerce timestamps
    df_processed = df_processed.copy()
    df_processed["timestamp"] = pd.to_datetime(df_processed["timestamp"], errors="coerce")
    
    # Get unique metric_ids
    unique_metrics = df_processed["metric_id"].unique()
    n_metrics = len(unique_metrics)

    if n_metrics == 0:
        return go.Figure()
    
    # Get full time range for x-axis (calculate from data if provided range is not provided)
    if full_time_range:
        x_min, x_max = full_time_range
    else:
        x_min = df_processed["timestamp"].min()
        x_max = df_processed["timestamp"].max()
    x_min = pd.Timestamp(x_min)
    x_max = pd.Timestamp(x_max)
    
    # Get color palette
    colors = get_color_palette(n_metrics)
    color_map = {metric: colors[i] for i, metric in enumerate(unique_metrics)}
    
    # Vertical spacing between subplots
    MIN_SUBPLOT_HEIGHT = 240  # Pixels per subplot row (scroll container avoids squashing)
    SUBPLOT_GAP_PX = 52       # Fixed pixel gap between subplot rows
    total_height = MIN_SUBPLOT_HEIGHT * n_metrics + SUBPLOT_GAP_PX * max(n_metrics - 1, 0)
    vertical_spacing = (SUBPLOT_GAP_PX / total_height) if n_metrics > 1 else 0.05
    
    # Create subplots with formatted titles
    # Using shared_xaxes=True so zooming one subplot zooms all others to the same time region
    formatted_titles = [_format_metric_title(metric_id) for metric_id in unique_metrics]
    fig = make_subplots(
        rows=n_metrics, cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=[f"<b>{title}</b>" for title in formatted_titles],
    )

    # Pre-calculate y-axis ranges for each metric using vectorized groupby (much faster than loop)
    y_stats = df_processed.groupby("metric_id")["value"].agg(["min", "max"])
    
    # Check if this is a memory category
    is_memory_category = category == "memory"
    
    y_ranges = {}
    for metric_id in unique_metrics:
        if metric_id not in y_stats.index:
            y_ranges[metric_id] = {"min": 0 if is_memory_category else -1, "max": 1}
            continue
        
        y_min = y_stats.loc[metric_id, "min"]
        y_max = y_stats.loc[metric_id, "max"]
        y_range = y_max - y_min if y_max != y_min else abs(y_max) if y_max != 0 else 1
        y_padding = 0.1 * y_range if y_range > 0 else 0.1
        
        # Ensure valid range (min < max)
        calculated_min = y_min - y_padding
        calculated_max = y_max + y_padding
        if calculated_min >= calculated_max:
            calculated_min = y_min - 0.1 if y_min != 0 else -0.1
            calculated_max = y_max + 0.1 if y_max != 0 else 0.1
        
        # For memory metrics, ensure minimum is not negative
        if is_memory_category:
            calculated_min = max(0, calculated_min)
        
        y_ranges[metric_id] = {
            "min": calculated_min,
            "max": calculated_max
        }
    
    # If sharing Y-axis, calculate global range across all metrics
    shared_y_range = None
    if share_yaxis and y_ranges:
        global_min = min(r["min"] for r in y_ranges.values())
        global_max = max(r["max"] for r in y_ranges.values())
        # For memory metrics, ensure minimum is not negative
        if is_memory_category:
            global_min = max(0, global_min)
        shared_y_range = {"min": global_min, "max": global_max}

    # Gray highlighted zone for process active period
    # Use layout shapes (vrect) instead of traces — these always span the full Y-axis
    # regardless of zoom level, and don't clutter the legend
    if proc_start and proc_end:
        for idx in range(1, n_metrics + 1):
            # yref="y{n} domain" makes the shape span 0-100% of the subplot's Y-axis
            yref = f"y{idx} domain" if idx > 1 else "y domain"
            xref = f"x{idx}" if idx > 1 else "x"
            fig.add_shape(
                type="rect",
                x0=proc_start, x1=proc_end,
                y0=0, y1=1,
                xref=xref,
                yref=yref,
                fillcolor="rgba(136, 192, 208, 0.12)",
                line=dict(width=0),
                layer="below",
            )
        # Add a single trace for the legend entry without plotting a data point.
        # Using visible="legendonly" makes Plotly dim the label, which has poor
        # contrast in both light and dark themes.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="rgba(136, 192, 208, 0.4)", symbol="square"),
                name="Process Active",
                showlegend=True,
                legendgroup="process_active",
            ),
            row=1, col=1,
        )

    # Pre-group data by metric_id for faster iteration (sort once, not per metric)
    df_sorted = df_processed.sort_values(["metric_id", "timestamp"])
    grouped = {mid: grp for mid, grp in df_sorted.groupby("metric_id", observed=True, sort=False)}
    
    # Determine total points to decide rendering strategy
    # Per-metric rendering: use WebGL when the series is huge.
    total_points = len(df_processed)
    use_webgl = total_points > 10000  # Use WebGL for large datasets
    show_markers_global = total_points < 5000  # Only show markers for smaller datasets
    
    # Add traces for each metric
    for idx, metric_id in enumerate(unique_metrics, start=1):
        metric_data = grouped.get(metric_id, pd.DataFrame())
        if metric_data.empty:
            continue

        n_pts = len(metric_data)
        use_webgl = n_pts > 10000  # Use WebGL for large series based on total number of points in each metric
        show_markers = show_markers_global and n_pts < 5000
            
        color = color_map[metric_id]

        # Convert hex color to rgba for fillcolor
        if color.startswith('#'):
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgba_fill = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)"
        elif color.startswith('rgba'):
            # Already rgba, just adjust opacity
            rgba_fill = color.rsplit(',', 1)[0] + ', 0.15)'
        elif color.startswith('rgb'):
            # Convert rgb to rgba
            rgba_fill = color.replace('rgb', 'rgba').replace(')', ', 0.15)')
        else:
            # Default fallback
            rgba_fill = "rgba(136, 192, 208, 0.15)"

        # Choose scatter type: Scattergl (WebGL) for large series, Scatter otherwise
        ScatterClass = go.Scattergl if use_webgl else go.Scatter
        
        # Build trace config - markers only for smaller datasets (performance)
        trace_config = dict(
            x=pd.to_datetime(metric_data["timestamp"], errors="coerce"),
            y=metric_data["value"],
            mode="lines+markers" if show_markers else "lines",
            name=metric_id,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{metric_id}</b><br>Time: %{{x|%H:%M:%S.%L}}<br>Value: %{{y:.4f}}<extra></extra>",
            showlegend=False,
        )
        
        # Add markers only for smaller datasets
        if show_markers:
            trace_config["marker"] = dict(
                color=color,
                size=6,
                symbol="circle",
                line=dict(width=1, color="rgba(255, 255, 255, 0.5)"),
            )
        
        # Fill only works well with regular Scatter (not Scattergl)
        if not use_webgl:
            trace_config["fill"] = "tozeroy"
            trace_config["fillcolor"] = rgba_fill

        fig.add_trace(ScatterClass(**trace_config), row=idx, col=1)
        
        # Fix x-axis range to full measurement time
        # With shared_xaxes=True, zooming one subplot zooms all others to the same time region
        fig.update_xaxes(
            type="date",
            range=[x_min, x_max],
            gridcolor="rgba(76, 86, 106, 0.2)",
            showticklabels=True,  # Show tick labels on each subplot (overrides shared_xaxes default)
            # White dotted spike line on hover - only at data points
            showspikes=True,
            spikemode="across",
            spikesnap="data",  # Only show spike when hovering near actual data points
            spikethickness=1,
            spikecolor="white",
            spikedash="dot",
            row=idx, col=1
        )
        # Determine Y-axis label based on category
        if category == "energy":
            y_axis_label = "Value (J)"
        elif category == "power":
            y_axis_label = "Value (W)"
        elif category == "memory":
            y_axis_label = "Value (B)"
        elif category == "utilization":
            y_axis_label = "Value (%)"
        elif category == "temperature":
            y_axis_label = "Value (°C)"
        elif category == "perf_counters":
            y_axis_label = "Value (count)"
        else:
            y_axis_label = "Value"
        
        # Build y-axis configuration
        yaxis_config = dict(
            title_text=y_axis_label,
            fixedrange=False,
            gridcolor="rgba(76, 86, 106, 0.2)",
        )
        
        # If sharing Y-axis, set fixed range; otherwise use autorange
        if shared_y_range:
            yaxis_config["range"] = [shared_y_range["min"], shared_y_range["max"]]
            yaxis_config["autorange"] = False
        else:
            yaxis_config["autorange"] = True
        
        # For memory metrics, add custom tick formatting
        if category == "memory" and not metric_data.empty:
            if shared_y_range:
                # Use global range for tick formatting when sharing Y-axis
                tickvals, ticktext = get_bytes_tickvals_ticktext(shared_y_range["min"], shared_y_range["max"], num_ticks=5)
            else:
                y_min, y_max = metric_data["value"].min(), metric_data["value"].max()
                tickvals, ticktext = get_bytes_tickvals_ticktext(y_min, y_max, num_ticks=5)
            yaxis_config["tickvals"] = tickvals
            yaxis_config["ticktext"] = ticktext
        
        fig.update_yaxes(**yaxis_config, row=idx, col=1)
    
    # Add "Time" label only to the bottom subplot
    fig.update_xaxes(title_text="Time", row=n_metrics, col=1)

    # total_height was already calculated above from MIN_SUBPLOT_HEIGHT * n_metrics + gaps
    fig.update_layout(
        height=total_height,  # Total height for all subplots
        title=dict(text="<b>📈 Time series of all metrics</b>", x=0.5, font=dict(size=16)),
        paper_bgcolor="rgba(46, 52, 64, 0.95)",
        plot_bgcolor="rgba(59, 66, 82, 0.7)",
        font=dict(color="#d8dee9"),
        hovermode="closest",  # Show hover info for closest point with spike line
        margin=dict(l=50, r=20, t=60, b=40),  # Reduced margins for wider plots
        autosize=True,  # Enable autosize to fill container width
        width=None,  # Let it fill the container
        showlegend=True,  # Ensure legend is visible
        legend=dict(
            bgcolor="rgba(46, 52, 64, 0.8)",
            bordercolor="rgba(136, 192, 208, 0.3)",
            borderwidth=1,
            font=dict(color="#d8dee9"),
        ),
    )
    fig.update_xaxes(type="date", rangeslider=dict(visible=False), row=n_metrics, col=1)
    
    return fig