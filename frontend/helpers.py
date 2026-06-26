"""Frontend UI helpers for Dash dropdown formatting."""

import json

import pandas as pd

from backend.categories import available_category_values, CATEGORY_LABELS


def normalize_dropdown_value(x):
    """Normalize a Dash dropdown value to a clean string or None."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def available_category_options(df_processed: pd.DataFrame) -> list[dict[str, str]]:
    """Return category options in Dash dropdowns."""
    return [{"label": CATEGORY_LABELS[value], "value": value} for value in available_category_values(df_processed)]

def triggered_component_type(triggered_prop_id: str) -> str | None:
    """Extract the matched component `type` from a Dash callback prop id."""
    if ".value" not in triggered_prop_id:
        return None
    try:
        id_dict = json.loads(triggered_prop_id.split(".value")[0])
    except (json.JSONDecodeError, TypeError):
        return None
    component_type = id_dict.get("type")
    return component_type if isinstance(component_type, str) else None

def ensure_timestamp_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure timestamp column is datetime64 dtype.
    
    Args:
        df: DataFrame to ensure timestamp column is datetime64 dtype
    
    Returns:
        DataFrame with timestamp column as datetime64 dtype
    """
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def parse_process_time_range_store(
    store: Optional[dict],
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Unpack a process-time-range dcc.Store dict into a (start, end) timestamp pair."""
    if not store:
        return None, None
    start = pd.to_datetime(store["start"]) if store.get("start") else None
    end = pd.to_datetime(store["end"]) if store.get("end") else None
    return start, end