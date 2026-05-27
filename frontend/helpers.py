"""Frontend UI helpers for Dash dropdown formatting."""

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