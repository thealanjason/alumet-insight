"""Frontend UI helpers for Dash dropdown filtering."""

import pandas as pd

from backend.categories import available_category_values, CATEGORY_LABELS


def normalize_dropdown_value(x):
    """Normalize a dropdown value to a clean string or None."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def unique_nonempty(series: pd.Series) -> list:
    """Get unique non-empty string values from a series.

    Optimized using vectorized operations instead of loop.
    """
    str_series = series.astype(object).fillna("").astype(str).str.strip()
    mask = str_series != ""
    unique_vals = str_series[mask].unique()
    return sorted(unique_vals)


def available_category_options(df_processed: pd.DataFrame) -> list[dict[str, str]]:
    """Return category options in the shape consumed by Dash dropdowns."""
    return [{"label": CATEGORY_LABELS[value], "value": value} for value in available_category_values(df_processed)]