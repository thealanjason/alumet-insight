"""Backend helpers for the process-specific (grid) tab.

Handles filter column normalisation, cascading option computation,
single-series filtering, and CSV export preparation.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from backend.data import filter_to_time_range


def unique_nonempty(series: pd.Series) -> list[str]:
    """Get unique non-empty string values from a series, sorted."""
    str_series = series.astype(object).fillna("").astype(str).str.strip()
    mask = str_series != ""
    return sorted(str_series[mask].unique())


def normalize_filter_columns(dfm: pd.DataFrame) -> pd.DataFrame:
    """Add normalised filter columns rk, rid, ck, cid, la."""
    dfm = dfm.copy()
    for col, src in [
        ("rk", "resource_kind"),
        ("rid", "resource_id"),
        ("ck", "consumer_kind"),
        ("cid", "consumer_id"),
        ("la", "__late_attributes"),
    ]:
        dfm[col] = dfm[src].astype(object).fillna("").astype(str).str.strip()
    return dfm


def cascade_filter_options(
    dfm: pd.DataFrame,
    rk: Optional[str],
    rid: Optional[str],
    ck: Optional[str],
    cid: Optional[str],
    la: Optional[str],
    triggered_id: Optional[str] = None,
) -> dict:
    """Compute cascaded filter options from reources/consumers/late attributes."""
    rk_opts = unique_nonempty(dfm["rk"])
    rk_eff = rk if rk in rk_opts else (rk_opts[0] if len(rk_opts) == 1 else None)
    df1 = dfm if rk_eff is None else dfm[dfm["rk"] == rk_eff]

    rid_opts = unique_nonempty(df1["rid"])
    if triggered_id == "resource-kind-dropdown":
        rid_eff = None
    else:
        rid_eff = rid if rid in rid_opts else (rid_opts[0] if len(rid_opts) == 1 else None)
    df2 = df1 if rid_eff is None else df1[df1["rid"] == rid_eff]

    ck_opts = unique_nonempty(df2["ck"])
    ck_eff = ck if ck in ck_opts else (ck_opts[0] if len(ck_opts) == 1 else None)
    df3 = df2 if ck_eff is None else df2[df2["ck"] == ck_eff]

    cid_opts = unique_nonempty(df3["cid"])
    if triggered_id == "consumer-kind-dropdown":
        cid_eff = None
    else:
        cid_eff = cid if cid in cid_opts else (cid_opts[0] if len(cid_opts) == 1 else None)
    df4 = df3 if cid_eff is None else df3[df3["cid"] == cid_eff]

    la_opts = unique_nonempty(df4["la"])
    if triggered_id in ("resource-kind-dropdown", "resource-id-dropdown",
                         "consumer-kind-dropdown", "consumer-id-dropdown"):
        la_eff = None
    else:
        la_eff = la if la in la_opts else None

    return {
        "rk": {"options": rk_opts, "effective": rk_eff},
        "rid": {"options": rid_opts, "effective": rid_eff},
        "ck": {"options": ck_opts, "effective": ck_eff},
        "cid": {"options": cid_opts, "effective": cid_eff},
        "la": {"options": la_opts, "effective": la_eff},
    }


def filter_single_series(
    dfm: pd.DataFrame,
    rk: Optional[str],
    rid: Optional[str],
    ck: Optional[str],
    cid: Optional[str],
    la: Optional[str],
) -> tuple[pd.DataFrame, dict]:
    """Apply cascading filters and return (filtered_df, cascade_info)."""
    cascade = cascade_filter_options(dfm, rk, rid, ck, cid, la)
    rk_eff = cascade["rk"]["effective"]
    rid_eff = cascade["rid"]["effective"]
    ck_eff = cascade["ck"]["effective"]
    cid_eff = cascade["cid"]["effective"]
    la_eff = cascade["la"]["effective"]

    df = dfm
    if rk_eff is not None:
        df = df[df["rk"] == rk_eff]
    if rid_eff is not None:
        df = df[df["rid"] == rid_eff]
    if ck_eff is not None:
        df = df[df["ck"] == ck_eff]
    if cid_eff is not None:
        df = df[df["cid"] == cid_eff]
    if la_eff is not None:
        df = df[df["la"] == la_eff]
    return df, cascade


def prepare_download_df(
    df_original: pd.DataFrame,
    metric: str,
    rk: Optional[str],
    rid: Optional[str],
    ck: Optional[str],
    cid: Optional[str],
    la: Optional[str],
    proc_start: Optional[pd.Timestamp] = None,
    proc_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Filter and prepare a DataFrame suitable for CSV download."""
    dfm = df_original[df_original["metric"] == metric].copy()

    dfm["rk"] = dfm["resource_kind"].astype(str).replace("nan", "").str.strip()
    dfm["rid"] = dfm["resource_id"].astype(str).replace("nan", "").str.strip()
    dfm["ck"] = dfm["consumer_kind"].astype(str).replace("nan", "").str.strip()
    dfm["cid"] = dfm["consumer_id"].astype(str).replace("nan", "").str.strip()
    dfm["la"] = dfm["__late_attributes"].astype(str).replace("nan", "").str.strip()

    def _norm(v):
        return str(v).strip() if v else ""

    if rk:
        dfm = dfm[dfm["rk"] == _norm(rk)]
    if rid:
        dfm = dfm[dfm["rid"] == _norm(rid)]
    if ck:
        dfm = dfm[dfm["ck"] == _norm(ck)]
    if cid:
        dfm = dfm[dfm["cid"] == _norm(cid)]
    if la:
        dfm = dfm[dfm["la"] == _norm(la)]

    dfm = filter_to_time_range(dfm, proc_start, proc_end, require_bounds=False)

    if dfm.empty:
        return dfm

    dfm = dfm.sort_values("timestamp")

    export_cols = ["timestamp", "metric", "value"]
    for orig_col in ["resource_kind", "resource_id", "consumer_kind", "consumer_id", "__late_attributes"]:
        if orig_col in dfm.columns and dfm[orig_col].notna().any():
            export_cols.append(orig_col)

    return dfm[export_cols].copy()


def safe_metric_filename(metric: str) -> str:
    """Return a filesystem-safe filename stem for *metric*."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in metric)
