"""Derive the leaderboard: per-entry means, skill, and within-dataset average rank (SPEC §6).

MVP: average rank on the accuracy axis (mean skill-vs-GLM). Critical-difference diagrams /
bootstrap CIs are later additions; the schema and stored predictions already support them.
"""
from __future__ import annotations

import pandas as pd

_ENTRY_KEYS = ["dataset", "task", "model", "variant", "family"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over outer splits per entry, with split SD for the headline accuracy metric."""
    if df.empty:
        return df
    sec_cols = [c for c in df.columns if c.startswith("sec_")]
    agg = {"primary_value": ["mean", "std"], "skill_vs_glm": ["mean", "std"], "split_id": "count"}
    for c in ("train_time_s", "inference_time_s"):
        if c in df.columns:
            agg[c] = "mean"
    for c in sec_cols:
        agg[c] = "mean"
    g = df.groupby(_ENTRY_KEYS, dropna=False).agg(agg)
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    return g.rename(columns={"split_id_count": "n_splits"}).reset_index()


def leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Rank entries within each (dataset, task) by mean skill-vs-GLM (higher = better)."""
    agg = aggregate(df)
    if agg.empty:
        return agg
    agg["acc_rank"] = (
        agg.groupby(["dataset", "task"])["skill_vs_glm_mean"].rank(ascending=False, method="average")
    )
    cols = (
        _ENTRY_KEYS
        + ["acc_rank", "skill_vs_glm_mean", "skill_vs_glm_std", "primary_value_mean", "n_splits",
           "train_time_s_mean", "inference_time_s_mean"]
        + [c for c in agg.columns if c.startswith("sec_") and c.endswith("_mean")]
    )
    cols = [c for c in cols if c in agg.columns]
    return agg[cols].sort_values(["dataset", "task", "acc_rank"]).reset_index(drop=True)


def format_table(lb: pd.DataFrame) -> str:
    if lb.empty:
        return "(no results)"
    show = lb.copy()
    for c in show.select_dtypes("float").columns:
        show[c] = show[c].round(4)
    return show.to_string(index=False)
