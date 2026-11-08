"""Cross-dataset statistical comparison (TabArena A.5 / F): critical-difference (Friedman+Nemenyi),
pairwise win-rate matrix, and per-dataset Wilcoxon+Holm significance. Competitor = model/variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _competitor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["competitor"] = df["model"].astype(str) + "/" + df["variant"].astype(str)
    return df


def _mean_skill_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """rows = datasets, cols = competitors, value = mean skill-vs-GLM (higher better)."""
    d = _competitor(df)
    m = d.groupby(["dataset", "competitor"])["skill_vs_glm"].mean().unstack("competitor")
    return m


def critical_difference(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Friedman omnibus + Nemenyi CD over competitors present on ALL datasets.

    Returns dict with mean_rank table, friedman p, CD, and the 'not significantly different from
    best' clique. (Average rank uses higher-skill = better = rank 1.)
    """
    from scipy.stats import friedmanchisquare

    M = _mean_skill_matrix(df).dropna(axis=1, how="any")  # competitors present on every dataset
    out = {"n_datasets": int(M.shape[0]), "competitors": list(M.columns), "note": ""}
    k, N = M.shape[1], M.shape[0]
    if k < 3 or N < 3:
        out["note"] = f"need >=3 competitors-on-all-datasets and >=3 datasets (have {k}, {N})"
        out["table"] = pd.DataFrame()
        return out
    ranks = M.rank(axis=1, ascending=False)            # 1 = best per dataset
    mean_rank = ranks.mean(axis=0).sort_values()
    try:
        _, p = friedmanchisquare(*[M[c].to_numpy() for c in M.columns])
    except Exception:
        p = float("nan")
    try:
        from scipy.stats import studentized_range
        q = float(studentized_range.ppf(1 - alpha, k, np.inf)) / np.sqrt(2.0)
    except Exception:
        q = 2.569  # ~q_0.05 fallback for moderate k
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    best = mean_rank.iloc[0]
    table = pd.DataFrame({
        "competitor": mean_rank.index,
        "mean_rank": mean_rank.values.round(3),
        "not_sig_diff_from_best": (mean_rank.values - best) <= cd,
    })
    out.update(friedman_p=float(p), cd=float(round(cd, 3)), table=table.reset_index(drop=True))
    return out


def pairwise_winrate(df: pd.DataFrame) -> pd.DataFrame:
    """Win-rate matrix: P(row beats col) by mean skill, over datasets where both ran (ties=0.5)."""
    M = _mean_skill_matrix(df)
    comps = list(M.columns)
    out = pd.DataFrame(index=comps, columns=comps, dtype=float)
    for a in comps:
        for b in comps:
            if a == b:
                out.loc[a, b] = np.nan
                continue
            both = M[[a, b]].dropna()
            if len(both) == 0:
                out.loc[a, b] = np.nan
            else:
                wins = (both[a] > both[b]).sum() + 0.5 * (both[a] == both[b]).sum()
                out.loc[a, b] = wins / len(both)
    return out.astype(float).round(3)


def per_dataset_significance(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Per dataset: which competitors are NOT significantly worse than the best, by paired
    Wilcoxon signed-rank on per-split primary values, Holm-corrected. (Low power at few splits —
    pre-registered.)"""
    from scipy.stats import wilcoxon

    d = _competitor(df)
    rows = []
    for ds, g in d.groupby("dataset"):
        piv = g.pivot_table(index="split_id", columns="competitor", values="primary_value")
        if piv.shape[1] == 0:
            continue
        means = piv.mean(axis=0)                       # lower primary (deviance/log-loss) = better
        best = means.idxmin()
        pvals = {}
        for c in piv.columns:
            if c == best:
                continue
            pair = piv[[best, c]].dropna()
            try:
                if pair[best].equals(pair[c]) or len(pair) < 2:
                    pvals[c] = 1.0
                else:
                    pvals[c] = float(wilcoxon(pair[best], pair[c]).pvalue)
            except Exception:
                pvals[c] = 1.0
        # Holm correction
        ordered = sorted(pvals.items(), key=lambda kv: kv[1])
        m = len(ordered)
        sig = {}
        for i, (c, p) in enumerate(ordered):
            sig[c] = p <= alpha / (m - i)              # True => significantly worse than best
        for c in piv.columns:
            rows.append({"dataset": ds, "competitor": c, "mean_primary": round(float(means[c]), 5),
                         "is_best": c == best,
                         "not_sig_worse_than_best": (c == best) or (not sig.get(c, False))})
    return pd.DataFrame(rows)


# -- formatting -------------------------------------------------------------------------------
def format_cd(res: dict) -> str:
    if res.get("note"):
        return f"(critical-difference unavailable: {res['note']})"
    head = (f"Friedman p={res['friedman_p']:.4g}  CD={res['cd']}  "
            f"(k={len(res['competitors'])}, N={res['n_datasets']} datasets)")
    return head + "\n" + res["table"].to_string(index=False)


def format_winrate(wr: pd.DataFrame) -> str:
    return "(win-rate unavailable)" if wr.empty else wr.to_string()


def budget_curve(trajectories: list, points=(1, 2, 5, 10, 25, 50, 100, 200)) -> pd.DataFrame:
    """Average tuned best-so-far OOF primary (lower=better) vs #configs, per (dataset, model)."""
    if not trajectories:
        return pd.DataFrame()
    import collections

    by = collections.defaultdict(list)
    for t in trajectories:
        traj = [np.nan if v is None else float(v) for v in t["trajectory"]]
        by[(t["dataset"], t["model"])].append(traj)
    rows = []
    for (ds, mdl), trajs in by.items():
        L = max(len(x) for x in trajs)
        arr = np.array([x + [x[-1]] * (L - len(x)) if x else [np.nan] * L for x in trajs], dtype=float)
        mean = np.nanmean(arr, axis=0)
        row = {"dataset": ds, "model": mdl}
        for p in points:
            if p <= L:
                row[f"cfg{p}"] = round(float(mean[p - 1]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def format_budget_curve(bc: pd.DataFrame) -> str:
    return "(no tuned trajectories)" if bc.empty else bc.to_string(index=False)
