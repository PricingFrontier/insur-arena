"""TabArena-style Elo ladder for cross-technique comparison.

A "competitor" is a (model, variant). On each (dataset, task) the competitors are ranked by mean
skill-vs-GLM; every pair where both ran yields a win/loss "match". A Bradley-Terry model (logistic
fit on the pairwise design) gives strengths, rescaled to the Elo scale (400/ln10 per logit) and
anchored so `glm/default` = 1000. CIs come from bootstrapping over datasets. (Matches TabArena's
pairwise-win → Elo + bootstrap; anchor is the GLM baseline rather than TabArena's RandomForest.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_SCALE = 400.0 / np.log(10.0)


def _competitor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["competitor"] = df["model"].astype(str) + "/" + df["variant"].astype(str)
    return df


def _fit_bt(per_ds: list[np.ndarray], n: int) -> np.ndarray | None:
    """Bradley-Terry strengths from pairwise wins, via logistic regression on a +1/-1 design."""
    rows, ys = [], []
    for skills in per_ds:                       # skills: array len n, NaN where competitor absent
        present = np.where(~np.isnan(skills))[0]
        for a_i in range(len(present)):
            for b_i in range(a_i + 1, len(present)):
                a, b = present[a_i], present[b_i]
                if skills[a] == skills[b]:
                    continue
                win_a = 1.0 if skills[a] > skills[b] else 0.0
                row = np.zeros(n)
                row[a], row[b] = 1.0, -1.0
                # both orientations -> always both classes present, unbiased BT fit
                rows.append(row); ys.append(win_a)
                rows.append(-row); ys.append(1.0 - win_a)
    if not rows:
        return None
    from sklearn.linear_model import LogisticRegression

    # Mild L2 ridge: prevents Bradley-Terry blow-up under (near-)perfect separation with few
    # datasets; negligible bias as the dataset count grows.
    lr = LogisticRegression(fit_intercept=False, C=30.0, max_iter=2000)
    lr.fit(np.asarray(rows), np.asarray(ys))
    return lr.coef_.ravel()


def elo_ratings(df: pd.DataFrame, anchor: str = "glm/default", n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    if df.empty:
        return df
    d = _competitor(df)
    agg = d.groupby(["competitor", "dataset", "task"])["skill_vs_glm"].mean().reset_index()
    comps = sorted(agg["competitor"].unique())
    idx = {c: i for i, c in enumerate(comps)}
    n = len(comps)

    # one skill-vector (len n, NaN = absent) per (dataset, task)
    per_ds = []
    for _, g in agg.groupby(["dataset", "task"]):
        v = np.full(n, np.nan)
        for c, s in zip(g["competitor"], g["skill_vs_glm"]):
            v[idx[c]] = s
        per_ds.append(v)

    def to_elo(theta):
        if theta is None:
            return None
        elo = theta * _SCALE
        if anchor in idx:
            elo = elo - elo[idx[anchor]] + 1000.0
        return elo

    base = to_elo(_fit_bt(per_ds, n))
    if base is None:
        return pd.DataFrame(columns=["competitor", "elo", "elo_lo", "elo_hi", "n_datasets"])

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = [per_ds[i] for i in rng.integers(0, len(per_ds), len(per_ds))]
        e = to_elo(_fit_bt(sample, n))
        if e is not None:
            boots.append(e)
    boots = np.asarray(boots) if boots else np.empty((0, n))
    lo = np.percentile(boots, 2.5, axis=0) if len(boots) else np.full(n, np.nan)
    hi = np.percentile(boots, 97.5, axis=0) if len(boots) else np.full(n, np.nan)
    n_ds = [int(np.sum([~np.isnan(v[i]) for v in per_ds])) for i in range(n)]

    out = pd.DataFrame({
        "competitor": comps,
        "elo": np.round(base).astype(int),
        "elo_lo": np.round(lo),
        "elo_hi": np.round(hi),
        "n_datasets": n_ds,
    })
    for c in ("elo_lo", "elo_hi"):
        out[c] = out[c].astype("Int64")  # pandas nullable int (handles all-NaN boots)
    return out.sort_values("elo", ascending=False).reset_index(drop=True)


def format_elo(elo: pd.DataFrame) -> str:
    if elo is None or elo.empty:
        return "(elo unavailable — need >=2 competitors across datasets)"
    return elo.to_string(index=False)
