"""TabArena-style headline chart: per-technique Elo bars (default vs tuned), ordered by tuned.

Mirrors TabArena's Fig 1: vertical bars per method, regimes overlaid (the shorter regime drawn in
front so both colours show), bootstrap-CI whiskers, a dashed baseline reference line, methods sorted
left→right by tuned Elo, and a hatch for techniques not run on every dataset ("partially imputed").
We have Default + Tuned (no Tuned+Ensembled — single-model-per-split by design).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurarena.results.elo import elo_ratings

_COLORS = {"default": "#aec7e8", "tuned": "#f4a26b"}  # TabArena light-blue / orange-tan


def elo_bar_chart(df: pd.DataFrame, path: str, baseline: str = "glm/default",
                  baseline_elo: float = 1000.0, title: str | None = None) -> str | None:
    """Render the default/tuned Elo bar chart to `path`. Returns the path, or None if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except Exception:
        return None

    elo = elo_ratings(df)
    if elo.empty:
        return None
    elo = elo.copy()
    elo[["base", "variant"]] = elo["competitor"].str.split("/", n=1, expand=True)
    rec = {(r.base, r.variant): r for r in elo.itertuples()}
    maxd = int(elo["n_datasets"].max())

    piv = elo.pivot_table(index="base", columns="variant", values="elo")
    order_col = "tuned" if "tuned" in piv.columns else piv.columns[0]
    # Rank on tuned Elo, FALLING BACK to the technique's own best regime when it has no tuned
    # variant. Sorting on the tuned column alone left default-only techniques (glm, tabpfn3) with a
    # NaN key, which pandas parks at the end — i.e. at the strong end of an ascending axis, reading
    # as better than every tuned method. They are ranked on the score they actually have.
    rank = piv[order_col]
    if len(piv.columns) > 1:
        rank = rank.fillna(piv.max(axis=1, numeric_only=True))
    models = list(rank.sort_values(ascending=True).index)                  # best on the right
    variants = [v for v in ("default", "tuned") if v in set(elo["variant"])]

    lo_all = elo["elo_lo"].fillna(elo["elo"]).min()
    floor = float(np.floor((min(lo_all, baseline_elo) - 60) / 50.0) * 50)
    x = np.arange(len(models))
    w = 0.64

    fig, ax = plt.subplots(figsize=(max(7.0, 1.15 * len(models)), 4.4))
    for i, m in enumerate(models):
        regs = sorted([(v, rec[(m, v)]) for v in variants if (m, v) in rec], key=lambda kv: -kv[1].elo)
        for v, r in regs:                                                   # tallest first (behind)
            hatch = "////" if int(r.n_datasets) < maxd else None
            ax.bar(x[i], r.elo - floor, bottom=floor, width=w, color=_COLORS[v],
                   edgecolor="white", linewidth=0.6, hatch=hatch, zorder=2)
        for v, r in regs:
            lo = r.elo - (r.elo_lo if pd.notna(r.elo_lo) else r.elo)
            hi = (r.elo_hi if pd.notna(r.elo_hi) else r.elo) - r.elo
            ax.errorbar(x[i], r.elo, yerr=[[max(lo, 0)], [max(hi, 0)]], fmt="none",
                        ecolor="0.25", elinewidth=1.1, capsize=2.5, zorder=4)

    ax.axhline(baseline_elo, color="k", ls="--", lw=1.2, zorder=3)
    ax.text(len(models) - 0.5, baseline_elo, f" {baseline} = {int(baseline_elo)} ",
            fontsize=8, va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Elo")
    ax.set_ylim(bottom=floor)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [mpatches.Patch(facecolor=_COLORS[v], label=v.capitalize()) for v in variants]
    handles.append(mpatches.Patch(facecolor="white", edgecolor="0.4", hatch="////", label="Partially imputed"))
    ax.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=True, fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
