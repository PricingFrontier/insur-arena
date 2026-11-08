"""Draw the board's Elo chart from an elo_ladder.csv.

Usage:
    python scripts/elo_chart.py results/elo_ladder.csv results/elo_chart.png [more outputs ...]

TabArena-style bars: one bar per technique, sorted by its best Elo (best on the right), the tuned
rating as a wide bar with the default nested inside it, bootstrap intervals as whiskers, and the
GLM anchor as a reference line. Each technique has one hue in a light (default) and dark (tuned) shade; t-boost is
picked out in its own hue.

Lives in scripts/ on purpose: src/insurarena/results/plots.py draws the older chart, and any edit
under src/insurarena changes software_manifest() and so re-keys every cell on the board.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

NAMES = {
    "t-boost": "t-boost", "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "catboost": "CatBoost", "ebm": "EBM", "tabpfn3": "TabPFN", "glm": "GLM",
}
FOCUS = "t-boost"
SHADES = {  # (default, tuned)
    "focus": ("#f6b393", "#eb6834"),   # reference palette orange, tint + base
    "other": ("#b7d0ee", "#2a78d6"),   # reference palette blue, tint + base
}
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"


def elo_chart(ladder: pd.DataFrame, paths: list[str], baseline: str = "glm/default") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    df = ladder.copy()
    df[["base", "variant"]] = df["competitor"].str.split("/", n=1, expand=True)
    df["name"] = df["base"].map(NAMES).fillna(df["base"])
    anchor = float(df.loc[df["competitor"] == baseline, "elo"].iloc[0])
    full = df["n_datasets"].max()

    bars = df[df["competitor"] != baseline]
    order = bars.groupby("name")["elo"].max().sort_values().index.tolist()  # best on the right
    lo_all = min(bars["elo_lo"].fillna(bars["elo"]).min(), anchor)
    floor = (int(lo_all) // 100) * 100 - 100
    top = bars["elo_hi"].fillna(bars["elo"]).max()

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    outer, inner = 0.66, 0.36  # tuned bar at the back, the default nested inside it
    ticks = []
    for i, name in enumerate(order):
        group = bars[bars["name"] == name].set_index("variant")
        shades = SHADES["focus" if name == FOCUS else "other"]
        both = {"default", "tuned"} <= set(group.index)
        layers = [("tuned", outer, shades[1]), ("default", inner, shades[0])] if both \
            else [(group.index[0], outer, shades[0] if group.index[0] == "default" else shades[1])]
        for v, width, color in layers:
            r = group.loc[v]
            partial = r.n_datasets < full
            ax.bar(i, r.elo - floor, bottom=floor, width=width, color=color, zorder=2,
                   hatch="////" if partial else None, edgecolor="white", linewidth=1.2)
            lo = r.elo - (r.elo_lo if pd.notna(r.elo_lo) else r.elo)
            hi = (r.elo_hi if pd.notna(r.elo_hi) else r.elo) - r.elo
            x = i + (outer / 2 - 0.08 if (both and v == "tuned") else 0.0)
            ax.errorbar(x, r.elo, yerr=[[lo], [hi]], fmt="none", ecolor=INK, elinewidth=1,
                        capsize=3, capthick=1, alpha=0.55, zorder=3)
        values = " \u2192 ".join(f"{group.loc[v].elo:.0f}" for v in ("default", "tuned") if v in group.index)
        ticks.append(f"{name}\n{values}")

    ax.axhline(anchor, color=INK, lw=1, ls=(0, (4, 3)), zorder=3)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(ticks, fontsize=9.5, color=INK, linespacing=1.5)
    for label in ax.get_xticklabels():
        if label.get_text().startswith(FOCUS + "\n"):
            label.set_fontweight("bold")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(floor, top + 40)
    ax.set_ylabel("Elo", fontsize=9, color=INK)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    handles = [
        mpatches.Patch(facecolor=SHADES["other"][0], label="Default"),
        mpatches.Patch(facecolor=SHADES["other"][1], label="Tuned"),
    ]
    handles.append(plt.Line2D([], [], color=INK, lw=1, ls=(0, (4, 3)),
                              label=f"GLM baseline ({anchor:.0f})"))
    if (bars["n_datasets"] < full).any():
        handles.append(mpatches.Patch(facecolor="white", edgecolor=MUTED, hatch="////",
                                      label="Not run on every dataset"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, ncol=len(handles),
              handlelength=1.6, handleheight=1.0, borderaxespad=0.2)
    ax.text(1.0, 1.02, "95% bootstrap intervals over datasets", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ladder = pd.read_csv(sys.argv[1])
    elo_chart(ladder, sys.argv[2:])
    for path in sys.argv[2:]:
        print(f"chart   -> {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
