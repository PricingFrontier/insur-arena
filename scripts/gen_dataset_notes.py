"""Generate a per-dataset curation note + a metadata table from the manifest (TabArena §1 practice).

Usage:  PYTHONPATH=src python3 scripts/gen_dataset_notes.py
"""
from __future__ import annotations

import pathlib

from insurarena.datasets.manifest import MANIFEST

NOTES = pathlib.Path(__file__).resolve().parent.parent / "dataset_notes"


def _split_mode(e) -> str:
    if e.group_col:
        return f"GROUP (key=`{e.group_col}`) + stratified — policy/group repeats, no cross-split leakage"
    return "stratified-IID on claim occurrence" if e.task.value in ("freq", "clf") else "IID random"


def main() -> int:
    NOTES.mkdir(exist_ok=True)
    rows = []
    for e in MANIFEST:
        exp = f"**Exposure:** `{e.exposure_col}`" if e.exposure_col else "no exposure (full-unit)"
        grp = f" | **Group key:** `{e.group_col}`" if e.group_col else ""
        cap = (", ".join(f"{k}≤{v}" for k, v in e.cap.items())) or "—"
        pw = f" | **Tweedie power:** {e.tweedie_power}" if e.task.value == "pp" else ""
        md = f"""# {e.name}

- **Task:** {e.task.value}  |  **Source:** {e.source}  |  **Access:** `{e.access_method}:{e.locator}`
- **License:** {e.license}  |  **Redistributable:** {e.redistributable} (benchmark is fetch-only — data is never re-hosted)
- **Target:** `{e.target_col}`  |  {exp}{grp}
- **Dropped (id / leakage / other targets):** {", ".join(e.drop_cols) or "—"}
- **Capping:** {cap}{pw}
- **Split mode:** {_split_mode(e)}; no temporal column in this version.
- **IID audit:** {"random split appropriate (published IID use)" if not e.group_col else "grouped — group-aware split enforced"}.
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** {e.notes}
"""
        (NOTES / f"{e.name}.md").write_text(md)
        rows.append((e.name, e.task.value, e.source, e.access_method, str(e.exposure_col or "—"),
                     "Y" if e.redistributable else "fetch-only"))

    tbl = ["# Dataset metadata", "",
           "| dataset | task | source | access | exposure | redistribution |",
           "|---|---|---|---|---|---|"]
    tbl += [f"| {n} | {t} | {s} | {a} | {x} | {r} |" for n, t, s, a, x, r in rows]
    (NOTES / "METADATA.md").write_text("\n".join(tbl) + "\n")
    print(f"wrote {len(rows)} dataset notes + METADATA.md to {NOTES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
