# danishuni

- **Task:** sev  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/danishuni.rda::danishuni`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `Loss`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** —
- **Capping:** —
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Danish fire losses, 2167. Classic heavy tail.
