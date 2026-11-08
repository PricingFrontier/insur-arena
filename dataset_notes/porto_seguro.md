# porto_seguro

- **Task:** clf  |  **Source:** OpenML  |  **Access:** `openml_id:42206`
- **License:** comp. data on OpenML (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `target`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** id
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Porto Seguro safe-driver (Brazil), 595k, ~3.6% positive, anonymized features. Large end of range.
