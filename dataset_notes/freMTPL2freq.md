# freMTPL2freq

- **Task:** freq  |  **Source:** OpenML  |  **Access:** `openml_id:41214`
- **License:** CC0 on OpenML (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `ClaimNb`  |  **Exposure:** `Exposure`
- **Dropped (id / leakage / other targets):** IDpol
- **Capping:** ClaimNb≤4, Exposure≤1.0
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Canonical French MTPL frequency; ~94% zero.
