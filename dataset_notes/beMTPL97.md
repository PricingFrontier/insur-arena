# beMTPL97

- **Task:** freq  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/beMTPL97.rda::beMTPL97`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `nclaims`  |  **Exposure:** `expo`
- **Dropped (id / leakage / other targets):** id, amount, average, claim
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Belgian MTPL 1997 (Henckaerts). Has geo coords.
