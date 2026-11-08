# beMTPL97_pp

- **Task:** pp  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/beMTPL97.rda::beMTPL97`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `amount`  |  **Exposure:** `expo`
- **Dropped (id / leakage / other targets):** id, nclaims, average, claim
- **Capping:** — | **Tweedie power:** 1.7
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Belgian MTPL pure premium (amount is total claim amount).
