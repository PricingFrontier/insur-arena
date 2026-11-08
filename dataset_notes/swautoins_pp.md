# swautoins_pp

- **Task:** pp  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/swautoins.rda::swautoins`
- **License:** public domain  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `Payment`  |  **Exposure:** `Insured`
- **Dropped (id / leakage / other targets):** Claims
- **Capping:** — | **Tweedie power:** 1.7
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Swedish 1977 pure premium (Payment is total claim amount).
