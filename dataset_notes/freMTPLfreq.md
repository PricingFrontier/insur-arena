# freMTPLfreq

- **Task:** freq  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/freMTPLfreq.rda::freMTPLfreq`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `ClaimNb`  |  **Exposure:** `Exposure`
- **Dropped (id / leakage / other targets):** PolicyID
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** French MTPL v1 (Charpentier), 413k. Reproduces founding Noll-Salzmann-Wuthrich study.
