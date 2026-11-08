# swautoins

- **Task:** freq  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/swautoins.rda::swautoins`
- **License:** public domain  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `Claims`  |  **Exposure:** `Insured`
- **Dropped (id / leakage / other targets):** Payment
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Swedish 1977 aggregated rating cells (2182). Grouped-data edge case.
