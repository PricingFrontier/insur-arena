# sgautonb

- **Task:** freq  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/sgautonb.rda::sgautonb`
- **License:** open (Frees)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `Clm_Count`  |  **Exposure:** `Exp_weights`
- **Dropped (id / leakage / other targets):** LNWEIGHT, LossAmt
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Singapore auto, 7.5k.
