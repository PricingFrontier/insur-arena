# pg17_payfreq

- **Task:** multiclf  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/pg17trainpol.rda::pg17trainpol`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `pol_pay_freq`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** id_client, id_vehicle, id_policy, id_year
- **Capping:** —
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Pricing Game 2017 INSTALMENT PROPENSITY (Yearly/Monthly/Biannual/Quarterly), 4-class, 100k. Same table as pg17_coverage, different behavioural target.
