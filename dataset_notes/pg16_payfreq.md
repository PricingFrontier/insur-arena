# pg16_payfreq

- **Task:** multiclf  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/pg16trainpol.rda::pg16trainpol`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `PayFreq`  |  no exposure (full-unit) | **Group key:** `PolicyID`
- **Dropped (id / leakage / other targets):** LicNb, ClaimNb, BeginDate, EndDate, Exposure, VehiclCateg
- **Capping:** —
- **Split mode:** GROUP (key=`PolicyID`) + stratified — policy/group repeats, no cross-split leakage; no temporal column in this version.
- **IID audit:** grouped — group-aware split enforced.
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Pricing Game 2016 COMMERCIAL-FLEET instalment propensity (year/semester/quarter), 3-class, 87k. Policies span 2011-13 -> group by PolicyID.
