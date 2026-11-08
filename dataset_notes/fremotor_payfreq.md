# fremotor_payfreq

- **Task:** multiclf  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/fremotor1prem0304a.rda::fremotor1prem0304a`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `PayFreq`  |  no exposure (full-unit) | **Group key:** `IDpol`
- **Dropped (id / leakage / other targets):** PremWindscreen, PremDamAll, PremFire, PremAcc1, PremAcc2, PremLegal, PremTPLM, PremTPLV, PremServ, PremTheft, PremTot
- **Capping:** —
- **Split mode:** GROUP (key=`IDpol`) + stratified — policy/group repeats, no cross-split leakage; no temporal column in this version.
- **IID audit:** grouped — group-aware split enforced.
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** French motor INSTALMENT PROPENSITY: payment frequency (Annual/Half-yearly/Quarterly/Monthly), 4-class, 52k. Policies span 2003-04 -> group by IDpol.
