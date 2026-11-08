# euhealth

- **Task:** freq  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/euhealthinsurance.rda::euhealthinsurance`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `num_visits`  |  **Exposure:** `policy_years` | **Group key:** `id_anon`
- **Dropped (id / leakage / other targets):** num_analysis, num_dentistry, num_diagnostics, num_endoscopy, num_hospitalizations, num_mammography, num_operations, amt_analysis, amt_dentistry, amt_diagnostics, amt_endoscopy, amt_hospitalizations, amt_mammography, amt_operations, amt_visits
- **Capping:** —
- **Split mode:** GROUP (key=`id_anon`) + stratified — policy/group repeats, no cross-split leakage; no temporal column in this version.
- **IID audit:** grouped — group-aware split enforced.
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** European HEALTH, 176k. Outpatient-visit frequency per policy-year. Members repeat -> group.
