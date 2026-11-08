# ohlsson_pp

- **Task:** pp  |  **Source:** Rdatasets  |  **Access:** `url_csv:https://vincentarelbundock.github.io/Rdatasets/csv/insuranceData/dataOhlsson.csv`
- **License:** GPL>=2 (open)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `skadkost`  |  **Exposure:** `duration`
- **Dropped (id / leakage / other targets):** rownames, antskad
- **Capping:** — | **Tweedie power:** 1.7
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Swedish motorcycle pure premium (skadkost is total claim cost).
