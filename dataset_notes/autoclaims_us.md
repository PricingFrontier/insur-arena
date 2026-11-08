# autoclaims_us

- **Task:** sev  |  **Source:** Rdatasets  |  **Access:** `url_csv:https://vincentarelbundock.github.io/Rdatasets/csv/insuranceData/AutoClaims.csv`
- **License:** GPL>=2 (open)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `PAID`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** rownames
- **Capping:** —
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** US auto claim severity, 6.8k.
