# autobi_us

- **Task:** sev  |  **Source:** Rdatasets  |  **Access:** `url_csv:https://vincentarelbundock.github.io/Rdatasets/csv/insuranceData/AutoBi.csv`
- **License:** GPL>=2 (open)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `LOSS`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** rownames, CASENUM
- **Capping:** —
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** US auto bodily-injury severity, 1.3k (small end of range). Has NaNs -> flag-and-leave.
