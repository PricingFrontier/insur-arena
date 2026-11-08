# ausprivauto_clf

- **Task:** clf  |  **Source:** Rdatasets  |  **Access:** `url_csv:https://vincentarelbundock.github.io/Rdatasets/csv/insuranceData/dataCar.csv`
- **License:** GPL>=2 (open)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `clm`  |  **Exposure:** `exposure`
- **Dropped (id / leakage / other targets):** rownames, X_OBSTAT_, claimcst0, numclaims
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Claim occurrence (0/1) on the Australian auto book.
