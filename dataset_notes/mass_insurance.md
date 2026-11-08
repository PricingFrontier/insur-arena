# mass_insurance

- **Task:** freq  |  **Source:** Rdatasets  |  **Access:** `url_csv:https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Insurance.csv`
- **License:** GPL>=2 (open)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `Claims`  |  **Exposure:** `Holders`
- **Dropped (id / leakage / other targets):** rownames
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Aggregated UK third-party motor tariff cells (64 rows). Fast frequency smoke set.
