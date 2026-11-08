# give_me_some_credit

- **Task:** clf  |  **Source:** OpenML  |  **Access:** `openml_id:45577`
- **License:** comp. data on OpenML (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `SeriousDlqin2yrs`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** Unnamed: 0
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Consumer credit delinquency, 150k, ~6.7% positive.
