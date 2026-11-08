# credit_g

- **Task:** clf  |  **Source:** OpenML  |  **Access:** `openml_id:31`
- **License:** UCI / CC BY 4.0  |  **Redistributable:** True (benchmark is fetch-only — data is never re-hosted)
- **Target:** `class`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** —
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** German credit (Statlog). Adjacent credit-default. Positive = bad credit.
