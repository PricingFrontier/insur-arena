# prudential_life

- **Task:** multiclf  |  **Source:** GitHub mirror (Kaggle comp.)  |  **Access:** `url_csv:https://raw.githubusercontent.com/AntonUBC/Prudential-Life-Insurance-Assessment/2d2c7e4e2c82766d19dbe25a4cb9c36aab070b66/data/data/train.csv`
- **License:** Kaggle competition data (fetch-only, attributed)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `Response`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** Id
- **Capping:** —
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Prudential LIFE-underwriting risk, 59k, 8-level ordinal Response (imbalanced). Canonical insurance multiclass.
