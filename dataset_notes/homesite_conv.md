# homesite_conv

- **Task:** clf  |  **Source:** OpenML  |  **Access:** `openml_id:46857`
- **License:** Kaggle comp. data on OpenML (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `QuoteConversion_Flag`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** QuoteNumber, Original_Quote_Date
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Homesite QUOTE CONVERSION (US home), 260k, ~19% positive. Anonymized real quote book. -1 is a missing-value sentinel in several PersonalField/PropertyField columns.
