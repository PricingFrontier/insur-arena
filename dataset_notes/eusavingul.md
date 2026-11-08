# eusavingul

- **Task:** clf  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/eusavingULnoPS.rda::eusavingULnoPS`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `lapse.reason`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** policy.ID, issue.date, termination.date, premium.frequency, CPI.relvar1mth, CPI.relvar1qtr, CPI.relvar1yr, CPI.relvar2yr, EUidx.relvar1mth, EUidx.relvar1qtr, EUidx.relvar1yr, EUidx.relvar2yr, rate1Y.relvar1mth, rate1Y.relvar1qtr, rate2Y.relvar1mth, rate2Y.relvar1qtr, rate10Y.relvar1mth, rate10Y.relvar1qtr, unemploy.relvar1mth, unemploy.relvar1qtr, industry.relvar1mth, industry.relvar1qtr, RTV.relvar1mth, RTV.relvar1qtr
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** EU UNIT-LINKED savings SURRENDER (Milhaud), 21.9k, ~36% positive. Death claims (195) fold into the negative class. premium.frequency is a redacted placeholder ('unique') -> dropped.
