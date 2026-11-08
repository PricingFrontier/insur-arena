# ausautoBI

- **Task:** sev  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/ausautoBI8999.rda::ausautoBI8999`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `AggClaim`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** AccDate, ReportDate, FinDate, FinMth, OpTime
- **Capping:** —
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Australian auto BODILY-INJURY closed claims (de Jong-Heller), 22k. Liability severity.
