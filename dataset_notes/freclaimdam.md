# freclaimdam

- **Task:** sev  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/freclaimset3dam9207.rda::freclaimset3dam9207`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `inc_Y15`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** —
- **Capping:** —
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** French commercial-property DAMAGE claims, 110k. Ultimate incurred (Y15) from risk + early dev (Y0-Y1).
