# brvehins1

- **Task:** freq  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/brvehins1a.rda::brvehins1a`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `ClaimNbTotColl`  |  **Exposure:** `ExposTotal`
- **Dropped (id / leakage / other targets):** ExposFireRob, PremTotal, PremFireRob, SumInsAvg, ClaimNbRob, ClaimNbPartColl, ClaimNbFire, ClaimNbOther, ClaimAmountRob, ClaimAmountPartColl, ClaimAmountTotColl, ClaimAmountFire, ClaimAmountOther
- **Capping:** —
- **Split mode:** stratified-IID on claim occurrence; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** BRAZILIAN vehicle (aggregated cells), 393k. Total-collision claim frequency. First S. American set.
