# brvehins1_pp

- **Task:** pp  |  **Source:** CASdatasets  |  **Access:** `casdatasets_rda:https://github.com/dutangc/CASdatasets/raw/master/data/brvehins1a.rda::brvehins1a`
- **License:** CASdatasets (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `ClaimAmountTotColl`  |  **Exposure:** `ExposTotal`
- **Dropped (id / leakage / other targets):** ExposFireRob, PremTotal, PremFireRob, SumInsAvg, ClaimNbRob, ClaimNbPartColl, ClaimNbTotColl, ClaimNbFire, ClaimNbOther, ClaimAmountRob, ClaimAmountPartColl, ClaimAmountFire, ClaimAmountOther
- **Capping:** — | **Tweedie power:** 1.7
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Brazilian vehicle total-collision pure premium.
