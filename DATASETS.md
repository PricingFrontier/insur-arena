# InsurArena — Dataset Catalog

**45 datasets wired and validated; 40 are benchmark-eligible** — one real-data loader catalog, with
task-specific benchmark views. Beyond claims, the catalog now covers **behavioural pricing targets**:
quote conversion, retention/surrender, instalment (payment-frequency) propensity, and market
premium. Each loads through the generic engine (`scripts/validate_datasets.py`),
all via **pure-Python, no-auth** routes. The benchmark ships loaders, **never the data** (fetch-only;
see Licensing).

**Access methods:** `openml_id` (sklearn) · `url_csv` (Rdatasets CSV mirror + pinned GitHub raw CSV) ·
`casdatasets_rda` (`rdata`/`pyreadr`). Adding a dataset is one `DatasetEntry` row in `datasets/manifest.py`.

**All five tasks are runnable** — frequency (Poisson), severity (Gamma), pure premium (Tweedie),
classification (Bernoulli), and **multiclass (multinomial)** — via the task-aware adapters
(`glm`/`lightgbm`/`xgboost`/`catboost`/`ebm`).

## Benchmark eligibility

Official runs require the full, post-task-shaping dataset to have **at least 1,000 modeled rows and
at least 4 predictor columns**. Failing either minimum excludes the dataset from new runs and CLI
reports. The loader remains available so historical results can be reproduced with
`include_ineligible_datasets: true` (or `insurarena report --include-ineligible-datasets`).

| Excluded loader | Modeled rows | Predictors | Failed minimum |
|---|---:|---:|---|
| mass_insurance | 64 | 3 | rows and predictors |
| danishuni | 2,167 | 1 | predictors |
| norfire | 9,181 | 1 | predictors |
| lossalae | 1,500 | 1 | predictors |
| ohlsson_sev | 670 | 6 | rows |

The boundary is inclusive: `credit_g` (exactly 1,000 rows) and datasets with exactly 4 predictors
remain eligible. Dimensions are measured after target/exposure shaping and feature hygiene, before
any run-specific subsampling.

---

## Frequency — Poisson, exposure

| Dataset | Domain | Access | Locator | Rows | License |
|---|---|---|---|---|---|
| freMTPL2freq | auto (FR) | openml_id | 41214 | 678k | CC0 (fetch-only) |
| ausprivauto | auto (AU) | url_csv | insuranceData/dataCar.csv | 68k | GPL≥2 |
| mass_insurance | auto (UK, aggregated) | url_csv | MASS/Insurance.csv | 64 | GPL≥2 |
| freMTPLfreq | auto (FR v1) | casdatasets_rda | freMTPLfreq | 413k | CASdatasets (fetch-only) |
| norauto | auto (NO) | casdatasets_rda | norauto | 184k | CASdatasets (fetch-only) |
| beMTPL97 | auto (BE) | casdatasets_rda | beMTPL97 | 163k | CASdatasets (fetch-only) |
| beMTPL16 | auto (BE) | casdatasets_rda | beMTPL16 | 71k | CASdatasets (fetch-only) |
| swautoins | auto (SE, aggregated) | casdatasets_rda | swautoins | 2.2k | public domain |
| pg15training | auto (pricing game) | casdatasets_rda | pg15training | 100k | CASdatasets (fetch-only) |
| catelematic13 | telematics (CA) | casdatasets_rda | catelematic13 | 100k | CASdatasets (fetch-only) |
| sgautonb | auto (SG) | casdatasets_rda | sgautonb | 7.5k | open (Frees) |
| ohlsson | motorcycle (SE) | url_csv | insuranceData/dataOhlsson.csv | 64k | GPL≥2 |
| brvehins1 | vehicle (BR) | casdatasets_rda | brvehins1a | 393k | CASdatasets (fetch-only) |
| euhealth | health (EU) | casdatasets_rda | euhealthinsurance | 176k | CASdatasets (fetch-only) |

## Severity — Gamma, conditional on a claim

| Dataset | Domain | Access | Locator | Rows | License |
|---|---|---|---|---|---|
| allstate_sev | general (US) | openml_id | 42571 | 188k | comp. data (fetch-only) |
| danishuni | fire (DK) | casdatasets_rda | danishuni | 2.2k | CASdatasets (fetch-only) |
| norfire | fire (NO) | casdatasets_rda | norfire | 9.2k | CASdatasets (fetch-only) |
| lossalae | liability (US) | casdatasets_rda | lossalae | 1.5k | CASdatasets (fetch-only) |
| ausautoBI | bodily-injury (AU) | casdatasets_rda | ausautoBI8999 | 22k | CASdatasets (fetch-only) |
| autoclaims_us | auto (US) | url_csv | insuranceData/AutoClaims.csv | 6.8k | GPL≥2 |
| autobi_us | bodily-injury (US) | url_csv | insuranceData/AutoBi.csv | 1.3k | GPL≥2 |
| ohlsson_sev | motorcycle (SE) | url_csv | insuranceData/dataOhlsson.csv | 64k | GPL≥2 |
| freclaimdam | commercial-property reserving (FR) | casdatasets_rda | freclaimset3dam9207 | 110k | CASdatasets (fetch-only) |
| fremotor_prem | auto market premium (FR) | casdatasets_rda | fremotor1prem0304a | 52k | CASdatasets (fetch-only) |

`freclaimdam` is **individual-claim reserving data** (per-claim risk factors + a full paid/incurred
development history `Y0…Y15`), framed here as Gamma severity on the **ultimate incurred cost** (`inc_Y15`)
predicted from risk factors + **early development** (`paid_Y0/Y1`, `inc_Y0/Y1`) — a real reserving ML task
that needs no new task infrastructure.

`fremotor_prem` is **market-premium modelling**: the target is the real annual premium (`PremTot`) a
French motor insurer charged each policy, predicted from rating factors (incl. `PayFreq`) — Gamma
regression via the severity machinery. The ten per-cover premium components are dropped as
target-derived; policies span 2003–04, so splits group on `IDpol`.

## Pure premium — Tweedie, single-table loss/exposure

| Dataset | Domain | Access | Locator | Rows | License |
|---|---|---|---|---|---|
| ausprivauto_pp | auto (AU) | url_csv | insuranceData/dataCar.csv | 68k | GPL≥2 |
| swautoins_pp | auto (SE, aggregated) | casdatasets_rda | swautoins | 2.2k | public domain |
| beMTPL97_pp | auto (BE) | casdatasets_rda | beMTPL97 | 163k | CASdatasets (fetch-only) |
| ohlsson_pp | motorcycle (SE) | url_csv | insuranceData/dataOhlsson.csv | 64k | GPL≥2 |
| euhealth_pp | health (EU) | casdatasets_rda | euhealthinsurance | 176k | CASdatasets (fetch-only) |
| brvehins1_pp | vehicle (BR) | casdatasets_rda | brvehins1a | 393k | CASdatasets (fetch-only) |

## Classification — Bernoulli

| Dataset | Domain | Access | Locator | Rows | License |
|---|---|---|---|---|---|
| ausprivauto_clf | auto claim occurrence (AU) | url_csv | insuranceData/dataCar.csv | 68k | GPL≥2 |
| credit_g | credit default (DE) | openml_id | 31 | 1k | UCI / CC BY |
| give_me_some_credit | credit delinquency | openml_id | 45577 | 150k | comp. data (fetch-only) |
| tic2000 | caravan-insurance purchase | openml_id | 46281 | 9.8k | UCI / open |
| eudirectlapse | policy lapse/retention (EU) | casdatasets_rda | eudirectlapse | 23k | CASdatasets (fetch-only) |
| uslapseagent | life-policy surrender (US) | casdatasets_rda | uslapseagent | 29k | CASdatasets (fetch-only) |
| porto_seguro | auto claim filed (BR) | openml_id | 42206 | 595k | comp. data (fetch-only) |
| brautocoll | collision fatality (BR) | casdatasets_rda | brautocoll | 1.3k | CASdatasets (fetch-only) |
| homesite_conv | home QUOTE CONVERSION (US) | openml_id | 46857 | 261k | Kaggle comp. (fetch-only) |
| eusavingul | unit-linked SURRENDER (EU) | casdatasets_rda | eusavingULnoPS | 22k | CASdatasets (fetch-only) |

`homesite_conv` is the canonical **quote-conversion** task (Kaggle "Homesite Quote Conversion", real
anonymized quote book, ~19% convert). (A second conversion set, `health_lead` / OpenML 45064, was
dropped: that mirror omits the signal-carrying columns and no model beats an intercept — fails the
non-trivial criterion.) `eusavingul` adds a third retention product line (EU
unit-linked savings surrender, Milhaud) next to `eudirectlapse`/`uslapseagent`; its 20 macro
covariates are keyed to the termination date (in-force rows share one value → perfect class
separation), so they are dropped — the panel variants (`eusavingULnoPSperYr`) are the right home for
macro effects if a hazard-style framing is ever added.

## Multiclass — multinomial, K>2 classes

| Dataset | Domain | Task | Access | Locator | Rows | K | License |
|---|---|---|---|---|---|---|---|
| prudential_life | life underwriting risk grade (US) | multiclf | url_csv | pinned GitHub `train.csv` | 59k | 8 (ordinal) | Kaggle comp. (fetch-only) |
| pg17_coverage | motor coverage-tier choice (FR) | multiclf | casdatasets_rda | pg17trainpol | 100k | 4 | CASdatasets (fetch-only) |
| fremotor_payfreq | motor INSTALMENT propensity (FR) | multiclf | casdatasets_rda | fremotor1prem0304a | 52k | 4 | CASdatasets (fetch-only) |
| pg17_payfreq | motor INSTALMENT propensity (FR) | multiclf | casdatasets_rda | pg17trainpol | 100k | 4 | CASdatasets (fetch-only) |
| pg16_payfreq | fleet INSTALMENT propensity (FR) | multiclf | casdatasets_rda | pg16trainpol | 87k | 3 | CASdatasets (fetch-only) |

`prudential_life` is the canonical insurance multiclass (Kaggle "Prudential Life Insurance Assessment").
No OpenML/Hugging-Face mirror exists, so it's fetched from a **faithful raw-GitHub mirror pinned to a
commit SHA** for reproducibility (competition data → fetch-only, attributed). The multinomial task is
scored on **multinomial log-loss** (+ weighted accuracy); Gini/calibration don't extend to K>2 and are
omitted. Exposure-agnostic (no offset). `glm`/`lightgbm`/`xgboost`/`catboost`/`ebm` support it; the
foundation comparators auto-skip.

The three `*_payfreq` entries model **instalment propensity** — which premium payment frequency
(annual / half-yearly / quarterly / monthly) a policyholder chooses. `fremotor_payfreq` drops all
premium columns (charged premiums embed instalment loadings → target leakage); `pg17_payfreq` reuses
the `pg17trainpol` table with a second behavioural target; `pg16_payfreq` covers the commercial-fleet
segment (grouped on `PolicyID`, 2011–13 panel; `ClaimNb` dropped as post-decision).

---

## Coverage

- **Single loader catalog:** every real dataset is one `DatasetEntry` row in `src/insurarena/datasets/manifest.py`; eligible task lists are derived views, not separate benchmark groups.
- **Frequency:** 14 loaders / 13 eligible across FR/AU/UK/NO/BE/SE/SG/CA/BR/EU — auto, motorcycle, health. All real portfolios (no synthetic).
- **Severity:** 10 loaders / 6 eligible (auto/fire/liability/bodily-injury/motorcycle + individual-claim reserving + market premium).
- **Pure premium:** 6 loaders / 6 eligible single-table datasets (auto/motorcycle/health/BR-vehicle). The freMTPL2 freq×sev join is a follow-up.
- **Classification:** 10 loaders / 10 eligible (auto occurrence + credit + caravan + lapse/retention + life surrender + collision fatality + quote conversion + unit-linked surrender).
- **Multiclass:** 5 loaders / 5 eligible (life underwriting risk grade, motor coverage-tier choice, 3× instalment propensity).
- **Behavioural pricing:** conversion (`homesite_conv`), retention/surrender (`eudirectlapse`, `uslapseagent`, `eusavingul`), instalment propensity (`fremotor_payfreq`, `pg17_payfreq`, `pg16_payfreq`), market premium (`fremotor_prem`).
- Geography now spans FR/UK/US/AU/CA/SG/BE/NO/SE/DK/BR/EU; domains span auto/motorcycle/health/liability/life/property/savings.

## Validation

`scripts/validate_datasets.py` fetch-tests every entry through the generic engine. Two notes from the
last run, both resolved:
- `sgautonb` carried `LNWEIGHT = log(exposure)` as a feature — the **exposure-leak guard caught it**;
  it's now dropped in the manifest. (The guard working as intended.)
- `bnp_paribas` (Cardif) had missing target rows (merged train/test) → dropped from the manifest.

## Licensing posture

- **Never re-hosted.** Loaders fetch from origin; the leaderboard would publish *results*, not data.
- **CASdatasets / freMTPL / norauto / swautoins:** no clean OSI license → **fetch-only**, attributed.
- **OpenML competition mirrors (Allstate, GiveMeSomeCredit, Homesite, Porto Seguro):** fetch-only.
- **Open / redistributable:** credit_g (UCI/CC BY), MASS/insuranceData (GPL).
- **No synthetic data:** the benchmark carries only real portfolios — synthetic/semi-synthetic sets
  (mtpl_shap, medical_cost, workers_comp) are deliberately excluded so results reflect real insurance
  data. (`workers_comp` / OpenML 42876 was dropped 2026-07 when its description was found to state the
  claims are "synthetically generated".)

## How to add a dataset

Append one row to `src/insurarena/datasets/manifest.py`, then run `scripts/validate_datasets.py`:

```python
DatasetEntry(
    name="norauto", task=TaskType.FREQ, source="CASdatasets",
    access_method="casdatasets_rda",
    locator="https://github.com/dutangc/CASdatasets/raw/master/data/norauto.rda::norauto",
    target_col="NbClaim", full_n_rows=183999, full_n_predictors=4,
    exposure_col="Expo", drop_cols=("ClaimAmount",),
    license="CASdatasets (fetch-only)",
)
```

`sources.fetch` → `spec.build_bundle` handles the rest (task shaping, exposure, capping, grouping,
categorical detection). Categoricals are auto-detected, so entries only declare target/exposure/drops.
