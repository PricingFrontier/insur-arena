# InsurArena — Specification (v0.2 draft)

> A **TabArena-style benchmark of tabular modelling techniques on insurance-pricing datasets**,
> scored with actuarially-correct metrics. One fitted model per split — no benchmark-level bagging,
> blending, or stacking. The goal is a clean comparison of *techniques* across *datasets*; it is not
> an explainability study.

- **Modeled on:** TabArena-v0.1 ([arXiv:2506.16791](https://arxiv.org/abs/2506.16791)), TabRepo ([arXiv:2311.02971](https://arxiv.org/abs/2311.02971))
- **Status:** design draft. The MVP runs the frequency task end-to-end; the dataset catalog and
  generic loader are working (`DATASETS.md`).

---

## 1. Purpose

No mainstream tabular benchmark (TabArena, Grinsztajn et al., TabRepo, AMLB, TabZilla) contains a
single insurance task — none model **exposure offsets**, **Tweedie / compound-Poisson-Gamma**
targets, or **frequency × severity** structure. Conversely the actuarial literature has never run
modern tabular models on a broad, standardized insurance suite. InsurArena bridges that: **benchmark
modelling techniques (GLM, GBM, …) across many insurance datasets, scored on deviance, Gini, and
calibration with correct exposure handling.**

Two deliberate choices distinguish it from a generic tabular benchmark:

1. **Insurance datasets + actuarial metrics** (deviance / exposure-weighted Gini / calibration), not RMSE/accuracy.
2. **One model per split** — no bagging/stacking. Each entry is a single model, the unit you'd deploy.

It is **not** an explainability benchmark. A GLM is transparent and a GBM is a black box — that's a
known property of the technique, not something we score, gate, or engineer around. The benchmark
measures predictive performance per technique per dataset and lets the reader draw the obvious
interpretability conclusions themselves. It's **open source**: new techniques (other models,
ensembles, interpretable variants) or datasets are added by contributors.

---

## 2. The one methodological constraint

Each benchmark cell is **one model, fit once, from one hyper-parameter configuration**, with **no
benchmark-level resampling aggregation**. A single LightGBM (internally a boosting ensemble) is fine
— it's one fitted artifact. What's excluded is TabArena's 8-fold bag-averaging, its Caruana weighted
ensemble over configs, and cross-model blends — because those aren't a single deployable model.

### What we keep / change vs TabArena-v0.1

| TabArena | InsurArena |
|---|---|
| Nested-CV evaluation geometry (size-dependent outer repeats) | **Keep** (§5.1) |
| 8-fold inner CV → **bagged** child models averaged | **Remove.** Single inner holdout for early stopping; refit one model on train+val (§5.2) |
| `tuned + ensemble` (Caruana blend), cross-model ensemble, AutoGluon stacked reference | **Remove.** Report `default` and `tuned` only |
| Random search, author-vetted spaces, budget = **#configs** | **Keep** (§5.5) |
| Metric: ROC-AUC / RMSE | **Replace** with deviance + Gini + calibration (§3) |
| Stored predictions → offline **ensemble simulation** | Keep prediction storage for offline metric recompute; **no ensemble simulation** |
| Curation discipline, pinned data, fetch routes | **Keep + harden** for insurance (exposure, IID audit, licensing) |
| Foundation models refit on train+val; standard models bag | **Unify:** all families refit on train+val (no data-advantage asymmetry) |

RandomForest/ExtraTrees are pure bagging ensembles, so they're out of scope as "single models" — a
contributor can add them if they want.

---

## 3. Metrics

Actuarial model evaluation reports three things; we report all three and rank on the first.

| Axis | Question | Metric |
|---|---|---|
| **Accuracy** (ranked) | right *level* / honest forecast | strictly-consistent **deviance**, reported as **skill vs a frozen GLM** |
| **Discrimination** | right *order* of risk | **exposure-weighted Gini** (+ normalized Gini secondary) |
| **Calibration** | right aggregate level | **balance** `Σwŷ/Σwy` (→1) + max per-decile deviation |

### 3.1 Per-task primary loss

| Task | Objective | Primary loss | Weight |
|---|---|---|---|
| Frequency | Poisson, log link | mean **Poisson deviance** | exposure |
| Severity | Gamma, log link, `L>0` | mean **Gamma deviance** | claim count |
| Pure premium | Tweedie, `p∈(1,2)` (fixed per dataset) | mean **Tweedie deviance @ fixed p** | exposure |
| Classification | Bernoulli, logit | **log-loss** | exposure / none |
| Multiclass | Multinomial, logit, `K>2` | **multinomial log-loss** (+ weighted accuracy) | none |

Multiclass predictions are a `(n, K)` class-probability matrix; discrimination (Gini) and calibration
(balance) don't extend to `K>2`, so those axes are omitted for this task — it is ranked on multinomial
log-loss (skill vs a frozen multinomial GLM) alone.

Deviance, never RMSE/MAE: it matches the mean-variance structure (`Var ∝ μᵖ`), is the likelihood
optimized (train/eval aligned), and isn't dominated by a few catastrophic claims. For classification
the accuracy headline is log-loss (proper); Gini (`= 2·AUC − 1`) sits on the discrimination axis.

### 3.2 Cross-dataset normalization: skill vs a frozen GLM

For each `(model, dataset, split)`: `Skill = 1 − D(model) / D(GLM_baseline)`, where
`D(GLM_baseline)` is a **frozen, pre-registered** GLM (default config, excluded from the leaderboard)
fit under the identical split/weights/power. `Skill = 0` ties the GLM, `>0` beats it. Unit-free and
poolable across tasks. **D² vs intercept-only** is reported as a secondary "how hard is this dataset"
signal. The frozen GLM is separate from the ranked GLM row (no model scores against itself).

### 3.3 Exposure-weighting is a hard invariant

Every metric is exposure-weighted (claim-count-weighted for severity); the metrics module refuses to
compute them unweighted. Gini cumulates exposure on x and exposure×rate (= loss) on y; CIs bootstrap
at the policy/group level. Exposure enters models as offset/weight, **never a feature** — the loader
rejects an exposure column and flags features collinear with `log(exposure)`.

---

## 4. Models (techniques benchmarked)

A model adapter = one fitted technique with `fit`/`predict`, **task-aware** (it picks the right
distribution/objective from the dataset's task). Shipped techniques:

| Technique | family | Distributions covered |
|---|---|---|
| **GLM** (automated) | `glm` | Poisson · Gamma · Tweedie · logistic · multinomial |
| **LightGBM** | `gbm` | poisson · gamma · tweedie · binary · multiclass |
| **XGBoost** | `gbm` | count:poisson · reg:gamma · reg:tweedie · binary · multi:softprob |
| **CatBoost** | `gbm` | Poisson · Tweedie · Logloss · MultiClass (no native Gamma → skips severity) |
| **EBM** (GA2M) | `gam` | poisson · gamma · tweedie deviance · (multiclass) classifier (`outer_bags=1` — single GA2M) |
| **TabDPT** (foundation) | `foundation` | black-box comparator, all tasks (regressor + classifier); CPU, no auth |
| **TabICL** (foundation) | `foundation` | black-box comparator, classification only; CPU, no auth |
| **TabPFNv2** (foundation) | `foundation` | black-box comparator; license-gated (needs `TABPFN_TOKEN`) |

Foundation models are **exposure-agnostic** (no Poisson/Gamma/Tweedie loss, no sample weights) — they fit a generic regressor/classifier on the rate/severity/binary target and ignore exposure weighting, so they're rough comparators (how predictive is a foundation model here), not exposure-correct pricing models. Categoricals are ordinal-encoded and training is capped for tractability on full data.

The **automated GLM** uses a data-driven pipeline (auto type detection → quantile-binned continuous →
one-hot with rare-level grouping, tuned L2 penalty) — no per-dataset recipe. **EBM runs with
`outer_bags=1`** (a single GA2M, not its default 14-fold internal bag) — both for speed and to stay
consistent with one-model-per-split. The runner **skips any (technique, task) pair the technique
doesn't support** (e.g. CatBoost on severity). Adding a technique (a neural model, …) is one adapter.

Each carries a neutral `family` label for grouping — not a score. There is **no interpretability
scoring, no monotonicity constraint, and no Pareto axis** — interpretability is a known property of
the technique, read off its name.

---

## 5. Evaluation protocol

### 5.0 Dataset eligibility

An official benchmark dataset must contain at least **1,000 modeled rows and 4 predictor columns**
in its full post-task-shaping bundle. A dataset that fails either minimum is excluded from new runs,
cached-cell reports, and task catalog views. Exact-boundary datasets remain eligible. The manifest
retains undersized loaders and their measured dimensions for historical reproducibility; this
requires the explicit `include_ineligible_datasets: true` config/report opt-in.

### 5.1 Outer CV (size-dependent, TabArena-aligned)

Bucket on `N_eff` = number of groups:

| profile | `N_eff < 2,500` | `N_eff ≥ 2,500` |
|---|---|---|
| `full` (official) | 10×3-fold → 30 | 3×3-fold → 9 |
| `lite` (default) | 3×3-fold → 9 | 1×3-fold → 3 |
| `smoke` (CI) | 1×3-fold → 3 | 1×3-fold → 3 |

Reported number = **mean Skill across splits, with across-split SD**.

### 5.2 Config selection & inner split: nested tuning

There are two official regimes: **default** and **tuned**. The `tuned` config is selected separately
inside every outer split, by k-fold out-of-fold (OOF) validation on that split's outer-train pool.
The selected hyper-parameters, including the OOF-derived boosting iteration count when available,
apply to that split only. Each reported cell is still **one model refit on that split's outer-train**,
with no bagging, stacking, or benchmark-level ensembling.

This multiplies the cost of a tuned run by the number of outer splits, which is the price of keeping
every test fold out of selection. See [docs/nested-campaign.md](docs/nested-campaign.md).

### 5.3 Splitting strategy (insurance is often not IID)

Precedence **GROUP > TEMPORAL > STRATIFIED-IID**, pinned per dataset. Group on policy id where rows
repeat (and aggregate per policy before pure-premium joins); stratify on `1[N>0]` / loss-quantile;
temporal forward-chaining where a date exists (labelled, not pooled with IID). A custom splitter
handles group+stratify / group+temporal composition.

### 5.4 Exposure handling

Canonical: rate target + `sample_weight = exposure` (offset form equivalent for count-native GLMs;
never both). Per library: GLM `sample_weight`; LightGBM `weight`/`init_score=log(w)`; XGBoost
`base_margin`. Predictions stored on the rate scale. Tweedie `p` is fixed per dataset for **both**
train and eval (no free knob).

### 5.5 HPO (budget = #configs)

Random search over a **static, seeded config set** (default + `n_configs` sampled, identical across
datasets — enables caching, reproducibility, and tuning trajectories). Search spaces are
**TabArena's author-vetted spaces** (Tables C.1–C.6) for CatBoost/LightGBM/XGBoost/EBM; GBM trees use
`n_estimators` large + early stopping (not tuned), as in TabArena. Budget is in **configs** (`full`
200, `lite` 32) with an optional `time_budget_s` that stops the search early. A bad config is
skipped, never aborts the search. The **tuning trajectory** (best-so-far OOF vs #configs) is stored
for every tuned cell and reported. `default` and `tuned` are ranked separately — a rank column
never mixes HPO regimes.

**Selection protocol v2 (noise-aware).** Candidate OOF scores are recorded **per fold** on folds
shared across candidates, so comparisons are paired. Small tune splits repeat the inner k-fold with
fresh fold seeds (`hpo.inner_repeats: auto` targets ~30k scored rows, capped at 5 repeats) — the
extra fits land exactly where they cost fractions of a second. The static search then applies a
**default ship-gate**: the champion ships only if it beats the default candidate by more than one
standard error of the paired per-fold differences; otherwise the default's hyper-parameters ship
(with the default's OOF-derived iteration count). This caps the downside of tuning at ~zero on
small/sparse portfolios, where argmax-over-noisy-scores otherwise pays the winner's curse, while
leaving genuine wins untouched. The decision and its evidence (margin, SE, fold count) are stored
in the tuned choice cell. Coordinate (greedy) plans get the same treatment: the true default
is evaluated first as candidate 0, each stage move must beat the stage incumbent by >1 paired-fold
SE (the declared prefer/defended near-tie rules keep their authority), and the finished walk passes
the same default ship-gate. The default anchor and tuned-family base are protocol overhead:
`n_configs` still counts stage evaluations only (t-boost's full plan needs 14).

**Search-cost levers (never fidelity levers).** Three opt-in `hpo:` keys cut search wall-clock. All
three touch only how candidates are *ranked against each other*; the shipped model is always refit
on the full outer-train of every outer split, and `default` cells never see any of them. The two
that can in principle move a selection key the tuned cell, so flipping them re-selects rather than
silently reusing a stale choice.

| key | what it does | keyed? |
|---|---|---|
| `fold_workers` | runs one candidate's inner folds concurrently — fold scores bit-identical | no (wall-clock only) |
| `racing_se` | discards a candidate whose **fold-1** paired deficit vs the running incumbent exceeds this many pooled paired-diff SDs; remaining folds are skipped | yes |
| `racing_min_rows` | suppresses racing below this many dataset rows (the effective threshold is what keys the cell, so a suppressed dataset carries the no-racing key) | yes, where it binds |
| `selection_max_rows` | caps the rows the **search** scores on: whole groups (never splitting a panel entity), stratified on the dataset's own stratify label, deterministic in the tune seed | yes |

`selection_max_rows` is the one that trades information for time, so its justification is explicit:
selection needs a *ranking*, and on ≥200k-row portfolios the paired per-fold SEs driving the
ship-gate and racing decisions are ~1e-4 — orders of magnitude below the config differences the
search is resolving. Capping the search sample therefore changes the ranking far less than it
changes the cost, and it changes the deployed model not at all. It is off by default; runs that set
it must report the cap alongside their results, since a tuned cell selected under a cap is not
interchangeable with one selected on full data (hence the cell key).

Racing carries one hard constraint, learned the expensive way: **it may only arm where a single pass
over the inner folds is itself a trustworthy signal.** Discarding a candidate on its first fold
contradicts `inner_repeats: auto`, which repeats the inner k-fold precisely *because* one pass on a
small tune split is noise-dominated. Measured failure: on swautoins (2,182 rows) racing discarded a
champion that had cleared the ship-gate by more than one paired SE; the fallback — the default's
hyper-parameters on the tuned recipe family — scored **below the plain default variant**
(+0.0069 → −0.0041). `racing_min_rows` encodes the consistency condition: with the 3-fold outer
schedule the tune split trains on ~2n/3, so `repeats == 1` requires n ≥ 45,000.

Candidate **evaluation order** is a free variable the spec does not pin: adapters may declare a
relative-cost hint so cheap candidates are evaluated first, which is what makes `racing_se` pay
(a strong incumbent exists before the expensive candidates run). Order changes neither the config
set, nor candidate indices/seeds, nor cached scores; the reported tuning trajectory is rebuilt over
nominal config index so the budget curve keeps meaning "the first N configs of the fixed basis"
whatever order they ran in.

### 5.6 Preprocessing — consistent vs per-model

**Identical across models (pinned by hash):** dataset identity + content hash; target/large-loss
capping (e.g. freMTPL2 `ClaimNb≤4`, `Exposure≤1`); exposure routed to weight and removed from `X`;
cached split vectors; Tweedie power; metric weights. **Per-family (recorded):** categorical encoding,
binning, scaling.

### 5.7 Runtime profiles

`lite` (default) caps datasets to ≤100k rows (exposure-stratified subsample, recorded) with reduced
folds + 32-config HPO — runs on a workstation in a few GPU-hours. Only `full` results are eligible
for the official leaderboard. Compute is logged per cell.

---

## 6. Leaderboard & statistics

- **Normalization:** Skill-vs-frozen-GLM (§3.2).
- **Aggregation (primary):** two-level **average rank** (within dataset → within task → equal task
  weight), with **per-task mean Skill %** as the magnitude companion.
- **Significance (implemented, `results/stats.py`):** **per-dataset Wilcoxon signed-rank +
  Holm-Bonferroni** marks which techniques are not significantly worse than the best on each dataset;
  **Friedman + Nemenyi critical-difference** gives average-rank cliques across datasets (descriptive
  — low power at this scale, pre-registered); a **pairwise win-rate matrix** complements both. All
  printed by the CLI. CIs bootstrap at the group level (Elo CIs already do).
- **Elo ladder (cross-technique):** a TabArena-style Elo over (model, variant) competitors —
  pairwise wins by mean skill-vs-GLM on each (dataset, task), fit as **Bradley-Terry** (logistic),
  rescaled to the Elo scale and **anchored so `glm/default` = 1000** (TabArena anchors on
  RandomForest; we use the GLM baseline). CIs bootstrap over datasets. Implemented in
  `results/elo.py`; printed by the CLI. It earns its keep now that the roster spans several
  techniques.
- **Presentation:** a table per task (+ pooled), one rank column on the accuracy axis, with Gini and
  calibration as reported columns, plus the Elo ladder. No Pareto/tier/interpretability axis.

Example (freMTPL2freq, real data, 8k subsample):

```
model     variant  family  acc_rank  skill   gini    balance
lightgbm  default  gbm      ...      +2.1%   0.282   0.987
glm       tuned    glm      ...      +1.9%   0.276   1.006
lightgbm  ...      gbm      ...      +0.2%   0.237   0.990
glm       default  glm      ...       0.0%   0.268   1.018   (= frozen baseline)
```

---

## 7. Datasets

Sourced via a multi-channel sweep; **all pure-Python, no-auth** routes. The benchmark ships loaders +
pinned access, **never the data** (fetch-only). Adding a dataset is one `DatasetEntry` row in the
single real-data catalog (`datasets/manifest.py`); task-specific lists are derived views, not
separate benchmark populations.

The catalog covers frequency MTPL/auto/motorcycle/vehicle/health portfolios; severity tasks for
workers-comp, allstate, fire, liability, bodily-injury, motorcycle, and individual-claim reserving
(`freclaimdam`); pure-premium task views; insurance classification tasks for auto occurrence,
lapse/retention, life surrender, credit/default-adjacent data, and Porto Seguro; plus multiclass
life-underwriting risk grade and motor coverage-tier choice.

**Curation criteria** (from TabArena + insurance additions): ≥500 effective rows (positive-claim subset
for severity), real-not-synthetic, no leakage, has exposure or it's derivable (for freq/pp), IID audited,
license clearable. A size *ceiling* is no longer imposed — the suite deliberately spans small→large
(1.3k → 595k) and the runtime profile subsamples large sets (`lite` caps to 100k). **Licensing:** never
re-host; CASdatasets / freMTPL are fetch-only; Kaggle competition data is loader-only, fetched from a
stable no-auth mirror (OpenML, or a commit-pinned GitHub raw CSV) and attributed. Full catalog, access
routes, and per-dataset status: **`DATASETS.md`**.

---

## 8. Reproducibility & artifacts

**Per-cell result cache (incremental runs).** Each cell — one `(dataset, model, variant, split)` —
is cached under a signature hashing the dataset **content hash** (which encodes the subsample), the
model/variant/hp-override, the split id, and a **protocol fingerprint** (split schedule,
HPO settings, Tweedie power, protocol version). A run reuses every cell it has already
computed and only runs the rest — so **adding a model or dataset, or extending the subsample, or
changing the tuning protocol, re-runs just the affected cells.** Cache lives in `cache_dir/cells/`;
toggle with `use_cache`. (`results/cache.py`.)

Stored TabRepo-style (predictions enable offline metric/CD/calibration recompute — no ensemble
simulation). Per `(dataset, task, model, variant, split)`: metrics, skill, **train + inference time**
(→ an efficiency/Pareto view), resolved seed, predictions path, and a `failed` field. **Failure
tolerance:** any single cell that errors is recorded with its error and skipped — it never aborts the
run. **Tuning trajectories** (best-so-far OOF vs #configs) are stored for tuned selections. Seeds are a
stable **hash of the identifying tuple**; determinism holds within a pinned environment, with metrics
reproducing within tolerance (GBM/GPU float reductions aren't bit-exact). Dataset identity is pinned
by content hash (resolves the freMTPL2 678,013-vs-678,007 divergence; canonical = OpenML).

*Regulatory note:* insurance pricing is high-scrutiny (EU AI Act, EIOPA, NAIC/Colorado, ASOPs). That
is context for why actuaries care about model choice — it is **not** a benchmark deliverable here;
this benchmark measures predictive performance, and any governance/fairness tooling is an extension a
contributor can add.

---

## 9. Repo / package architecture

```
insur-arena/
  src/insurarena/
    tasks.py                     # TaskType + per-task contracts
    seeds.py scoring.py
    datasets/
      base.py        # DatasetBundle (enforces "exposure is not a feature")
      sources.py     # pure-Python fetch: openml_id | url_csv | casdatasets_rda
      spec.py        # DatasetEntry (manifest row) + build_bundle + GenericLoader
      manifest.py    # the declarative dataset manifest
      registry.py    # name -> loader; single catalog + task views
      loaders/synthetic.py        # offline synthetic data for tests
      preprocessing.py
    metrics/         # deviance · discrimination · calibration · registry
    models/          # glm · gbm(LightGBM) · xgb · catboost · ebm · tabpfn · base · registry
    eval/            # splitting (nested CV + kfold OOF) · hpo (static set, OOF select) · runner
    results/         # schema · store · cache (per-cell reuse) · leaderboard · elo · stats · plots
    config.py cli.py
  configs/  smoke.yaml · lite.yaml · full.yaml · stretch.yaml
  scripts/  validate_datasets.py · gen_dataset_notes.py
  dataset_notes/   # per-dataset curation notes + METADATA.md
  docs/CONTRIBUTING_DATASETS.md
  tests/
```

Key interfaces: `DatasetEntry` (declarative dataset) · `ModelAdapter.fit/predict` (one fitted model)
· `Metric(fn, direction, weighting)`.

---

## 10. Roadmap

- **Phase 1 (done):** frequency MVP — automated GLM + LightGBM, three-axis metrics, nested CV
  (single inner holdout, no bag), `default`+`tuned`, parquet results, leaderboard. Validated on
  synthetic + real freMTPL2freq.
- **Phase 2 (done):** task-aware adapters across all four tasks; **five techniques** (GLM, LightGBM,
  XGBoost, CatBoost, EBM) + a TabPFNv2 black-box comparator; multi-domain real-data suite; Elo ladder.
- **Phase 2.5 — TabArena-methodology adoption (done):** k-fold OOF config selection (not single
  holdout); TabArena author-vetted search spaces + static seeded config set + early-stopping;
  critical-difference (Friedman+Nemenyi), pairwise win-rate, per-dataset Wilcoxon+Holm; failure-
  tolerant cells + train/inference timing + tuning trajectories; data hygiene (datetime expansion,
  flag-and-leave missing, drop constant/dup); per-dataset curation notes + metadata + contribution
  checklist.
- **Phase 3:** neural-model adapter (custom Poisson/Gamma/Tweedie loss); freMTPL2 freq×sev join
  loader; more datasets; bootstrap CIs on the metric leaderboard; "living benchmark" submission flow.

---

## 11. Open decisions (defaults chosen — confirm or override)

| # | Decision | Default |
|---|---|---|
| D1 | Classification accuracy headline | **log-loss** (Gini on discrimination axis) |
| D2 | Temporal holdout for dated datasets | **out of scope v0.1** (current catalog entries used for official IID runs do not carry per-row prediction dates) |
| D3 | Dataset distribution | **fetch-only, never re-host**; results-only leaderboard |
| D4 | Default runtime profile | **`lite`** |
| D5 | Canonical freMTPL2 source | **OpenML 41214/41215** |
| D6 | RandomForest/ExtraTrees | **out of scope** (pure bagging); contributor-addable |

---

## 12. Primary sources

TabArena [arXiv:2506.16791] · TabRepo [arXiv:2311.02971] · Grinsztajn et al. [arXiv:2207.08815] ·
sklearn Tweedie insurance example · Noll–Salzmann–Wüthrich [SSRN 3164764] · Henckaerts et al.
[arXiv:1904.10890] · Lorentzen–Mayer–Wüthrich "Gini and Friends" [SSRN 4248143] ·
Denuit–Charpentier–Trufin autocalibration [arXiv:2103.03635] · Wüthrich–Merz *Statistical
Foundations of Actuarial Learning* · glum · CASdatasets · OpenML · Rdatasets.
