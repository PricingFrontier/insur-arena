# Dataset-Specific Adjustment Log

This file records any dataset-specific preprocessing, modelling, or scoring adjustments that go
beyond the generic benchmark recipe. The default position is to avoid bespoke fixes; prefer
automatic, fit-fold-only rules that apply across datasets. If a dataset-specific adjustment is
needed, document it here before or alongside the code change.

## Entry Template

```text
## <dataset>

- Status: proposed | implemented | rejected | superseded
- Date:
- Code location:
- Adjustment:
- Columns affected:
- Evidence:
- Rationale:
- Fairness notes:
- Benchmark impact:
```

## Policy

- Dataset-specific changes must be explicit in code and documented here.
- Learn any numeric bounds or transforms on the training fold only, then apply them to validation
  and test folds.
- Do not use target or test-fold information to choose bounds.
- Prefer generic robust GLM hygiene, such as continuous-feature winsorization or skew-aware
  transforms, over hand-picking columns for one model.
- Keep exposure and target handling especially visible, since changing them can change the
  estimand.

## Current RustyStats GLM Robustness Investigation

Status: implemented

Date: 2026-07-08

Context: RustyStats default GLM rerun exposed log-link prediction guardrail failures on
`catelematic13`, `freclaimdam`, `workers_comp`, and `ohlsson_pp`. Previous runs mostly did not fail,
but some produced extreme finite metrics, so the new failures appear to be guardrails surfacing
unstable automatic GLM fits rather than a new multiclass issue.

Implemented generic fix:

- Code location: `src/insurarena/models/rustystats_model.py`
- Adapter cache version: `RustyStatsGLM.adapter_version == "7"`
- Fit-fold-only winsorization for continuous numeric features, defaulting to the 0.5% and 99.5%
  training quantiles.
- Skew-aware `log1p` transform for non-negative, heavily right-skewed continuous features.
- Median imputation after the learned continuous transform.
- Default RustyStats prediction policy uses finite eta clipping rather than failing a whole
  benchmark cell after a recoverable extreme linear predictor.
- Pure-premium categorical smoothing: in automatic mode, if any otherwise-dummy low-card
  categorical level has a sparse but non-zero positive-loss count, the low-card dummy block is
  represented with target encoding. Zero-positive levels alone do not trigger this rule, which
  avoids treating structural-zero levels as evidence for smoothing.

These are generic RustyStats adapter rules, not dataset-specific branches. This file records why
the rules were added and which datasets motivated/validated them.

Non-failing dataset regression check:

- Date: 2026-07-08
- Config: `configs/rustystats_av7_others_check.yaml`
- Log: `runs_rustystats_av7_others_check.log`
- Comparison output: `runs/rustystats_av7_others_comparison.csv`
- Scope: 27 datasets excluding `catelematic13`, `freclaimdam`, `workers_comp`, and `ohlsson_pp`.
- Result: 285/285 RustyStats cells completed with no failures.
- Negative skill deltas were tiny: largest drop was `brvehins1` at `-0.000362`, followed by
  `ausprivauto` at `-0.000113`, `ausprivauto_clf` at `-0.000082`, `beMTPL97` at `-0.000064`,
  `euhealth` at `-0.000010`, and `euhealth_pp` at `-0.000003`.
- Several non-failing datasets improved materially, especially `give_me_some_credit` (`+0.122109`)
  and `ohlsson_sev` (`+0.054939`).
- Decision: keep the generic default changes rather than adding dataset-specific branches only for
  the original failures.

## catelematic13

- Status: implemented
- Date: 2026-07-08
- Code location: `src/insurarena/models/rustystats_model.py`
- Adjustment: generic continuous winsorization plus skew-aware `log1p` transforms, learned only on
  the training fold.
- Columns affected: continuous telematics and rating variables, selected automatically by the
  generic continuous-feature rule.
- Evidence: RustyStats guardrail failures were lower-tail eta on zero-claim rows. A targeted
  experiment found `alpha=1.0` with lower-bound-tolerant prediction produced all successful splits
  and positive mean skill, while clipping alone produced catastrophic skill.
- Rationale: the issue looks like insufficient shrinkage/unstable smooth or spline behavior rather
  than a single corrupt column.
- Fairness notes: bounds are learned on the training fold only and applied unchanged to validation
  and test folds.
- Benchmark impact: affected-dataset sweep after implementation: 9/9 successful splits, mean primary
  deviance `0.00072173`, mean skill-vs-GLM `+0.184128`.

## freclaimdam

- Status: implemented
- Date: 2026-07-08
- Code location: `src/insurarena/models/rustystats_model.py`
- Adjustment: prefer skew-aware `log1p` plus winsorization for heavy-tailed early-development amount
  predictors.
- Columns affected: heavy-tailed continuous fields selected automatically; observed motivating
  fields included `paid_Y0`, `paid_Y1`, `inc_Y0`, and `inc_Y1`.
- Evidence: failures were upper-tail eta on rows with million-scale early paid/incurred values.
  Clipping avoided hard failures but still gave poor mean skill, and stronger ridge alone did not
  remove the upper-tail problem.
- Rationale: early-development amount fields are valid reserving predictors, but raw amount scale is
  a brittle input for an automatic log-link GLM.
- Fairness notes: do not drop these columns unless we decide the task definition should exclude
  early-development predictors. `log1p`/winsorization should be fit-fold-only and available as a
  generic continuous-feature rule.
- Benchmark impact: affected-dataset sweep after implementation: 9/9 successful splits, mean primary
  deviance `1.09659`, mean skill-vs-GLM `-0.0691628`.

## workers_comp

- Status: implemented (dataset since REMOVED from the catalog, 2026-07-13 — its OpenML description
  states the 100k claims are "synthetically generated"; no-synthetic policy)
- Date: 2026-07-08
- Code location: `src/insurarena/models/rustystats_model.py`
- Adjustment: generic continuous winsorization and skew-aware transforms; no hard domain cap added.
- Columns affected: likely `HoursWorkedPerWeek` and other right-skewed numeric claim/risk fields,
  selected automatically.
- Evidence: failures were upper-tail eta. Representative extreme rows included implausible
  `HoursWorkedPerWeek` values such as 462.
- Rationale: tree/boosting methods tolerate these values, but automatic GLMs with log links can
  extrapolate violently. Robust continuous preprocessing is standard GLM hygiene.
- Fairness notes: a hard domain cap on hours would be dataset-specific and should only be used if a
  generic train-fold quantile cap is insufficient.
- Benchmark impact: affected-dataset sweep after implementation: 9/9 successful splits, mean primary
  deviance `3.63162`, mean skill-vs-GLM `+0.00706663`.

## ohlsson_pp

- Status: implemented
- Date: 2026-07-08
- Code location: `src/insurarena/models/rustystats_model.py`
- Adjustment: automatic pure-premium categorical smoothing for low-card dummy blocks when a
  train-fold level has sparse but non-zero positive-loss counts.
- Columns affected: selected automatically per fold. In the validation sweep this selected
  `bonuskl`, `kon`, `mcklass`, and `zon`.
- Evidence: failures were lower-tail eta. Most extreme rows had zero target, but at least one
  positive-loss row also had extreme low eta. Loosening or clipping the guard produced catastrophic
  Tweedie metrics rather than a useful fit. Target encoding the low-card categorical block removed
  the held-out cliff.
- Rationale: this looks like a pure-premium/Tweedie instability, not merely an over-strict lower
  eta guardrail.
- Fairness notes: changing exposure filtering or target scale can change the estimand, so any
  adjustment here needs extra care and a rerun against other pure-premium datasets. The implemented
  rule uses only training-fold target counts and does not inspect validation/test outcomes.
- Benchmark impact: affected-dataset sweep after implementation: 9/9 successful splits, mean primary
  deviance `45.9743`, mean skill-vs-GLM `-0.146527`. Pure-premium sanity sweep: `ausprivauto_pp`,
  `beMTPL97_pp`, `euhealth_pp`, and `brvehins1_pp` all had 9/9 successful splits and did not trigger
  the low-card smoothing rule; `brvehins1_pp` mean skill-vs-GLM was `-0.0373131`.
