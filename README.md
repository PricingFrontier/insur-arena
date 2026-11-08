# InsurArena

A **TabArena-style benchmark of tabular modelling techniques on insurance-pricing datasets.**

It answers a simple question: **how do different modelling techniques (GLM, GBM, …) perform across
real insurance datasets, scored the way actuaries actually score them?** — Poisson/Gamma/Tweedie
deviance, exposure-weighted Gini, and calibration, not RMSE/accuracy.

## What it is (and isn't)

- **It is** a clean comparison of modelling *techniques* on *many* insurance datasets, with
  actuarially-correct metrics and exposure handling. TabArena's methodology, adapted for insurance.
- **One fitted model per split** — no benchmark-level bagging, blending, or stacking. Each entry is
  a single model, the unit you would actually deploy.
- **It isn't** an explainability study. GBMs are black boxes and GLMs are transparent — analysts
  know this; the benchmark doesn't score it or try to change it. It just measures predictive
  performance per technique per dataset.
- **It's open source** — if you want to add a technique (other models, ensembles, interpretable
  variants) or a dataset, you add it to the benchmark.

## How it works

- **Datasets** are fetched at run time from their origin (OpenML / Rdatasets CSV / CASdatasets) —
  pure-Python, no auth, never re-hosted. Adding one is a single `DatasetEntry` row in
  `datasets/manifest.py` (see `DATASETS.md`). Official runs require at least 1,000 modeled rows and
  4 predictors; retained undersized loaders are available only for historical reproduction.
- **Results are cached per cell** (dataset × model × variant × split), with tuned-once HPO choices
  cached per dataset × model, so re-running after adding a model/dataset — or changing the
  subsample/tuning — only computes the new or changed work.
- **Tasks:** frequency (Poisson + exposure), severity (Gamma), pure premium (Tweedie),
  classification (Bernoulli), and multiclass (multinomial) — **all five runnable** via task-aware adapters.
- **Metrics:** primary = deviance, reported as **skill vs a frozen GLM baseline**
  (`1 − D_model/D_GLM`); plus exposure-weighted Gini and calibration (balance). Aggregated by
  average rank, a **TabArena-style Elo ladder** (`glm/default` = 1000), **critical-difference
  (Friedman+Nemenyi)**, a **pairwise win-rate matrix**, and per-dataset **Wilcoxon+Holm** significance.
- **Techniques:** automated GLM (data-driven binning + one-hot, tuned penalty), LightGBM, XGBoost,
  CatBoost, EBM — each a single task-aware adapter — plus an (optional) TabPFNv2 black-box comparator.
  Tuned configs are picked once per dataset × model by **k-fold OOF** over TabArena's author-vetted
  search spaces, then reused across outer splits. Adding a technique (e.g. a neural model) is one file.

## Status

Working across all five tasks (frequency, severity, pure premium, classification, multiclass). **45
real insurance + adjacent datasets** (no synthetic data) are sourced and validated through the generic
engine; **40 meet the benchmark size policy** and carry the published board (`DATASETS.md`). Run it:

```bash
pip install -e ".[gbm,io,dev]"          # or: uv pip install lightgbm rdata pyarrow
python -m insurarena.cli run --config configs/smoke.yaml   # offline synthetic, seconds
python -m insurarena.cli run --config configs/lite.yaml    # real freMTPL2freq (frequency)
python -m insurarena.cli run --config configs/stretch.yaml # freq + severity + pure-prem + classification
python -m pytest -q                                        # offline tests
```

The full design is in **[SPEC.md](./SPEC.md)**; the dataset catalog in **[DATASETS.md](./DATASETS.md)**.

## What's in the repo

| Path | What it holds |
|---|---|
| `src/insurarena/` | The engine: dataset loaders + manifest, splitting, metrics, HPO, and one file per model adapter. |
| `configs/` | Four quick profiles (`smoke`, `lite`, `stretch`, `full`) plus **one campaign config per model**, named for the model it runs: `catboost.yaml`, `ebm.yaml`, `lightgbm.yaml`, `t-boost.yaml`, `xgboost.yaml`. Each covers all 40 eligible datasets and is what produced the published board. |
| `results/` | The published board — `leaderboard_cells.csv` (one row per cell), `elo_ladder.csv`, `elo_chart.png`. |
| `scripts/` | `elo_ladder.py` (build/print the ladder), `resolved_report.py`, `validate_datasets.py`, `gen_dataset_notes.py`. |
| `dataset_notes/` | Per-dataset provenance, shaping and hygiene notes, generated from the manifest. |
| `tests/` | Offline tests — metrics, splitting, adapters, and the HPO/gate protocol invariants. |
| `SPEC.md`, `DATASETS.md` | The methodology and the dataset catalog. |

`runs/` is local run state (per-cell caches, fetched data, predictions) and is not tracked; the
committed `results/` table is the published form. Modelled on TabArena
([arXiv:2506.16791](https://arxiv.org/abs/2506.16791)).

## Benchmark your model

Every published score is in **`results/leaderboard_cells.csv`** — ONE row per
`(model, variant, dataset, split_id)`, with `skill_vs_glm`, the primary metric and its value,
`n_train`/`n_test`, the split `seed`, and the dataset `source_hash`. That file is the comparison
surface: you do not need our per-cell caches (they are not in git).

```bash
git clone <this repo> && cd insur-arena
uv venv --python 3.11 && uv pip install -e ".[all,viz,dev]"   # or: pip install -e ".[all,viz,dev]"
python -m pytest -q                                           # offline; no downloads
```

Tests that exercise an optional engine fail rather than skip when it is missing. Without the
optional engines installed, this is the green run:

```bash
python -m pytest -q --ignore=tests/test_triboost_fidelity_gate.py \
  --ignore=tests/test_triboost_rare_floor_gate.py --ignore=tests/test_triboost_count_channel.py \
  --ignore=tests/test_rustystats_adapter.py \
  --deselect "tests/test_adapters.py::test_adapter_conformance[tabdpt]" \
  --deselect "tests/test_adapters.py::test_adapter_conformance[tabicl]" \
  --deselect "tests/test_adapters.py::test_adapter_conformance[rustystats]"
```

`tri_boost`, `tabdpt` and `tabicl` are optional engines. The `rustystats` adapter is deselected
too: it is an optional GLM engine, is not on the published board, and does not currently fit
against `rustystats==0.8.13` (`prior_weight` typing, `rustystats_model.py:453`).

**Datasets are fetched at load time** from OpenML / Rdatasets / CASdatasets — nothing is re-hosted
here, so a first run needs network. No credentials are required for those three; CASdatasets `.rda`
files need the `casdata` extra (`rdata`). TabPFN3 is the one gated comparator (license acceptance +
`TABPFN_TOKEN`). `tri_boost` is **not** on PyPI as used here — build it from the tri-boost repo with
`maturin develop --release` into this venv (pins that matter: `rustystats==0.8.13`, `numpy==2.5.2`).

**Splits are deterministic and dataset-derived.** For dataset `D`:

```python
from insurarena.datasets.registry import build_loader
from insurarena.eval.splitting import outer_splits
from insurarena.seeds import seed_for
bundle = build_loader("credit_g", cache_dir="runs/benchmark").load()
splits = outer_splits(bundle, {"small_threshold": 2500, "small": {"folds": 3, "repeats": 10},
                               "large": {"folds": 3, "repeats": 3}}, seed_for("credit_g"))
```

`seed_for(D)` is a SHA-256 hash of the dataset name (no arithmetic seeds), and the schedule block is
the `split:` section of the campaign configs (`configs/t-boost.yaml` et al): under
2,500 groups → 3×10 = **30 splits**, otherwise 3×3 = **9 splits**. Folds are group-honest and
stratified where the bundle declares it (`GroupKFold` / `StratifiedGroupKFold` / `StratifiedKFold`).
`split_id` is the position in that list, so split *k* here is split *k* there. Check
`bundle.meta["hash"]` against the `source_hash` column before comparing.

**Two ways to compete.** Either add an adapter — one file in `src/insurarena/models/`, one line in
`src/insurarena/models/registry.py`, then run `python -m insurarena.cli run --config <your.yaml>`
— or fit externally on each `(train_idx, test_idx)` and score yourself. The metric is
**skill vs the frozen GLM baseline**, per split:

    skill_vs_glm = 1 - D_model / D_GLM

where `D` is the task's primary deviance on the test fold (Poisson / Gamma / Tweedie / Bernoulli /
multinomial — see `insurarena.metrics.deviance`). Reuse our `D_GLM` rather than refitting it: the
`glm/default` row for that `(dataset, split_id)` carries it directly as `primary_value` (its own
skill is 0 by construction); from any other row, `D_GLM = primary_value / (1 - skill_vs_glm)`.
Positive beats the GLM; the ladder aggregates the per-split mean per dataset.

**Then place yourself on the ladder.** Append your rows (same columns, `model`/`variant` of your
choosing) and recompute:

```bash
python scripts/elo_ladder.py --cells results/leaderboard_cells.csv   # ladder from the table alone
python scripts/elo_ladder.py --export results                        # re-extract from runs/benchmark
```

`results/elo_ladder.csv` holds the current Bradley-Terry Elo (bootstrap CIs, `glm/default` = 1000)
and `results/elo_chart.png` is rendered from that same table.

## License

MIT — see [LICENSE](./LICENSE). Datasets are fetched from their origins under their own terms;
none are re-hosted here.

For the corrected nested rerun protocol, retained pricing evidence and serial timing pass, see [nested campaign](docs/nested-campaign.md).
