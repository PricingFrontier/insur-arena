# Contributing a dataset

InsurArena is open source — add a dataset by appending one `DatasetEntry` row to
`src/insurarena/datasets/manifest.py`, then run `PYTHONPATH=src python3 scripts/validate_datasets.py`
and `python3 scripts/gen_dataset_notes.py`. Datasets are **fetch-only** (we ship a loader, never the
data). Adapted from TabArena's dataset-contribution checklist (paper §E.3), with insurance additions.

## Checklist (answer each before adding)

1. **Access & license** — is there a public, no-auth fetch route (OpenML id, a stable CSV URL, a
   CASdatasets `.rda`)? Does the license permit benchmark *use* (we don't redistribute the data)?
   Kaggle competition data is loader-only.
2. **Sample size** — ≥ 500 effective rows (groups, not raw rows). For severity, the positive-claim
   subset must clear ~500. Very large datasets are allowed, but note the runtime implications.
3. **Modality** — genuinely tabular (not featurized image/text/audio).
4. **Real, not synthetic** — flag semi-synthetic/textbook data (e.g. the medical-charges set).
5. **Predictive task** — published for prediction, not hypothesis-testing / visualization.
6. **Exposure** — does it have an exposure column (or a uniform policy term → exposure = 1)? Required
   for `freq`/`pp`; optional for `sev`/`clf`. **Exposure is scale, never a feature** — the loader
   rejects an exposure column in `X` and flags features collinear with `log(exposure)`.
7. **Distribution / task** — frequency (Poisson), severity (Gamma), pure premium (Tweedie),
   classification (Bernoulli), or multiclass (multinomial)? Set `task` + (for `pp`) `tweedie_power`.
8. **Leakage** — list post-outcome / target-derived columns to `drop_cols`; identify a `group_col`
   if a policy appears in multiple rows (multi-claim severity, longitudinal panels).
9. **Temporal structure** — is there a date column and is future-prediction the real task? If so the
   dataset needs temporal (not IID) splitting — currently out of scope; flag it.
10. **Preprocessing already applied** — prefer the rawest version; record any pinned capping/cleaning
    (e.g. freMTPL2's `ClaimNb≤4`, `Exposure≤1`) in `cap` so results are reproducible.
11. **Non-trivial** — the canonical GLM must beat an intercept-only model; exclude datasets a model
    trivially solves or where several tie at the ceiling.

## Then

- `validate_datasets.py` must load it cleanly (correct task, exposure detected, no leakage).
- `gen_dataset_notes.py` regenerates the curation note + `dataset_notes/METADATA.md`.
- Add it to the single manifest catalog and to the appropriate task section in `DATASETS.md`.
