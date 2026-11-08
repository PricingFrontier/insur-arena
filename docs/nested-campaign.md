# Rerunning the pricing benchmark

`run_benchmark` now runs nested tuning for every tuned model on every outer split.
The split schedule is shared by all competitors. Candidate caches include exact
outer-training row hashes and software fingerprints. Historical first-split HPO
choices and cells are incompatible and will not be reused. Cross-outer-split
finale pooling is disabled; the existing within-partition candidate search and
ship gate remain. Early-stopping probes, encoders, calibration and final fitting
receive only the current outer training partition. Dataset hygiene is the one
exception: constant and duplicate columns are dropped once per dataset, from the
features alone (never the target), before any split. This is the hygiene the
whole board was scored on.

This changes cost substantially: a nine-split dataset has nine searches per tuned
competitor. The benchmark does not launch automatically when configuration files
are edited. Run the smoke tests before starting the full campaign.

## Response and scoring contract

All adapters are scored by the same external metric functions. Frequency and
pure-premium responses stored as totals become rates with exposure weights;
severity uses mean severity with claim-count weights. Exposure is not multiplied
into sample weights twice. Predictions and inputs must meet the task's finite,
domain and shape requirements. Tweedie scoring power comes from the task spec.
Package-native training losses may differ, particularly proxy-loss adapters;
these are not described as identical optimization objectives.

For a separate t-boost frequency comparison, copy the same dataset/split/config
recipe into two configurations with distinct output directories. Set
`hp: {frequency_formulation: rate_weight}` in one and
`hp: {frequency_formulation: count_offset}` in the other. The latter supplies raw
counts and a fixed exposure offset while both adapters return rates to the same
external scorer. Only frequency supports this option. Keep every other parameter
identical. This comparison is supplementary, not extra tuning on outer test data.

## Retained evidence

Both sequential and process-pool runs write signature-keyed evidence under
`storage.cache_dir/evidence/`. JSON includes protocol, selected parameters, row
hashes, timing scope, A/E and available model inspection information. Successful
cells with `storage.store_predictions: true` also retain a compressed NumPy dump
with canonical and original targets, predictions, scoring weights, exposure,
source row positions and group IDs where available. Load with `allow_pickle=False`.

Raw predictor values are not retained by default. Explicitly list permitted
segment columns using `storage.segment_columns: [Region, VehAge]`; these columns
must exist in every selected dataset. Use separate configurations where schemas
differ. Missing evidence invalidates a metric-cache hit so a resume cannot claim
complete artifacts it no longer has. Failures produce JSON evidence and are not
cached as successful cells. Retained predictions support subsequent segment A/E,
lift and calibration analysis without refitting.

## TabPFN cells

`tabpfn3` is the one comparator run against a hosted, paid API (Prior Labs), default variant
only. The API does not return the fitted model and the runs do not store predictions, so a
TabPFN cell's evidence file records `"evidence_retained": false` instead of the census and
prediction evidence other cells carry. Anything reading evidence must treat
`evidence_retained: false` as absent, never as empty.

The board is the one run `runs/benchmark`, whose manifest lists its cells directly
(`insurarena.results.canonical`), so TabPFN cells stay on the board as the harness changes. Running `configs/tabpfn3.yaml`
calls the API again for every cell whose key has moved, which after any edit under
`src/insurarena` is all of them.

## Timing pass

Accuracy-workload durations are operational measurements, not isolated serving
latency. Run a separate configuration with:

```yaml
evaluation_mode: timing
max_workers: 1
threads_per_worker: 4
timing:
  repeats: 7
  batch_size: 1024
```

Timing mode always makes fresh final fits and prediction measurements. It can
reuse compatible inner tuning choices from the accuracy run in the same cache
directory. It executes one cell at a time, warms the prediction path, then records
single-policy and batch prediction medians and upper quantiles. Input slicing and
loss evaluation are outside those repeated prediction timers. Shut down unrelated
workloads before treating these measurements as hardware comparisons; the runner
cannot control other applications on the host.

Tuning/search retrieval, final fit, prediction, external scoring and inspection
are separate measurements. Final fit includes preprocessing, any early-stopping
probe/refit, and table compilation performed internally by a model's fit method.
Inspection/export time is additional post-fit work; it cannot retrospectively
separate native compilation from fitting. A tuning-cache hit is labelled and its
lookup duration must not be interpreted as search cost.

The runner records package versions, harness/t-boost Python fingerprints and the
t-boost extension fingerprint in cache identities. Freeze the environment and
source before a campaign. A benchmark rerun still does not establish that a dataset
was untouched during library development.

## Smoke check and known limits

Run `python -m insurarena.cli run --config configs/nested-validation-smoke.yaml`
for the synthetic GLM/LightGBM/t-boost check. Its deliberately small fit caps are
for execution testing, not default-model accuracy claims. Enable prediction
storage explicitly when adapting existing campaign YAMLs: several historical
configurations set `store_predictions: false`.

The CLI's partial report reads `active_campaign.json` and its durable result rows,
so changing protocol or model selection does not mix historical cells into the
current leaderboard. Row-index split manifests are stored separately in `splits/`.
Run one coordinator per cache directory; workers within that coordinator are
supported. Timing artifacts have a fresh run identifier and do not overwrite
previous measurements.

Artifact sizes state their serialization method. Native model formats may exclude
adapter preprocessing; standard Python adapters without such a format are measured
using pickle protocol 5 on the locally fitted object, without retaining or loading
a pickle file. Those sizes are not interchangeable deployment formats.
Independent public-table reconstruction currently covers numeric single-output
t-boost models, including float32 boundaries, missing values and factored boxes.
Categorical and multiclass reconstruction is reported as unsupported, rather than
reported as passed. Their available table census and prediction evidence are still
retained.

Known environment gaps: the optional TabDPT and TabICL packages are not installed,
and the installed RustyStats rejects `prior_weight='auto'` in its compiled
target-encoding binding, so those three adapters fail conformance; TabPFN3 is
skipped by its availability gate when no API token is present. Resolve the
optional-engine environment before including those engines in a campaign.
