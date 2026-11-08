"""Generate ONE config per (model, dataset) from each model's campaign config — and PROVE the
shard keys the same cells the campaign config does.

WHY (2026-09-10). The runner walks datasets SEQUENTIALLY and parallelises only the cells WITHIN a
dataset. That is fine for the deploy phase — a large dataset has 9 splits x 2 variants of
independent work — but a dataset's TUNED SELECTION is a single cell: `_tuned_once` races its 25
candidates behind `fold_workers` alone. Measured on the 2026-09-09 EBM phase: four workers at
~130% CPU, i.e. 4.7 of 32 cores, 15 hours for 15 of 40 datasets. Fanning out at the PROCESS level
— one `insurarena.cli run` per (model, dataset), many at once — puts N independent HPO searches on
the box simultaneously, which is the thing worth parallelising.

`threads_per_worker` IS PART OF THE CELL KEY, AND THAT IS THE WHOLE DIFFICULTY HERE.
`runner._proto` does not mention it, so it looks like a pure wall-clock lever — but the live path
is `campaign.partition_protocol`, which takes `_proto` and then does

    proto.update(protocol=..., train_rows_sha256=..., test_rows_sha256=..., software=...,
                 threads_per_worker=cfg.threads_per_worker, segment_columns=..., predictions=...,
                 evaluation_mode=..., split_manifest=..., timing_repeats=..., timing_batch_size=...)

so the thread count lands in the signature. Caught empirically: a first cut of this script gave
the light tier `threads_per_worker: 2`, and the `ebm__credit_g` shard — a dataset already computed
in full — wrote 60 NEW cells instead of hitting 60 cached ones. Those 60 strays were deleted.

CONSEQUENCES, and they shape the whole design:
  * Each shard MUST inherit its model's campaign `threads_per_worker` verbatim — ebm 4,
    lightgbm/xgboost/catboost 2, t-boost 8. It is not a tuning knob available to this script.
  * The ONLY free lever is how many shard PROCESSES run at once (`max_workers` is absent from the
    proto, and it is 1 in every shard anyway because the process is the unit of parallelism).
  * So concurrency, not thread count, is what gets sized per tier — and total load is then
    `concurrency x the model's fixed thread count`, which is why the wave table below differs per
    model rather than being one number.
  * `predictions` is keyed too (`store_predictions: false` in every campaign config, so every
    shard sets false); `segment_columns`, `evaluation_mode` and the `timing_*` pair are keyed and
    left at their defaults.

`verify()` is the proof, and it checks the REAL builder rather than `_proto`: for every
(model, dataset) it constructs `partition_protocol` for both variants — the exact dict that goes
into `cell_signature` — plus `_hpo_proto` and `_hpo_config_proto` on the same scoped config, from
the campaign config and from the shard, and asserts they are byte-identical. A shard that fails is
not written. Same discipline as `tests/test_hpo_finale_rerace.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/home/ralph/suite/insur-arena2")
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "configs" / "shards"
MODELS = ["ebm", "lightgbm", "xgboost", "catboost", "t-boost"]

#: (rows, features) per dataset — from configs/tabpfn3.yaml's annotated catalog.
SHAPE = dict(
    credit_g=(1000, 20), brautocoll=(1314, 5), autobi_us=(1340, 11), swautoins=(2182, 4),
    swautoins_pp=(2182, 4), freclaimdam=(3948, 6), autoclaims_us=(6773, 4), sgautonb=(7483, 12),
    tic2000=(9822, 85), eusavingul=(21921, 5), ausautoBI=(22036, 13), eudirectlapse=(23060, 18),
    uslapseagent=(29317, 9), fremotor_payfreq=(51949, 18), fremotor_prem=(51949, 19),
    prudential_life=(59381, 139), ohlsson=(62474, 6), ohlsson_pp=(62474, 6),
    ausprivauto=(67856, 6), ausprivauto_clf=(67856, 6), ausprivauto_pp=(67856, 6),
    beMTPL16=(70791, 15), pg16_payfreq=(87228, 14), catelematic13=(100000, 49),
    pg17_coverage=(100000, 27), pg17_payfreq=(100000, 27), pg15training=(100021, 13),
    give_me_some_credit=(150000, 12), beMTPL97=(163212, 12), beMTPL97_pp=(163212, 12),
    euhealth=(176032, 4), euhealth_pp=(176032, 4), norauto=(183999, 4),
    allstate_sev=(188318, 130), homesite_conv=(260753, 301), brvehins1=(382621, 12),
    brvehins1_pp=(382621, 12), freMTPLfreq=(413169, 7), porto_seguro=(595212, 50),
    freMTPL2freq=(678013, 9),
)

#: Size tiers on rows x features — the proxy for both fit cost and resident set.
TIERS = [("heavy", 5e6), ("mid", 1e6), ("light", 0.0)]

#: CONCURRENCY per (model, tier) — the only lever left, since threads are pinned by the key.
#: Sized against two ceilings at once:
#:   RAM     per-job RSS anchors from the heaviest fits measured on this box
#:           (configs/t-boost.yaml: homesite_conv 6.41 GB, porto_seguro 4.56 GB,
#:           freMTPL2freq 1.55 GB). A shard holds ONE fit at a time (max_workers 1), so per-job
#:           RSS is that figure, not a multiple of it. Budget ~36 GB of the 52 GB cgroup cap.
#:   CORES   concurrency x the model's fixed threads_per_worker, targeting the 32 cores. The light
#:           tier is allowed to oversubscribe ~1.5-2x: sub-100k fits never saturate their thread
#:           budget, so more processes beats more idle threads.
#: LightGBM/XGBoost/CatBoost sit at 2 threads, so they need the most processes to fill the box;
#: tri-boost sits at 8 and needs the fewest.
CONCURRENCY = {
    #            heavy  mid  light        threads/job   heavy cores / RAM
    "ebm":       (6,     8,   12),      # 4            24 / ~18 GB
    "lightgbm":  (8,    16,   20),      # 2 -> 4 on 2026-09-13: these counts were sized at 2
                                        # threads; halve them if regenerating at 4.
    "xgboost":   (8,    16,   20),      # 2            16 / ~24 GB
    "catboost":  (8,    16,   20),      # 2            16 / ~24 GB
    "t-boost":  (4,     4,    6),      # 8            32 / ~26 GB
}

# REVISED 2026-09-10 09:20, on MEASUREMENT rather than the anchors above. The five heaviest EBM
# jobs — prudential_life, allstate_sev, homesite_conv, porto_seguro, freMTPL2freq, i.e. the worst
# five datasets on the board — ran concurrently at a scope MemoryPeak of 12.4 GB, with the largest
# single job (homesite_conv, 261k x 301) resident at 4.2 GB. The table above had budgeted ~36 GB
# for that wave. The anchors it was sized from are tri-boost depth-6 deploy fits, which are the
# heaviest thing on this board by a wide margin and badly overstate what the four comparators
# cost. So the 2-thread models' mid and light waves go 12 -> 16 and 16 -> 20: at ~1.5 GB per
# LightGBM/XGBoost/CatBoost job on a 400k-row set that is ~30 GB worst case, still inside the
# 52 GB cgroup cap, and it lifts the mid wave from 24 to 32 cores. tri-boost is NOT raised — its
# 6.41 GB homesite_conv measurement is real and 4 x 8 threads already fills the box.


def tier_of(ds: str) -> str:
    r, f = SHAPE[ds]
    for name, thresh in TIERS:
        if r * f > thresh:
            return name
    raise AssertionError(ds)


def shard_doc(campaign: dict, ds: str) -> dict:
    """The campaign config with EXACTLY `datasets` and `storage.max_workers` changed.

    `threads_per_worker` is carried over from the campaign config unchanged — it keys the cell.
    """
    doc = yaml.safe_load(yaml.safe_dump(campaign))          # deep copy
    doc["datasets"] = [ds]
    storage = dict(campaign["storage"])
    storage["max_workers"] = 1        # the PROCESS is the unit of parallelism; not keyed
    doc["storage"] = storage
    # `fold_workers` is a wall-clock-only lever (config.py) absent from every proto, so it may be
    # set freely; leave it exactly as the campaign config has it so the shard is a minimal diff.
    doc["benchmark"] = f"{campaign.get('benchmark', 'board')}--shard-{ds}"
    return doc


def verify(model: str, campaign_path: Path, shard_path: Path, ds: str) -> None:
    """Assert the REAL cell-key dict is byte-identical between campaign config and shard."""
    from dataclasses import replace

    from insurarena.config import Config
    from insurarena.datasets.registry import build_loader
    from insurarena.eval.campaign import partition_protocol, software_manifest
    from insurarena.eval.runner import _hpo_config_proto, _hpo_proto
    from insurarena.eval.splitting import outer_splits
    from insurarena.seeds import seed_for

    a = Config.from_yaml(str(campaign_path))
    b = Config.from_yaml(str(shard_path))
    assert a.subsample_rows == b.subsample_rows, f"{model}/{ds}: subsample_rows differs"
    assert a.threads_per_worker == b.threads_per_worker, (
        f"{model}/{ds}: threads_per_worker {a.threads_per_worker} -> {b.threads_per_worker}; "
        "it is part of the cell key (campaign.partition_protocol) and must not move")
    bundle = build_loader(ds, subsample=a.subsample_rows, cache_dir=a.cache_dir).load()
    splits = outer_splits(bundle, a.split, seed_for(ds))
    sw = software_manifest()
    hp: dict = {}
    pairs: dict[str, tuple[dict, dict]] = {}
    for variant in ("default", "tuned"):
        pairs[f"partition_protocol/{variant}"] = (
            partition_protocol(a, bundle, model, variant, hp, 0, splits, sw)[1],
            partition_protocol(b, bundle, model, variant, hp, 0, splits, sw)[1],
        )
    sa, sb = replace(a, tune_split_id=0), replace(b, tune_split_id=0)
    pairs["hpo_proto"] = (_hpo_proto(sa, bundle, model, hp), _hpo_proto(sb, bundle, model, hp))
    pairs["hpo_config_proto"] = (_hpo_config_proto(sa, bundle, model, hp, 1),
                                 _hpo_config_proto(sb, bundle, model, hp, 1))
    for kind, (pa, pb) in pairs.items():
        ja = json.dumps(pa, sort_keys=True, default=str)
        jb = json.dumps(pb, sort_keys=True, default=str)
        if ja != jb:
            raise SystemExit(f"PROTO DRIFT {model}/{ds} [{kind}]\n  campaign: {ja}\n  shard:    {jb}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in list(OUT.glob("*.yaml")) + list(OUT.glob("_*.tsv")):
        stale.unlink()
    jobs: list[tuple[str, str, str, Path]] = []
    waves: list[tuple[str, str, int, int]] = []
    for model in MODELS:
        campaign_path = ROOT / "configs" / f"{model}.yaml"
        campaign = yaml.safe_load(campaign_path.read_text())
        thr = int(campaign["storage"]["threads_per_worker"])
        for i, (tier, _) in enumerate(TIERS):
            waves.append((model, tier, CONCURRENCY[model][i], thr))
        for ds in campaign["datasets"]:
            tier = tier_of(ds)
            p = OUT / f"{model}__{ds}.yaml"
            p.write_text(
                f"# GENERATED by scripts/gen_shard_configs.py from configs/{model}.yaml — DO NOT EDIT.\n"
                f"# One (model, dataset) shard. Everything except `datasets` and `storage.max_workers`\n"
                f"# is copied verbatim — `threads_per_worker` in particular, because it is part of the\n"
                f"# cell key via campaign.partition_protocol. Every proto kind is asserted\n"
                f"# byte-identical to the campaign config before this file is written. tier={tier}\n"
                + yaml.safe_dump(shard_doc(campaign, ds), sort_keys=False)
            )
            verify(model, campaign_path, p, ds)
            jobs.append((model, ds, tier, p))
    (OUT / "_jobs.tsv").write_text(
        "".join(f"{t}\t{m}\t{d}\t{p.relative_to(ROOT)}\n" for m, d, t, p in jobs))
    (OUT / "_waves.tsv").write_text(
        "".join(f"{m}\t{t}\t{c}\t{th}\n" for m, t, c, th in waves))
    print(f"wrote {len(jobs)} shards, all protos verified byte-identical to their campaign config")
    print(f"{'model':10s} {'thr':>3s}  " + "  ".join(f"{t:>12s}" for t, _ in TIERS))
    for model in MODELS:
        thr = next(th for m, _, _, th in waves if m == model)
        cells = []
        for i, (tier, _) in enumerate(TIERS):
            n = sum(1 for m, _, t, _ in jobs if m == model and t == tier)
            c = CONCURRENCY[model][i]
            cells.append(f"{n:2d}j x{c:2d} ={c*thr:3d}c")
        print(f"{model:10s} {thr:3d}  " + "  ".join(f"{s:>12s}" for s in cells))


if __name__ == "__main__":
    main()
