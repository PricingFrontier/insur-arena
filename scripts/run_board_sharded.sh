#!/usr/bin/env bash
# FULL BOARD REBUILD, sharded — one `insurarena.cli run` per (model, dataset).
#
# SUPERSEDES scripts/chain_board_2026-09-09.sh, which ran one process per MODEL over all 40
# datasets. That shape left the box 85% idle: the runner walks datasets sequentially and
# parallelises only the cells within one dataset, but a dataset's TUNED SELECTION is a single cell
# whose 25 candidates race behind `fold_workers` alone. Measured on the EBM phase — 4 workers at
# ~130% CPU = 4.7 of 32 cores, 15 hours for 15 of 40 datasets, of which 7h50m went on the five
# sub-2500 sets. Fanning out per (model, dataset) puts N independent HPO searches on the box at
# once, which is the thing worth parallelising.
#
# RESULT-NEUTRAL, AND PROVEN SO. `scripts/gen_shard_configs.py` builds each shard by copying the
# model's campaign config and replacing only `datasets` and `storage`, then asserts all four proto
# kinds (`_proto` default/tuned, `_hpo_proto`, `_hpo_config_proto`) are byte-identical to the
# campaign config's before writing the file. Worker counts are absent from every fingerprint by
# design, so the cells this produces are the cells the unsharded chain would have produced.
#
# SAFE TO RUN CONCURRENTLY. `ResultCache.put` writes a tempfile in the cache dir and `os.replace`s
# it into position, so a reader sees the whole old record or the whole new one. The 465
# `__glm_baseline__` cells every shard needs were all computed by the glm phase on 2026-09-09, so
# concurrent jobs cache-hit them rather than racing to recompute.
#
# THREADS ARE NOT A LEVER HERE. `threads_per_worker` is part of the cell key — `runner._proto`
# omits it, but `campaign.partition_protocol` folds it into the signature, so changing it
# invalidates every cached cell. Proven the hard way: a first cut of the generator retiered the
# thread count and the already-complete `ebm__credit_g` shard wrote 60 fresh cells instead of
# hitting 60 cached ones. Each shard therefore inherits its model's campaign thread count
# verbatim (ebm 4, lightgbm/xgboost/catboost 2, t-boost 8), and the ONLY thing sized per tier is
# how many shard PROCESSES run at once. See scripts/gen_shard_configs.py's CONCURRENCY table for
# the derivation; the waves are read from configs/shards/_waves.tsv so the two cannot drift.
#
# Sized against two ceilings: RAM (per-job RSS anchors from the heaviest fits measured on this
# box — homesite_conv 6.41 GB, porto_seguro 4.56 GB, freMTPL2freq 1.55 GB, and a shard holds ONE
# fit at a time because max_workers is 1), budgeting ~36 GB of the 52 GB cgroup cap; and cores,
# targeting 32 with the light tier allowed to oversubscribe ~1.5x because sub-100k fits never
# saturate their thread budget. Heavy runs FIRST within each model so the long jobs get the box
# rather than becoming the tail.
#
# NOT USING THE GPU, deliberately. The 3080 (10 GB, 2.6 GB already resident) cannot be used here
# without changing results: no adapter has a device knob — `models/xgb.py` hardcodes
# `tree_method="hist"`, and the CatBoost adapter sets no `task_type` — so GPU execution means
# editing adapters, which bumps `adapter_version`, re-keys every cell, and would leave the board
# with two of six competitors fit on different hardware arithmetic than the rest. 10 GB also
# serialises exactly the heavy sets that most need the parallelism. The GPU's real use on this
# benchmark is TabPFN, whose 984 cells are already cached and are the only thing the clear-out
# kept.
set -uo pipefail
cd /home/ralph/suite/insur-arena2

ROOT=/home/ralph/suite/insur-arena2
LOG=runs_board_sharded.log
UNIT=tbshard
JOBS=configs/shards/_jobs.tsv
LOGDIR=runs/shard_logs
log() { echo "[$(date -Is)] $*" >> "$LOG"; }

mkdir -p "$LOGDIR"

while pgrep -f "insurarena[.]cli run" >/dev/null; do
  log "another insurarena run is active; holding"
  sleep 120
done
if systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null; then
  log "REFUSING: $UNIT.scope is already active"; exit 1
fi
if systemctl --user list-units --all "$UNIT.scope" --no-legend 2>/dev/null | grep -q .; then
  log "REFUSING: $UNIT.scope is loaded but not active — systemctl --user reset-failed $UNIT.scope"; exit 1
fi
[ -s "$JOBS" ] || { log "REFUSING: $JOBS missing — run scripts/gen_shard_configs.py first"; exit 1; }

systemd-run --user --scope --expand-environment=no \
  -p MemoryMax=52G --unit="$UNIT" \
  bash -c '
    set -uo pipefail
    cd /home/ralph/suite/insur-arena2
    LOGDIR=runs/shard_logs
    DONE=$LOGDIR/_done.tsv
    : > "$DONE"

    # THE active_campaign.json RACE, and why this is handled here rather than in the source.
    # `campaign.run_campaign` persists a run manifest with
    #     tmp = manifest_path.with_suffix(".tmp"); tmp.write_text(...); os.replace(tmp, manifest_path)
    # and that tmp name is FIXED — runs/benchmark/active_campaign.tmp — so two processes sharing a
    # cache_dir clobber each other: A renames the file away, B os.replace()s a path that no longer
    # exists and dies with FileNotFoundError. It cost 3 of the 12 ebm/mid jobs on the first pass.
    #
    # It cannot be fixed at source. `software_manifest()` sha256s EVERY .py under src/insurarena,
    # and that digest goes into `partition_protocol` -> the cell key, so editing campaign.py
    # invalidates all 12,256 cached cells. Nor can each shard be given its own cache_dir: line 242
    # sets `cfg._split_manifest = str(split_path)` — the ABSOLUTE path of the split file under
    # cache_dir — and that string is in the key too, so a per-shard cache dir re-keys everything
    # just as thoroughly. (Symlinking the shared subdirectories does not help: .resolve() would
    # still yield the shard path.)
    #
    # What makes it tractable is WHERE the writes happen. The only two call sites are line 214
    # (once) and line 255, which sits in the signature-ENUMERATION loop — it runs 1 + variants x
    # splits times, back to back, BEFORE any fitting starts, and never again. So each job emits one
    # sub-second burst at startup and is then silent for minutes to hours. Two defences:
    #   1. JITTER the start so bursts do not overlap. Milliseconds of exposure, seconds of spread.
    #   2. RETRY, but ONLY on this specific traceback — a retry is nearly free because every cell
    #      computed before the crash is already cached (verified: a fully-cached shard replays in
    #      1 second). Any other failure is recorded as a real failure and NOT retried, so this
    #      cannot mask a genuine bug.
    run_one() {
      local tier="$1" model="$2" ds="$3" cfg="$4" threads="$5"
      local log="$LOGDIR/${model}__${ds}.log"
      local t0=$(date +%s) rc=1 attempt=0
      sleep $(( (RANDOM % 25) + 1 ))
      while [ $attempt -lt 4 ]; do
        attempt=$(( attempt + 1 ))
        # run_campaign sets these from threads_per_worker anyway; exported here too so a job
        # thread pool can never be built at 32 before the config is read.
        OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
        NUMEXPR_NUM_THREADS=$threads RAYON_NUM_THREADS=$threads \
          nice -n 10 ./.venv/bin/python -m insurarena.cli run --config "$cfg" > "$log" 2>&1
        rc=$?
        [ $rc -eq 0 ] && break
        grep -q "active_campaign.tmp" "$log" || break     # a real failure: do not retry
        echo "RETRY $model/$ds (attempt $attempt hit the active_campaign race)" >&2
        sleep $(( attempt * 20 + RANDOM % 20 ))
      done
      printf "%s\t%s\t%s\t%s\t%ss\t%s\n" "$tier" "$model" "$ds" "$rc" "$(( $(date +%s) - t0 ))" "try$attempt" >> "$DONE"
      [ $rc -eq 0 ] || echo "JOB FAILED rc=$rc: $model/$ds" >&2
      return 0
    }
    export -f run_one
    export LOGDIR DONE

    rc=0
    for model in ebm lightgbm xgboost catboost t-boost; do
      # NOTE: no IFS override — $'\t' embeds a single quote, which would close the
      # `bash -c '...'` body this whole loop lives inside. Default IFS splits on tabs too.
      while read -r wm tier par thr; do
        [ "$wm" = "$model" ] || continue
        n=$(awk -F"\t" -v t="$tier" -v m="$model" "\$1==t && \$2==m" configs/shards/_jobs.tsv | wc -l)
        echo "=== $model / $tier — $n jobs, $par concurrent x $thr threads — $(date -Is) ==="
        awk -F"\t" -v t="$tier" -v m="$model" "\$1==t && \$2==m {print \$2\" \"\$3\" \"\$4}" \
            configs/shards/_jobs.tsv \
          | xargs -P "$par" -I{} bash -c "read -r mm dd cc <<< \"{}\"; run_one $tier \"\$mm\" \"\$dd\" \"\$cc\" $thr"
        echo "=== $model / $tier done — $(date -Is) ==="
      done < configs/shards/_waves.tsv
      echo "=== MODEL $model COMPLETE — $(date -Is) ==="
    done
    exit $rc
  ' >> "$LOG" 2>&1
rc=$?

fails=$(awk -F'\t' '$4 != 0' "$LOGDIR/_done.tsv" 2>/dev/null | wc -l)
if [ "$rc" -ne 0 ] || [ "$fails" -ne 0 ]; then
  log "sharded board FAILED (driver exit $rc, $fails failed jobs) — board NOT rebuilt"
  awk -F'\t' '$4 != 0 {print "  FAILED: "$2"/"$3" rc="$4}' "$LOGDIR/_done.tsv" >> "$LOG" 2>/dev/null
  exit 1
fi

log "all 200 shards finished clean — rebuilding the board"
./.venv/bin/python scripts/elo_ladder.py --export results >> "$LOG" 2>&1 \
  || { log "elo_ladder --export FAILED"; exit 1; }
log "board rebuilt -> results/leaderboard_cells.csv, results/elo_ladder.csv, results/elo_chart.png"
