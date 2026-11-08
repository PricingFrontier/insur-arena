#!/usr/bin/env bash
# ONE DRIVER PER MODEL (2026-09-15) — replaces the five overlapping shard drivers.
#
# WHY. Drivers 1-5 each had a hard-coded model list, and whenever one finished its own model it
# walked into another driver's: driver 1 duplicated LightGBM work for ~2 h on homesite_conv and
# was running porto_seguro alongside driver 4. Ralph asked for one driver per model with as many
# workers as the box allows. Usage:  scripts/run_board_model.sh <catboost|xgboost|lightgbm>
#
# NOTHING WAS KILLED MID-FIT. The cutover SIGKILLed only the schedulers (wave loops and xargs) and
# the job wrappers still in their start-up jitter sleep; the 18 fits already running were left to
# finish in their old scopes ("orphans", listed in configs/shards/_orphans_<model>.txt). This
# driver's job list (configs/shards/_jobs_<model>.tsv) is every dataset for the model that was
# neither complete nor in flight at cutover, so it never duplicates an orphan.
#
# THE SWEEP. After its waves, the driver waits until no process is still running any
# `<model>__*.yaml` shard (i.e. every orphan has exited), then re-runs every dataset of the model
# that is still incomplete. That catches an orphan that failed. Complete datasets replay from cache
# in seconds, so the sweep costs nothing when everything went through.
#
# ONE WAVE PER MODEL (all remaining datasets, mid first then light): xargs refills a slot the
# moment any job ends, so no dataset waits for a slower one. The first cut used mid/light waves
# and xgboost's 23 light datasets were queued behind a single 10-20 h mid job (freMTPLfreq).
# Same shard files, same keys, same active_campaign race handling as before. Concurrency comes
# from configs/shards/_waves_<model>.tsv. It is sized on MEMORY rather than cores: the box is
# already CPU-saturated by the orphans, and the one real hazard is catboost/allstate_sev, which
# has spiked to 30 GB. Oversubscribing CPU is deliberate so cores fill as orphans drain.
set -uo pipefail
MODEL="${1:?model}"
cd /home/ralph/suite/insur-arena2
case "$MODEL" in catboost) MEM=22G ;; xgboost) MEM=12G ;; lightgbm) MEM=16G ;; *) echo "unknown model $MODEL"; exit 2 ;; esac
LOG=runs_board_$MODEL.log
UNIT=tbm-$MODEL
[ -s configs/shards/_waves_$MODEL.tsv ] || { echo "missing waves file" >> "$LOG"; exit 1; }
if systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null; then echo "REFUSING: $UNIT.scope already active" >> "$LOG"; exit 1; fi
systemctl --user reset-failed "$UNIT.scope" 2>/dev/null
mkdir -p runs/shard_logs_$MODEL
systemd-run --user --scope --expand-environment=no -p MemoryMax=$MEM --unit="$UNIT" \
  bash -c '
    set -uo pipefail
    MODEL="$1"; cd /home/ralph/suite/insur-arena2
    LOGDIR=runs/shard_logs_$MODEL; DONE=$LOGDIR/_done.tsv; touch "$DONE"
    run_one() {
      local tier="$1" model="$2" ds="$3" cfg="$4" threads="$5"
      local log="$LOGDIR/${model}__${ds}.log"
      local t0=$(date +%s) rc=1 attempt=0
      sleep $(( (RANDOM % 25) + 1 ))
      while [ $attempt -lt 4 ]; do
        attempt=$(( attempt + 1 ))
        OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
        NUMEXPR_NUM_THREADS=$threads RAYON_NUM_THREADS=$threads \
          nice -n 10 ./.venv/bin/python -m insurarena.cli run --config "$cfg" > "$log" 2>&1
        rc=$?
        [ $rc -eq 0 ] && break
        grep -q "active_campaign.tmp" "$log" || break
        echo "RETRY $model/$ds (attempt $attempt hit the active_campaign race)" >&2
        sleep $(( attempt * 20 + RANDOM % 20 ))
      done
      printf "%s\t%s\t%s\t%s\t%ss\t%s\n" "$tier" "$model" "$ds" "$rc" "$(( $(date +%s) - t0 ))" "try$attempt" >> "$DONE"
      [ $rc -eq 0 ] || echo "JOB FAILED rc=$rc: $model/$ds" >&2
      return 0
    }
    export -f run_one; export LOGDIR DONE
    while read -r wm tier par thr; do
      n=$(awk -F"\t" -v t="$tier" "\$1==t" configs/shards/_jobs_$MODEL.tsv | wc -l)
      echo "=== $MODEL / $tier — $n jobs, $par concurrent — $(date -Is) ==="
      [ "$n" -gt 0 ] || continue
      awk -F"\t" -v t="$tier" "\$1==t {print \$2\" \"\$3\" \"\$4}" configs/shards/_jobs_$MODEL.tsv \
        | xargs -P "$par" -I{} bash -c "read -r mm dd cc <<< \"{}\"; run_one $tier \"\$mm\" \"\$dd\" \"\$cc\" $thr"
      echo "=== $MODEL / $tier done — $(date -Is) ==="
    done < configs/shards/_waves_$MODEL.tsv
    echo "=== waves done; waiting for in-flight $MODEL fits from the old drivers — $(date -Is) ==="
    while pgrep -f "insurarena.cli run --config configs/shards/(straggler/)?${MODEL}__" >/dev/null; do sleep 120; done
    echo "=== sweep: re-running any still-incomplete $MODEL dataset — $(date -Is) ==="
    ./.venv/bin/python - "$MODEL" > configs/shards/_sweep_$MODEL.tsv <<PY
import json, os, sys, collections
m = sys.argv[1]; small = {"credit_g","brautocoll","autobi_us","swautoins","swautoins_pp"}
per = collections.Counter()
for e in os.scandir("runs/benchmark/cells"):
    try: d = json.load(open(e.path))
    except Exception: continue
    r = d.get("row") if isinstance(d, dict) and isinstance(d.get("row"), dict) else None
    if r and r.get("model") == m and r.get("split_id") is not None: per[(r["dataset"], r["variant"])] += 1
for line in open("configs/shards/_jobs.tsv"):
    tier, mm, ds, cfg = line.rstrip("\n").split("\t")
    if mm != m: continue
    need = 30 if ds in small else 9
    if per[(ds, "default")] < need or per[(ds, "tuned")] < need: print(f"{tier} {mm} {ds} {cfg}")
PY
    n=$(wc -l < configs/shards/_sweep_$MODEL.tsv)
    thr=$(awk -F"\t" "NR==1{print \$4}" configs/shards/_waves_$MODEL.tsv); par=$(awk -F"\t" "NR==1{print \$3}" configs/shards/_waves_$MODEL.tsv)
    echo "sweep: $n incomplete dataset(s)"
    [ "$n" -gt 0 ] && xargs -P "$par" -I{} bash -c "read -r tt mm dd cc <<< \"{}\"; run_one \$tt \"\$mm\" \"\$dd\" \"\$cc\" $thr" < configs/shards/_sweep_$MODEL.tsv
    echo "=== MODEL $MODEL COMPLETE — $(date -Is) ==="
  ' _ "$MODEL" >> "$LOG" 2>&1
