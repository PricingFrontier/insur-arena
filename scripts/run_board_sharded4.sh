#!/usr/bin/env bash
# FOURTH DRIVER — LightGBM only, alongside drivers 1-3.
#
# WHY. LightGBM is the LAST model in both driver 1's queue (after EBM, whose light wave is blocked
# behind brvehins1's tuning) and driver 2's (after CatBoost and XGBoost) — days away either way.
# On 2026-09-13 18:05, with t-boost complete, the arena was using ~21.6 of 32 cores. Ralph asked
# for LightGBM to be started now with as many threads as it needs.
#
# THREADS: configs/lightgbm.yaml moved to threads_per_worker 4 + fold_workers 3 at the same time,
# while ZERO LightGBM cells existed, so nothing was invalidated. threads_per_worker is in every
# deploy cell's key and must not move again; fold_workers is wall-clock only. All 40 LightGBM shards
# were regenerated and re-verified against the new campaign config (every proto kind
# byte-identical). Drivers 1 and 2 read the same shard files, so when they eventually reach
# LightGBM they cache-hit this driver's cells rather than recomputing at a different thread count.
#
# CONCURRENCY heavy 1 / mid 2 / light 3 -> peak 12 / 24 / 36 threads (4 threads x 3 parallel inner
# folds per job). Sized to the ~10 cores free at launch rather than to the whole box, growing as the
# other drivers' current waves drain. MemoryMax=16G: LightGBM's binned datasets are small; even
# with three inner folds materialised at once the heaviest set stays well inside it.
#
# Same safety story as drivers 2 and 3: process-safe cell cache; jitter + retry only on the
# active_campaign race; its own unit, log, job-log directory and _done.tsv. Does NOT rebuild the
# board — driver 1 owns that.
set -uo pipefail
cd /home/ralph/suite/insur-arena2

LOG=runs_board_sharded4.log
UNIT=tbshard4
LOGDIR=runs/shard_logs4
log() { echo "[$(date -Is)] $*" >> "$LOG"; }

mkdir -p "$LOGDIR"

# NOTE: deliberately NO "another insurarena run is active" hold — coexisting with driver 1 is the
# entire point of this script. The double-launch guard on THIS unit is kept.
if systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null; then
  log "REFUSING: $UNIT.scope is already active"; exit 1
fi
if systemctl --user list-units --all "$UNIT.scope" --no-legend 2>/dev/null | grep -q .; then
  log "REFUSING: $UNIT.scope is loaded but not active — systemctl --user reset-failed $UNIT.scope"; exit 1
fi
[ -s configs/shards/_waves4.tsv ] || { log "REFUSING: configs/shards/_waves4.tsv missing"; exit 1; }

systemd-run --user --scope --expand-environment=no \
  -p MemoryMax=16G --unit="$UNIT" \
  bash -c '
    set -uo pipefail
    cd /home/ralph/suite/insur-arena2
    LOGDIR=runs/shard_logs4
    DONE=$LOGDIR/_done.tsv
    : > "$DONE"

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

    for model in lightgbm; do
      while read -r wm tier par thr; do
        [ "$wm" = "$model" ] || continue
        n=$(awk -F"\t" -v t="$tier" -v m="$model" "\$1==t && \$2==m" configs/shards/_jobs.tsv | wc -l)
        echo "=== $model / $tier — $n jobs, $par concurrent x $thr threads — $(date -Is) ==="
        awk -F"\t" -v t="$tier" -v m="$model" "\$1==t && \$2==m {print \$2\" \"\$3\" \"\$4}" \
            configs/shards/_jobs.tsv \
          | xargs -P "$par" -I{} bash -c "read -r mm dd cc <<< \"{}\"; run_one $tier \"\$mm\" \"\$dd\" \"\$cc\" $thr"
        echo "=== $model / $tier done — $(date -Is) ==="
      done < configs/shards/_waves4.tsv
      echo "=== MODEL $model COMPLETE — $(date -Is) ==="
    done
  ' >> "$LOG" 2>&1
rc=$?

fails=$(awk -F'\t' '$4 != 0' "$LOGDIR/_done.tsv" 2>/dev/null | wc -l)
log "driver 4 finished (exit $rc, $fails failed jobs) — board NOT rebuilt here; driver 1 owns that"
awk -F'\t' '$4 != 0 {print "  FAILED: "$2"/"$3" rc="$4}' "$LOGDIR/_done.tsv" >> "$LOG" 2>/dev/null
exit $(( rc != 0 || fails != 0 ))
