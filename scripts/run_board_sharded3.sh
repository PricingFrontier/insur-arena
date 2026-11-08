#!/usr/bin/env bash
# THIRD DRIVER — XGBoost only, alongside drivers 1 and 2.
#
# WHY. `xargs -P` finishes a whole wave before starting the next, so each driver blocks on its
# slowest job. On 2026-09-12 that left driver 1 waiting on EBM's last two mid datasets
# (brvehins1 2/9 tuned, brvehins1_pp 5/9) and driver 2 on t-boost's last heavy dataset
# (homesite_conv 8/9) — three jobs, load ~10 on 32 cores, ~21 cores idle — while 120 jobs
# (CatBoost, XGBoost, LightGBM) sat untouched behind them.
#
# WHY XGBOOST. It is the model neither other driver reaches soon: driver 1's post-EBM order starts
# at lightgbm, driver 2's post-t-boost order starts at catboost. When either eventually walks into
# xgboost it cache-hits this driver's finished cells (same shards, same keys; a genuine collision
# is wasteful but never corrupting).
#
# SAME SAFETY STORY AS DRIVER 2, copied verbatim: process-safe cell cache; jitter + retry-only-on-
# the-active_campaign-race; its own unit, log, job-log directory and _done.tsv. Does NOT rebuild
# the board — driver 1 owns that.
#
# SIZING. XGBoost shards inherit `threads_per_worker: 2` from configs/xgboost.yaml (keyed — see
# gen_shard_configs.py). Concurrency heavy 4 / mid 8 / light 10 = 8 / 16 / 20 cores, chosen to
# fit the idle capacity next to driver 2's t-boost waves rather than to fill the box alone.
# MemoryMax=16G: xgboost `hist` holds a quantised matrix, far lighter than t-boost's banks — even
# homesite_conv (261k x 301) is ~0.3 GB of float32 input. The cap exists so that three scopes on a
# 62 GB box cannot jointly overcommit when driver 2 revisits a large fit (its measured peak was
# 40.9 GB, on the heavy wave that is now nearly finished).
set -uo pipefail
cd /home/ralph/suite/insur-arena2

LOG=runs_board_sharded3.log
UNIT=tbshard3
LOGDIR=runs/shard_logs3
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
[ -s configs/shards/_waves3.tsv ] || { log "REFUSING: configs/shards/_waves3.tsv missing"; exit 1; }

systemd-run --user --scope --expand-environment=no \
  -p MemoryMax=16G --unit="$UNIT" \
  bash -c '
    set -uo pipefail
    cd /home/ralph/suite/insur-arena2
    LOGDIR=runs/shard_logs3
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

    for model in xgboost; do
      while read -r wm tier par thr; do
        [ "$wm" = "$model" ] || continue
        n=$(awk -F"\t" -v t="$tier" -v m="$model" "\$1==t && \$2==m" configs/shards/_jobs.tsv | wc -l)
        echo "=== $model / $tier — $n jobs, $par concurrent x $thr threads — $(date -Is) ==="
        awk -F"\t" -v t="$tier" -v m="$model" "\$1==t && \$2==m {print \$2\" \"\$3\" \"\$4}" \
            configs/shards/_jobs.tsv \
          | xargs -P "$par" -I{} bash -c "read -r mm dd cc <<< \"{}\"; run_one $tier \"\$mm\" \"\$dd\" \"\$cc\" $thr"
        echo "=== $model / $tier done — $(date -Is) ==="
      done < configs/shards/_waves3.tsv
      echo "=== MODEL $model COMPLETE — $(date -Is) ==="
    done
  ' >> "$LOG" 2>&1
rc=$?

fails=$(awk -F'\t' '$4 != 0' "$LOGDIR/_done.tsv" 2>/dev/null | wc -l)
log "driver 3 finished (exit $rc, $fails failed jobs) — board NOT rebuilt here; driver 1 owns that"
awk -F'\t' '$4 != 0 {print "  FAILED: "$2"/"$3" rc="$4}' "$LOGDIR/_done.tsv" >> "$LOG" 2>/dev/null
exit $(( rc != 0 || fails != 0 ))
