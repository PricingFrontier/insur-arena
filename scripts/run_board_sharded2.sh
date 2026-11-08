#!/usr/bin/env bash
# SECOND DRIVER — runs alongside scripts/run_board_sharded.sh, not instead of it.
#
# WHY. Driver 1 walks models sequentially and is stuck on EBM's tail: three mid-tier jobs
# (brvehins1, brvehins1_pp, catelematic13) each need ~6-8 more tuned cells, and a tuned cell is
# expensive for a reason that is by design rather than a defect — `campaign.partition_protocol`
# sets `scoped = replace(cfg, tune_split_id=sid)` and `tune_split_id` is in the tuned cell key, so
# `_tuned_once` runs a FRESH 25-candidate search inside every outer split. Nine outer splits means
# nine HPO searches per dataset. Measured on brvehins1: the tuned DEPLOY fit is 14 s, while the
# cell takes hours. Meanwhile LightGBM, XGBoost, CatBoost and t-boost — 160 of the 200 shards —
# are queued behind that, and the box sits at load ~6 of 32 cores with 55 GB of 62 free.
#
# SAFE TO RUN CONCURRENTLY WITH DRIVER 1:
#   * the per-cell cache is process-safe (unique tempfile + os.replace), and it is the ONLY shared
#     mutable state that matters;
#   * the one hazard — the fixed-name `active_campaign.tmp` write documented in driver 1 — is
#     handled by the same jitter + targeted-retry wrapper, copied here verbatim;
#   * this driver has its OWN unit, log, job-log directory and _done.tsv, so nothing is clobbered.
#
# MODELS IN REVERSE ORDER (t-boost, catboost, xgboost, lightgbm) ON PURPOSE. Driver 1's loop is
# `for model in ebm lightgbm xgboost catboost t-boost` and it is already running, so its model
# list cannot be changed — when it finishes EBM it WILL walk into lightgbm. Starting from the
# opposite end means the two drivers converge from opposite directions and mostly cache-hit each
# other's finished work instead of racing to compute the same uncached cell twice. (A genuine
# collision is wasteful, never corrupting: both processes write the same signature with the same
# content, and os.replace makes that a no-op.) t-boost first is also the useful order — it is the
# model the board exists to place.
#
# DOES NOT REBUILD THE BOARD. Driver 1 owns that step, because it is the one that ends up having
# walked all five models. Run `.venv/bin/python scripts/elo_ladder.py --export results` by hand if
# driver 1 is no longer around when this finishes.
#
# MemoryMax=50G (was 40G): two scopes share one 62 GB box, but driver 1 only ever draws ~3 GB while
# EBM is its tail, and driver 2 carries the heavy t-boost fits. MEASURED 2026-09-11 at the 40G
# cap: allstate_sev 11.0 GB RSS and homesite_conv 8.2 GB, well above the 6.41 GB anchor the waves
# were sized from, taking the scope to 40.3 GB — 94% of cap. It never breached (memory.events
# max=0, oom_kill=0, pressure 0.00), but that is too little headroom to leave in place.
# NOTE the cap is re-applied from THIS LINE on every relaunch, so raising it live with
# `systemctl --user set-property` does not survive — a reboot on 2026-09-11 silently put the
# scope back to 40G and it had to be raised again by hand.
set -uo pipefail
cd /home/ralph/suite/insur-arena2

LOG=runs_board_sharded2.log
UNIT=tbshard2
LOGDIR=runs/shard_logs2
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
[ -s configs/shards/_waves2.tsv ] || { log "REFUSING: configs/shards/_waves2.tsv missing"; exit 1; }

systemd-run --user --scope --expand-environment=no \
  -p MemoryMax=50G --unit="$UNIT" \
  bash -c '
    set -uo pipefail
    cd /home/ralph/suite/insur-arena2
    LOGDIR=runs/shard_logs2
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

    for model in t-boost catboost xgboost lightgbm; do
      while read -r wm tier par thr; do
        [ "$wm" = "$model" ] || continue
        n=$(awk -F"\t" -v t="$tier" -v m="$model" "\$1==t && \$2==m" configs/shards/_jobs.tsv | wc -l)
        echo "=== $model / $tier — $n jobs, $par concurrent x $thr threads — $(date -Is) ==="
        awk -F"\t" -v t="$tier" -v m="$model" "\$1==t && \$2==m {print \$2\" \"\$3\" \"\$4}" \
            configs/shards/_jobs.tsv \
          | xargs -P "$par" -I{} bash -c "read -r mm dd cc <<< \"{}\"; run_one $tier \"\$mm\" \"\$dd\" \"\$cc\" $thr"
        echo "=== $model / $tier done — $(date -Is) ==="
      done < configs/shards/_waves2.tsv
      echo "=== MODEL $model COMPLETE — $(date -Is) ==="
    done
  ' >> "$LOG" 2>&1
rc=$?

fails=$(awk -F'\t' '$4 != 0' "$LOGDIR/_done.tsv" 2>/dev/null | wc -l)
log "driver 2 finished (exit $rc, $fails failed jobs) — board NOT rebuilt here; driver 1 owns that"
awk -F'\t' '$4 != 0 {print "  FAILED: "$2"/"$3" rc="$4}' "$LOGDIR/_done.tsv" >> "$LOG" 2>/dev/null
exit $(( rc != 0 || fails != 0 ))
