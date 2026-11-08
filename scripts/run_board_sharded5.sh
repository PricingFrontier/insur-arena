#!/usr/bin/env bash
# FIFTH DRIVER — LightGBM MID + LIGHT tiers only (2026-09-15).
#
# WHY. Driver 1 finished EBM at 14:04 and, as its hard-coded model loop dictates, walked into
# LightGBM — which driver 4 already owned. Measured duplication: both drivers worked
# lightgbm/homesite_conv until 16:04 (driver 1 for ~2 h), and both were on lightgbm/porto_seguro
# at 16:50, with freMTPL2freq next in both queues. The fix retired the duplicates instead of
# letting them race:
#   * driver 1 paused (SIGSTOP on its wave loop) so it starts nothing new; its two in-flight
#     LightGBM HEAVY jobs (porto_seguro, freMTPL2freq) finish, then its scope is stopped;
#   * driver 4 stopped right after a committed porto_seguro cell (straggler_handoff.py
#     --stop-unit), so it never repeats those two heavy sets;
#   * THIS driver takes LightGBM's 12 mid + 23 light datasets, which neither of the above reaches.
#
# Same shard files as every other LightGBM path (threads_per_worker 4, fold_workers 3 — keyed
# identically), so all finished cells cache-hit. Same race handling as drivers 2-4.
# Concurrency mid 3 / light 5 (x up to 12 threads with parallel folds): the box is saturated, so
# this is sized to add LightGBM throughput without swamping CatBoost's two heavy stragglers.
# MemoryMax=20G. Does NOT rebuild the board.
set -uo pipefail
cd /home/ralph/suite/insur-arena2

LOG=runs_board_sharded5.log
UNIT=tbshard5
LOGDIR=runs/shard_logs5
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
[ -s configs/shards/_waves5.tsv ] || { log "REFUSING: configs/shards/_waves5.tsv missing"; exit 1; }

systemd-run --user --scope --expand-environment=no \
  -p MemoryMax=20G --unit="$UNIT" \
  bash -c '
    set -uo pipefail
    cd /home/ralph/suite/insur-arena2
    LOGDIR=runs/shard_logs5
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
      done < configs/shards/_waves5.tsv
      echo "=== MODEL $model COMPLETE — $(date -Is) ==="
    done
  ' >> "$LOG" 2>&1
rc=$?

fails=$(awk -F'\t' '$4 != 0' "$LOGDIR/_done.tsv" 2>/dev/null | wc -l)
log "driver 5 finished (exit $rc, $fails failed jobs) — board NOT rebuilt here; driver 1 owns that"
awk -F'\t' '$4 != 0 {print "  FAILED: "$2"/"$3" rc="$4}' "$LOGDIR/_done.tsv" >> "$LOG" 2>/dev/null
exit $(( rc != 0 || fails != 0 ))
