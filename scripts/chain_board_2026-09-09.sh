#!/usr/bin/env bash
# FULL BOARD REBUILD, 2026-09-09 — every competitor recomputed from an empty cache.
#
# WHY: the per-cell cache was cleared down to `tabpfn3` only (984 cells survive; nothing else
# does). So this is not a top-up and not a half-cache-hit run like av43/av44 — EVERY cell of every
# model below is computed from scratch, against a t-boost extension rebuilt today from the working
# tree (`cargo build --release -p t-boost-py`, engine fingerprint 0.0.0+unknown+8cbbf67fcb05b730).
#
# WHAT RUNS, and why in this order:
#   1. glm       default          THE ANCHOR, and it goes first for two reasons. `canonical.py`
#                                 requires `glm/default` = 1000 to have FULL catalog coverage or
#                                 the ladder cannot be built at all; and `runner._baseline_deviances`
#                                 caches a per-(dataset, split) `__glm_baseline__` fit that every
#                                 `skill_vs_glm` on the board divides by. Computing the anchor first
#                                 primes that cache for all five comparators, so none of them pays
#                                 for the baseline fits.
#   2. ebm       default + tuned  Cheapest tuned comparator on the catalog.
#   3. lightgbm  default + tuned  `default` is NEW to this config (see its header).
#   4. xgboost   default + tuned
#   5. catboost  default + tuned
#   6. t-boost  default + tuned  LAST, and deliberately. It is by far the longest phase (~20h; it
#                                 carries freMTPL2freq, porto_seguro and homesite_conv at
#                                 max_workers 5), and putting it last means the whole comparator
#                                 board exists before it starts. If it fails or has to be re-run
#                                 against a different engine build, nothing else has to move.
#
# Deploy cells: 465 per model-variant (5 sub-2500 sets x 30 splits + 35 x 9). 11 model-variants
# here = 5,115 deploy cells, plus 40 `tuned__hpo` selections and ~1,000 `tuned__hpo_config`
# candidate/finale evals per tuned model.
#
# SEQUENTIAL, in ONE scope, because every phase wants the whole box — each config sizes its own
# `max_workers`/`threads_per_worker` on the assumption that it is alone. Two campaigns writing the
# same cell cache concurrently is the failure the double-launch guard below exists to prevent.
#
# MemoryMax=52G: measured on the t-boost build at max_workers 6 (74% of the cap on the worst
# fit, homesite_conv at 6.41 GB peak RSS per worker); t-boost.yaml runs max_workers 5, which puts
# that same worst case at 62%. OMP_NUM_THREADS=4 keeps a fit inside its worker.
set -uo pipefail
cd /home/ralph/suite/insur-arena2

LOG=runs_board_2026-09-09.log
UNIT=tbboard0909
log() { echo "[$(date -Is)] $*" >> "$LOG"; }

# Don't pile onto anything else heavy.
while pgrep -f "insurarena[.]cli run" >/dev/null; do
  log "another insurarena run is active; holding"
  sleep 120
done

# REFUSE TO DOUBLE-LAUNCH. Learned the hard way on 2026-09-01: an invocation whose `systemd-run`
# lost the unit name ("Unit ... was already loaded or has a fragment file") fell straight through
# to the completion message below and logged a FAILED launch as a finished campaign. The unit name
# is checked first and the exit status is checked after.
if systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null; then
  log "REFUSING: $UNIT.scope is already active — a chain is already running"
  exit 1
fi
if systemctl --user list-units --all "$UNIT.scope" --no-legend 2>/dev/null | grep -q .; then
  log "REFUSING: $UNIT.scope is loaded but not active — reset it first:"
  log "  systemctl --user reset-failed $UNIT.scope"
  exit 1
fi

# --expand-environment=no pins TODAY's systemd behavior: the inner `$cfg`/`$PY` belong to the
# bash -c body, not to systemd, and systemd warns that a future release will start expanding them
# by default — which would substitute empty strings and run `--config configs/.yaml`.
systemd-run --user --scope --expand-environment=no \
  -p MemoryMax=52G --setenv=OMP_NUM_THREADS=4 --unit="$UNIT" \
  bash -c '
    set -uo pipefail
    cd /home/ralph/suite/insur-arena2
    PY=./.venv/bin/python
    rc=0
    for cfg in glm ebm lightgbm xgboost catboost t-boost; do
      echo "=== board 2026-09-09 phase: $cfg — $(date -Is) ==="
      nice -n 10 $PY -m insurarena.cli run --config configs/$cfg.yaml \
        > runs_board0909_$cfg.log 2>&1 || { echo "PHASE FAILED: $cfg"; rc=1; }
      echo "=== phase $cfg done rc=$? — $(date -Is) ==="
    done
    exit $rc
  ' >> "$LOG" 2>&1
rc=$?

# CHECK THE STATUS. Without this the failure mode above is silent and looks like success.
if [ "$rc" -ne 0 ]; then
  log "board chain FAILED (exit $rc) — the board was NOT rebuilt; read $LOG and runs_board0909_*.log"
  exit "$rc"
fi

log "all six phases finished — rebuilding the board"
./.venv/bin/python scripts/elo_ladder.py --export results >> "$LOG" 2>&1 \
  || { log "elo_ladder --export FAILED"; exit 1; }
log "board rebuilt -> results/leaderboard_cells.csv, results/elo_ladder.csv, results/elo_chart.png"
