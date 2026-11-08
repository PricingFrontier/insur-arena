"""Hand a long-running shard job over to a split-parallel rerun WITHOUT losing any work.

WHY (2026-09-14). Two jobs were the long poles of the board: ebm/brvehins1 (5 tuned outer splits
left, run one after another at ~1.8 cores) and xgboost/homesite_conv (8 left, >14 h per split at
2 threads). A job walks its outer splits serially (its shard runs max_workers 1), but every tuned
split is an independent search. `max_workers` is absent from every cache key — proved twice
before this script ran: `gen_shard_configs.verify` found all proto kinds byte-identical, and a
max_workers=4 replay of the fully-cached ebm/credit_g shard served 60/60 rows from cache and wrote
zero cells. So a rerun with max_workers N reuses every finished default fit, tuned refit, per-split
selection and candidate evaluation, and parallelises only what is left.

THE ONE THING THAT COULD BE LOST is the candidate evaluation in flight when the old job is
stopped (EBM on brvehins1 has had single candidates run for hours). So this does not kill on a
timer: it watches the cell cache and stops the old job within one poll of that job COMMITTING a
cell. A cell is attributed to the job by its contents, since cells do not name their writer:
  * deploy rows and per-split selections carry `model` + `dataset` -> exact;
  * candidate evaluations carry only `hp` + scores, so they are matched on hp keys unique to the
    model among the jobs running (EBM: `smoothing_rounds`, or the bare `interactions` anchor —
    brvehins1 is the ONLY EBM job alive; XGBoost: `min_child_weight`, or the
    learning_rate/max_depth/n_estimators default anchor) and, for XGBoost, on the score level:
    homesite_conv inner log loss sits at ~0.17 while porto_seguro — the only other XGBoost
    log-loss job — sits at ~0.152, so the cut is 0.162.
The loss after a commit is at most one poll interval of the next candidate.

Optionally SIGSTOPs another process first (driver 1's wave loop), so that stopping ebm/brvehins1
does not immediately release driver 1 into EBM's 23-job light wave while memory is still tight.
Resume it later with `kill -CONT <pid>`.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, signal, subprocess, sys, time

CELLS = "runs/benchmark/cells"


def log(fh, msg):
    fh.write(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}\n"); fh.flush()


def matches(kind: str, d: dict) -> bool:
    row = d.get("row") if isinstance(d.get("row"), dict) else {}
    hp = d.get("hp") if isinstance(d.get("hp"), dict) else {}
    fs = d.get("fold_scores") or []
    if kind == "ebm-brvehins1":
        if row.get("model") == "ebm" and row.get("dataset") == "brvehins1": return True
        if d.get("model") == "ebm" and d.get("dataset") == "brvehins1": return True
        return d.get("kind") == "tuned_hpo_config" and ("smoothing_rounds" in hp or set(hp) == {"interactions"})
    if kind == "xgb-homesite":
        if row.get("model") == "xgboost" and row.get("dataset") == "homesite_conv": return True
        if d.get("model") == "xgboost" and d.get("dataset") == "homesite_conv": return True
        xgb_shape = "min_child_weight" in hp or set(hp) == {"learning_rate", "max_depth", "n_estimators"}
        return (d.get("kind") == "tuned_hpo_config" and xgb_shape and d.get("metric") == "log_loss"
                and bool(fs) and sum(fs) / len(fs) > 0.162)
    if kind == "lgb-porto":
        # LightGBM's only log-loss job alive is porto_seguro (drivers 1 and 4 both running it).
        if row.get("model") == "lightgbm" and row.get("dataset") == "porto_seguro": return True
        if d.get("model") == "lightgbm" and d.get("dataset") == "porto_seguro": return True
        lgb_shape = "num_leaves" in hp or "bagging_fraction" in hp
        return d.get("kind") == "tuned_hpo_config" and lgb_shape and d.get("metric") == "log_loss"
    raise ValueError(kind)


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f: return f.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--match", required=True); ap.add_argument("--config")
    ap.add_argument("--unit"); ap.add_argument("--memmax")
    ap.add_argument("--threads"); ap.add_argument("--pause-pid", type=int)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--stop-unit", help="instead of SIGTERMing --pid and launching a rerun, stop this whole "
                    "systemd scope at the boundary (retires a duplicate driver)")
    a = ap.parse_args()
    os.makedirs("runs/straggler_logs", exist_ok=True)
    fh = open(f"runs/straggler_logs/{a.name}.handoff.log", "a")
    cmd = open(f"/proc/{a.pid}/cmdline").read().replace("\0", " ")
    if a.name.split("__")[-1] not in cmd.replace(".yaml", ""):
        log(fh, f"REFUSING: pid {a.pid} is not the {a.name} job ({cmd[:120]})"); sys.exit(1)
    log(fh, f"watching pid {a.pid} for a committed {a.match} cell (poll {a.poll}s)")
    seen_mtime = time.time()
    while True:
        if not alive(a.pid):
            log(fh, "old job exited on its own before a boundary was seen — nothing to stop")
            break
        newest = seen_mtime; hit = None
        for e in os.scandir(CELLS):
            m = e.stat().st_mtime
            if m <= seen_mtime or not e.name.endswith(".json"): continue
            newest = max(newest, m)
            try: d = json.load(open(e.path))
            except Exception: continue
            if isinstance(d, dict) and matches(a.match, d): hit = (e.name, d.get("kind") or "row")
        seen_mtime = newest
        if hit:
            log(fh, f"commit boundary: {hit[1]} cell {hit[0]} — stopping old job now")
            if a.pause_pid:
                os.kill(a.pause_pid, signal.SIGSTOP); log(fh, f"SIGSTOP pause-pid {a.pause_pid}")
            if a.stop_unit:
                subprocess.run(["systemctl", "--user", "stop", f"{a.stop_unit}.scope"])
                log(fh, f"stopped {a.stop_unit}.scope — duplicate driver retired, no rerun")
                return
            os.kill(a.pid, signal.SIGTERM)
            for _ in range(60):
                if not alive(a.pid): break
                time.sleep(1)
            else:
                os.kill(a.pid, signal.SIGKILL); log(fh, "old job ignored SIGTERM for 60s — SIGKILL")
            log(fh, "old job stopped")
            break
        time.sleep(a.poll)
    wrapper = (
        "set -uo pipefail; cd /home/ralph/suite/insur-arena2; L=runs/straggler_logs/%(n)s.log; rc=1; "
        "for a in 1 2 3 4; do sleep $((RANDOM%%15+1)); "
        "OMP_NUM_THREADS=%(t)s OPENBLAS_NUM_THREADS=%(t)s MKL_NUM_THREADS=%(t)s NUMEXPR_NUM_THREADS=%(t)s RAYON_NUM_THREADS=%(t)s "
        "nice -n 10 ./.venv/bin/python -m insurarena.cli run --config %(c)s > $L 2>&1; rc=$?; "
        "[ $rc -eq 0 ] && break; grep -q active_campaign.tmp $L || break; "
        "echo \"RETRY attempt $a (active_campaign race)\"; done; echo \"rerun finished rc=$rc\"; exit $rc"
    ) % {"n": a.name, "t": a.threads, "c": a.config}
    subprocess.run(["systemctl", "--user", "reset-failed", f"{a.unit}.scope"], capture_output=True)
    log(fh, f"launching rerun: {a.config} in {a.unit}.scope MemoryMax={a.memmax}")
    r = subprocess.run(["systemd-run", "--user", "--scope", "--expand-environment=no",
                        "-p", f"MemoryMax={a.memmax}", f"--unit={a.unit}", "bash", "-c", wrapper],
                       stdout=fh, stderr=fh)
    log(fh, f"rerun scope exited rc={r.returncode}")


if __name__ == "__main__":
    main()
