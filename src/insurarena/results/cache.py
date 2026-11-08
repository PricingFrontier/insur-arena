"""Per-cell result cache so runs reuse work (TabRepo idea).

A "cell" is one (dataset, model, variant, split). Its signature hashes the dataset *content* hash
(which already encodes the subsample), the model+variant+hp-override, the split id, and a protocol
fingerprint (split schedule, n_configs/inner_folds for tuned, Tweedie power, protocol version). A
cell is recomputed only when one of those changes — so adding a new model or dataset, increasing the
subsample, or changing the tuning protocol re-runs just the affected cells; everything else is reused.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from insurarena.model_ids import hash_key

PROTOCOL_VERSION = "0.3"   # bump to invalidate all cached cells (e.g. on a results-affecting change)


def cell_signature(dataset_hash: str, model: str, variant: str, split_id: int, proto: dict) -> str:
    key = json.dumps(
        {"h": dataset_hash, "m": hash_key(model), "v": variant, "s": int(split_id),
         "p": proto, "pv": PROTOCOL_VERSION},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:24]


class ResultCache:
    """Cell store.

    `model` is optional and used ONLY for the engine-build fingerprint (see
    `results.engine_fingerprint`), which is written into the payload and checked on read. It is
    deliberately NOT part of `cell_signature` — a rebuild of a compiled engine warns instead of
    invalidating ~9,000 cells, and the warning is what makes an otherwise silent stale read
    visible.
    """

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.dir = Path(os.path.expanduser(cache_dir)) / "cells"
        self.enabled = enabled
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, sig: str, model: str | None = None) -> dict | None:
        if not self.enabled:
            return None
        p = self.dir / f"{sig}.json"
        if not p.exists():
            return None
        try:
            payload = json.loads(p.read_text())
        except Exception:
            return None
        if model:
            from .engine_fingerprint import check
            payload = check(payload, model, sig)
        return payload

    def put(self, sig: str, payload: dict, model: str | None = None) -> None:
        if not self.enabled:
            return
        if model:
            from .engine_fingerprint import stamp
            payload = stamp(payload, model)
        # Concurrent cells may share a baseline or candidate. Readers must see
        # either the complete previous record or the complete new record.
        with tempfile.NamedTemporaryFile(mode="w", dir=self.dir, delete=False) as f:
            tmp = Path(f.name)
            json.dump(payload, f, default=float)
        try:
            os.replace(tmp, self.dir / f"{sig}.json")
        finally:
            tmp.unlink(missing_ok=True)
