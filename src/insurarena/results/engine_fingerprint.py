"""Optional-engine build fingerprints, recorded BESIDE each cached cell — never in its signature.

WHY THIS EXISTS. A cell's identity (`results.cache.cell_signature`) is built from the dataset
hash, the model/variant, the split id and a protocol dict whose model-version component is a
HAND-TYPED `adapter_version` on the adapter class. Nothing in it fingerprints the compiled engine
the fit actually ran against. For a pure-Python engine that is fine — the adapter and the engine
move together in one repo. For `t-boost` it is not: the engine is a separately-built Rust
extension that is not even a declared dependency of this package, so a rebuild that moves a fit
(a Rust refactor, a dependency bump, an accidental default change) silently invalidates every
cached cell while leaving every cell KEY identical. The cache would then serve stale results with
no error and no warning — the one failure a version-keyed cache cannot detect on its own.

That is not hypothetical: on 2026-09-04 the t-boost extension was rebuilt to fix a
`monotone_constraints` defect, and the only thing that established the cached cells were still
valid was a hand-run 16-cell byte-identity replay.

WHY A WARNING AND NOT A KEY. Putting the fingerprint in the proto would be stricter but would
re-key ~9,000 cells on every rebuild, including the overwhelming majority of rebuilds that change
nothing observable (a comment, a clippy fix, a faster histogram that is bit-identical). The cost
of that is certain and large; the cost of a silent stale read is uncertain and rare but severe.
Recording + warning gets the detection without the recompute, and leaves the decision with a
human who can run a byte-identity replay to settle it.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache

_WARNED: set[tuple[str, str, str]] = set()


@lru_cache(maxsize=None)
def engine_fingerprint(model: str) -> str | None:
    """A stable identifier for the compiled engine `model` will fit against, or None.

    None means "no separately-compiled engine to track" — the common case — and is recorded as
    nothing at all, so cells for pure-Python adapters are byte-identical to before this change.
    """
    if model != "t-boost":
        return None
    try:
        import t_boost
    except Exception:
        return None
    version = str(getattr(t_boost, "__version__", "?"))
    # The version alone is far too coarse: an editable `maturin develop` rebuild keeps 0.6.0
    # across arbitrary source changes, which is exactly the case that motivated this module. Hash
    # the shipped extension itself.
    so_hash = "?"
    try:
        base = os.path.dirname(os.path.abspath(t_boost.__file__))
        cands = sorted(f for f in os.listdir(base) if f.endswith((".so", ".pyd")))
        if cands:
            h = hashlib.sha256()
            with open(os.path.join(base, cands[0]), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            so_hash = h.hexdigest()[:16]
    except Exception:
        pass
    return f"{version}+{so_hash}"


def stamp(payload: dict, model: str) -> dict:
    """Record the current engine fingerprint on a payload about to be cached."""
    fp = engine_fingerprint(model)
    if fp is not None:
        payload = {**payload, "engine_fingerprint": fp}
    return payload


def check(payload: dict, model: str, sig: str) -> dict:
    """Warn ONCE per (model, cached, current) triple when a cell was computed by another build.

    Deliberately not an error and deliberately not a cache miss: a differing fingerprint means
    "this cell was produced by a different engine build", which is USUALLY benign. The point is
    that it stops being invisible.
    """
    if not isinstance(payload, dict):
        return payload
    cached = payload.get("engine_fingerprint")
    if cached is None:
        return payload                      # pre-fingerprint cell, or no compiled engine
    current = engine_fingerprint(model)
    if current is None or cached == current:
        return payload
    key = (model, str(cached), str(current))
    if key not in _WARNED:
        _WARNED.add(key)
        import warnings
        warnings.warn(
            f"cached {model} cells were computed against engine build {cached}, but the engine "
            f"currently importable is {current}. Cell signatures do NOT include the engine build, "
            f"so these results are being reused across a rebuild. If that rebuild moved any fit, "
            f"the reused values are stale. Verify with a byte-identity replay before trusting a "
            f"board built from this cache (first seen on cell {sig}).",
            RuntimeWarning, stacklevel=2,
        )
    return payload
