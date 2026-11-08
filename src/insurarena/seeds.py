"""Deterministic seeds by stable hash of a tuple (SPEC §9) — not arithmetic (collides)."""
from __future__ import annotations

import hashlib

from insurarena.model_ids import hash_key


def seed_for(*parts: object) -> int:
    """Reproducible 32-bit seed from arbitrary identifying parts.

    A renamed model id is hashed under its legacy spelling (see `model_ids`), so renaming a
    model never changes its seeds.
    """
    key = "|".join(hash_key(p) if isinstance(p, str) else str(p) for p in parts)
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big")
