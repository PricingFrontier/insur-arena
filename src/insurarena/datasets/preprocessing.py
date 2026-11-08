"""Pinned, versioned preprocessing helpers (clipping, hashing) — not folklore (SPEC §5.6)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def content_hash(df: pd.DataFrame) -> str:
    """Stable content hash of a dataframe — pins the exact data version."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(row_hashes).tobytes())
    h.update("|".join(map(str, df.columns)).encode())
    return h.hexdigest()[:16]


def clip_series(s: pd.Series, upper: float | None = None, lower: float | None = None) -> pd.Series:
    return s.clip(lower=lower, upper=upper)
