"""Model ids renamed after their cells were cached under the old name.

A model id is not only a label. It feeds the RNG (`seeds.seed_for` for every deploy and tuning
fit, `eval.hpo._config_seed` for the sampled tuning candidates) and the cell key
(`results.cache.cell_signature`). Renaming one outright would change every fit's seed and the
candidate set, so the cached cells would no longer be what the renamed code computes, and a
re-signed cache would claim results no rerun could reproduce.

So the old spelling is frozen here as the HASH KEY, and used only for hashing. Everything a
person or a config sees uses the new id.
"""
from __future__ import annotations

#: new id -> the spelling its seeds and cell signatures were computed under (frozen forever)
LEGACY_HASH_KEYS: dict[str, str] = {
    "t-boost": "triboost",   # renamed 2026-09-24 (the product was renamed tri-boost -> t-boost)
}


def hash_key(name: str) -> str:
    """The string to hash for model id `name` (its legacy spelling if it was renamed)."""
    return LEGACY_HASH_KEYS.get(name, name)

_CURRENT = {legacy: new for new, legacy in LEGACY_HASH_KEYS.items()}


def current_id(name: str) -> str:
    """The current id for a model name read back from a stored row.

    Rows cached before a rename carry the legacy spelling; anything that reads rows back
    (cache hits, resolved reports, raw extraction) maps them through this.
    """
    return _CURRENT.get(name, name)
