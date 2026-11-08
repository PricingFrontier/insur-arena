"""Pure-Python fetch layer: (access_method, locator) -> raw DataFrame, no auth where possible.

Supported methods:
  openml_id        locator = OpenML data id            -> sklearn.fetch_openml
  url_csv          locator = http(s) URL to CSV/CSV.gz  -> pandas.read_csv
  rpackage_csv     locator = Rdatasets CSV URL          -> pandas.read_csv (alias of url_csv)
  casdatasets_rda  locator = "<.rda url>::<object>"      -> rdata / pyreadr (optional deps)
"""
from __future__ import annotations

import os
import tempfile
import urllib.request

import pandas as pd


def fetch(method: str, locator: str) -> pd.DataFrame:
    if method == "openml_id":
        from sklearn.datasets import fetch_openml

        df = fetch_openml(data_id=int(locator), as_frame=True).frame.copy()
    elif method in ("url_csv", "rpackage_csv"):
        df = _read_csv_url(locator)
    elif method == "casdatasets_rda":
        df = _read_rda_url(locator)
    else:
        raise ValueError(f"unknown access method {method!r}")
    # Normalize column names to plain str (the rda reader yields numpy.str_, which
    # sklearn's ColumnTransformer rejects as a column specification).
    df.columns = [str(c) for c in df.columns]
    return df


def _read_csv_url(url: str) -> pd.DataFrame:
    comp = "gzip" if url.endswith(".gz") else "infer"
    return pd.read_csv(url, compression=comp)


def _read_rda_url(spec: str) -> pd.DataFrame:
    """spec = '<raw .rda url>::<object name>' (object optional -> first dataframe found)."""
    url, _, obj = spec.partition("::")
    raw = urllib.request.urlopen(url, timeout=120).read()  # noqa: S310 (trusted mirror URLs)
    with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        objs = _load_rda(path)
    finally:
        os.unlink(path)
    if obj and obj in objs:
        return _to_df(objs[obj])
    for v in objs.values():
        df = _to_df(v)
        if df is not None and len(df):
            return df
    raise ValueError(f"no dataframe object found in {url}")


def _load_rda(path: str) -> dict:
    try:
        import rdata

        if hasattr(rdata, "read_rda"):
            return dict(rdata.read_rda(path))
        parsed = rdata.parser.parse_file(path)
        return dict(rdata.conversion.convert(parsed))
    except ImportError:
        import pyreadr

        return dict(pyreadr.read_r(path))


def _to_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        return pd.DataFrame(obj)
    except Exception:
        return None
