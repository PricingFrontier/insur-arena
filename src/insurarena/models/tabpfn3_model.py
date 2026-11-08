"""TabPFN-3 — tabular foundation model, BLACK-BOX comparator. Two backends:
  - backend="api" (CANONICAL DEFAULT): the hosted Prior Labs API via `tabpfn_client`==0.3.3.
    Sends training/test data to Prior Labs servers — approved for this arena's open-data
    benchmark use by Ralph. No local GPU or gated weights needed. Classification supports up
    to ~160 classes (a server-side limit the client fetches from GET /tabpfn/get_model_limits
    and enforces itself; not hardcoded here).
  - backend="local": the original `tabpfn`==8.1.0 OSS package, license-gated local weights,
    GPU recommended. Classification is capped at 10 classes (package's own
    MAX_NUMBER_OF_CLASSES). Its fit path has never changed. Kept for parity, offline use, and
    as a fallback when no API token is available.

Exposure story (documented, not hidden; identical across backends): TabPFN has no native
sample_weight, so exposure enters the model in two optional ways instead of a loss-level offset:
  (a) exposure_mode="resample" — the fit-time context is drawn WITH replacement, probability
      proportional to the exposure weight, approximating sample_weight in-context. Regression
      tasks only (freq/sev/pp); classification tasks and degenerate (zero-variance) weights fall
      back to plain subsampling. DEFAULT OFF ("none"): measured on the v3 api backend, resample
      loses to plain subsampling everywhere once the context already holds the full/capped
      training set — worth revisiting only if max_rows is lowered a lot.
  (b) rebalance=True — after fitting, one extra inference pass over the full (un-resampled) train
      X computes a multiplicative level-correction factor = sum(w*y) / sum(w*y_hat), clipped to
      [0.25, 4], applied to every prediction. DEFAULT ON: the same probe showed a clear, fairly
      consistent win (primary-metric deviance 0.679 vs 0.810 on sgautonb; 0.800 vs 0.928 on a
      partial ausprivauto run; neutral on swautoins, never worse).
Predictions stay on the NATURAL scale (rate/cost/probability) regardless of backend/mode, per the
adapter contract.

V3 handles up to 1M rows x 200 features; local-backend context is capped at max_rows for GPU
memory (this box has 8GB) — the api backend reuses the same max_rows cap for the fit context (it
has its own, usually larger, server-side row/cell limits, validated by the client itself).
Categoricals are ordinal-encoded to float codes using the FIT-time category levels (stable across
train/predict; a level unseen at fit time maps to -1 — see `_encode`) AND their column positions
are passed via categorical_features_indices so TabPFN still treats them as categorical.

api backend implementation notes (introspected from the installed `tabpfn_client` 0.3.3 source):
  - Auth: `tabpfn_client.set_access_token(token)`, token resolved by our own small helper
    (TABPFN_TOKEN env -> ~/.cache/tabpfn/auth_token -> ~/.tabpfn/token; independent of the local
    `tabpfn` package's own token lookup so backend="api" works without `tabpfn`/torch installed).
    Guarded by a module-level flag — set_access_token is cheap (local state + a maybe-write of an
    unchanged token file, no network round trip) but there is no reason to repeat it every fit.
  - Progress spinner: `tabpfn_client.estimator.run_task()` writes a live "NN:SS Fitting/Predicting
    ... <spinner>" line to stdout unless `TABPFN_CLIENT_CI_MODE` is truthy (constants.py
    `ci_mode_enabled()`, `functools.cache`d — must be set before the first fit/predict in the
    process). We set it (via `setdefault`, so an operator can force it back on for interactive
    debugging) as the primary suppression. Belt-and-suspenders: every api fit/predict/predict_proba
    call is additionally wrapped in `contextlib.redirect_stdout` in case anything else in the
    client prints; stderr (and therefore real tracebacks) is never touched.
  - Retries: the client already retries connection errors and 5xx-ish failures internally via
    `backoff` (see `tabpfn_client.client.RetryableServerError`). HTTP 429 has no dedicated
    exception type there — it surfaces as a bare `RuntimeError` mentioning the status code. Our
    own outer retry (`_call_with_api_retries`, up to 3 attempts, 5s/15s backoff) is a backstop for:
    exhausted inner retries; 429s the inner layer doesn't retry at all; and
    `httpx.RemoteProtocolError` ("Server disconnected without sending a response"), seen in
    production after the inner backoff gave up. We deliberately do NOT retry the sibling
    `httpx.LocalProtocolError` — client.py's own comments flag it as a distinct, client-side h2
    state-machine issue they worked around by pinning HTTP/1.1 rather than retrying, so treating
    it as transient would be wrong.
  - Row cap: the server caps test-set rows (documented around 200k); predict is chunked via
    `hp["api_predict_chunk"]` (default 100_000) through `_chunked_call`/`_row_chunks`, which also
    carries the rebalance step's extra full-train-X inference pass.
  - Supported constructor kwargs (checked against `tabpfn_client.estimator` directly):
    `categorical_features_indices`, `random_state`, `n_estimators` are all accepted, as is
    `create_default_for_version(ModelVersion, **overrides)` for pinning a model version — same
    shape as the OSS package's own method, but note it is a *different* `ModelVersion` enum
    (`tabpfn_client.constants.ModelVersion`, not `tabpfn.constants.ModelVersion`). There is no
    `device` kwarg at all — inference runs server-side.

Lazy imports; optional deps `tabpfn_client` (api backend) / `tabpfn` (local backend, license-gated:
one-time acceptance + TABPFN_TOKEN, or a cached ~/.cache/tabpfn/auth_token; GPU recommended).
"""
from __future__ import annotations

import contextlib
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter
from insurarena.scoring import task_targets
from insurarena.tasks import TaskType

_MAX_ROWS = 100_000          # LOCAL-backend context cap — bounded by this box's 8GB GPU
_MAX_ROWS_API = 500_000      # API-backend context cap — inference runs SERVER-side (V3 handles 1M x
                             # 200), so the 8GB-GPU limit never applied here. 500k covers every
                             # catalog train split (largest: freMTPL2freq ~452k @ 3-fold). The old
                             # code reused the local 100k cap for the api backend too, silently
                             # discarding 24-64% of the training rows on the 6 largest sets.
_DEDUP_VARY_TOL = 0.05       # group-dedup fires only if < this fraction of multi-row groups have a
                             # target that VARIES within the group (else 1-row-per-group loses real
                             # signal — e.g. fremotor_prem premium varies 99.9% within policy).
_API_PREDICT_CHUNK = 100_000  # api backend: rows per predict call (server caps around 200k)
_API_MAX_TEST_CELLS = 19_000_000  # api backend hard limit: a predict call rejects if rows*features
                                  # exceeds 20M cells; chunk by cells (margin below 20M) so wide
                                  # datasets (e.g. homesite_conv, 301 feat -> <=63k rows/chunk) work.
_API_RETRY_DELAYS = (5.0, 15.0)  # -> 3 attempts total (2 retries) on transient api errors

_api_authed = False   # module-level: set_access_token is cheap but pointless to repeat every fit


def _find_api_token() -> str | None:
    """TABPFN_TOKEN env -> ~/.cache/tabpfn/auth_token -> ~/.tabpfn/token. Deliberately independent
    of the (separate, optional) local `tabpfn` package's own token lookup."""
    tok = os.environ.get("TABPFN_TOKEN")
    if tok and tok.strip():
        return tok.strip()
    for p in (Path.home() / ".cache" / "tabpfn" / "auth_token", Path.home() / ".tabpfn" / "token"):
        if p.is_file():
            tok = p.read_text().strip()
            if tok:
                return tok
    return None


def _ensure_api_auth() -> None:
    """Authenticate the tabpfn_client singleton once per process (see module docstring)."""
    global _api_authed
    if _api_authed:
        return
    token = _find_api_token()
    if not token:
        raise RuntimeError(
            "tabpfn3 backend='api': no token found (checked TABPFN_TOKEN env, "
            "~/.cache/tabpfn/auth_token, ~/.tabpfn/token). Set one of these to use the api backend."
        )
    os.environ.setdefault("TABPFN_CLIENT_CI_MODE", "1")   # suppress the client's spinner
    import tabpfn_client

    tabpfn_client.set_access_token(token)
    _api_authed = True


def _suppress_stdout(fn, *args, **kwargs):
    """Run fn with stdout redirected to a throwaway buffer. Belt-and-suspenders on top of
    TABPFN_CLIENT_CI_MODE (see module docstring); stderr/exceptions are untouched."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class TabPFN3(ModelAdapter):
    name = "tabpfn3"
    family = "foundation"
    adapter_version = "1"   # reset 2026-09-09 (arena rebuilt from scratch); prior history in git
    needs_es_probe = False               # in-context model, no iterative early stopping -> skip probe
    supported_tasks = (TaskType.FREQ, TaskType.SEV, TaskType.PP, TaskType.CLF, TaskType.MULTICLF)

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        """Ordinal-encode categoricals using the FIT-time category levels (self._cat_categories) so
        train/predict codes stay aligned; a level unseen at fit time maps to -1."""
        X = X[self._columns].copy()
        for c in self._cat:
            X[c] = pd.Categorical(X[c], categories=self._cat_categories[c]).codes.astype(float)
        return np.nan_to_num(X.astype(float).to_numpy(), nan=0.0)

    @staticmethod
    def _weighted_resample_idx(w: np.ndarray, n_ctx: int, rng: np.random.Generator) -> np.ndarray:
        """Row indices drawn WITH replacement, probability proportional to `w` — approximates
        sample_weight in-context for a model with no native weighting. Pure/testable in isolation."""
        w = np.clip(np.asarray(w, dtype=float), 0.0, None)
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            return rng.integers(0, len(w), size=n_ctx)
        return rng.choice(len(w), size=n_ctx, replace=True, p=w / total)

    @classmethod
    def _fit_row_idx(cls, n: int, w: np.ndarray, task: TaskType, mode: str, max_rows: int,
                      rng: np.random.Generator) -> np.ndarray:
        """Which rows enter the TabPFN fit context: exposure-weighted resample when eligible
        (mode="resample", a regression task, non-degenerate weights), else a plain cap/no-op."""
        w = np.asarray(w, dtype=float)
        non_degenerate = w.size > 0 and (np.max(w) - np.min(w)) > 0   # cheap zero-variance check
        eligible = mode == "resample" and task not in (TaskType.CLF, TaskType.MULTICLF) and non_degenerate
        if eligible:
            return cls._weighted_resample_idx(w, min(n, max_rows), rng)
        if n > max_rows:
            return rng.choice(n, max_rows, replace=False)
        return np.arange(n)

    @staticmethod
    def _group_rep_idx(group_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One rng-chosen representative row index per group, sorted. Longitudinal sets repeat a
        policy across yearly rows (pg16_payfreq: 4.8 rows/policy, PayFreq constant within EVERY
        policy) — feeding the duplicates to an in-context learner is memorization, not evidence,
        and made TabPFN overconfident (log loss 1.35 vs 1.07 for the priors alone). Pure/testable."""
        perm = rng.permutation(len(group_ids))
        _, first = np.unique(np.asarray(group_ids)[perm], return_index=True)
        return np.sort(perm[first])

    @staticmethod
    def _dedup_safe(group_ids: np.ndarray, y: np.ndarray, tol: float = _DEDUP_VARY_TOL) -> bool:
        """True when keeping one row per group is (near-)lossless: among groups with >1 row, the
        fraction whose target takes more than one distinct value is below `tol`. Constant-within-
        group targets (pg16 PayFreq, 0.0) pass; within-group-varying ones (fremotor_prem premium,
        0.999) fail. Pure/testable — no model or network needed."""
        g = np.asarray(group_ids)
        y = np.asarray(y, dtype=float)
        order = np.argsort(g, kind="stable")
        gs, ys = g[order], y[order]
        bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
        multi = varies = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a > 1:
                multi += 1
                if np.unique(ys[a:b]).size > 1:
                    varies += 1
        return multi == 0 or (varies / multi) < tol

    @staticmethod
    def _solve_logit_shift(logit: np.ndarray, w: np.ndarray, target: float,
                           lo: float, hi: float, iters: int = 60) -> float:
        """Intercept shift delta (bisection, monotone objective) so that
        sum(w * sigmoid(logit + delta)) == target, clamped to [lo, hi]. Pure/testable."""
        def total(d: float) -> float:
            return float(np.sum(w / (1.0 + np.exp(-(logit + d)))))
        if total(lo) >= target:
            return lo
        if total(hi) <= target:
            return hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if total(mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _row_chunks(n: int, chunk: int) -> list[tuple[int, int]]:
        """(start, stop) row-slice boundaries of size <= chunk covering range(n). Pure, testable."""
        chunk = max(1, int(chunk))
        return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]

    @classmethod
    def _chunked_call(cls, call_fn, X: np.ndarray, chunk: int) -> np.ndarray:
        """Apply call_fn to row-chunks of X and concatenate along axis 0, preserving 1-D (n,) or
        2-D (n, K) shape. Pure w.r.t. call_fn — testable with a stub, no network/model needed."""
        n = len(X)
        if n <= chunk:
            return np.asarray(call_fn(X), dtype=float)
        parts = [np.asarray(call_fn(X[a:b]), dtype=float) for a, b in cls._row_chunks(n, chunk)]
        return np.concatenate(parts, axis=0)

    @staticmethod
    def _is_retryable_api_error(exc: Exception) -> bool:
        """True for transient tabpfn_client errors worth retrying: httpx connection/timeout/
        remote-protocol errors, the client's own RetryableServerError (5xx-ish), or HTTP 429 --
        which the client surfaces as a bare RuntimeError mentioning the status code (confirmed by
        reading tabpfn_client/client.py: 429 has no dedicated exception type there).

        httpx.RemoteProtocolError (server-side, e.g. "Server disconnected without sending a
        response") is retried; its sibling httpx.LocalProtocolError is NOT -- that one is a
        client-side protocol violation (tabpfn_client's own client.py documents a known h2/PING
        case), not a transient server hiccup, so retrying it would just mask a real bug."""
        import httpx
        from tabpfn_client.client import RetryableServerError

        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError,
                             RetryableServerError)):
            return True
        return isinstance(exc, RuntimeError) and "429" in str(exc)

    @classmethod
    def _call_with_api_retries(cls, fn, delays=_API_RETRY_DELAYS, sleep=None):
        """Call fn() with up to len(delays)+1 attempts, sleeping delays[i] between attempts,
        retrying only on _is_retryable_api_error; the final failure always propagates."""
        sleep = sleep or time.sleep
        for delay in (*delays, None):
            try:
                return fn()
            except Exception as e:
                if delay is None or not cls._is_retryable_api_error(e):
                    raise
                sleep(delay)

    def _raw_predict(self, Xp: np.ndarray, proba: bool) -> np.ndarray:
        """Single choke point for calling the underlying model's predict/predict_proba. Local
        backend: direct call. Api backend: chunked, retried, and stdout-suppressed."""
        call = self.model.predict_proba if proba else self.model.predict
        if self._backend == "local":
            return np.asarray(call(Xp), dtype=float)
        chunk = int(self.hp.get("api_predict_chunk", _API_PREDICT_CHUNK))
        n_feat = Xp.shape[1] if Xp.ndim > 1 else 1     # cap chunk so rows*features <= the 20M-cell API limit
        if n_feat > 0:
            chunk = min(chunk, max(1, _API_MAX_TEST_CELLS // n_feat))
        chunk_call = lambda x: self._call_with_api_retries(lambda: _suppress_stdout(call, x))  # noqa: E731
        return self._chunked_call(chunk_call, Xp, chunk)

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "TabPFN3":
        self._backend = self.hp.get("backend", "api")
        if self._backend not in ("api", "local"):
            raise ValueError(f"tabpfn3: unknown backend {self._backend!r} (expected 'api' or 'local')")

        self._columns = list(train.X.columns)
        self._cat = [c for c in train.cat_features if c in self._columns]
        cat_set = set(self._cat)
        cat_idx = [i for i, c in enumerate(self._columns) if c in cat_set]
        self._cat_categories = {c: train.X[c].astype("category").cat.categories for c in self._cat}
        self._task = train.task.task
        self._is_multi = self._task == TaskType.MULTICLF
        self._is_clf = self._task in (TaskType.CLF, TaskType.MULTICLF)

        y, w, _ = task_targets(train)                      # rate/severity/label + exposure weight
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        X_full = self._encode(train.X)

        seed = int(self.hp.get("seed", 0))
        rng = np.random.default_rng(seed)
        # Backend-specific context cap: the api backend runs server-side and is NOT bound by the
        # local 8GB GPU (see _MAX_ROWS_API). An explicit hp max_rows still wins.
        mr = self.hp.get("max_rows")
        max_rows = int(mr) if mr else (_MAX_ROWS_API if self._backend == "api" else _MAX_ROWS)
        mode = self.hp.get("exposure_mode", "none")
        # Group-level context: one representative row per policy — but ONLY when that is (near-)
        # lossless, i.e. the target is constant within groups (see _dedup_safe). Longitudinal sets
        # whose target varies year-to-year (e.g. fremotor_prem) keep all rows. The cap/resample
        # logic then runs on the resulting pool.
        pool = np.arange(len(X_full))
        if self.hp.get("dedup_groups", True) and train.group_ids is not None:
            gids = np.asarray(train.group_ids)
            if len(np.unique(gids)) < len(gids) and self._dedup_safe(gids, y):
                pool = self._group_rep_idx(gids, rng)
        idx = pool[self._fit_row_idx(len(pool), w[pool], self._task, mode, max_rows, rng)]
        X_ctx, y_ctx = X_full[idx], y[idx]

        # Deviance-safe prediction floor for PP/SEV: a single near-zero prediction on a positive-loss
        # row blows up Tweedie/Gamma deviance (the old 1e-9 clip let ohlsson_pp hit skill -821k on one
        # split). Deliberately NOT applied to FREQ — low Poisson rates are legitimate there, and a
        # 3%-of-mean floor would over-predict the safest policies. FREQ keeps the plain 1e-9 clip.
        self._pred_floor = 1e-9
        if self._task in (TaskType.PP, TaskType.SEV):
            wsum = float(np.sum(w))
            mean_y = float(np.sum(w * y)) / wsum if wsum > 0 else 0.0
            frac = float(self.hp.get("pred_floor_frac", 3e-2))
            if mean_y > 0 and frac > 0:
                self._pred_floor = max(1e-9, frac * mean_y)

        common = {"categorical_features_indices": cat_idx, "random_state": seed}
        n_estimators = self.hp.get("n_estimators")
        if n_estimators is not None:                       # None -> package/server default (8)
            common["n_estimators"] = int(n_estimators)
        model_version = self.hp.get("model_version")

        if self._backend == "local":
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.constants import ModelVersion

            common["device"] = self.hp.get("device", "auto")
            Model = TabPFNClassifier if self._is_clf else TabPFNRegressor
            self.model = (Model.create_default_for_version(ModelVersion(model_version), **common)
                          if model_version else Model(**common))
            self.model.fit(X_ctx, y_ctx)
        else:
            _ensure_api_auth()
            import tabpfn_client
            from tabpfn_client.constants import ModelVersion as ApiModelVersion

            Model = tabpfn_client.TabPFNClassifier if self._is_clf else tabpfn_client.TabPFNRegressor
            self.model = (Model.create_default_for_version(ApiModelVersion(model_version), **common)
                          if model_version else Model(**common))
            self._call_with_api_retries(lambda: _suppress_stdout(self.model.fit, X_ctx, y_ctx))

        self._rebalance_factor = None
        self._logit_shift = None
        if self.hp.get("rebalance") and not self._is_multi:
            if self._is_clf:
                # Binary clf analogue of the regression level step (same clip authority, log-odds
                # space): TabPFN's probabilities ran at 0.59x the observed base rate on
                # ausprivauto_clf, and the clf path previously had no level correction at all.
                p = np.clip(self._raw_predict(X_full, proba=True)[:, 1], 1e-6, 1.0 - 1e-6)
                target = float(np.sum(w * y))
                if 0.0 < target < float(np.sum(w)):
                    shift = self._solve_logit_shift(np.log(p / (1.0 - p)), w, target,
                                                    -np.log(4.0), np.log(4.0))
                    self._logit_shift = float(shift)
            else:
                y_hat = self._raw_predict(X_full, proba=False)   # extra inference pass (chunked if api)
                denom = float(np.sum(w * y_hat))
                factor = float(np.sum(w * y)) / denom if denom else 1.0
                self._rebalance_factor = float(np.clip(factor, 0.25, 4.0))

        self.best_iteration_ = 0
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        Xp = self._encode(X)
        if self._is_multi:
            return self._raw_predict(Xp, proba=True)                                  # (n, K)
        if self._is_clf:
            p = np.clip(self._raw_predict(Xp, proba=True)[:, 1], 1e-6, 1.0 - 1e-6)
            if self._logit_shift is not None:
                p = 1.0 / (1.0 + np.exp(-(np.log(p / (1.0 - p)) + self._logit_shift)))
            return np.clip(p, 0.0, 1.0)
        pred = self._raw_predict(Xp, proba=False)
        if self._rebalance_factor is not None:
            pred = pred * self._rebalance_factor
        return np.clip(pred, getattr(self, "_pred_floor", 1e-9), None)

    def explain(self) -> Explanation:
        return Explanation(kind="none")        # black box

    @staticmethod
    def default_hp() -> dict:
        return {
            "backend": "api",            # "api" | "local" — see module docstring; api is canonical
            "device": "auto",            # local backend only; api runs server-side (no device knob)
            "model_version": None,       # None -> package/server default (v3); both backends support it
            "n_estimators": None,        # None -> default (8, auto-scaled locally); both backends support it
            "exposure_mode": "none",     # "resample" | "none" — see module docstring
            "max_rows": None,            # None -> backend default (api: _MAX_ROWS_API, local: _MAX_ROWS)
            "rebalance": True,           # measured win on the v3 api backend — see module docstring
            "dedup_groups": True,        # v3: one context row per policy — gated on target invariance (v4)
            "pred_floor_frac": 3e-2,     # v4: PP/SEV floor = frac x weighted mean target (freq exempt).
                                         # Sweep on ohlsson_pp (sid6 pathological / sid0 healthy):
                                         # healthy splits are floor-insensitive up to 3e-2
                                         # (-0.0006 skill) while the collapse split goes
                                         # -821k -> -0.07; 1e-1 starts clipping real signal.
            "api_predict_chunk": _API_PREDICT_CHUNK,   # api backend only
        }

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        return TabPFN3.default_hp()   # foundation model — default only (no HPO), as in TabArena
