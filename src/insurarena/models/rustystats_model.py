"""RustyStats GLM/GAM-style actuarial model.

This is kept as a separate diagnostic competitor from the frozen ``glm`` baseline. It uses
RustyStats' native exposure handling, conservative categorical encodings, explicit low-df
continuous terms when the training signal supports them, and benchmark-scale predictions
(rates / severities / probabilities).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from insurarena.datasets.base import DatasetBundle
from insurarena.models.base import Explanation, ModelAdapter, filter_tiny_exposure_fit_rows
from insurarena.models.glm import _LOW_CARD, _auto_roles
from insurarena.tasks import TaskType

_RESPONSE = "__y"
_EXPOSURE = "__exposure"
_WEIGHT = "__weight"


class RustyStatsGLM(ModelAdapter):
    name = "rustystats"
    family = "glm"
    adapter_version = "1"   # reset 2026-09-09 (arena rebuilt from scratch); prior history in git
    supported_tasks = (TaskType.FREQ, TaskType.SEV, TaskType.PP, TaskType.CLF, TaskType.MULTICLF)
    needs_es_probe = False

    def _learn_schema(self, train: DatasetBundle) -> None:
        self._columns = list(train.X.columns)
        self._cat, self._cont = _auto_roles(
            train.X, train.cat_features, int(self.hp.get("low_card", _LOW_CARD))
        )
        self._cat = [c for c in self._cat if c in self._columns]
        self._cont = [c for c in self._cont if c in self._columns]
        self._medians = {}
        self._cont_transforms = {}
        for c in self._cont:
            s = (
                pd.to_numeric(train.X[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .astype(float)
            )
            transform = self._learn_continuous_transform(s)
            self._cont_transforms[c] = transform
            self._medians[c] = float(transform["median"])

    def _learn_continuous_transform(self, s: pd.Series) -> dict:
        robust = bool(self.hp.get("robust_continuous", True))
        lower_q = float(self.hp.get("continuous_winsor_lower", 0.005))
        upper_q = float(self.hp.get("continuous_winsor_upper", 0.995))
        if not 0.0 <= lower_q < upper_q <= 1.0:
            raise ValueError("continuous winsor quantiles must satisfy 0 <= lower < upper <= 1")

        values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
        finite = values.dropna().astype(float)
        use_log1p = robust and self._use_log1p_transform(finite)
        if use_log1p:
            values = np.log1p(values.clip(lower=0.0))
            finite = values.dropna().astype(float)

        lower = upper = None
        if robust and len(finite):
            lower = float(finite.quantile(lower_q))
            upper = float(finite.quantile(upper_q))
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                lower = upper = None
        if lower is not None and upper is not None:
            values = values.clip(lower=lower, upper=upper)

        finite_after = values.dropna().astype(float)
        median = float(finite_after.median()) if len(finite_after) else 0.0
        return {
            "log1p": bool(use_log1p),
            "lower": lower,
            "upper": upper,
            "median": median if np.isfinite(median) else 0.0,
        }

    def _use_log1p_transform(self, finite: pd.Series) -> bool:
        if not bool(self.hp.get("continuous_log1p_skewed", True)):
            return False
        min_unique = int(self.hp.get("continuous_log1p_min_unique", 20))
        if len(finite) < min_unique or int(finite.nunique(dropna=True)) < min_unique:
            return False
        if float(finite.min()) < 0.0:
            return False
        p50, p75, p90, p99 = [float(v) for v in finite.quantile([0.50, 0.75, 0.90, 0.99])]
        if p99 <= 0.0:
            return False
        median_ratio = p99 / max(p50, 1.0)
        upper_tail_ratio = p99 / max(p90, 1e-12)
        iqr_tail_ratio = p99 / max(p75, 1.0)
        return (
            median_ratio >= float(self.hp.get("continuous_log1p_p99_median_ratio", 50.0))
            or upper_tail_ratio >= float(self.hp.get("continuous_log1p_p99_p90_ratio", 10.0))
            or (p50 <= 1.0 and iqr_tail_ratio >= 50.0)
        )

    @staticmethod
    def _apply_continuous_transform(s: pd.Series, transform: dict) -> pd.Series:
        values = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
        if bool(transform.get("log1p", False)):
            values = np.log1p(values.clip(lower=0.0))
        lower = transform.get("lower")
        upper = transform.get("upper")
        if lower is not None and upper is not None:
            values = values.clip(lower=float(lower), upper=float(upper))
        return values.fillna(float(transform.get("median", 0.0))).astype(float)

    def _prep_X(self, X: pd.DataFrame):
        import polars as pl

        Xp = X.loc[:, self._columns].copy(deep=False)
        out = pd.DataFrame(index=Xp.index)
        for c in self._cont:
            out[c] = self._apply_continuous_transform(Xp[c], self._cont_transforms[c])
        for c in self._cat:
            out[c] = Xp[c].astype("string").fillna("nan").astype(str)
        return pl.from_pandas(out.reset_index(drop=True))

    def _signal_count(self, train: DatasetBundle) -> float:
        y = np.asarray(train.y, dtype=float)
        if self._task == TaskType.FREQ:
            return float(np.nansum(np.clip(y, 0.0, np.inf)))
        if self._task == TaskType.PP:
            return float(np.sum(np.isfinite(y) & (y > 0.0)))
        if self._task == TaskType.CLF:
            p = float(np.nanmean(np.clip(y, 0.0, 1.0))) if y.size else 0.0
            return float(len(y) * min(p, 1.0 - p))
        if self._task == TaskType.MULTICLF:
            counts = np.bincount(np.asarray(train.y, dtype=int))
            return float(counts.min()) if counts.size else 0.0
        return float(len(y))

    def _small_or_sparse(self, train: DatasetBundle) -> bool:
        if len(train.y) < int(self.hp.get("spline_min_rows", 10_000)):
            return True
        return self._signal_count(train) < float(self.hp.get("spline_min_signal", 500.0))

    def _default_alpha(self, train: DatasetBundle) -> float:
        if self._small_or_sparse(train):
            return 0.1
        if self._task == TaskType.SEV:
            return 0.03
        return 0.01 if len(train.y) >= 100_000 else 0.03

    def _continuous_term(self, train: DatasetBundle) -> dict:
        mode = str(self.hp.get("continuous", "auto"))
        if self._task == TaskType.MULTICLF and not bool(self.hp.get("multiclass_splines", False)):
            return {"type": "linear"}
        if mode == "linear" or self._small_or_sparse(train):
            return {"type": "linear"}
        if mode == "smooth":
            return {"type": str(self.hp.get("spline", "bs"))}
        return {
            "type": str(self.hp.get("spline", "ns")),
            "df": int(self.hp.get("spline_df", 4)),
        }

    def _terms(self, train: DatasetBundle) -> dict[str, dict]:
        terms: dict[str, dict] = {}
        cont_term = self._continuous_term(train)
        for c in self._cont:
            terms[c] = dict(cont_term)

        low_card = int(self.hp.get("low_card_categorical", self.hp.get("low_card", _LOW_CARD)))
        dummy_budget = int(self.hp.get("max_categorical_dummy_features", 200))
        force_te = bool(self.hp.get("force_target_encoding", False))
        pp_force_te = self._pp_should_target_encode_categoricals()
        pp_te_columns = set(getattr(self, "_pp_target_encode_columns", set()))
        used_dummy_width = 0
        prior_weight = self.hp.get("prior_weight", "auto")
        for c in self._cat:
            n_levels = int(self._prep_category_count(train.X[c]))
            dummy_width = max(n_levels - 1, 1)
            if (
                c not in pp_te_columns
                and not pp_force_te
                and not force_te
                and n_levels <= low_card
                and used_dummy_width + dummy_width <= dummy_budget
            ):
                terms[c] = {"type": "categorical"}
                used_dummy_width += dummy_width
            else:
                terms[c] = {"type": "target_encoding", "prior_weight": prior_weight}
        return terms

    @staticmethod
    def _prep_category_count(s: pd.Series) -> int:
        return int(s.astype("string").fillna("nan").nunique(dropna=False))

    def _pp_should_target_encode_categoricals(self) -> bool:
        if self._task != TaskType.PP:
            return False
        mode = self.hp.get("pp_target_encode_categoricals", "auto")
        if isinstance(mode, str):
            mode = mode.lower()
            if mode in {"1", "true", "yes", "always"}:
                return True
            if mode in {"0", "false", "no", "never"}:
                return False
        elif isinstance(mode, bool):
            return mode
        return bool(getattr(self, "_pp_force_target_encoding", False))

    def _pp_auto_target_encode_columns(self, train: DatasetBundle) -> set[str]:
        if self._task != TaskType.PP or not self._cat:
            return set()
        mode = self.hp.get("pp_target_encode_categoricals", "auto")
        if not isinstance(mode, str) or mode.lower() != "auto":
            return set()

        y = np.asarray(train.y, dtype=float)
        positive = np.isfinite(y) & (y > 0.0)
        max_positive = int(self.hp.get("pp_te_sparse_positive_max", 5))
        min_rows = int(self.hp.get("pp_te_sparse_level_min_rows", 20))
        if max_positive < 1 or min_rows < 1:
            return set()

        dummy_candidates: list[str] = []
        has_sparse_positive_level = False
        low_card = int(self.hp.get("low_card_categorical", self.hp.get("low_card", _LOW_CARD)))
        dummy_budget = int(self.hp.get("max_categorical_dummy_features", 200))
        used_dummy_width = 0
        for c in self._cat:
            n_levels = int(self._prep_category_count(train.X[c]))
            dummy_width = max(n_levels - 1, 1)
            dummy_candidate = (
                n_levels <= low_card and used_dummy_width + dummy_width <= dummy_budget
            )
            if not dummy_candidate:
                continue
            dummy_candidates.append(c)
            used_dummy_width += dummy_width
            if self._pp_has_sparse_positive_levels(
                train.X[c],
                positive,
                min_rows=min_rows,
                max_positive=max_positive,
            ):
                has_sparse_positive_level = True
        return set(dummy_candidates) if has_sparse_positive_level else set()

    @staticmethod
    def _pp_has_sparse_positive_levels(
        s: pd.Series,
        positive: np.ndarray,
        *,
        min_rows: int,
        max_positive: int,
    ) -> bool:
        x = s.astype("string").fillna("nan")
        grouped = pd.DataFrame({"x": x, "positive": positive}).groupby("x", dropna=False)[
            "positive"
        ].agg(["size", "sum"])
        if grouped.empty:
            return False
        sparse = grouped[(grouped["size"] >= min_rows) & (grouped["sum"] > 0)]
        return bool((sparse["sum"] <= max_positive).any())

    def _fit_kwargs(self, train: DatasetBundle) -> dict:
        alpha = self.hp.get("alpha", None)
        return {
            "alpha": float(self._default_alpha(train) if alpha is None else alpha),
            "l1_ratio": float(self.hp.get("l1_ratio", 0.0)),
            "max_iter": int(self.hp.get("max_iter", 400)),
            "tol": float(self.hp.get("tol", 1e-8)),
            "compute_covariance": False,
        }

    def _multiclass_fit_kwargs(self, train: DatasetBundle) -> dict:
        fit_kw = self._fit_kwargs(train)
        multiclass_alpha = self.hp.get("multiclass_alpha", None)
        fit_kw["alpha"] = float(fit_kw["alpha"] if multiclass_alpha is None else multiclass_alpha)
        fit_kw["solver"] = str(self.hp.get("multiclass_solver", self.hp.get("solver", "auto")))

        optional = {
            "multiclass_hessian_memory_limit_bytes": ("hessian_memory_limit_bytes", int),
            "multiclass_max_dense_parameters": ("max_dense_parameters", int),
            "matrix_free_cg_max_iter": ("matrix_free_cg_max_iter", int),
            "matrix_free_cg_tol": ("matrix_free_cg_tol", float),
            "matrix_free_cg_damping": ("matrix_free_cg_damping", float),
        }
        for hp_name, (fit_name, caster) in optional.items():
            if hp_name in self.hp:
                fit_kw[fit_name] = caster(self.hp[hp_name])
        return fit_kw

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0.0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_neg = np.exp(z[~pos])
        out[~pos] = exp_neg / (1.0 + exp_neg)
        return out

    def _fit_logit_shift(
        self, pred: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None
    ) -> float:
        pred = np.clip(np.asarray(pred, dtype=float), 1e-6, 1.0 - 1e-6)
        y = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
        w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
        target = float(np.sum(w * y) / max(np.sum(w), 1e-12))
        if target <= 1e-6 or target >= 1.0 - 1e-6:
            return 0.0
        eta = self._logit(pred)
        lo, hi = -20.0, 20.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rate = float(np.sum(w * self._sigmoid(eta + mid)) / max(np.sum(w), 1e-12))
            if rate < target:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    def _predict_kwargs(self) -> dict:
        out = {"on_extreme_eta": str(self.hp.get("on_extreme_eta", "clip"))}
        if "eta_clip" in self.hp:
            out["eta_clip"] = self.hp["eta_clip"]
        return out

    def _rs_predict(self, model, data, **kwargs):
        predict_kwargs = self._predict_kwargs()
        try:
            return model.predict(data, **kwargs, **predict_kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "on_extreme_eta" not in msg and "eta_clip" not in msg:
                raise
            return model.predict(data, **kwargs)

    def _pp_fit_has_extreme_eta(self, model, data) -> bool:
        mode = self.hp.get("pp_target_encode_categoricals", "auto")
        if not isinstance(mode, str) or mode.lower() != "auto" or not self._cat:
            return False
        try:
            eta = np.asarray(model.predict_linear(data, exposure=np.ones(data.height)), dtype=float)
        except Exception:
            return True
        if not np.all(np.isfinite(eta)):
            return True
        eta_clip = self.hp.get("eta_clip", 50.0)
        if isinstance(eta_clip, (tuple, list)):
            lower, upper = float(eta_clip[0]), float(eta_clip[1])
        else:
            bound = abs(float(eta_clip))
            lower, upper = -bound, bound
        return bool(np.any((eta < lower) | (eta > upper)))

    def fit(self, train: DatasetBundle, val: DatasetBundle | None = None) -> "RustyStatsGLM":
        import polars as pl
        import rustystats as rs

        train = filter_tiny_exposure_fit_rows(train)
        val = filter_tiny_exposure_fit_rows(val) if val is not None else None
        self._task = train.task.task
        self._learn_schema(train)
        self._pp_force_target_encoding = False
        self._pp_target_encode_columns = self._pp_auto_target_encode_columns(train)
        data = self._prep_X(train.X)
        terms = self._terms(train)

        if self._task == TaskType.MULTICLF:
            y = np.asarray(train.y, dtype=int)
            n_classes = int(train.meta.get("n_classes", int(y.max()) + 1))
            classes = [str(i) for i in range(n_classes)]
            observed_codes = sorted(set(int(value) for value in y))
            fit_classes = [str(value) for value in observed_codes]
            w = np.asarray(train.sample_weight, dtype=float)
            self._classes = classes
            self._fit_classes = fit_classes
            self._fit_class_indices = np.asarray(observed_codes, dtype=int)
            self._constant_multiclass_proba = None

            if len(fit_classes) < 2:
                counts = np.bincount(y, weights=w, minlength=n_classes).astype(float)
                counts = counts + 1e-9
                self._constant_multiclass_proba = counts / counts.sum()
                self.model = None
                self._fitted = True
                return self

            fit_kw = self._multiclass_fit_kwargs(train)
            response = [str(int(value)) for value in y]
            d = data.with_columns(pl.Series(_RESPONSE, response))
            model_kwargs = {}
            if not np.allclose(w, 1.0):
                d = d.with_columns(pl.Series(_WEIGHT, w))
                model_kwargs["weights"] = _WEIGHT
            spec = rs.multinomial_dict(
                response=_RESPONSE,
                terms=terms,
                data=d,
                classes=fit_classes,
                reference=fit_classes[0],
                seed=int(self.hp.get("seed", 0)),
                **model_kwargs,
            )
            self.model = spec.fit(**fit_kw)
            self._fitted = True
            return self

        family = {
            TaskType.FREQ: "poisson",
            TaskType.SEV: "gamma",
            TaskType.PP: "tweedie",
            TaskType.CLF: "binomial",
        }[self._task]
        fit_y = np.asarray(train.y, dtype=float)
        model_kwargs = {}
        if self._task in (TaskType.FREQ, TaskType.PP):
            exposure_arr = np.asarray(train.exposure, dtype=float)
            data = data.with_columns(
                pl.Series(_RESPONSE, fit_y),
                pl.Series(_EXPOSURE, exposure_arr),
            )
            model_kwargs["exposure"] = _EXPOSURE
        else:
            data = data.with_columns(pl.Series(_RESPONSE, fit_y))
            w = np.asarray(train.sample_weight, dtype=float)
            if not np.allclose(w, 1.0):
                data = data.with_columns(pl.Series(_WEIGHT, w))
                model_kwargs["weights"] = _WEIGHT
        if self._task == TaskType.PP:
            p = float(train.task.tweedie_power or 1.5)
            model_kwargs["var_power"] = p
            model_kwargs["allow_extended_tweedie"] = True
            if bool(self.hp.get("pp_exposure_weight_fix", True)):
                w = np.power(np.clip(exposure_arr, 1e-12, np.inf), p - 1.0)
                sw = np.asarray(train.sample_weight, dtype=float)
                if not np.allclose(sw, 1.0):
                    w = w * sw
                data = data.with_columns(pl.Series(_WEIGHT, w))
                model_kwargs["weights"] = _WEIGHT

        spec = rs.glm_dict(
            response=_RESPONSE,
            terms=terms,
            data=data,
            family=family,
            seed=int(self.hp.get("seed", 0)),
            **model_kwargs,
        )
        self.model = spec.fit(**self._fit_kwargs(train))
        if self._task == TaskType.PP and self._pp_fit_has_extreme_eta(self.model, data):
            self._pp_force_target_encoding = True
            self._pp_target_encode_columns = set(self._cat)
            terms = self._terms(train)
            spec = rs.glm_dict(
                response=_RESPONSE,
                terms=terms,
                data=data,
                family=family,
                seed=int(self.hp.get("seed", 0)),
                **model_kwargs,
            )
            self.model = spec.fit(**self._fit_kwargs(train))
        if self._task == TaskType.CLF and bool(self.hp.get("calibrate_binary", True)):
            w = np.asarray(train.sample_weight, dtype=float)
            pred_train = self._rs_predict(self.model, data)
            self._binary_logit_shift = self._fit_logit_shift(pred_train, fit_y, w)
        else:
            self._binary_logit_shift = 0.0
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, exposure: np.ndarray | None = None) -> np.ndarray:
        data = self._prep_X(X)
        if self._task == TaskType.MULTICLF:
            constant = getattr(self, "_constant_multiclass_proba", None)
            if constant is not None:
                return np.tile(np.asarray(constant, dtype=float), (len(X), 1))
            p_fit = np.asarray(self.model.predict_proba(data), dtype=float)
            p = np.zeros((len(X), len(self._classes)), dtype=float)
            p[:, np.asarray(self._fit_class_indices, dtype=int)] = p_fit
            p = np.clip(p, 1e-9, 1.0)
            return p / p.sum(axis=1, keepdims=True)
        if self._task in (TaskType.FREQ, TaskType.PP):
            return np.asarray(
                self._rs_predict(self.model, data, exposure=np.ones(len(X))),
                dtype=float,
            )
        pred = np.asarray(self._rs_predict(self.model, data), dtype=float)
        if self._task == TaskType.CLF:
            shift = float(getattr(self, "_binary_logit_shift", 0.0))
            pred = self._sigmoid(self._logit(pred) + shift)
            return np.clip(pred, 1e-6, 1.0 - 1e-6)
        return pred

    def explain(self) -> Explanation:
        try:
            artifact = self.model.coef_table()
        except Exception:
            try:
                artifact = self.model.summary()
            except Exception:
                return Explanation(kind="none")
        return Explanation(kind="coefficients", artifact=artifact)

    @staticmethod
    def default_hp() -> dict:
        return {
            "alpha": None,
            "prior_weight": "auto",
            "continuous": "auto",
            "spline": "ns",
            "spline_df": 4,
            "low_card_categorical": _LOW_CARD,
            "max_categorical_dummy_features": 200,
            "pp_exposure_weight_fix": True,
            "calibrate_binary": True,
            "multiclass_solver": "auto",
            "robust_continuous": True,
            "continuous_winsor_lower": 0.005,
            "continuous_winsor_upper": 0.995,
            "continuous_log1p_skewed": True,
            "on_extreme_eta": "clip",
            "pp_target_encode_categoricals": "auto",
            "pp_te_sparse_positive_max": 5,
            "pp_te_sparse_level_min_rows": 20,
        }

    @staticmethod
    def sample_hp(rng: np.random.Generator) -> dict:
        return {"alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(1.0))))}
