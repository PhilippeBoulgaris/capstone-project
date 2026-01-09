# src/ml.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Iterable, Any

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from .config import Config


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_month_end_index(dates: pd.Series) -> pd.Series:
    """Normalize dates to month-end (tz-naive) for consistent time grouping."""
    d = pd.to_datetime(pd.Series(dates), errors="coerce")
    d = d.dt.tz_localize(None)
    return d.dt.to_period("M").dt.to_timestamp("M")


def _ensure_1d_array(x) -> np.ndarray:
    arr = np.asarray(x)
    return arr.ravel()


def make_date_grouped_tscv(
    dates_train: pd.Series,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build a TimeSeriesSplit CV grouped by month-end date:
    - all samples in the same month go in the same fold.
    Returns a list of (train_idx, valid_idx) arrays referencing row indices
    of the ORIGINAL training set.

    IMPORTANT:
    - If dates contain NaT, folds are built on valid rows only.
      Caller should align X_train/y_train to the same mask if NaTs exist.
    """
    d = _to_month_end_index(dates_train).reset_index(drop=True)

    valid_mask = ~d.isna()
    if not bool(valid_mask.all()):
        d_valid = d[valid_mask].reset_index(drop=True)
    else:
        d_valid = d

    uniq = pd.Index(d_valid.unique()).sort_values()
    if len(uniq) < (n_splits + 1):
        return []

    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []

    # Build folds in "month space", then map back to row indices
    for train_m_idx, valid_m_idx in tscv.split(uniq):
        train_months = set(uniq[train_m_idx])
        valid_months = set(uniq[valid_m_idx])

        # row indices within *valid* rows
        train_idx_valid = np.flatnonzero(d_valid.isin(train_months).to_numpy())
        valid_idx_valid = np.flatnonzero(d_valid.isin(valid_months).to_numpy())

        if len(train_idx_valid) == 0 or len(valid_idx_valid) == 0:
            continue

        # Map back to original indices if NaTs existed
        if not bool(valid_mask.all()):
            orig_idx = np.flatnonzero(valid_mask.to_numpy())
            train_idx = orig_idx[train_idx_valid]
            valid_idx = orig_idx[valid_idx_valid]
        else:
            train_idx, valid_idx = train_idx_valid, valid_idx_valid

        folds.append((train_idx, valid_idx))

    return folds


def _default_n_jobs(cfg: Config) -> int:
    """
    Reproducibility-first default:
    - Use cfg.N_JOBS if it exists.
    - Else use 1 (deterministic).
    """
    return int(getattr(cfg, "N_JOBS", 1))


# -----------------------------------------------------------------------------
# ML Trainer
# -----------------------------------------------------------------------------

class MLTrainer:
    def __init__(self, config: Config):
        self.config = config
        seed = int(getattr(config, "RANDOM_STATE", 42))

        # Models (random_state everywhere it's supported)
        self.models: Dict[str, Pipeline] = {
            "Logistic Regression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    random_state=seed,
                    # note: lbfgs is deterministic; keep defaults unless you have a reason
                )),
            ]),
            "Random Forest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    random_state=seed,
                )),
            ]),
            "Gradient Boosting": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(
                    random_state=seed,
                )),
            ]),
            "Neural Network": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(
                    max_iter=800,
                    random_state=seed,
                )),
            ]),
        }

        # Hyperparameter search spaces
        self.param_distributions: Dict[str, Dict[str, list]] = {
            "Random Forest": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [5, 10, 15, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "Gradient Boosting": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
            "Neural Network": {
                "model__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
                "model__alpha": [0.0001, 0.001, 0.01],
                "model__learning_rate_init": [0.001, 0.01],
            },
        }

    def fit_model(
        self,
        model_name: str,
        X_train,
        y_train,
        dates_train: Optional[pd.Series] = None,
    ) -> Tuple[Any, Optional[dict], Optional[float]]:
        """
        Fit one model.
        If model has a search space, run RandomizedSearchCV with time-series CV.

        If dates_train is provided, CV is grouped by month to prevent leakage across
        tickers inside the same month.
        """
        base_model = self.models[model_name]

        if model_name in self.param_distributions:
            # Build CV
            cv = None
            if dates_train is not None:
                folds = make_date_grouped_tscv(dates_train, n_splits=int(self.config.N_CV_SPLITS))
                if folds:
                    cv = folds

            if cv is None:
                cv = TimeSeriesSplit(n_splits=int(self.config.N_CV_SPLITS))

            # Deterministic by default (n_jobs=1 unless you explicitly override in Config)
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.param_distributions[model_name],
                n_iter=int(self.config.N_ITER_SEARCH),
                cv=cv,
                scoring="roc_auc",
                random_state=int(self.config.RANDOM_STATE),
                n_jobs=_default_n_jobs(self.config),
            )
            search.fit(X_train, y_train)
            return search.best_estimator_, search.best_params_, float(search.best_score_)

        base_model.fit(X_train, y_train)
        return base_model, None, None

    @staticmethod
    def safe_predict_proba(model, X) -> np.ndarray:
        """
        Return P(class=1). Works for models with predict_proba or decision_function.
        """
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            # binary: columns [P0, P1]
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1]
            return _ensure_1d_array(p)

        if hasattr(model, "decision_function"):
            scores = _ensure_1d_array(model.decision_function(X))
            # logistic squashing
            return 1.0 / (1.0 + np.exp(-scores))

        # last resort: treat predictions as probs (not ideal, but avoids crash)
        return _ensure_1d_array(model.predict(X)).astype(float)

    def train_universe(
        self,
        universe_name: str,
        X_train,
        y_train,
        X_test,
        y_test,
        dates_train: Optional[pd.Series] = None,
    ) -> Dict[str, dict]:
        results: Dict[str, dict] = {}

        print(f"\n===== TRAINING UNIVERSE: {universe_name} =====")
        print(f"  Train samples: {len(y_train):,} | Test: {len(y_test):,}")

        # basic guards
        y_train_arr = _ensure_1d_array(y_train)
        y_test_arr = _ensure_1d_array(y_test)

        if len(y_train_arr) < 200 or len(np.unique(y_train_arr)) < 2:
            print("  ⚠️ Not enough training data -> skipping")
            return results
        if len(y_test_arr) < 50 or len(np.unique(y_test_arr)) < 2:
            print("  ⚠️ Not enough test data -> skipping")
            return results

        for model_name in self.models.keys():
            print(f"  Training {model_name} [{universe_name}]...")

            best_model, best_params, best_cv = self.fit_model(
                model_name, X_train, y_train, dates_train=dates_train
            )

            if best_params is not None:
                print(f"    → Best params: {best_params}")
                print(f"    → CV AUC: {best_cv:.3f}")

            y_pred = best_model.predict(X_test)
            y_proba = self.safe_predict_proba(best_model, X_test)

            acc = float(accuracy_score(y_test_arr, y_pred))
            auc = float(roc_auc_score(y_test_arr, y_proba))

            key = f"{model_name} ({universe_name}-trained)"
            results[key] = {
                "model": best_model,
                "accuracy": acc,
                "auc": auc,
                "predictions": _ensure_1d_array(y_pred),
                "probabilities": _ensure_1d_array(y_proba),
                "universe": universe_name,
                "best_params": best_params,
                "best_cv_auc": best_cv,
            }
            print(f"    ✓ Test Accuracy: {acc:.3f} | AUC: {auc:.3f}")

        return results


# -----------------------------------------------------------------------------
# Portfolio construction from meta
# -----------------------------------------------------------------------------

def build_weights_from_meta(
    meta: pd.DataFrame,
    prob: np.ndarray,
    top_percentile: float,
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """
    Build long-only equal-weight portfolios each decision month based on predicted probabilities.
    Deterministic tie-break: (prob desc, ticker asc).

    Returns:
        {month_end_timestamp: {ticker: weight}}
    """
    if meta is None or len(meta) == 0:
        return {}

    mm = meta.copy()
    mm["prob"] = _ensure_1d_array(prob)

    mm["date"] = _to_month_end_index(mm["date"])

    # Defensive: ensure required cols exist
    if "ticker" not in mm.columns:
        raise ValueError("meta must contain a 'ticker' column")
    if "date" not in mm.columns:
        raise ValueError("meta must contain a 'date' column")

    weights: Dict[pd.Timestamp, Dict[str, float]] = {}

    # Iterate months in sorted order for determinism
    for d in pd.Index(mm["date"].dropna().unique()).sort_values():
        dd = mm.loc[mm["date"] == d, ["ticker", "prob"]].dropna(subset=["ticker", "prob"]).copy()
        if len(dd) == 0:
            continue

        n = max(1, int(np.floor(len(dd) * float(top_percentile))))

        # Deterministic tie-break
        dd2 = dd.sort_values(["prob", "ticker"], ascending=[False, True], kind="mergesort")
        top = dd2.head(n)

        k = int(len(top))
        if k <= 0:
            continue

        w = 1.0 / k
        weights[pd.Timestamp(d)] = {str(t): w for t in top["ticker"].tolist()}

    return weights
