# src/ml.py
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple
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


class MLTrainer:
    def __init__(self, config: Config):
        self.config = config

        self.models = {
            "Logistic Regression": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE)),
            ]),
            "Random Forest": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(random_state=config.RANDOM_STATE)),
            ]),
            "Gradient Boosting": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=config.RANDOM_STATE)),
            ]),
            "Neural Network": Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(max_iter=800, random_state=config.RANDOM_STATE)),
            ]),
        }

        self.param_distributions = {
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

    def fit_model(self, model_name: str, X_train, y_train) -> Tuple[object, Optional[dict], Optional[float]]:
        base_model = self.models[model_name]
        if model_name in self.param_distributions:
            tscv = TimeSeriesSplit(n_splits=self.config.N_CV_SPLITS)
            search = RandomizedSearchCV(
                base_model,
                self.param_distributions[model_name],
                n_iter=self.config.N_ITER_SEARCH,
                cv=tscv,
                scoring="roc_auc",
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            return search.best_estimator_, search.best_params_, float(search.best_score_)
        else:
            base_model.fit(X_train, y_train)
            return base_model, None, None

    @staticmethod
    def safe_predict_proba(model, X) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    def train_universe(self, universe_name: str, X_train, y_train, X_test, y_test) -> Dict:
        results = {}
        print(f"\n===== TRAINING UNIVERSE: {universe_name} =====")
        print(f"  Train samples: {len(y_train):,} | Test: {len(y_test):,}")

        if len(y_train) < 200 or len(np.unique(y_train)) < 2 or len(y_test) < 50 or len(np.unique(y_test)) < 2:
            print("  ⚠️ Not enough data -> skipping")
            return results

        for model_name in self.models.keys():
            print(f"  Training {model_name} [{universe_name}]...")
            best_model, best_params, best_cv = self.fit_model(model_name, X_train, y_train)

            if best_params is not None:
                print(f"    → Best params: {best_params}")
                print(f"    → CV AUC: {best_cv:.3f}")

            y_pred = best_model.predict(X_test)
            y_proba = self.safe_predict_proba(best_model, X_test)

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            key = f"{model_name} ({universe_name}-trained)"
            results[key] = {
                "model": best_model,
                "accuracy": acc,
                "auc": auc,
                "predictions": y_pred,
                "probabilities": y_proba,
                "universe": universe_name,
            }
            print(f"    ✓ Test Accuracy: {acc:.3f} | AUC: {auc:.3f}")

        return results


def build_weights_from_meta(meta: pd.DataFrame, prob: np.ndarray, top_percentile: float) -> Dict[pd.Timestamp, Dict[str, float]]:
    mm = meta.copy()
    mm["prob"] = prob
    weights = {}
    for d in mm["date"].unique():
        dd = mm[mm["date"] == d]
        if len(dd) == 0:
            continue
        n = max(1, int(len(dd) * top_percentile))
        top = dd.nlargest(n, "prob")
        weights[d] = {t: 1.0 / len(top) for t in top["ticker"]}
    return weights
