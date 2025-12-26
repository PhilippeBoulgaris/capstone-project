# src/utils.py
# -*- coding: utf-8 -*-

from typing import Dict, Optional
import numpy as np
import pandas as pd


def short_model_name(k: str) -> str:
    """
    Convertit:
    - "Random Forest (Yahoo-trained)" -> "RF"
    - "Logistic Regression (...)" -> "LogReg"
    etc.
    """
    s = str(k)
    s = s.replace("Logistic Regression", "LogReg")
    s = s.replace("Random Forest", "RF")
    s = s.replace("Gradient Boosting", "GB")
    s = s.replace("Neural Network", "NN")
    if "(" in s:
        s = s.split("(")[0].strip()
    return s.strip()


def pick_best_model_key(results: Dict, universe_label: str, metric: str = "auc") -> Optional[str]:
    """
    Retourne la clé du modèle avec la meilleure métrique dans results.
    Ex: "Random Forest (Yahoo-trained)"
    """
    best_key, best_val = None, -np.inf
    for k, v in results.items():
        if f"({universe_label}-trained)" not in k:
            continue
        val = v.get(metric, np.nan)
        if pd.notna(val) and val > best_val:
            best_val, best_key = val, k
    return best_key
