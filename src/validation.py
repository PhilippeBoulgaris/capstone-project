# src/validation.py
# -*- coding: utf-8 -*-

from typing import List
import pandas as pd


def numeric_feature_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    ex = set(exclude)
    return [
        c for c in df.columns
        if c not in ex and pd.api.types.is_numeric_dtype(df[c])
    ]


def assert_has_min_samples(y, min_train: int = 200, min_test: int = 50) -> None:
    # simple guardrails (trainer also checks, but this makes debugging easier)
    if len(y) == 0:
        raise ValueError("Empty target vector.")
