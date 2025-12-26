# src/portfolios.py
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def build_equal_weight_static(
    dates: pd.DatetimeIndex,
    tickers: List[str],
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """
    Weights decided at each date d (for holding next month):
    {d: {ticker: 1/N}}
    """
    tickers = [t for t in tickers if t is not None]
    if len(tickers) == 0:
        return {}
    w = {t: 1.0 / len(tickers) for t in tickers}
    return {d: dict(w) for d in dates}


def build_top_percentile_universe_from_scores(
    scores: pd.Series,
    top_percentile: float,
) -> List[str]:
    """
    scores index = tickers, higher score = more selected.
    """
    scores = scores.dropna()
    if len(scores) == 0:
        return []
    n = max(1, int(len(scores) * float(top_percentile)))
    return scores.nlargest(n).index.tolist()


def build_random_weights_each_month(
    dates: pd.DatetimeIndex,
    tickers: List[str],
    top_percentile: float,
    seed: int = 42,
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """
    Each month d: randomly pick top_percentile of tickers, equal-weight.
    """
    rng = np.random.default_rng(seed)
    tickers = list(tickers)
    if len(tickers) == 0:
        return {}

    weights = {}
    for d in dates:
        n = max(1, int(len(tickers) * float(top_percentile)))
        sel = rng.choice(tickers, size=n, replace=False)
        weights[d] = {t: 1.0 / n for t in sel}
    return weights
