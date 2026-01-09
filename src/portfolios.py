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
    Each date d: randomly pick top_percentile of tickers, equal-weight (LONG-ONLY).
    Guaranteed: weights >= 0, finite, sum(weights)=1.0.
    """
    if not (0 < float(top_percentile) <= 1.0):
        raise ValueError(f"top_percentile must be in (0, 1], got {top_percentile}")

    # Clean tickers: drop None/NaN, unique
    tickers_clean = []
    seen = set()
    for t in tickers:
        if t is None:
            continue
        t = str(t).strip()
        if t == "" or t.lower() == "nan":
            continue
        if t in seen:
            continue
        seen.add(t)
        tickers_clean.append(t)

    if len(tickers_clean) == 0:
        return {}

    rng = np.random.default_rng(seed)
    n_total = len(tickers_clean)
    n = max(1, int(np.floor(n_total * float(top_percentile))))
    n = min(n, n_total)

    weights: Dict[pd.Timestamp, Dict[str, float]] = {}

    # Ensure dates are Timestamp and sorted for deterministic output
    dates = pd.DatetimeIndex(dates).sort_values()

    for d in dates:
        sel = rng.choice(tickers_clean, size=n, replace=False)

        # Equal weights, force exact sum=1 (avoid floating drift)
        w = np.full(n, 1.0 / n, dtype=float)
        w[-1] = 1.0 - float(w[:-1].sum())

        # Build dict and assert invariants
        wd = {str(t): float(wi) for t, wi in zip(sel, w)}
        s = sum(wd.values())
        if (not np.isfinite(s)) or abs(s - 1.0) > 1e-10:
            # last-resort normalization (shouldn't happen)
            wd = {k: float(v) for k, v in wd.items() if np.isfinite(v) and v > 0}
            ss = sum(wd.values())
            if ss <= 0:
                continue
            wd = {k: v / ss for k, v in wd.items()}

        # Final safety: long-only
        wd = {k: float(v) for k, v in wd.items() if np.isfinite(v) and v >= 0}

        weights[pd.Timestamp(d)] = wd

    return weights

