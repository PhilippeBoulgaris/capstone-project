# src/sec_style.py
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def _zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    return (s - mu) / (sd + 1e-12)


def compute_sec_value_score(cs: pd.DataFrame) -> pd.Series:
    """
    Value: inverse multiples -> higher score = cheaper
    Works cross-sectionally (cs is one date slice).
    """
    cols = [c for c in ["pe", "pb", "ps", "ev_ebitda"] if c in cs.columns]
    if not cols:
        return pd.Series(index=cs.index, dtype=float)

    inv = cs[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        inv[c] = 1.0 / inv[c].where(inv[c] > 0)

    inv = inv.replace([np.inf, -np.inf], np.nan)

    for c in cols:
        lo = inv[c].quantile(0.01)
        hi = inv[c].quantile(0.99)
        inv[c] = inv[c].clip(lower=lo, upper=hi)

    for c in cols:
        med = inv[c].median()
        inv[c] = inv[c].fillna(med if pd.notna(med) else 0.0)

    z = inv.apply(_zscore_series)
    return z.mean(axis=1)


def compute_sec_growth_score(cs: pd.DataFrame) -> pd.Series:
    """
    Growth proxy from SEC panel: profitability-ish
    (roe, net_margin). Cross-sectional.
    """
    cols = [c for c in ["roe", "net_margin"] if c in cs.columns]
    if not cols:
        return pd.Series(index=cs.index, dtype=float)

    g = cs[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    for c in cols:
        lo = g[c].quantile(0.01)
        hi = g[c].quantile(0.99)
        g[c] = g[c].clip(lower=lo, upper=hi)

    for c in cols:
        med = g[c].median()
        g[c] = g[c].fillna(med if pd.notna(med) else 0.0)

    z = g.apply(_zscore_series)
    return z.mean(axis=1)


def build_sec_style_weights(
    sec_scored: pd.DataFrame,
    score_col: str,
    top_percentile: float,
    allowed_tickers: Optional[List[str]] = None,
) -> Dict[pd.Timestamp, Dict[str, float]]:
    """
    Monthly weights dict: at each date d, pick top top_percentile by score
    and hold for next month.
    """
    if sec_scored is None or sec_scored.empty:
        return {}

    p = sec_scored.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce")
    p = p.dropna(subset=["date", "ticker"])
    if allowed_tickers is not None:
        p = p[p["ticker"].isin(set(allowed_tickers))].copy()

    if p.empty or score_col not in p.columns:
        return {}

    weights = {}
    for d, cs in p.groupby("date"):
        cs = cs.dropna(subset=[score_col])
        if cs.empty:
            continue

        n = max(1, int(len(cs) * float(top_percentile)))
        top = cs.nlargest(n, score_col)

        tick_list = top["ticker"].astype(str).tolist()
        if len(tick_list) == 0:
            continue

        weights[d] = {t: 1.0 / len(tick_list) for t in tick_list}

    return weights
