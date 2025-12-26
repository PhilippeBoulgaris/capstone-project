# src/performance.py
# -*- coding: utf-8 -*-

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from .config import Config


def to_equity(logret: pd.Series) -> pd.Series:
    logret = logret.dropna()
    if len(logret) == 0:
        return pd.Series(dtype=float)
    return np.exp(logret.cumsum())


def compute_drawdown(eq: pd.Series) -> pd.Series:
    eq = eq.dropna()
    if len(eq) < 2:
        return pd.Series(dtype=float)
    return (eq / eq.cummax()) - 1.0


class PerformanceAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def backtest_portfolio(
        self,
        weights_dict: Dict[pd.Timestamp, Dict[str, float]],
        returns: pd.DataFrame,
        costs_bps: Optional[int] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        weights_dict: {date d: {ticker: weight}} weights decided at date d,
        applied to returns at next month d_next.
        returns: monthly log returns, index = month end (or month start), aligned.
        """
        if costs_bps is None:
            costs_bps = self.config.COSTS_BPS

        idx = returns.index
        if len(idx) < 3:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        next_map = {idx[i]: idx[i + 1] for i in range(len(idx) - 1)}

        r_list, turnover_list, used_dates = [], [], []
        prev_w = None

        for d in sorted(weights_dict.keys()):
            if d not in next_map:
                continue
            d_next = next_map[d]
            if d_next not in returns.index:
                continue

            w = pd.Series(weights_dict[d], dtype=float).reindex(returns.columns, fill_value=0.0)
            w = w / w.sum() if w.sum() > 0 else w

            r = float((w * returns.loc[d_next]).sum())

            if prev_w is None:
                tc, turnover = 0.0, 1.0
            else:
                turnover = float((w - prev_w).abs().sum())
                tc = (costs_bps / 10000.0) * turnover

            r_list.append(r - tc)
            turnover_list.append(turnover)
            used_dates.append(d_next)
            prev_w = w

        return pd.Series(r_list, index=used_dates), pd.Series(turnover_list, index=used_dates)

    def compute_enhanced_metrics(self, equity_curve: pd.Series, bench_returns: pd.Series) -> Dict:
        if equity_curve is None or len(equity_curve) < 2:
            final_val = 1.0 if equity_curve is None or len(equity_curve) == 0 else float(equity_curve.iloc[-1])
            return self._empty_metrics(final_val)

        logr = np.log(equity_curve / equity_curve.shift(1)).dropna()
        if len(logr) < 2:
            return self._empty_metrics(float(equity_curve.iloc[-1]))

        rf_monthly = self.config.RF_RATE_ANNUAL / 12.0

        mu = float(logr.mean() * 12)
        sig = float(logr.std(ddof=0) * np.sqrt(12))
        sharpe = (mu - self.config.RF_RATE_ANNUAL) / (sig + 1e-12)

        downside = logr[logr < rf_monthly]
        downside_std = float(downside.std(ddof=0) * np.sqrt(12)) if len(downside) > 1 else sig
        sortino = (mu - self.config.RF_RATE_ANNUAL) / (downside_std + 1e-12)

        mdd = float((equity_curve / equity_curve.cummax() - 1).min())
        pct_neg = float((logr < 0).sum() / len(logr) * 100)
        pct_large_loss = float((logr < -0.10).sum() / len(logr) * 100)

        bench_aligned = bench_returns.reindex(logr.index, fill_value=0.0)
        simple_alpha = float((logr - bench_aligned).mean() * 12)

        excess_ret = logr - rf_monthly
        excess_bench = bench_aligned - rf_monthly

        if len(excess_ret) > 5 and excess_bench.std() > 1e-9:
            slope, intercept, _, _, std_err = stats.linregress(excess_bench, excess_ret)
            capm_beta, capm_alpha = float(slope), float(intercept * 12)
            alpha_tstat = float(intercept / (std_err + 1e-12))
        else:
            capm_beta, capm_alpha, alpha_tstat = 1.0, 0.0, 0.0

        return {
            "CAGR": mu,
            "Volatility": sig,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": mdd,
            "Pct Negative Months": pct_neg,
            "Pct Large Losses": pct_large_loss,
            "Simple Alpha": simple_alpha,
            "CAPM Alpha": capm_alpha,
            "CAPM Beta": capm_beta,
            "Alpha t-stat": alpha_tstat,
            "Final Value": float(equity_curve.iloc[-1]),
        }

    @staticmethod
    def _empty_metrics(final_value: float = 1.0) -> Dict:
        return {
            "CAGR": 0.0,
            "Volatility": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Max Drawdown": 0.0,
            "Pct Negative Months": 0.0,
            "Pct Large Losses": 0.0,
            "Simple Alpha": 0.0,
            "CAPM Alpha": 0.0,
            "CAPM Beta": 1.0,
            "Alpha t-stat": 0.0,
            "Final Value": final_value,
        }
