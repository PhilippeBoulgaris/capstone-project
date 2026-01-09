# src/performance.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_month_end_index(idx) -> pd.DatetimeIndex:
    """Normalize any datetime-like index to month-end (tz-naive)."""
    di = pd.DatetimeIndex(pd.to_datetime(idx, errors="coerce")).tz_localize(None)
    return di.to_period("M").to_timestamp("M")


def to_equity(logret: pd.Series) -> pd.Series:
    """
    Convert a log-return series to an equity curve (starting at 1.0).
    """
    r = pd.Series(logret, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return pd.Series(dtype=float)
    # Keep original index but sanitize to month-end if it's datetime-like
    try:
        r.index = _to_month_end_index(r.index)
        r = r[~r.index.duplicated(keep="last")].sort_index()
    except Exception:
        pass
    return np.exp(r.cumsum())


def compute_drawdown(eq: pd.Series) -> pd.Series:
    """
    Compute drawdown series from an equity curve (levels).
    """
    e = pd.Series(eq, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(e) < 2:
        return pd.Series(dtype=float)
    try:
        e.index = _to_month_end_index(e.index)
        e = e[~e.index.duplicated(keep="last")].sort_index()
    except Exception:
        pass
    return (e / e.cummax()) - 1.0


class PerformanceAnalyzer:
    """
    - Backtest monthly-rebalanced portfolios from a weights dict (decision date -> weights).
    - Compute enhanced performance metrics from an equity curve and benchmark log returns.

    Notes on turnover
    -----------------
    Turnover is computed as one-way turnover:
        turnover_t = 0.5 * sum_i | w_target_t - w_post_{t-1} |
    where w_post_{t-1} are the PREVIOUS target weights drifted through the
    realized returns of the intervening month (i.e., post-return weights).
    """

    def __init__(self, config):
        # config should expose:
        # - COSTS_BPS (int)
        # - RF_RATE_ANNUAL (float, simple annual rate e.g. 0.02)
        self.config = config

    @staticmethod
    def _to_timestamp(x: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
        if x is None:
            return None
        return pd.Timestamp(x)

    @staticmethod
    def _empty_metrics(final_value: float = 1.0) -> Dict[str, float]:
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
            "CAPM Beta": 0.0,
            "Alpha t-stat": 0.0,
            "Final Value": float(final_value),
            "N Months": 0.0,
        }

    @staticmethod
    def _clean_and_normalize_weights(w: pd.Series) -> Optional[pd.Series]:
        """
        Make weights finite, long-only, fully invested. Return None if unusable.
        """
        w2 = (
            pd.Series(w, dtype=float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        w2[w2 < 0.0] = 0.0

        s = float(w2.sum())
        if (not np.isfinite(s)) or (s <= 0.0):
            return None

        return w2 / s

    @staticmethod
    def _infer_month_anchor(idx: pd.DatetimeIndex) -> str:
        """
        Infer whether the index is closer to MonthEnd or MonthBegin.
        Returns "ME" or "MS".
        """
        if len(idx) == 0:
            return "ME"
        days = pd.Series(idx.day)
        ms_score = float((days <= 3).mean())
        me_score = float((days >= 25).mean())
        return "MS" if ms_score > me_score else "ME"

    @classmethod
    def _align_month_index(cls, idx: pd.DatetimeIndex, anchor: str) -> pd.DatetimeIndex:
        """
        Align any datetime index to month start ("MS") or month end ("ME"), preserving month.
        """
        idx = pd.DatetimeIndex(pd.to_datetime(idx)).tz_localize(None)
        if anchor == "MS":
            return idx.to_period("M").to_timestamp(how="start")
        return idx.to_period("M").to_timestamp(how="end")

    def backtest_portfolio(
        self,
        weights_dict: Dict[pd.Timestamp, Dict[str, float]],
        returns: pd.DataFrame,
        costs_bps: Optional[int] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Backtest a monthly-rebalanced portfolio.

        Conventions
        -----------
        - `returns` are MONTHLY LOG returns.
        - Weights decided at date d are applied to returns at next month d_next.
        - Turnover is one-way (drift-adjusted): 0.5 * sum |w_new - w_old_post|.
        - Transaction costs are applied in log space:
              r_net = r_gross - (costs_bps/10000) * turnover

        Index alignment
        ---------------
        - All dates are aligned to MonthEnd or MonthStart based on `returns.index`.
        """
        if costs_bps is None:
            costs_bps = int(getattr(self.config, "COSTS_BPS", 0))

        rets = returns.copy()
        rets.index = pd.DatetimeIndex(pd.to_datetime(rets.index)).tz_localize(None)
        rets = rets.sort_index()

        # Align returns index to a consistent monthly anchor
        anchor = self._infer_month_anchor(rets.index)
        rets.index = self._align_month_index(rets.index, anchor)
        rets = rets[~rets.index.duplicated(keep="last")].sort_index()

        # Optional windowing (align window endpoints to same anchor for consistency)
        sd = self._to_timestamp(start_date)
        ed = self._to_timestamp(end_date)
        if sd is not None:
            sd_al = self._align_month_index(pd.DatetimeIndex([sd]), anchor)[0]
            rets = rets.loc[rets.index >= sd_al]
        if ed is not None:
            ed_al = self._align_month_index(pd.DatetimeIndex([ed]), anchor)[0]
            rets = rets.loc[rets.index <= ed_al]

        idx = rets.index
        if len(idx) < 3:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Map decision date -> next month (applied date)
        next_map = {idx[i]: idx[i + 1] for i in range(len(idx) - 1)}

        # Align decision dates from weights_dict to the same monthly anchor as returns
        wdict_aligned: Dict[pd.Timestamp, Dict[str, float]] = {}
        for d, w in weights_dict.items():
            ts = pd.Timestamp(d).tz_localize(None) if isinstance(d, pd.Timestamp) else pd.Timestamp(d)
            ts_aligned = self._align_month_index(pd.DatetimeIndex([ts]), anchor)[0]
            wdict_aligned[ts_aligned] = w

        r_list: list[float] = []
        turnover_list: list[float] = []
        used_dates: list[pd.Timestamp] = []

        prev_w: Optional[pd.Series] = None

        for d in sorted(wdict_aligned.keys()):
            if d not in next_map:
                continue

            d_next = next_map[d]
            if d_next not in rets.index:
                continue

            raw_w = (
                pd.Series(wdict_aligned[d], dtype=float)
                .reindex(rets.columns, fill_value=0.0)
            )
            w = self._clean_and_normalize_weights(raw_w)
            if w is None:
                continue

            # Gross portfolio LOG return next month
            r_gross = float((w * rets.loc[d_next]).sum())

            # Turnover + transaction cost (drift-adjusted)
            if prev_w is None:
                turnover = 0.0
                tc = 0.0
            else:
                # Drift previous target weights through realized returns of month d.
                # This approximates the "pre-trade" weights before rebalancing at d.
                if d in rets.index:
                    gross = np.exp(rets.loc[d])  # LOG returns -> gross returns
                    prev_val = prev_w * gross
                    s = float(prev_val.sum())
                    prev_w_post = (prev_val / s) if (np.isfinite(s) and s > 0.0) else prev_w
                else:
                    prev_w_post = prev_w

                turnover = 0.5 * float((w - prev_w_post).abs().sum())
                tc = (float(costs_bps) / 10000.0) * turnover

            r_list.append(r_gross - tc)
            turnover_list.append(turnover)
            used_dates.append(d_next)

            prev_w = w

        return pd.Series(r_list, index=used_dates), pd.Series(turnover_list, index=used_dates)

    @staticmethod
    def _capm_alpha_beta_tstat(
        excess_ret: pd.Series,
        excess_bench: pd.Series,
    ) -> Tuple[float, float, float]:
        """
        OLS CAPM regression on monthly excess LOG returns:
            y_t = alpha + beta * x_t + eps_t
        Returns (alpha_log_annual, beta, alpha_tstat).
        """
        y = pd.Series(excess_ret, dtype=float)
        x = pd.Series(excess_bench, dtype=float)

        df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if len(df) < 12:
            return 0.0, 0.0, 0.0

        x_arr = df["x"].to_numpy()
        y_arr = df["y"].to_numpy()
        n = int(len(y_arr))

        x_mean = float(x_arr.mean())
        y_mean = float(y_arr.mean())

        Sxx = float(((x_arr - x_mean) ** 2).sum())
        if Sxx <= 1e-18:
            return 0.0, 0.0, 0.0

        beta = float(((x_arr - x_mean) * (y_arr - y_mean)).sum() / Sxx)
        alpha_monthly = float(y_mean - beta * x_mean)

        resid = y_arr - (alpha_monthly + beta * x_arr)

        k = 2
        dof = max(1, n - k)
        sigma2 = float((resid**2).sum() / dof)

        # Standard error of alpha (with intercept)
        se_alpha = float(np.sqrt(sigma2 * (1.0 / n + (x_mean**2) / (Sxx + 1e-18))))
        alpha_t = float(alpha_monthly / (se_alpha + 1e-18))

        alpha_log_annual = float(alpha_monthly * 12.0)
        return alpha_log_annual, beta, alpha_t

    def compute_enhanced_metrics(
        self,
        equity_curve: pd.Series,
        bench_returns: pd.Series,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        rebase: bool = True,
    ) -> Dict[str, float]:
        """
        Compute annualized performance and risk metrics from an equity curve and benchmark returns.

        Inputs
        ------
        - equity_curve: LEVEL series (1.0 -> ...).
        - bench_returns: MONTHLY LOG returns.
        """
        if equity_curve is None:
            return self._empty_metrics(1.0)

        eq = (
            pd.Series(equity_curve, dtype=float)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if eq.empty or len(eq) < 2:
            final_val = 1.0 if eq.empty else float(eq.iloc[-1])
            return self._empty_metrics(final_val)

        # Normalize equity index to month-end for clean alignment with benchmark
        try:
            eq.index = _to_month_end_index(eq.index)
            eq = eq[~eq.index.duplicated(keep="last")].sort_index()
        except Exception:
            eq = eq.sort_index()

        # Optional windowing (month-end normalized)
        if start_date is not None:
            sd = pd.Timestamp(start_date).tz_localize(None).to_period("M").to_timestamp("M")
            eq = eq.loc[eq.index >= sd]
        if end_date is not None:
            ed = pd.Timestamp(end_date).tz_localize(None).to_period("M").to_timestamp("M")
            eq = eq.loc[eq.index <= ed]

        if len(eq) < 2:
            final_val = 1.0 if eq.empty else float(eq.iloc[-1])
            return self._empty_metrics(final_val)

        if rebase:
            first = float(eq.iloc[0])
            if np.isfinite(first) and first > 0:
                eq = eq / first

        logr = np.log(eq / eq.shift(1)).dropna()
        n_months = int(len(logr))
        if n_months < 2:
            return self._empty_metrics(float(eq.iloc[-1]))

        final_value = float(eq.iloc[-1])
        cagr = float(final_value ** (12.0 / n_months) - 1.0) if final_value > 0 else 0.0

        rf_annual_simple = float(getattr(self.config, "RF_RATE_ANNUAL", 0.0))
        rf_annual_log = float(np.log1p(rf_annual_simple))
        rf_monthly_log = float(np.log1p(rf_annual_simple / 12.0))

        mu_log_annual = float(logr.mean() * 12.0)
        sig_log_annual = float(logr.std(ddof=0) * np.sqrt(12.0))

        sharpe = float((mu_log_annual - rf_annual_log) / (sig_log_annual + 1e-12))

        downside = logr[logr < rf_monthly_log]
        if len(downside) > 1:
            downside_std_annual = float(downside.std(ddof=0) * np.sqrt(12.0))
        else:
            downside_std_annual = sig_log_annual
        sortino = float((mu_log_annual - rf_annual_log) / (downside_std_annual + 1e-12))

        mdd = float((eq / eq.cummax() - 1.0).min())
        pct_neg = float((logr < 0).mean() * 100.0)
        pct_large_loss = float((logr < -0.10).mean() * 100.0)

        # Align benchmark: month-end + drop months missing in either series (NO zero filling)
        bench_log = pd.Series(bench_returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        try:
            bench_log.index = _to_month_end_index(bench_log.index)
            bench_log = bench_log[~bench_log.index.duplicated(keep="last")].sort_index()
        except Exception:
            bench_log = bench_log.sort_index()

        df_ab = pd.concat([logr.rename("r"), bench_log.rename("b")], axis=1).dropna()
        if len(df_ab) < 2:
            return self._empty_metrics(final_value)

        r_aligned = df_ab["r"]
        b_aligned = df_ab["b"]

        alpha_log_annual = float((r_aligned - b_aligned).mean() * 12.0)
        simple_alpha = float(np.expm1(alpha_log_annual))

        excess_ret = r_aligned - rf_monthly_log
        excess_bench = b_aligned - rf_monthly_log
        capm_alpha_log_annual, capm_beta, alpha_tstat = self._capm_alpha_beta_tstat(
            excess_ret, excess_bench
        )
        capm_alpha = float(np.expm1(capm_alpha_log_annual))

        return {
            "CAGR": cagr,
            "Volatility": sig_log_annual,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": mdd,
            "Pct Negative Months": pct_neg,
            "Pct Large Losses": pct_large_loss,
            "Simple Alpha": simple_alpha,
            "CAPM Alpha": capm_alpha,
            "CAPM Beta": float(capm_beta),
            "Alpha t-stat": float(alpha_tstat),
            "Final Value": final_value,
            "N Months": float(n_months),
        }
