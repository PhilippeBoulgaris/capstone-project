# src/fundamentals_yahoo.py
# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf


class FundamentalFeatureEngineering:
    @staticmethod
    def fetch_yahoo_fundamentals(ticker: str) -> Dict:
        try:
            info = yf.Ticker(ticker).info
            if not info:
                return {"ticker": ticker}
        except Exception as e:
            print(f"    ⚠️ Yahoo error {ticker}: {e}")
            return {"ticker": ticker}

        mcap = info.get("marketCap")
        fcf = info.get("freeCashflow")
        p_fcf = (mcap / fcf) if (mcap and fcf and fcf != 0) else np.nan

        eps = info.get("trailingEps")
        eps_growth = info.get("earningsQuarterlyGrowth")

        equity = info.get("totalStockholderEquity")
        net_income = info.get("netIncomeToCommon")
        roe = (net_income / equity) if (net_income and equity and equity != 0) else np.nan

        total_assets = info.get("totalAssets")
        roa = (net_income / total_assets) if (net_income and total_assets and total_assets != 0) else np.nan

        total_debt = info.get("totalDebt")
        debt_equity = (total_debt / equity) if (total_debt and equity and equity != 0) else np.nan

        revenue = info.get("totalRevenue")
        profit_margin = (net_income / revenue) if (net_income and revenue and revenue != 0) else np.nan

        return {
            "ticker": ticker,
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "p_fcf": p_fcf,
            "div_yield": info.get("dividendYield"),
            "eps_growth": eps_growth,
            "sector": info.get("sector"),
            "roe": roe,
            "roa": roa,
            "debt_equity": debt_equity,
            "profit_margin": profit_margin,
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "eps": eps,
            "beta": info.get("beta"),
            "market_cap": mcap,
        }

    @staticmethod
    def _zscore_df(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.mean()) / (df.std(ddof=0) + 1e-12)

    @staticmethod
    def compute_value_score(fund: pd.DataFrame) -> pd.Series:
        cols = [c for c in ["pe", "pb", "ps", "ev_ebitda", "p_fcf"] if c in fund.columns]
        if not cols:
            return pd.Series(index=fund.index, dtype=float)

        inv = fund[cols].apply(pd.to_numeric, errors="coerce")
        for c in cols:
            inv[c] = 1.0 / inv[c].where(inv[c] > 0)

        inv = inv.replace([np.inf, -np.inf], np.nan)
        inv = inv.clip(lower=inv.quantile(0.01), upper=inv.quantile(0.99), axis=1)

        for c in inv.columns:
            med = inv[c].median()
            inv[c] = inv[c].fillna(med if pd.notna(med) else 0.0)

        return FundamentalFeatureEngineering._zscore_df(inv).mean(axis=1)

    @staticmethod
    def compute_growth_score(fund: pd.DataFrame) -> pd.Series:
        cols = [c for c in ["eps_growth", "roe", "roa"] if c in fund.columns]
        if not cols:
            return pd.Series(index=fund.index, dtype=float)

        g = fund[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        g = g.clip(lower=g.quantile(0.01), upper=g.quantile(0.99), axis=1)

        for c in g.columns:
            med = g[c].median()
            g[c] = g[c].fillna(med if pd.notna(med) else 0.0)

        return FundamentalFeatureEngineering._zscore_df(g).mean(axis=1)

    @staticmethod
    def compute_quality_score(fund: pd.DataFrame) -> pd.Series:
        cols = [
            c for c in [
                "roe", "roa", "profit_margin",
                "current_ratio", "quick_ratio",
                "gross_margin", "operating_margin",
            ] if c in fund.columns
        ]
        if not cols:
            return pd.Series(index=fund.index, dtype=float)

        q = fund[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        q = q.clip(lower=q.quantile(0.01), upper=q.quantile(0.99), axis=1)

        for c in q.columns:
            med = q[c].median()
            q[c] = q[c].fillna(med if pd.notna(med) else 0.0)

        return FundamentalFeatureEngineering._zscore_df(q).mean(axis=1)
