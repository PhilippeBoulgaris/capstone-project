# -*- coding: utf-8 -*-

import os
import json
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy import stats

warnings.filterwarnings("ignore")


# =============================
# CONFIG
# =============================

@dataclass
class Config:
    START: str = "2010-01-01"
    TRAIN_TEST_SPLIT: str = "2018-01-01"
    INTERVAL: str = "1mo"
    COSTS_BPS: int = 20
    RF_RATE_ANNUAL: float = 0.02
    RANDOM_STATE: int = 42
    N_CV_SPLITS: int = 4
    N_ITER_SEARCH: int = 10
    TOP_PERCENTILE: float = 0.2
    N_RANDOM_SIMULATIONS: int = 100
    BENCHMARK: str = "^GSPC"

    SEC_USER_AGENT: str = "Philippe Boulgaris boulgaris02@gmail.com - Capstone project"
    SEC_CACHE_DIR: str = "./sec_cache"
    SEC_MIN_INTERVAL_SEC: float = 0.2
    SEC_TIMEOUT_SEC: int = 30
    SEC_MAX_RETRIES: int = 8
    SEC_LAG_DAYS: int = 90

    # Visuals extras
    ENABLE_ROC_AND_CM: bool = False  # True -> génère ROC curves + matrices confusion

    SUB_PERIODS: Optional[List[Tuple[str, str, str]]] = None

    def __post_init__(self):
        if self.SUB_PERIODS is None:
            self.SUB_PERIODS = [
                ("2010-01-01", "2015-12-31", "Period 1: 2010-2015"),
                ("2016-01-01", "2020-12-31", "Period 2: 2016-2020"),
                ("2021-01-01", "2024-12-31", "Period 3: 2021-2024"),
            ]


# =============================
# UNIVERSE
# =============================

SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "ADBE", "CRM", "CSCO", "ACN", "AMD", "IBM", "INTC", "TXN", "QCOM", "INTU",
    "NOW", "AMAT", "MU", "ADI", "LRCX", "PANW", "SNPS", "CDNS", "KLAC", "NXPI",
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "CVS",
    "AMGN", "BMY", "MDT", "GILD", "CI", "ISRG", "REGN", "VRTX", "BSX", "SYK",
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK",
    "SPGI", "C", "CB", "PGR", "MMC", "SCHW", "CME", "ICE", "MCO", "USB",
    "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG", "MAR", "ABNB",
    "GM", "F", "ORLY", "AZO", "YUM", "DHI", "LEN", "ROST", "HLT", "DPZ",
    "WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "CL", "MO", "KMB",
    "GIS", "KHC", "HSY", "K", "STZ", "SYY", "TAP", "CPB", "CAG", "HRL",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES",
    "CAT", "BA", "GE", "RTX", "HON", "UNP", "LMT", "UPS", "DE", "ADP",
    "MMM", "GD", "NOC", "ETN", "ITW", "EMR", "CSX", "NSC", "WM", "FDX",
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "DOW", "DD", "NUE", "VMC",
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "WELL", "DLR", "SPG", "AVB",
    "NEE", "SO", "DUK", "AEP", "SRE", "D", "EXC", "XEL", "ED", "PEG",
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "MTCH",
]


# =============================
# DATA LOADER
# =============================

class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def download_prices(self, tickers: List[str]) -> pd.DataFrame:
        print(f"\n[1/9] Downloading prices for {len(tickers)} tickers...")
        data = yf.download(
            tickers=tickers,
            start=self.config.START,
            interval=self.config.INTERVAL,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if isinstance(data.columns, pd.MultiIndex):
            px = data["Close"].dropna(how="all")
        else:
            px = data.dropna(how="all")
        print(f"✓ Prices: {px.shape[0]} months × {px.shape[1]} tickers")
        return px

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        rets = np.log(prices / prices.shift(1)).dropna(how="all")
        print(f"✓ Log returns computed over {len(rets)} months")
        return rets


# =============================
# YAHOO FUNDAMENTALS + SCORES
# =============================

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
        cols = [c for c in ["roe", "roa", "profit_margin", "current_ratio", "quick_ratio", "gross_margin", "operating_margin"]
                if c in fund.columns]
        if not cols:
            return pd.Series(index=fund.index, dtype=float)
        q = fund[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        q = q.clip(lower=q.quantile(0.01), upper=q.quantile(0.99), axis=1)
        for c in q.columns:
            med = q[c].median()
            q[c] = q[c].fillna(med if pd.notna(med) else 0.0)
        return FundamentalFeatureEngineering._zscore_df(q).mean(axis=1)


# =============================
# DATASET BUILDERS
# =============================

class MLDatasetBuilder:
    """Yahoo: fundamentals statiques (snapshot) -> panel monthly par recopie"""
    def __init__(self, config: Config):
        self.config = config

    def build_yahoo_dataset(self, asset_rets: pd.DataFrame, bench_rets: pd.Series, fund_data: pd.DataFrame) -> pd.DataFrame:
        print("\n[3/9] Building ML dataset (Yahoo fundamentals only)...")

        fund_static = fund_data.copy()
        if "ticker" not in fund_static.columns:
            fund_static = fund_static.reset_index().rename(columns={"index": "ticker"})

        idx = asset_rets.index
        next_map = {idx[i]: idx[i + 1] for i in range(len(idx) - 1)}

        rows = []
        num_cols = [c for c in fund_static.columns if c not in ["ticker", "sector"]]

        for d in asset_rets.index:
            if d not in next_map:
                continue
            nd = next_map[d]

            for t in asset_rets.columns:
                fr = fund_static[fund_static["ticker"] == t]
                if fr.empty:
                    continue
                fr = fr.iloc[0]

                stock_ret = float(asset_rets.loc[nd, t])
                bench_ret = float(bench_rets.loc[nd]) if nd in bench_rets.index else 0.0

                row = {"date": d, "ticker": t, "target": 1 if stock_ret > bench_ret else 0, "next_month_return": stock_ret}
                for c in num_cols:
                    row[c] = pd.to_numeric(fr.get(c, np.nan), errors="coerce")
                rows.append(row)

        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        feat_cols = [c for c in df.columns if c not in ["date", "ticker", "target", "next_month_return"]]

        # Filtre soft + impute médiane (mais la "garantie no-NaN" est dans le Pipeline ML)
        if feat_cols:
            keep_min = max(1, int(np.ceil(0.5 * len(feat_cols))))
            df = df[df[feat_cols].notna().sum(axis=1) >= keep_min].copy()
            med = df[feat_cols].median(numeric_only=True)
            df[feat_cols] = df[feat_cols].fillna(med)

        print(f"✓ Yahoo ML dataset: {len(df):,} samples over {df['date'].nunique():,} months | features={len(feat_cols)}")
        return df


# =============================
# ML TRAINER (✅ Pipeline Imputer)
# =============================

class MLTrainer:
    def __init__(self, config: Config):
        self.config = config

        # Pipelines: imputer toujours, scaler seulement si utile
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

    def fit_model(self, model_name: str, X_train, y_train):
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
    def safe_predict_proba(model, X):
        # model est un Pipeline, donc predict_proba existe si le dernier step le supporte
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        # fallback (rare)
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    def train_universe(self, universe_name: str, X_train, y_train, X_test, y_test):
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


# =============================
# PERFORMANCE ANALYZER
# =============================

class PerformanceAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def backtest_portfolio(self, weights_dict: Dict, returns: pd.DataFrame, costs_bps: Optional[int] = None):
        if costs_bps is None:
            costs_bps = self.config.COSTS_BPS

        idx = returns.index
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
            return self._empty_metrics(1.0 if equity_curve is None or len(equity_curve) == 0 else float(equity_curve.iloc[-1]))

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


# =============================
# VISUALS
# =============================

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_equity_curves(self, curves_dict: Dict[str, pd.Series], filename: str, title: str, figsize=(11, 6)):
        plt.figure(figsize=figsize)
        for name, eq in curves_dict.items():
            if eq is not None and len(eq) >= 2:
                (eq / eq.iloc[0]).plot(label=name, linewidth=2.0)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.ylabel("Cumulative Value ($1)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()

    def plot_drawdown_curves(self, curves_dict: Dict[str, pd.Series], filename: str, title: str, figsize=(11, 5.5)):
        plt.figure(figsize=figsize)
        any_plotted = False
        for name, eq in curves_dict.items():
            if eq is None or len(eq) < 2:
                continue
            dd = compute_drawdown(eq)
            if len(dd) < 2:
                continue
            dd.plot(label=name, linewidth=2.0)
            any_plotted = True

        if not any_plotted:
            return

        plt.title(title, fontsize=13, fontweight="bold")
        plt.ylabel("Drawdown")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()


class MLVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # Déjà existant : comparaison Yahoo vs SEC sur Accuracy et AUC (2 subplots)
    def plot_ml_comparison(self, yahoo_results: Dict, sec_results: Dict, filename: str):
        model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Network"]
        yahoo_acc = [yahoo_results.get(f"{m} (Yahoo-trained)", {}).get("accuracy", np.nan) for m in model_names]
        sec_acc = [sec_results.get(f"{m} (SEC-trained)", {}).get("accuracy", np.nan) for m in model_names]
        yahoo_auc = [yahoo_results.get(f"{m} (Yahoo-trained)", {}).get("auc", np.nan) for m in model_names]
        sec_auc = [sec_results.get(f"{m} (SEC-trained)", {}).get("auc", np.nan) for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax1 = axes[0]
        ax1.bar(x - width/2, yahoo_acc, width, label="Yahoo")
        ax1.bar(x + width/2, sec_acc, width, label="SEC")
        ax1.set_title("Accuracy: Yahoo vs SEC")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["LogReg", "RF", "GB", "NN"])
        ax1.axhline(0.5, linestyle="--")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.bar(x - width/2, yahoo_auc, width, label="Yahoo")
        ax2.bar(x + width/2, sec_auc, width, label="SEC")
        ax2.set_title("AUC: Yahoo vs SEC")
        ax2.set_xticks(x)
        ax2.set_xticklabels(["LogReg", "RF", "GB", "NN"])
        ax2.axhline(0.5, linestyle="--")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ ML comparison plot saved: {filename}")

    # NOUVEAU : comparaison des modèles entre eux dans un univers (Yahoo OU SEC)
    def plot_ml_metrics_single(self, results: Dict, universe_label: str, filename: str):
        model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Network"]

        acc = []
        auc = []
        short = []

        for m in model_names:
            key = f"{m} ({universe_label}-trained)"
            acc.append(results.get(key, {}).get("accuracy", np.nan))
            auc.append(results.get(key, {}).get("auc", np.nan))
            short.append(short_model_name(key))

        x = np.arange(len(short))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax1 = axes[0]
        ax1.bar(x, acc)
        ax1.set_title(f"Accuracy — {universe_label} (models compared)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(short)
        ax1.axhline(0.5, linestyle="--")
        ax1.grid(alpha=0.3)

        ax2 = axes[1]
        ax2.bar(x, auc)
        ax2.set_title(f"AUC — {universe_label} (models compared)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short)
        ax2.axhline(0.5, linestyle="--")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ ML metrics ({universe_label}) saved: {filename}")

    # OPTIONNEL : ROC curves pour un univers
    def plot_roc_curves(self, y_true: np.ndarray, results: Dict, universe_label: str, filename: str):
        plt.figure(figsize=(7, 6))
        model_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Network"]

        plotted = False
        for m in model_names:
            key = f"{m} ({universe_label}-trained)"
            if key not in results:
                continue
            proba = results[key].get("probabilities", None)
            if proba is None or len(proba) != len(y_true):
                continue

            fpr, tpr, _ = roc_curve(y_true, proba)
            auc_val = results[key].get("auc", np.nan)
            plt.plot(fpr, tpr, label=f"{short_model_name(key)} (AUC={auc_val:.3f})")
            plotted = True

        if not plotted:
            plt.close()
            return

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"ROC Curves — {universe_label}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ ROC plot ({universe_label}) saved: {filename}")

    # OPTIONNEL : matrice de confusion
    def plot_confusion(self, y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        disp.plot(ax=ax, values_format="d")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ Confusion matrix saved: {filename}")


# =============================
# HELPERS
# =============================

def short_model_name(k: str) -> str:
    """
    Convertit "Random Forest (Yahoo-trained)" -> "RF"
    et "Logistic Regression (...)" -> "LogReg"
    """
    s = str(k)
    s = s.replace("Logistic Regression", "LogReg")
    s = s.replace("Random Forest", "RF")
    s = s.replace("Gradient Boosting", "GB")
    s = s.replace("Neural Network", "NN")
    # retire suffixe "(Yahoo-trained)" etc.
    if "(" in s:
        s = s.split("(")[0].strip()
    return s.strip()


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


def pick_best_model_key(results: Dict, universe_label: str, metric: str = "auc") -> Optional[str]:
    """
    Retourne la clé du modèle avec la meilleure métrique (auc ou accuracy) dans results.
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


# =============================
# SEC CLIENT + BUILDERS
# =============================

class SECClient:
    def __init__(self, config: Config):
        self.cfg = config
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
        self.session.mount("https://", adapter)

        self.cache_dir = Path(self.cfg.SEC_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "companyfacts").mkdir(parents=True, exist_ok=True)
        self._last_t = 0.0

    def _sleep_rate_limit(self):
        dt = time.time() - self._last_t
        if dt < self.cfg.SEC_MIN_INTERVAL_SEC:
            time.sleep(self.cfg.SEC_MIN_INTERVAL_SEC - dt)
        self._last_t = time.time()

    @staticmethod
    def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8"):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding=encoding)
        tmp.replace(path)

    def _get(self, url: str) -> Optional[requests.Response]:
        if "@" not in self.cfg.SEC_USER_AGENT:
            raise RuntimeError("Config.SEC_USER_AGENT invalide: mets un email.")

        headers = {
            "User-Agent": self.cfg.SEC_USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        last_err = None
        for attempt in range(self.cfg.SEC_MAX_RETRIES):
            self._sleep_rate_limit()
            try:
                r = self.session.get(url, headers=headers, timeout=self.cfg.SEC_TIMEOUT_SEC)
                if r.status_code == 200:
                    return r

                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        time.sleep(max(float(retry_after), 0.0))
                        continue
                    except Exception:
                        pass

                if r.status_code in (403, 429) or (500 <= r.status_code <= 599):
                    backoff = (0.5 * (2 ** attempt)) + random.uniform(0, 0.5)
                    time.sleep(backoff)
                    continue

                backoff = (0.3 * (2 ** attempt)) + random.uniform(0, 0.3)
                time.sleep(backoff)
            except Exception as e:
                last_err = e
                backoff = (0.5 * (2 ** attempt)) + random.uniform(0, 0.5)
                time.sleep(backoff)

        print(f"[SEC] FAIL url={url} err={last_err}")
        return None

    def get_json(self, url: str) -> Optional[dict]:
        r = self._get(url)
        if r is None:
            return None
        try:
            return r.json()
        except Exception:
            return None

    def fetch_ticker_cik_map(self, force_refresh: bool = False) -> Dict[str, int]:
        cache_path = self.cache_dir / "company_tickers.json"
        data = None

        if cache_path.exists() and not force_refresh:
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                data = None

        if data is None:
            url = "https://www.sec.gov/files/company_tickers.json"
            data = self.get_json(url)
            if data is None:
                raise RuntimeError("Impossible de télécharger company_tickers.json")
            try:
                self._atomic_write_text(cache_path, json.dumps(data), encoding="utf-8")
            except Exception:
                pass

        out = {}
        for _, rec in data.items():
            t = str(rec.get("ticker", "")).upper().strip()
            cik = rec.get("cik_str", None)
            if t and cik is not None:
                try:
                    out[t.replace(" ", "").replace("/", "")] = int(cik)
                except Exception:
                    continue
        return out

    @staticmethod
    def cik10(cik: int) -> str:
        return str(int(cik)).zfill(10)

    def fetch_companyfacts(self, cik: int, use_cache: bool = True, ttl_days: Optional[int] = None) -> Optional[dict]:
        path = self.cache_dir / "companyfacts" / f"companyfacts_{self.cik10(cik)}.json"

        if use_cache and path.exists():
            try:
                if ttl_days is not None:
                    age_sec = time.time() - path.stat().st_mtime
                    if age_sec > ttl_days * 86400:
                        raise ValueError("Cache expired")
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{self.cik10(cik)}.json"
        data = self.get_json(url)
        if data is None:
            return None

        try:
            self._atomic_write_text(path, json.dumps(data), encoding="utf-8")
        except Exception:
            pass
        return data


def _pick_units(facts: dict, tag: str) -> dict:
    try:
        us = facts.get("facts", {}).get("us-gaap", {})
        if tag in us:
            return us[tag].get("units", {})
    except Exception:
        pass
    try:
        ifrs = facts.get("facts", {}).get("ifrs-full", {})
        if tag in ifrs:
            return ifrs[tag].get("units", {})
    except Exception:
        pass
    return {}


def _choose_unit(units: dict, preferred=("USD", "shares")) -> list:
    if not isinstance(units, dict) or len(units) == 0:
        return []
    for u in preferred:
        if u in units:
            return units[u]
    for k in sorted(units.keys()):
        return units[k]
    return []


def _to_annual_fy(obs: list, allow_forms=("10-K", "20-F", "40-F")) -> pd.DataFrame:
    rows = []
    if not isinstance(obs, list):
        return pd.DataFrame(columns=["fyear", "fye_end", "val"])

    for o in obs:
        fy = o.get("fy")
        fp = o.get("fp")
        form = (o.get("form") or "").strip()
        end = o.get("end")
        val = o.get("val")

        if fy is None or fp != "FY":
            continue
        if form and (form not in allow_forms):
            continue
        if val is None or end is None:
            continue

        try:
            year = int(fy)
        except Exception:
            continue

        rows.append({"fyear": year, "fye_end": pd.to_datetime(end, errors="coerce"), "val": val})

    if not rows:
        return pd.DataFrame(columns=["fyear", "fye_end", "val"])

    df = pd.DataFrame(rows).dropna(subset=["fye_end"])
    if df.empty:
        return pd.DataFrame(columns=["fyear", "fye_end", "val"])

    df = df.sort_values(["fyear", "fye_end"])
    return df.drop_duplicates(subset=["fyear"], keep="last")


class SECFundamentalsBuilder:
    def __init__(self, config: Config):
        self.cfg = config
        self.sec = SECClient(config)
        self.alias = {"BRK-B": ["BRK.B", "BRK-B", "BRKB"], "BF-B": ["BF.B", "BF-B", "BFB"]}

    def _ticker_variants(self, t: str) -> List[str]:
        t = str(t).upper().strip()
        out = [t]
        if "-" in t:
            out.append(t.replace("-", "."))
        if "." in t:
            out.append(t.replace(".", "-"))
        out += self.alias.get(t, [])

        out2 = []
        for s in out:
            s2 = s.replace(" ", "").replace("/", "").strip().upper()
            if s2:
                out2.append(s2)

        uniq, seen = [], set()
        for s in (out + out2):
            s = str(s).upper().strip()
            if s and s not in seen:
                uniq.append(s)
                seen.add(s)
        return uniq

    def build_annual_sec_table(self, tickers: List[str], start_year=2010, end_year=2024) -> pd.DataFrame:
        print("\n[2bis/9] Downloading SEC fundamentals (annual FY)...")
        t2c = self.sec.fetch_ticker_cik_map(force_refresh=False)

        rows_all = []
        n_ok, n_skip = 0, 0

        for t in tickers:
            cik, used = None, None
            for v in self._ticker_variants(t):
                vv = v.replace(" ", "").replace("/", "").upper()
                if vv in t2c:
                    cik, used = t2c[vv], vv
                    break
            if cik is None:
                n_skip += 1
                continue

            facts = self.sec.fetch_companyfacts(cik, use_cache=True, ttl_days=None)
            if facts is None:
                n_skip += 1
                continue

            rev_units = (_pick_units(facts, "Revenues")
                         or _pick_units(facts, "SalesRevenueNet")
                         or _pick_units(facts, "RevenueFromContractWithCustomerExcludingAssessedTax"))
            rev = _to_annual_fy(_choose_unit(rev_units, ("USD",))).rename(columns={"val": "revenue"})

            ni = _to_annual_fy(_choose_unit(_pick_units(facts, "NetIncomeLoss"), ("USD",))).rename(columns={"val": "net_income"})

            eq_units = (_pick_units(facts, "StockholdersEquity")
                        or _pick_units(facts, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"))
            eq = _to_annual_fy(_choose_unit(eq_units, ("USD",))).rename(columns={"val": "equity"})

            ltd = _to_annual_fy(_choose_unit(_pick_units(facts, "LongTermDebt"), ("USD",))).rename(columns={"val": "ltd"})
            std_units = _pick_units(facts, "DebtCurrent") or _pick_units(facts, "LongTermDebtCurrent")
            std = _to_annual_fy(_choose_unit(std_units, ("USD",))).rename(columns={"val": "std"})

            cash = _to_annual_fy(_choose_unit(_pick_units(facts, "CashAndCashEquivalentsAtCarryingValue"), ("USD",))).rename(columns={"val": "cash"})
            sh = _to_annual_fy(_choose_unit(_pick_units(facts, "CommonStockSharesOutstanding"), ("shares",))).rename(columns={"val": "shares"})

            ebitda = _to_annual_fy(_choose_unit(_pick_units(facts, "EarningsBeforeInterestTaxesDepreciationAmortization"), ("USD",))).rename(columns={"val": "ebitda"})
            if ebitda.empty:
                op = _to_annual_fy(_choose_unit(_pick_units(facts, "OperatingIncomeLoss"), ("USD",))).rename(columns={"val": "op_income"})
                da = _to_annual_fy(_choose_unit(_pick_units(facts, "DepreciationDepletionAndAmortization"), ("USD",))).rename(columns={"val": "da"})
                if (not op.empty) and (not da.empty):
                    tmp = op.merge(da[["fyear", "da"]], on="fyear", how="inner")
                    tmp["ebitda"] = pd.to_numeric(tmp["op_income"], errors="coerce") + pd.to_numeric(tmp["da"], errors="coerce")
                    ebitda = tmp[["fyear", "fye_end", "ebitda"]]

            base = rev[["fyear", "fye_end", "revenue"]].copy() if not rev.empty else pd.DataFrame(columns=["fyear", "fye_end", "revenue"])
            for add in [
                ni[["fyear", "net_income"]] if not ni.empty else pd.DataFrame(columns=["fyear", "net_income"]),
                eq[["fyear", "equity"]] if not eq.empty else pd.DataFrame(columns=["fyear", "equity"]),
                ltd[["fyear", "ltd"]] if not ltd.empty else pd.DataFrame(columns=["fyear", "ltd"]),
                std[["fyear", "std"]] if not std.empty else pd.DataFrame(columns=["fyear", "std"]),
                cash[["fyear", "cash"]] if not cash.empty else pd.DataFrame(columns=["fyear", "cash"]),
                sh[["fyear", "shares"]] if not sh.empty else pd.DataFrame(columns=["fyear", "shares"]),
            ]:
                base = base.merge(add, on="fyear", how="outer")

            if "ebitda" in ebitda.columns and not ebitda.empty:
                base = base.merge(ebitda[["fyear", "ebitda"]], on="fyear", how="outer")

            base["ticker"] = t
            base["CIK"] = self.sec.cik10(cik)
            base["SymbolUsedForCIK"] = used
            base = base[(base["fyear"] >= start_year) & (base["fyear"] <= end_year)].copy()
            if base.empty:
                n_skip += 1
                continue

            for c in ["revenue", "net_income", "equity", "ltd", "std", "cash", "shares", "ebitda"]:
                if c in base.columns:
                    base[c] = pd.to_numeric(base[c], errors="coerce")

            base["debt"] = base[["ltd", "std"]].fillna(0).sum(axis=1)
            rows_all.append(base)
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  SEC progress: ok={n_ok} | skipped={n_skip} | last={t}")

        if not rows_all:
            return pd.DataFrame()

        out = pd.concat(rows_all, ignore_index=True)
        out = out.sort_values(["ticker", "fyear", "fye_end"]).drop_duplicates(["ticker", "fyear"], keep="last")
        return out

    def annual_to_monthly_panel(self, annual: pd.DataFrame, month_index: pd.DatetimeIndex) -> pd.DataFrame:
        if annual is None or annual.empty:
            return pd.DataFrame()

        a = annual.copy()
        a["fye_end"] = pd.to_datetime(a["fye_end"], errors="coerce")
        a = a.dropna(subset=["fye_end"])
        if a.empty:
            return pd.DataFrame()

        a["available_from"] = a["fye_end"] + pd.Timedelta(days=int(self.cfg.SEC_LAG_DAYS))
        a = a.sort_values(["ticker", "available_from"])

        panel_rows = []
        for t, sub in a.groupby("ticker", sort=False):
            sub = sub.dropna(subset=["available_from"]).sort_values("available_from")
            if sub.empty:
                continue
            for d in month_index:
                eligible = sub[sub["available_from"] <= d]
                if eligible.empty:
                    continue
                last = eligible.iloc[-1].to_dict()
                last["date"] = d
                panel_rows.append(last)

        if not panel_rows:
            return pd.DataFrame()

        panel = pd.DataFrame(panel_rows)
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel = panel.sort_values(["ticker", "date", "available_from"]).drop_duplicates(["ticker", "date"], keep="last")
        return panel


class MLDatasetBuilderPanel:
    """SEC: panel mensuel as-of -> dataset supervised (no look-ahead)"""
    def __init__(self, config: Config):
        self.config = config

    def build_dataset_from_panel(
        self,
        asset_rets: pd.DataFrame,
        bench_rets: pd.Series,
        feature_panel: pd.DataFrame,
        feature_cols: List[str],
        min_non_nan_features: Optional[int] = None,
        impute: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        print("\n[3bis/9] Building ML dataset (SEC fundamentals panel)...")

        if feature_panel is None or feature_panel.empty:
            print("⚠️ SEC feature_panel empty -> SEC ML dataset empty.")
            return pd.DataFrame()

        p = feature_panel.copy()
        p["date"] = pd.to_datetime(p["date"], errors="coerce")
        p = p.dropna(subset=["date", "ticker"])
        p = p[p["date"].isin(asset_rets.index)].copy()
        if p.empty:
            print("⚠️ SEC panel has no overlap dates with returns index.")
            return pd.DataFrame()

        idx = asset_rets.index
        next_map = {idx[i]: idx[i + 1] for i in range(len(idx) - 1)}

        rows = []
        for _, r in p.iterrows():
            d = r["date"]
            t = r["ticker"]
            if t not in asset_rets.columns:
                continue
            if d not in next_map:
                continue
            nd = next_map[d]
            if nd not in asset_rets.index:
                continue

            bench_ret = float(bench_rets.loc[nd]) if nd in bench_rets.index else 0.0
            stock_ret = float(asset_rets.loc[nd, t])

            row = {"date": d, "ticker": t}
            for c in feature_cols:
                row[c] = r.get(c, np.nan)
            row["target"] = 1 if stock_ret > bench_ret else 0
            row["next_month_return"] = stock_ret
            rows.append(row)

        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        if df.empty:
            print("⚠️ SEC ML dataset empty after join.")
            return df

        if verbose:
            na_rate = df[feature_cols].isna().mean().sort_values(ascending=False)
            print(f"  Raw SEC rows: {len(df):,} | months: {df['date'].nunique():,} | tickers: {df['ticker'].nunique():,}")
            print("  NaN rate (top):")
            print(na_rate.head(min(10, len(na_rate))).to_string())

        if min_non_nan_features is None:
            min_non_nan_features = max(1, int(np.ceil(0.5 * len(feature_cols))))

        before_soft = len(df)
        df = df[df[feature_cols].notna().sum(axis=1) >= min_non_nan_features].copy()
        if verbose:
            print(f"  After soft filter (>= {min_non_nan_features} non-NaN feats): {before_soft:,} → {len(df):,}")

        if df.empty:
            print("⚠️ SEC ML dataset empty after soft filter -> lower min_non_nan_features.")
            return df

        if impute:
            med = df[feature_cols].median(numeric_only=True)
            df[feature_cols] = df[feature_cols].fillna(med)

        df = df.dropna(subset=["date", "ticker", "target", "next_month_return"]).copy()
        print(f"✓ SEC panel ML dataset: {len(df):,} samples over {df['date'].nunique():,} months")
        return df


# =============================
# MAIN
# =============================

def main():
    t0 = time.perf_counter()
    config = Config()

    if "@" not in config.SEC_USER_AGENT:
        raise RuntimeError("Please set Config.SEC_USER_AGENT with your email (SEC requirement).")

    tickers = list(dict.fromkeys(SP500_TICKERS))
    print("=" * 80)
    print("🤖 BACKTESTING COMPLET: VALUE vs GROWTH + ML COMPARAISON YAHOO vs SEC")
    print("FONDAMENTAUX SEULEMENT - PAS DE MOMENTUM/TECHNIQUES")
    print("=" * 80)
    print(f"✓ Loaded {len(tickers)} stocks")
    print(f"Period: {config.START} → today (monthly)")
    print(f"Train/Test split: {config.TRAIN_TEST_SPLIT}")
    print("=" * 80)

    OUT_DIR = "./backtest_fundamentals_only_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Prices
    loader = DataLoader(config)
    px = loader.download_prices(tickers + [config.BENCHMARK])
    rets = loader.compute_returns(px)

    bench_rets = rets[config.BENCHMARK].copy() if config.BENCHMARK in rets.columns else pd.Series(dtype=float)
    asset_cols = [t for t in tickers if t in rets.columns]
    asset_rets = rets[asset_cols].copy()
    px_assets = px[asset_cols].reindex(asset_rets.index)

    # Yahoo fundamentals
    print("\n[2/9] Fetching Yahoo fundamentals...")
    fe = FundamentalFeatureEngineering()
    fund_list = []
    for i, t in enumerate(asset_rets.columns, 1):
        fund_list.append(fe.fetch_yahoo_fundamentals(t))
        if i % 20 == 0:
            print(f"    Processed {i}/{len(asset_rets.columns)} tickers...")
    fund_yahoo = pd.DataFrame(fund_list).set_index("ticker")

    for c in fund_yahoo.columns:
        if c != "sector":
            fund_yahoo[c] = pd.to_numeric(fund_yahoo[c], errors="coerce")

    print("  Computing Yahoo scores...")
    fund_yahoo["score_value"] = fe.compute_value_score(fund_yahoo)
    fund_yahoo["score_growth"] = fe.compute_growth_score(fund_yahoo)
    fund_yahoo["score_quality"] = fe.compute_quality_score(fund_yahoo)

    # universes
    valid_val = fund_yahoo["score_value"].dropna()
    valid_gro = fund_yahoo["score_growth"].dropna()
    top_val = max(1, int(len(valid_val) * config.TOP_PERCENTILE)) if len(valid_val) else 0
    top_gro = max(1, int(len(valid_gro) * config.TOP_PERCENTILE)) if len(valid_gro) else 0
    val_names = valid_val.nlargest(top_val).index.tolist() if top_val else []
    gro_names = valid_gro.nlargest(top_gro).index.tolist() if top_gro else []
    print(f"✓ Value universe size: {len(val_names)} | Growth: {len(gro_names)}")

    # SEC pipeline
    sec_panel = pd.DataFrame()
    sec_feature_cols = ["pb", "pe", "ps", "ev_ebitda", "roe", "debt_to_equity", "net_margin"]

    try:
        sec_builder = SECFundamentalsBuilder(config)
        sec_annual = sec_builder.build_annual_sec_table(asset_rets.columns.tolist(), start_year=2010, end_year=2024)

        if not sec_annual.empty:
            sec_panel = sec_builder.annual_to_monthly_panel(sec_annual, asset_rets.index)

            def _mcap(row):
                d = row.get("date")
                t = row.get("ticker")
                pr = float(px_assets.loc[d, t]) if (d in px_assets.index and t in px_assets.columns) else np.nan
                sh = row.get("shares", np.nan)
                return pr * sh

            sec_panel["market_cap"] = sec_panel.apply(_mcap, axis=1)
            sec_panel["ev"] = sec_panel["market_cap"] + sec_panel["debt"] - sec_panel["cash"]

            sec_panel["pb"] = sec_panel["market_cap"] / sec_panel["equity"].replace(0, np.nan)
            sec_panel["pe"] = sec_panel["market_cap"] / sec_panel["net_income"].replace(0, np.nan)
            sec_panel["ps"] = sec_panel["market_cap"] / sec_panel["revenue"].replace(0, np.nan)
            if "ebitda" in sec_panel.columns:
                sec_panel["ev_ebitda"] = sec_panel["ev"] / sec_panel["ebitda"].replace(0, np.nan)

            sec_panel["roe"] = sec_panel["net_income"] / sec_panel["equity"].replace(0, np.nan)
            sec_panel["debt_to_equity"] = sec_panel["debt"] / sec_panel["equity"].replace(0, np.nan)
            sec_panel["net_margin"] = sec_panel["net_income"] / sec_panel["revenue"].replace(0, np.nan)

            for c in sec_feature_cols:
                if c in sec_panel.columns:
                    sec_panel.loc[np.isinf(sec_panel[c]), c] = np.nan

            sec_feature_cols = [c for c in sec_feature_cols if c in sec_panel.columns]
            print(f"✓ SEC panel rows: {len(sec_panel):,} | SEC features: {sec_feature_cols}")
        else:
            print("⚠️ SEC annual table empty -> skipping SEC pipeline.")
    except Exception as e:
        print(f"⚠️ SEC pipeline error -> skipping SEC. Error: {e}")

    # ---------------------------------------------------------
    # NEW: SEC-based Value vs Growth portfolios (monthly, as-of)
    # ---------------------------------------------------------
    def _zscore_series(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        mu = s.mean()
        sd = s.std(ddof=0)
        return (s - mu) / (sd + 1e-12)

    def _compute_sec_value_score(cs: pd.DataFrame) -> pd.Series:
        # Value: inverse multiples -> higher score = cheaper
        cols = [c for c in ["pe", "pb", "ps", "ev_ebitda"] if c in cs.columns]
        if not cols:
            return pd.Series(index=cs.index, dtype=float)

        inv = cs[cols].apply(pd.to_numeric, errors="coerce")
        for c in cols:
            inv[c] = 1.0 / inv[c].where(inv[c] > 0)

        inv = inv.replace([np.inf, -np.inf], np.nan)

        # cross-sectional winsorize per date
        for c in cols:
            lo = inv[c].quantile(0.01)
            hi = inv[c].quantile(0.99)
            inv[c] = inv[c].clip(lower=lo, upper=hi)

        # fill NaN with median (cross-section)
        for c in cols:
            med = inv[c].median()
            inv[c] = inv[c].fillna(med if pd.notna(med) else 0.0)

        z = inv.apply(_zscore_series)
        return z.mean(axis=1)

    def _compute_sec_growth_score(cs: pd.DataFrame) -> pd.Series:
        # Growth/Profitability proxy (SEC doesn't give clean EPS growth monthly here)
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

    def _build_sec_style_weights(sec_panel_in: pd.DataFrame, score_col: str) -> Dict[pd.Timestamp, Dict[str, float]]:
        """
        Builds monthly weights dict: at each date d, pick top TOP_PERCENTILE by score
        and hold for next month (backtest_portfolio() already uses next_map).
        """
        if sec_panel_in is None or sec_panel_in.empty:
            return {}

        p = sec_panel_in.copy()
        p["date"] = pd.to_datetime(p["date"], errors="coerce")
        p = p.dropna(subset=["date", "ticker"])
        p = p[p["ticker"].isin(asset_rets.columns)].copy()
        if p.empty or score_col not in p.columns:
            return {}

        weights = {}
        for d, cs in p.groupby("date"):
            cs = cs.dropna(subset=[score_col])
            if cs.empty:
                continue

            n = max(1, int(len(cs) * config.TOP_PERCENTILE))
            top = cs.nlargest(n, score_col)

            tick_list = top["ticker"].astype(str).tolist()
            tick_list = [t for t in tick_list if t in asset_rets.columns]
            if len(tick_list) == 0:
                continue

            weights[d] = {t: 1.0 / len(tick_list) for t in tick_list}

        return weights

    eq_val_sec, eq_gro_sec = None, None
    if sec_panel is not None and len(sec_panel) > 0:
        sec_style_cols_needed = ["pb", "pe", "ps", "ev_ebitda", "roe", "net_margin"]
        avail = [c for c in sec_style_cols_needed if c in sec_panel.columns]
        if len(avail) >= 2:
            print("\n[2ter/9] Building SEC Value/Growth scores (monthly cross-section)...")

            sec_scores = sec_panel[["date", "ticker"] + avail].copy()

            # compute scores per month (cross-section)
            val_scores = []
            gro_scores = []
            for d, cs in sec_scores.groupby("date"):
                cs2 = cs.reset_index(drop=True).copy()
                cs2["score_value_sec"] = _compute_sec_value_score(cs2)
                cs2["score_growth_sec"] = _compute_sec_growth_score(cs2)
                val_scores.append(cs2[["date", "ticker", "score_value_sec"]])
                gro_scores.append(cs2[["date", "ticker", "score_growth_sec"]])

            val_scores = pd.concat(val_scores, ignore_index=True) if val_scores else pd.DataFrame()
            gro_scores = pd.concat(gro_scores, ignore_index=True) if gro_scores else pd.DataFrame()

            sec_scored = sec_scores.merge(val_scores, on=["date", "ticker"], how="left") \
                                   .merge(gro_scores, on=["date", "ticker"], how="left")

            # build monthly weights (top percentile)
            w_val_sec = _build_sec_style_weights(sec_scored, "score_value_sec")
            w_gro_sec = _build_sec_style_weights(sec_scored, "score_growth_sec")

            # backtest
            perf = PerformanceAnalyzer(config)  # <-- INITIALISATION AJOUTÉE ICI
            logret_val_sec, _ = perf.backtest_portfolio(w_val_sec, asset_rets, costs_bps=config.COSTS_BPS)
            logret_gro_sec, _ = perf.backtest_portfolio(w_gro_sec, asset_rets, costs_bps=config.COSTS_BPS)

            eq_val_sec = to_equity(logret_val_sec)
            eq_gro_sec = to_equity(logret_gro_sec)

            print(f"✓ SEC Value months: {eq_val_sec.index.nunique() if eq_val_sec is not None else 0}")
            print(f"✓ SEC Growth months: {eq_gro_sec.index.nunique() if eq_gro_sec is not None else 0}")
        else:
            print("⚠️ SEC Value/Growth: not enough SEC columns available for scoring.")
    else:
        print("⚠️ SEC Value/Growth: SEC panel empty -> skipping.")

    # Build ML datasets
    print("\n[3/9] Building ML datasets...")
    builder_y = MLDatasetBuilder(config)
    df_ml_yahoo = builder_y.build_yahoo_dataset(asset_rets, bench_rets, fund_yahoo.reset_index())

    df_ml_sec = pd.DataFrame()
    if len(sec_panel) > 0 and len(sec_feature_cols) > 0:
        builder_s = MLDatasetBuilderPanel(config)
        df_ml_sec = builder_s.build_dataset_from_panel(
            asset_rets=asset_rets,
            bench_rets=bench_rets,
            feature_panel=sec_panel[["date", "ticker"] + sec_feature_cols].copy(),
            feature_cols=sec_feature_cols,
            min_non_nan_features=max(1, int(np.ceil(0.5 * len(sec_feature_cols)))),
            impute=True,
            verbose=True,
        )
    else:
        print("⚠️ SEC ML skipped (no panel/features).")

    # ML training (Yahoo)
    print("\n[4/9] ML training (Yahoo fundamentals)...")
    trainer = MLTrainer(config)

    feature_cols_y = [
        c for c in df_ml_yahoo.columns
        if c not in ["date", "ticker", "target", "next_month_return", "sector"]
        and pd.api.types.is_numeric_dtype(df_ml_yahoo[c])
    ]

    train_mask_y = df_ml_yahoo["date"] < config.TRAIN_TEST_SPLIT
    X_train_y = df_ml_yahoo.loc[train_mask_y, feature_cols_y]
    y_train_y = df_ml_yahoo.loc[train_mask_y, "target"]
    X_test_y = df_ml_yahoo.loc[~train_mask_y, feature_cols_y]
    y_test_y = df_ml_yahoo.loc[~train_mask_y, "target"]
    meta_test_y = df_ml_yahoo.loc[~train_mask_y, ["date", "ticker", "next_month_return"]].copy()

    ml_results_y = trainer.train_universe("Yahoo", X_train_y, y_train_y, X_test_y, y_test_y)

    # ML training (SEC)
    ml_results_sec = {}
    meta_test_s = pd.DataFrame()
    y_test_s = None  # pour visus

    if len(df_ml_sec) > 0:
        print("\n[4bis/9] ML training (SEC fundamentals)...")
        trainer_s = MLTrainer(config)

        feature_cols_s = [
            c for c in df_ml_sec.columns
            if c not in ["date", "ticker", "target", "next_month_return"]
            and pd.api.types.is_numeric_dtype(df_ml_sec[c])
        ]

        train_mask_s = df_ml_sec["date"] < config.TRAIN_TEST_SPLIT
        X_train_s = df_ml_sec.loc[train_mask_s, feature_cols_s]
        y_train_s = df_ml_sec.loc[train_mask_s, "target"]
        X_test_s = df_ml_sec.loc[~train_mask_s, feature_cols_s]
        y_test_s = df_ml_sec.loc[~train_mask_s, "target"]
        meta_test_s = df_ml_sec.loc[~train_mask_s, ["date", "ticker", "next_month_return"]].copy()

        ml_results_sec = trainer_s.train_universe("SEC", X_train_s, y_train_s, X_test_s, y_test_s)

    # Backtesting
    print("\n[5/9] Backtesting Value/Growth + ML portfolios...")
    perf = PerformanceAnalyzer(config)  # <-- INITIALISATION (ou réutilisation)

    # Value EW
    if len(val_names) > 0:
        w_val = {d: {t: 1 / len(val_names) for t in val_names} for d in asset_rets.index[:-1]}
        logret_val, turnover_val = perf.backtest_portfolio(w_val, asset_rets, costs_bps=config.COSTS_BPS)
        eq_val = to_equity(logret_val)
    else:
        logret_val, turnover_val = pd.Series(dtype=float), pd.Series(dtype=float)
        eq_val = pd.Series([1.0], index=asset_rets.index[:1])

    # Growth EW
    if len(gro_names) > 0:
        w_gro = {d: {t: 1 / len(gro_names) for t in gro_names} for d in asset_rets.index[:-1]}
        logret_gro, turnover_gro = perf.backtest_portfolio(w_gro, asset_rets, costs_bps=config.COSTS_BPS)
        eq_gro = to_equity(logret_gro)
    else:
        logret_gro, turnover_gro = pd.Series(dtype=float), pd.Series(dtype=float)
        eq_gro = pd.Series([1.0], index=asset_rets.index[:1])

    # EW all stocks
    all_names = list(asset_rets.columns)
    w_sp = {d: {t: 1 / len(all_names) for t in all_names} for d in asset_rets.index[:-1]}
    logret_sp, turnover_sp = perf.backtest_portfolio(w_sp, asset_rets, costs_bps=0)
    eq_sp = to_equity(logret_sp)

    # Random avg
    np.random.seed(config.RANDOM_STATE)
    rnd_logrets, rnd_turns = [], []
    for _ in range(config.N_RANDOM_SIMULATIONS):
        weights_random = {}
        for d in asset_rets.index[:-1]:
            n_select = max(1, int(len(all_names) * config.TOP_PERCENTILE))
            selected = np.random.choice(all_names, size=n_select, replace=False)
            weights_random[d] = {t: 1 / len(selected) for t in selected}
        lr, tr = perf.backtest_portfolio(weights_random, asset_rets, costs_bps=config.COSTS_BPS)
        rnd_logrets.append(lr)
        rnd_turns.append(tr)

    logret_random = pd.concat(rnd_logrets, axis=1).mean(axis=1) if rnd_logrets else pd.Series(dtype=float)
    turnover_random = pd.concat(rnd_turns, axis=1).mean(axis=1) if rnd_turns else pd.Series(dtype=float)
    eq_random = to_equity(logret_random)

    # S&P Benchmark equity curve
    eq_bench = to_equity(bench_rets.dropna()) if bench_rets is not None and len(bench_rets) else pd.Series(dtype=float)

    # ML weights
    def build_weights_from_meta(meta: pd.DataFrame, prob: np.ndarray):
        mm = meta.copy()
        mm["prob"] = prob
        weights = {}
        for d in mm["date"].unique():
            dd = mm[mm["date"] == d]
            if len(dd) == 0:
                continue
            n = max(1, int(len(dd) * config.TOP_PERCENTILE))
            top = dd.nlargest(n, "prob")
            weights[d] = {t: 1 / len(top) for t in top["ticker"]}
        return weights

    ml_backtest_logrets = {}
    ml_backtest_turns = {}

    for name, res in ml_results_y.items():
        w = build_weights_from_meta(meta_test_y, res["probabilities"])
        lr, tr = perf.backtest_portfolio(w, asset_rets, costs_bps=config.COSTS_BPS)
        ml_backtest_logrets[f"{short_model_name(name)} [Yahoo]"] = lr
        ml_backtest_turns[f"{short_model_name(name)} [Yahoo]"] = tr

    for name, res in ml_results_sec.items():
        if len(meta_test_s) == 0:
            continue
        w = build_weights_from_meta(meta_test_s, res["probabilities"])
        lr, tr = perf.backtest_portfolio(w, asset_rets, costs_bps=config.COSTS_BPS)
        ml_backtest_logrets[f"{short_model_name(name)} [SEC]"] = lr
        ml_backtest_turns[f"{short_model_name(name)} [SEC]"] = tr

    eq_ml = {k: to_equity(v) for k, v in ml_backtest_logrets.items()}

    # Metrics
    print("\n[6/9] Computing metrics...")
    all_stats = {
        "VALUE (Traditional)": perf.compute_enhanced_metrics(eq_val, bench_rets),
        "GROWTH (Traditional)": perf.compute_enhanced_metrics(eq_gro, bench_rets),
        "EW (All Stocks)": perf.compute_enhanced_metrics(eq_sp, bench_rets),
        "RANDOM Portfolio (Avg)": perf.compute_enhanced_metrics(eq_random, bench_rets),
    }

    all_stats["VALUE (Traditional)"]["Avg Annual Turnover"] = float(turnover_val.mean() * 12) if len(turnover_val) else 0.0
    all_stats["GROWTH (Traditional)"]["Avg Annual Turnover"] = float(turnover_gro.mean() * 12) if len(turnover_gro) else 0.0
    all_stats["EW (All Stocks)"]["Avg Annual Turnover"] = float(turnover_sp.mean() * 12) if len(turnover_sp) else 0.0
    all_stats["RANDOM Portfolio (Avg)"]["Avg Annual Turnover"] = float(turnover_random.mean() * 12) if len(turnover_random) else 0.0

    for strat, eq in eq_ml.items():
        st = perf.compute_enhanced_metrics(eq, bench_rets)
        tr = ml_backtest_turns.get(strat, pd.Series(dtype=float))
        st["Avg Annual Turnover"] = float(tr.mean() * 12) if len(tr) else 0.0
        all_stats[strat] = st

    stats_df = pd.DataFrame(all_stats).T.sort_values("Sharpe", ascending=False)

    csv_path = os.path.join(OUT_DIR, f"performance_table_{ts}.csv")
    stats_df.to_csv(csv_path, index=True)
    print(f"✓ Performance table saved: {csv_path}")

    print("\n" + "=" * 80)
    print("⚡ PERFORMANCE TABLE (Fundamentals Only)")
    print("=" * 80)
    print(stats_df.to_string())
    print("=" * 80)

    # Visuals (existantes + nouvelles)
    print("\n[8/9] Generating visuals...")
    viz = Visualizer(OUT_DIR)
    ml_viz = MLVisualizer(OUT_DIR)

    # (1) Equity curves — toutes stratégies
    curves_all = {
        "VALUE (Traditional)": eq_val,
        "GROWTH (Traditional)": eq_gro,
        "EW (All Stocks)": eq_sp,
        "RANDOM Portfolio (Avg)": eq_random,
    }
    curves_all.update(eq_ml)
    viz.plot_equity_curves(curves_all, f"equity_ALL_{ts}.png", "Equity Curves — All Strategies (Fundamentals Only)")
    print("✓ Equity curves saved")

    # (2) Comparaison Yahoo vs SEC (déjà dans ton code)
    if ml_results_y and ml_results_sec:
        ml_viz.plot_ml_comparison(ml_results_y, ml_results_sec, f"ml_comparison_{ts}.png")

    # =============================
    # NOUVELLE VISUALISATION: Value vs Growth (Yahoo + SEC + S&P + Random)
    # =============================
    curves_vg = {
        "VALUE (Yahoo)": eq_val,
        "GROWTH (Yahoo)": eq_gro,
        "VALUE (SEC)": eq_val_sec,
        "GROWTH (SEC)": eq_gro_sec,
        "S&P 500 (Benchmark)": eq_bench,
        "RANDOM Portfolio (Avg)": eq_random,
    }

    # drop Nones safely (plot_equity_curves already checks length, but let's be clean)
    curves_vg = {k: v for k, v in curves_vg.items() if v is not None and len(v) >= 2}

    if len(curves_vg) >= 2:
        viz.plot_equity_curves(
            curves_vg,
            f"equity_VALUE_vs_GROWTH_YAHOO_SEC_SP_RANDOM_{ts}.png",
            "Value vs Growth — Yahoo vs SEC (Benchmark: S&P 500, + Random)"
        )
        print("✓ Value vs Growth (Yahoo+SEC+S&P+Random) equity plot saved")
    else:
        print("⚠️ Not enough curves to plot Value vs Growth (Yahoo+SEC).")

    # =============================
    # AUTRES VISUALISATIONS
    # =============================

    # (A) ML intra-Yahoo
    if ml_results_y:
        ml_viz.plot_ml_metrics_single(ml_results_y, "Yahoo", f"ml_metrics_YAHOO_{ts}.png")

    # (B) ML intra-SEC
    if ml_results_sec:
        ml_viz.plot_ml_metrics_single(ml_results_sec, "SEC", f"ml_metrics_SEC_{ts}.png")

    # (C) Best Yahoo vs Best SEC (equity OOS)
    best_y = pick_best_model_key(ml_results_y, "Yahoo", metric="auc") if ml_results_y else None
    best_s = pick_best_model_key(ml_results_sec, "SEC", metric="auc") if ml_results_sec else None

    curves_best = {}
    if best_y is not None:
        key_y = f"{short_model_name(best_y)} [Yahoo]"
        curves_best[f"Best Yahoo (AUC): {short_model_name(best_y)}"] = eq_ml.get(key_y, None)

    if best_s is not None:
        key_s = f"{short_model_name(best_s)} [SEC]"
        curves_best[f"Best SEC (AUC): {short_model_name(best_s)}"] = eq_ml.get(key_s, None)

    # Ajoute EW comme référence
    curves_best["EW (All Stocks)"] = eq_sp

    if sum([(v is not None and len(v) >= 2) for v in curves_best.values()]) >= 2:
        viz.plot_equity_curves(
            curves_best,
            f"equity_BEST_Y_vs_S_{ts}.png",
            "Best ML — Yahoo vs SEC (Out-of-sample)"
        )

        # (D) Best Yahoo vs Best SEC (drawdown OOS)
        viz.plot_drawdown_curves(
            curves_best,
            f"drawdown_BEST_Y_vs_S_{ts}.png",
            "Drawdowns — Best ML Yahoo vs Best ML SEC (Out-of-sample)"
        )

    # (Optionnel) ROC + confusion matrices
    if config.ENABLE_ROC_AND_CM:
        if ml_results_y:
            ml_viz.plot_roc_curves(y_test_y.values, ml_results_y, "Yahoo", f"roc_YAHOO_{ts}.png")
        if ml_results_sec and y_test_s is not None:
            ml_viz.plot_roc_curves(y_test_s.values, ml_results_sec, "SEC", f"roc_SEC_{ts}.png")

        if best_y is not None:
            ypred = ml_results_y[best_y]["predictions"]
            ml_viz.plot_confusion(
                y_true=y_test_y.values,
                y_pred=ypred,
                title=f"Confusion Matrix — Best Yahoo: {short_model_name(best_y)}",
                filename=f"cm_BEST_YAHOO_{ts}.png"
            )

        if best_s is not None and y_test_s is not None:
            ypred = ml_results_sec[best_s]["predictions"]
            ml_viz.plot_confusion(
                y_true=y_test_s.values,
                y_pred=ypred,
                title=f"Confusion Matrix — Best SEC: {short_model_name(best_s)}",
                filename=f"cm_BEST_SEC_{ts}.png"
            )

    print("\n✅ DONE ✅")
    print(f"📂 Outputs: {os.path.abspath(OUT_DIR)}")
    print(f"⏱ Total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()