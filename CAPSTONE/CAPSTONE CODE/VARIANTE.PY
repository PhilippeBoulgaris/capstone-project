# ==================== IMPORTS ====================
import os
import warnings
warnings.filterwarnings("ignore")
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

import joblib
from scipy import stats

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ seaborn not available - using simplified heatmaps")


# ==================== CONFIGURATION ====================

@dataclass
class Config:
    """Centralized project configuration"""
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
    SUB_PERIODS: List[Tuple[str, str, str]] = None

    def __post_init__(self):
        if self.SUB_PERIODS is None:
            self.SUB_PERIODS = [
                ("2010-01-01", "2015-12-31", "Period 1: 2010-2015"),
                ("2016-01-01", "2020-12-31", "Period 2: 2016-2020"),
                ("2021-01-01", "2024-12-31", "Period 3: 2021-2024"),
            ]


# ==================== TICKERS ====================
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


# ==================== CLASSES ====================

class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def download_prices(self, tickers: List[str]) -> pd.DataFrame:
        print(f"\n[1/8] Downloading data for {len(tickers)} tickers...")
        data = yf.download(
            tickers=tickers,
            start=self.config.START,
            interval=self.config.INTERVAL,
            auto_adjust=True,
            progress=False
        )
        px = data["Close"].dropna(how="all") if isinstance(data.columns, pd.MultiIndex) else data.dropna(how="all")
        print(f"✓ Prices: {px.shape[0]} months × {px.shape[1]} tickers")
        return px

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        rets = np.log(prices / prices.shift(1)).dropna(how="all")
        print(f"✓ Log returns computed over {len(rets)} periods")
        return rets


class FeatureEngineering:
    @staticmethod
    def fetch_fundamentals(ticker: str) -> Dict:
        """Yahoo Finance snapshot (⚠️ not historical, same as your code 2)"""
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}
        mcap = info.get("marketCap")
        fcf = info.get("freeCashflow")
        p_fcf = (mcap / fcf) if (mcap and fcf and fcf != 0) else None
        return {
            "ticker": ticker,
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "ps": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "p_fcf": p_fcf,
            "div_yield": info.get("dividendYield"),
            "eps_growth": info.get("earningsQuarterlyGrowth"),
            "sector": info.get("sector"),
        }

    @staticmethod
    def calculate_technical_features(prices: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        # latest snapshot, same as your code 2
        return {
            "momentum_3m": prices.pct_change(3).iloc[-1],
            "momentum_6m": prices.pct_change(6).iloc[-1],
            "momentum_12m": prices.pct_change(12).iloc[-1],
            "volatility_3m": returns.rolling(3).std().iloc[-1],
            "volatility_12m": returns.rolling(12).std().iloc[-1],
        }

    @staticmethod
    def compute_value_score(fund_data: pd.DataFrame) -> pd.Series:
        value_cols = [c for c in ["pe", "pb", "ps", "ev_ebitda", "p_fcf"] if c in fund_data.columns]
        inv = fund_data[value_cols].copy()
        for c in value_cols:
            inv[c] = 1.0 / fund_data[c].where(fund_data[c] > 0)

        inv = inv.replace([np.inf, -np.inf], np.nan).fillna(inv.median(numeric_only=True))
        inv = inv.clip(lower=inv.quantile(0.01), upper=inv.quantile(0.99), axis=1)
        Z_val = (inv - inv.mean()) / inv.std(ddof=0)
        return Z_val.mean(axis=1)

    @staticmethod
    def compute_growth_score(fund_data: pd.DataFrame) -> pd.Series:
        growth_cols = [c for c in ["pe", "pb", "ps", "eps_growth"] if c in fund_data.columns]
        g = fund_data[growth_cols].copy()
        g = g.replace([np.inf, -np.inf], np.nan).fillna(g.median(numeric_only=True))
        g = g.clip(lower=g.quantile(0.01), upper=g.quantile(0.99), axis=1)
        Z_gro = (g - g.mean()) / g.std(ddof=0)
        return Z_gro.mean(axis=1)


class MLDatasetBuilder:
    def __init__(self, config: Config):
        self.config = config

    def build_dataset(self, asset_rets: pd.DataFrame, bench_rets: pd.Series, fund_data: pd.DataFrame) -> pd.DataFrame:
        print("\n[3/8] Building ML dataset...")
        ml_data = []
        for date in asset_rets.index[12:-1]:
            date_idx = asset_rets.index.get_loc(date)
            next_date = asset_rets.index[date_idx + 1]
            bench_ret = float(bench_rets.loc[next_date]) if next_date in bench_rets.index else 0.0
            for ticker in asset_rets.columns:
                if ticker not in fund_data.index:
                    continue
                ml_data.append(self._extract_features(
                    ticker, date, date_idx, asset_rets, fund_data, next_date, bench_ret
                ))

        df_ml = pd.DataFrame(ml_data).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"✓ ML dataset: {len(df_ml)} samples over {df_ml['date'].nunique()} months")
        print(f"✓ Target distribution: {df_ml['target'].value_counts().to_dict()}")
        return df_ml

    def _extract_features(self, ticker, date, date_idx, asset_rets, fund_data, next_date, bench_ret):
        features = {
            "date": date,
            "ticker": ticker,
            "pe": fund_data.loc[ticker, "pe"],
            "pb": fund_data.loc[ticker, "pb"],
            "ps": fund_data.loc[ticker, "ps"],
            "div_yield": fund_data.loc[ticker, "div_yield"],
            "eps_growth": fund_data.loc[ticker, "eps_growth"],
            "score_value": fund_data.loc[ticker, "score_value"],
            "score_growth": fund_data.loc[ticker, "score_growth"],
            "momentum_1m": asset_rets.loc[date, ticker],
            "momentum_3m": asset_rets.iloc[max(0, date_idx-3):date_idx+1][ticker].sum(),
            "momentum_6m": asset_rets.iloc[max(0, date_idx-6):date_idx+1][ticker].sum(),
            "volatility_3m": asset_rets.iloc[max(0, date_idx-3):date_idx+1][ticker].std(),
            "volatility_12m": asset_rets.iloc[max(0, date_idx-12):date_idx+1][ticker].std(),
        }
        stock_ret = float(asset_rets.loc[next_date, ticker])
        features["target"] = 1 if stock_ret > bench_ret else 0
        features["next_month_return"] = stock_ret
        return features


class MLTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.param_distributions = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            },
            "Neural Network": {
                'hidden_layer_sizes': [(32,), (64, 32), (128, 64)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            "Random Forest": RandomForestClassifier(random_state=config.RANDOM_STATE),
            "Gradient Boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            "Neural Network": MLPClassifier(max_iter=500, random_state=config.RANDOM_STATE)
        }

    def fit_model(self, model_name: str, X_train, y_train):
        base_model = self.models[model_name]
        if model_name in self.param_distributions:
            print("    → Hyperparameter tuning with TimeSeriesSplit...")
            tscv = TimeSeriesSplit(n_splits=self.config.N_CV_SPLITS)
            search = RandomizedSearchCV(
                base_model,
                self.param_distributions[model_name],
                n_iter=self.config.N_ITER_SEARCH,
                cv=tscv,
                scoring='roc_auc',
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            return search.best_estimator_, search.best_params_, float(search.best_score_)
        else:
            base_model.fit(X_train, y_train)
            return base_model, None, None

    @staticmethod
    def safe_predict_proba(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    def train_universe(self, universe_name: str, X_train, y_train, X_test, y_test):
        results = {}
        print(f"\n===== TRAINING UNIVERSE: {universe_name} =====")
        print(f"  Train samples: {len(y_train)} | test: {len(y_test)}")

        # guard (same as your code 2)
        if len(y_train) < 50 or len(np.unique(y_train)) < 2 or len(y_test) < 20 or len(np.unique(y_test)) < 2:
            print("  ⚠️ Not enough data → skipping universe")
            return results

        for model_name in self.models.keys():
            print(f"  Training {model_name} [{universe_name}]...")
            best_model, best_params, best_cv = self.fit_model(model_name, X_train, y_train)

            if best_params:
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
                "universe": universe_name
            }
            print(f"    ✓ Test Accuracy: {acc:.3f} | AUC: {auc:.3f}")

        return results


class PerformanceAnalyzer:
    def __init__(self, config: Config):
        self.config = config

    def backtest_portfolio(self, weights_dict: Dict, returns: pd.DataFrame, costs_bps: Optional[int] = None) -> Tuple[pd.Series, pd.Series]:
        """
        Returns:
          - monthly net log-returns after transaction costs (Series)
          - turnover (Series)
        """
        if costs_bps is None:
            costs_bps = self.config.COSTS_BPS

        r_list, turnover_list, used_dates = [], [], []
        prev_w = None

        for d in sorted(weights_dict.keys()):
            if d not in returns.index:
                continue

            w = pd.Series(weights_dict[d], dtype=float).reindex(returns.columns, fill_value=0.0)
            w = w / w.sum() if w.sum() > 0 else w

            r = float((w * returns.loc[d]).sum())

            if prev_w is None:
                tc, turnover = 0.0, 1.0
            else:
                turnover = float((w - prev_w).abs().sum())
                tc = (costs_bps / 10000.0) * turnover

            r_list.append(r - tc)
            turnover_list.append(turnover)
            used_dates.append(d)
            prev_w = w

        return pd.Series(r_list, index=used_dates), pd.Series(turnover_list, index=used_dates)

    def compute_enhanced_metrics(self, equity_curve: pd.Series, bench_returns: pd.Series) -> Dict:
        if len(equity_curve) == 0:
            return self._empty_metrics()

        logr = np.log(equity_curve / equity_curve.shift(1)).dropna()
        if len(logr) == 0:
            return self._empty_metrics(float(equity_curve.iloc[-1]))

        rf_monthly = self.config.RF_RATE_ANNUAL / 12.0

        mu = float(logr.mean() * 12)
        sig = float(logr.std(ddof=0) * np.sqrt(12))
        sharpe = (mu - self.config.RF_RATE_ANNUAL) / (sig + 1e-12)

        downside = logr[logr < rf_monthly]
        downside_std = float(downside.std(ddof=0) * np.sqrt(12)) if len(downside) > 0 else sig
        sortino = (mu - self.config.RF_RATE_ANNUAL) / (downside_std + 1e-12)

        mdd = float((equity_curve / equity_curve.cummax() - 1).min())
        pct_neg = float((logr < 0).sum() / len(logr) * 100)
        pct_large_loss = float((logr < -0.10).sum() / len(logr) * 100)

        bench_aligned = bench_returns.reindex(logr.index, fill_value=0.0)
        simple_alpha = float((logr - bench_aligned).mean() * 12)

        excess_ret = logr - rf_monthly
        excess_bench = bench_aligned - rf_monthly

        if len(excess_ret) > 1 and excess_bench.std() > 1e-9:
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
            "CAPM Beta": 0.0,
            "Alpha t-stat": 0.0,
            "Final Value": final_value,
        }


class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def plot_equity_curves(self, curves_dict: Dict[str, pd.Series], filename: str, title: str, figsize=(11, 6)):
        plt.figure(figsize=figsize)
        for name, eq in curves_dict.items():
            if len(eq) >= 2:
                (eq / eq.iloc[0]).plot(label=name, linewidth=2.0)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.ylabel("Cumulative Value ($1)")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()

    def plot_barh(self, s: pd.Series, filename: str, title: str, xlabel: str, benchmark_name: Optional[str] = None):
        plt.figure(figsize=(8, 5))
        s.sort_values().plot(kind="barh", edgecolor="black")
        plt.title(title, fontsize=12, fontweight="bold")
        plt.xlabel(xlabel)
        if benchmark_name and benchmark_name in s.index:
            plt.axvline(s.loc[benchmark_name], color="black", linewidth=2, linestyle="--", label="Benchmark")
            plt.legend()
        plt.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()

    def plot_capm_alpha(self, stats_df: pd.DataFrame, filename: str, title: str):
        df = stats_df[["CAPM Alpha", "Alpha t-stat"]].sort_values("CAPM Alpha")
        plt.figure(figsize=(8, 5))
        df["CAPM Alpha"].plot(kind="barh", edgecolor="black")
        plt.axvline(0, color="black", linewidth=1)

        for i, (_, row) in enumerate(df.iterrows()):
            t = row["Alpha t-stat"]
            sig = "***" if abs(t) > 2.576 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.645 else ""
            if sig:
                plt.text(row["CAPM Alpha"], i, f" {sig}", va="center", fontsize=9)

        plt.title(title, fontweight="bold")
        plt.xlabel("Annualized CAPM Alpha")
        plt.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()

    def plot_turnover_vs_alpha(self, stats_df: pd.DataFrame, filename: str, title: str):
        plt.figure(figsize=(8, 5))
        plt.scatter(
            stats_df["Avg Annual Turnover"],
            stats_df["Simple Alpha"] * 100,
            s=120, alpha=0.75, edgecolors="black"
        )
        for name in stats_df.index:
            plt.annotate(
                name,
                (stats_df.loc[name, "Avg Annual Turnover"],
                 stats_df.loc[name, "Simple Alpha"] * 100),
                fontsize=8
            )
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel("Average Annual Turnover")
        plt.ylabel("Simple Alpha (%)")
        plt.title(title, fontweight="bold")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()

    def plot_robustness_bars(self, rob_df: pd.DataFrame, filename: str, title: str):
        if len(rob_df) == 0:
            return
        tmp = rob_df.set_index("Period").dropna(axis=1, how="all")
        plt.figure(figsize=(10, 5))
        tmp.plot(kind="bar", width=0.85, edgecolor="black")
        plt.axhline(0, color="black", linewidth=1)
        plt.title(title, fontweight="bold")
        plt.ylabel("Sharpe Ratio")
        plt.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1))
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close()


# ==================== HELPERS (MAIN/APPENDIX labels like code 2) ====================

def pretty_name(k: str) -> str:
    name = k
    name = name.replace("Logistic Regression", "LogReg")
    name = name.replace("Random Forest", "RF")
    name = name.replace("Gradient Boosting", "GB")
    name = name.replace("Neural Network", "NN")
    name = name.replace(" (All-trained)", " | train:All")
    name = name.replace(" (Value-trained)", " | train:Value")
    name = name.replace(" (Growth-trained)", " | train:Growth")
    name = name.replace(" (All)", " | use:All")
    name = name.replace(" (Value)", " | use:Value")
    name = name.replace(" (Growth)", " | use:Growth")
    return name

def override_main_label(raw_key: str) -> str:
    if raw_key.startswith("Logistic Regression"):
        return "LogReg (ML on Value)"
    if raw_key.startswith("Random Forest"):
        return "RF (ML on Value)"
    if raw_key.startswith("Gradient Boosting"):
        return "GB (ML on Value)"
    if raw_key.startswith("Neural Network"):
        return "NN (ML on Value)"
    return pretty_name(raw_key)


# ==================== MAIN ====================

def main():
    t0 = time.perf_counter()
    config = Config()

    SP500_TICKERS_UNIQUE = list(set(SP500_TICKERS))

    print("=" * 80)
    print("CAN WE BEAT THE MARKET? VALUE vs GROWTH + MACHINE LEARNING")
    print("Analysis with Robust Risk Metrics & Hyperparameter Optimization")
    print("Public data from Yahoo Finance")
    print("=" * 80)
    print(f"✓ Loaded {len(SP500_TICKERS_UNIQUE)} stocks")
    print(f"Period: {config.START} → today (monthly)")
    print(f"Train/Test split: {config.TRAIN_TEST_SPLIT}")
    print("=" * 80)

    OUT_DIR = "./sp500_ml_comparison_enhanced"
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    def to_equity(logret: pd.Series) -> pd.Series:
        logret = logret.dropna()
        if len(logret) == 0:
            return pd.Series(dtype=float)
        return np.exp(logret.cumsum())

    # 1) DATA
    loader = DataLoader(config)
    px = loader.download_prices(SP500_TICKERS_UNIQUE + [config.BENCHMARK])
    rets = loader.compute_returns(px)

    bench_rets = rets[config.BENCHMARK].copy() if config.BENCHMARK in rets.columns else pd.Series(dtype=float)
    asset_cols = [t for t in SP500_TICKERS_UNIQUE if t in rets.columns]
    asset_rets = rets[asset_cols].copy()

    # 2) FEATURES
    print("\n[2/8] Feature engineering...")
    fe = FeatureEngineering()
    fund = pd.DataFrame([fe.fetch_fundamentals(t) for t in asset_rets.columns]).set_index("ticker")

    tech_features = fe.calculate_technical_features(px, rets)
    for feat_name, feat_values in tech_features.items():
        fund[feat_name] = feat_values.reindex(fund.index)

    fund["score_value"] = fe.compute_value_score(fund)
    fund["score_growth"] = fe.compute_growth_score(fund)

    valid_scores_val = fund["score_value"].dropna()
    top_20pct_val = max(1, int(len(valid_scores_val) * config.TOP_PERCENTILE))
    val_names = valid_scores_val.nlargest(top_20pct_val).index.tolist()
    val_set = set(val_names)

    valid_scores_gro = fund["score_growth"].dropna()
    top_20pct_gro = max(1, int(len(valid_scores_gro) * config.TOP_PERCENTILE))
    gro_names = valid_scores_gro.nlargest(top_20pct_gro).index.tolist()
    gro_set = set(gro_names)

    print(f"✓ Value universe size: {len(val_set)} | Growth: {len(gro_set)}")

    # 3) ML DATASET
    builder = MLDatasetBuilder(config)
    df_ml = builder.build_dataset(asset_rets, bench_rets, fund)

    feature_cols = [c for c in df_ml.columns if c not in ["date", "ticker", "target", "next_month_return"]]
    train_mask = df_ml["date"] < config.TRAIN_TEST_SPLIT

    def split(df: pd.DataFrame, mask: pd.Series, universe: Optional[set] = None):
        sub = df.loc[mask].copy()
        if universe is not None:
            sub = sub[sub["ticker"].isin(universe)]
        X = sub[feature_cols]
        y = sub["target"]
        meta = sub[["date", "ticker", "next_month_return"]].copy()
        return X, y, meta

    X_train, y_train, _ = split(df_ml, train_mask, None)
    X_test, y_test, test_meta_all = split(df_ml, ~train_mask, None)

    X_train_val, y_train_val, _ = split(df_ml, train_mask, val_set)
    X_test_val, y_test_val, test_meta_val = split(df_ml, ~train_mask, val_set)

    X_train_gro, y_train_gro, _ = split(df_ml, train_mask, gro_set)
    X_test_gro, y_test_gro, test_meta_gro = split(df_ml, ~train_mask, gro_set)

    print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"✓ Features: {feature_cols}")

    # Scaling
    scaler_all = StandardScaler()
    X_train_scaled = scaler_all.fit_transform(X_train)
    X_test_scaled = scaler_all.transform(X_test)

    scaler_val = StandardScaler()
    X_train_val_scaled = scaler_val.fit_transform(X_train_val) if len(X_train_val) else np.empty((0, len(feature_cols)))
    X_test_val_scaled = scaler_val.transform(X_test_val) if len(X_test_val) else np.empty((0, len(feature_cols)))

    scaler_gro = StandardScaler()
    X_train_gro_scaled = scaler_gro.fit_transform(X_train_gro) if len(X_train_gro) else np.empty((0, len(feature_cols)))
    X_test_gro_scaled = scaler_gro.transform(X_test_gro) if len(X_test_gro) else np.empty((0, len(feature_cols)))

    # 4) TRAIN ML (All / Value / Growth)
    print("\n[4/8] ML training (All / Value / Growth)...")
    trainer = MLTrainer(config)
    ml_results = {}
    ml_results.update(trainer.train_universe("All", X_train_scaled, y_train, X_test_scaled, y_test))
    ml_results.update(trainer.train_universe("Value", X_train_val_scaled, y_train_val, X_test_val_scaled, y_test_val))
    ml_results.update(trainer.train_universe("Growth", X_train_gro_scaled, y_train_gro, X_test_gro_scaled, y_test_gro))

    # Save models
    for name, result in ml_results.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        model_path = os.path.join(OUT_DIR, f"model_{safe_name}_{ts}.pkl")
        joblib.dump(result["model"], model_path)

    # 4.5) Feature importance exports + plots (RF + GB) — like code 2
    print("\n[4.5/8] Feature importance (RF/GB if available)...")
    for key, res in ml_results.items():
        model = res["model"]
        universe = res["universe"]

        if not hasattr(model, "feature_importances_"):
            continue

        imp = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        base = "rf" if key.startswith("Random Forest") else ("gb" if key.startswith("Gradient Boosting") else "model")

        imp_path = os.path.join(OUT_DIR, f"feature_importance_{base}_{universe.lower()}_{ts}.csv")
        imp.to_csv(imp_path, index=False)

        # plot TOP 10
        top10 = imp.head(10).sort_values("importance")
        plt.figure(figsize=(7, 4))
        top10.set_index("feature")["importance"].plot(kind="barh", edgecolor="black")
        plt.title(f"Feature Importance — {key}", fontweight="bold")
        plt.xlabel("Importance")
        plt.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        fig_path = os.path.join(
            OUT_DIR,
            f"feature_importance_{key.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}_{ts}.png"
        )
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

    # 5) BACKTESTING
    print("\n[5/8] Backtesting (Value/Growth/EW/Random + ML)...")
    perf = PerformanceAnalyzer(config)

    # Traditional VALUE (with costs)
    w_val = {d: {t: 1/len(val_names) for t in val_names} for d in asset_rets.index[1:]}
    logret_val, turnover_val = perf.backtest_portfolio(w_val, asset_rets, costs_bps=config.COSTS_BPS)
    eq_val = to_equity(logret_val)

    # Traditional GROWTH (with costs)
    w_gro = {d: {t: 1/len(gro_names) for t in gro_names} for d in asset_rets.index[1:]}
    logret_gro, turnover_gro = perf.backtest_portfolio(w_gro, asset_rets, costs_bps=config.COSTS_BPS)
    eq_gro = to_equity(logret_gro)

    # S&P 500 (EW across universe) baseline — like code 2: costs = 0
    all_names = list(asset_rets.columns)
    w_sp = {d: {t: 1/len(all_names) for t in all_names} for d in asset_rets.index[1:]}
    logret_sp, turnover_sp = perf.backtest_portfolio(w_sp, asset_rets, costs_bps=0)
    eq_sp = to_equity(logret_sp)

    # Random baseline avg(N runs)
    np.random.seed(config.RANDOM_STATE)
    rnd_logrets, rnd_turns = [], []
    for _ in range(config.N_RANDOM_SIMULATIONS):
        weights_random = {}
        for d in asset_rets.index[1:]:
            n_select = max(1, int(len(all_names) * config.TOP_PERCENTILE))
            selected = np.random.choice(all_names, size=n_select, replace=False)
            weights_random[d] = {t: 1/len(selected) for t in selected}
        lr, tr = perf.backtest_portfolio(weights_random, asset_rets, costs_bps=config.COSTS_BPS)
        rnd_logrets.append(lr)
        rnd_turns.append(tr)

    logret_random = pd.concat(rnd_logrets, axis=1).mean(axis=1)
    turnover_random = pd.concat(rnd_turns, axis=1).mean(axis=1)
    eq_random = to_equity(logret_random)

    # ML strategies: train universe + use universe
    ml_backtest_logrets = {}
    ml_backtest_turns = {}

    def build_weights_from_meta(meta: pd.DataFrame, prob: np.ndarray, use_set: Optional[set]):
        mm = meta.copy()
        mm["prob"] = prob
        weights = {}
        for d in mm["date"].unique():
            dd = mm[mm["date"] == d]
            if use_set is not None:
                dd = dd[dd["ticker"].isin(use_set)]
            if len(dd) == 0:
                continue
            n = max(1, int(len(dd) * config.TOP_PERCENTILE))
            top = dd.nlargest(n, "prob")
            weights[d] = {t: 1/len(top) for t in top["ticker"]}
        return weights

    for name, res in ml_results.items():
        universe_train = res["universe"]
        prob = res["probabilities"]

        if universe_train == "Value":
            base_meta = test_meta_val
            use_unis = [("Value", val_set)]
        elif universe_train == "Growth":
            base_meta = test_meta_gro
            use_unis = [("Growth", gro_set)]
        else:
            base_meta = test_meta_all
            use_unis = [("All", None), ("Value", val_set), ("Growth", gro_set)]

        for use_name, use_set in use_unis:
            weights = build_weights_from_meta(base_meta, prob, use_set)
            strat_name = f"{name} ({use_name})"
            lr, tr = perf.backtest_portfolio(weights, asset_rets, costs_bps=config.COSTS_BPS)
            ml_backtest_logrets[strat_name] = lr
            ml_backtest_turns[strat_name] = tr

    eq_ml = {k: to_equity(v) for k, v in ml_backtest_logrets.items()}

    # 6) METRICS
    print("\n[6/8] Computing advanced metrics...")
    all_stats = {
        "VALUE (Traditional)": perf.compute_enhanced_metrics(eq_val, bench_rets),
        "GROWTH (Traditional)": perf.compute_enhanced_metrics(eq_gro, bench_rets),
        "S&P 500 (EW)": perf.compute_enhanced_metrics(eq_sp, bench_rets),
        "RANDOM Portfolio (Avg)": perf.compute_enhanced_metrics(eq_random, bench_rets),
    }

    all_stats["VALUE (Traditional)"]["Avg Annual Turnover"] = float(turnover_val.mean() * 12) if len(turnover_val) else 0.0
    all_stats["GROWTH (Traditional)"]["Avg Annual Turnover"] = float(turnover_gro.mean() * 12) if len(turnover_gro) else 0.0
    all_stats["S&P 500 (EW)"]["Avg Annual Turnover"] = float(turnover_sp.mean() * 12) if len(turnover_sp) else 0.0
    all_stats["RANDOM Portfolio (Avg)"]["Avg Annual Turnover"] = float(turnover_random.mean() * 12) if len(turnover_random) else 0.0

    for strat, eq in eq_ml.items():
        stats_dict = perf.compute_enhanced_metrics(eq, bench_rets)
        tr = ml_backtest_turns.get(strat, pd.Series(dtype=float))
        stats_dict["Avg Annual Turnover"] = float(tr.mean() * 12) if len(tr) else 0.0
        all_stats[strat] = stats_dict

    stats_df = pd.DataFrame(all_stats).T.sort_values("Sharpe", ascending=False)
    stats_df.to_csv(os.path.join(OUT_DIR, f"performance_table_{ts}.csv"), index=True)

    print("\n" + "=" * 80)
    print("⚡ ENHANCED PERFORMANCE TABLE")
    print("=" * 80)
    print(stats_df.to_string())
    print("=" * 80)

    # verdict (like code 2: beats market + beats random via CAGR)
    sp500_cagr = float(stats_df.loc["S&P 500 (EW)", "CAGR"])
    random_cagr = float(stats_df.loc["RANDOM Portfolio (Avg)", "CAGR"])
    random_sharpe = float(stats_df.loc["RANDOM Portfolio (Avg)", "Sharpe"])

    strategies_only = stats_df.drop(["S&P 500 (EW)", "RANDOM Portfolio (Avg)"], errors="ignore")
    beats_market = strategies_only[strategies_only["CAGR"] > sp500_cagr]
    beats_random = strategies_only[strategies_only["CAGR"] > random_cagr]
    beats_both = strategies_only[(strategies_only["CAGR"] > sp500_cagr) & (strategies_only["CAGR"] > random_cagr)]

    # 7) ROBUSTNESS
    print("\n[7/8] Robustness by sub-periods (Sharpe)...")
    logret_map = {
        "VALUE (Traditional)": logret_val,
        "GROWTH (Traditional)": logret_gro,
        "S&P 500 (EW)": logret_sp,
        "RANDOM Portfolio (Avg)": logret_random,
    }
    logret_map.update(ml_backtest_logrets)

    robustness_rows = []
    for start_date, end_date, label in config.SUB_PERIODS:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        bench_p = bench_rets.loc[(bench_rets.index >= start_ts) & (bench_rets.index <= end_ts)]

        row = {"Period": label}
        for strat, lr in logret_map.items():
            lr_p = lr.loc[(lr.index >= start_ts) & (lr.index <= end_ts)].dropna()
            if len(lr_p) < 12:
                row[strat] = np.nan
            else:
                eq_p = np.exp(lr_p.cumsum())
                row[strat] = perf.compute_enhanced_metrics(eq_p, bench_p)["Sharpe"]
        robustness_rows.append(row)

    robustness_df = pd.DataFrame(robustness_rows)
    robustness_df.to_csv(os.path.join(OUT_DIR, f"robustness_sharpe_{ts}.csv"), index=False)
    print(robustness_df.to_string(index=False))

    # 8) VISUALS + REPORT (MAIN + APPENDIX, like code 2)
    print("\n[8/8] Figures + report...")
    viz = Visualizer(OUT_DIR)

    # equity curves all
    curves_all = {
        "VALUE (Traditional)": eq_val,
        "GROWTH (Traditional)": eq_gro,
        "S&P 500 (EW)": eq_sp,
        "RANDOM Portfolio (Avg)": eq_random,
    }
    curves_all.update(eq_ml)

    viz.plot_equity_curves(curves_all, f"equity_ALL_{ts}.png", "Equity Curves — All Strategies")

    # MAIN selection: train All + use Value (4 models)
    ml_keys = list(eq_ml.keys())

    def find_main(prefix: str):
        cands = [k for k in ml_keys if k.startswith(prefix) and "(All-trained)" in k and " (Value)" in k]
        return cands[0] if len(cands) else None

    main_ml = [
        find_main("Logistic Regression"),
        find_main("Random Forest"),
        find_main("Gradient Boosting"),
        find_main("Neural Network"),
    ]
    main_ml = [k for k in main_ml if k is not None]
    if len(main_ml) == 0:
        main_ml = [k for k in ml_keys if "(All-trained)" in k and " (Value)" in k]

    BASELINES = ["S&P 500 (EW)", "RANDOM Portfolio (Avg)", "VALUE (Traditional)", "GROWTH (Traditional)"]
    keep_main = BASELINES + main_ml

    # rename (like code 2)
    rename = {
        "S&P 500 (EW)": "Benchmark (EW)",
        "RANDOM Portfolio (Avg)": "Random (Avg)",
        "VALUE (Traditional)": "Value",
        "GROWTH (Traditional)": "Growth",
    }
    for k in ml_keys:
        rename[k] = pretty_name(k)
    for k in main_ml:
        rename[k] = override_main_label(k)

    curves_main = {rename[k]: curves_all[k] for k in keep_main if k in curves_all}
    viz.plot_equity_curves(
        curves_main,
        f"MAIN_equity_ML_useVALUE_{ts}.png",
        "MAIN — ML train:All use:Value vs Benchmark/Value/Growth/Random"
    )

    # MAIN metrics plots
    stats_df_main = stats_df.loc[[k for k in keep_main if k in stats_df.index]].copy()
    stats_df_main.index = [rename.get(i, i) for i in stats_df_main.index]
    bench_name = rename.get("S&P 500 (EW)", "S&P 500 (EW)")

    viz.plot_barh(stats_df_main["Sortino"], f"MAIN_sortino_{ts}.png", "Sortino Ratio — MAIN (use:Value)", "Sortino (↑)", bench_name)
    viz.plot_capm_alpha(stats_df_main, f"MAIN_capm_alpha_{ts}.png", "CAPM Alpha — MAIN (use:Value)")
    viz.plot_turnover_vs_alpha(stats_df_main, f"MAIN_turnover_vs_alpha_{ts}.png", "Turnover vs Alpha — MAIN")

    viz.plot_barh(stats_df_main["Max Drawdown"] * 100, f"MAIN_max_drawdown_{ts}.png", "Maximum Drawdown — MAIN", "Max Drawdown (%)", bench_name)
    viz.plot_barh(stats_df_main["Pct Negative Months"], f"MAIN_negative_months_{ts}.png", "% Negative Months — MAIN", "%", bench_name)
    viz.plot_barh(stats_df_main["Pct Large Losses"], f"MAIN_large_losses_{ts}.png", "% Months with loss >10% — MAIN", "%", bench_name)

    # robustness MAIN (subset)
    if len(robustness_df) > 0:
        keep_cols = ["Period"] + [k for k in keep_main if k in robustness_df.columns]
        rob_main = robustness_df[keep_cols].copy().rename(columns=rename)
        viz.plot_robustness_bars(rob_main, f"MAIN_robustness_sharpe_{ts}.png", "Robustness — Sharpe by sub-period (MAIN)")

    # APPENDIX by use universe
    def curves_use(tag: str):
        ml_use = [k for k in ml_keys if f" ({tag})" in k]
        keep = BASELINES + ml_use
        out = {}
        for k in keep:
            if k in curves_all:
                out[rename.get(k, k)] = curves_all[k]
        return out

    viz.plot_equity_curves(curves_use("All"), f"APP_equity_useALL_{ts}.png", "APPENDIX — use:All")
    viz.plot_equity_curves(curves_use("Value"), f"APP_equity_useVALUE_{ts}.png", "APPENDIX — use:Value")
    viz.plot_equity_curves(curves_use("Growth"), f"APP_equity_useGROWTH_{ts}.png", "APPENDIX — use:Growth")

    # FINAL REPORT (detailed like code 2)
    best_strat = stats_df.index[0]
    best_sharpe = float(stats_df.loc[best_strat, "Sharpe"])
    best_cagr = float(stats_df.loc[best_strat, "CAGR"])
    best_alpha = float(stats_df.loc[best_strat, "CAPM Alpha"])
    best_tstat = float(stats_df.loc[best_strat, "Alpha t-stat"])

    report_lines = [
        "=" * 80,
        "CAN WE BEAT THE MARKET? - ENHANCED ANALYSIS",
        "Value vs Growth + Machine Learning vs S&P 500",
        "=" * 80,
        "",
        "📋 ENHANCED FEATURES:",
        "   ✓ Sortino Ratio (downside risk)",
        "   ✓ CAPM Alpha & Beta with t-statistics",
        "   ✓ Turnover tracking & cost analysis",
        "   ✓ Large loss frequency (>10% monthly loss)",
        "   ✓ Hyperparameter tuning with TimeSeriesSplit",
        "   ✓ Robustness testing across sub-periods",
        "",
        "⚠️ LIMITATIONS:",
        "   • Survivorship bias (current ticker list)",
        "   • Look-ahead bias on fundamentals (Yahoo Finance snapshot)",
        "   • Simplified transaction cost model",
        "",
        "=" * 80,
        "ENHANCED PERFORMANCE METRICS",
        "=" * 80,
        "",
    ]

    for strat, metrics in stats_df.iterrows():
        report_lines.append(f"{strat}:")
        report_lines.append(f"  • CAGR:                {metrics['CAGR']:.2%}")
        report_lines.append(f"  • Volatility:          {metrics['Volatility']:.2%}")
        report_lines.append(f"  • Sharpe Ratio:        {metrics['Sharpe']:.2f}")
        report_lines.append(f"  • Sortino Ratio:       {metrics['Sortino']:.2f}")
        report_lines.append(f"  • Max Drawdown:        {metrics['Max Drawdown']:.2%}")
        report_lines.append(f"  • % Negative Months:   {metrics['Pct Negative Months']:.1f}%")
        report_lines.append(f"  • % Large Losses:      {metrics['Pct Large Losses']:.1f}%")
        report_lines.append(f"  • Simple Alpha:        {metrics['Simple Alpha']:.2%}")
        report_lines.append(f"  • CAPM Alpha:          {metrics['CAPM Alpha']:.2%}")
        report_lines.append(f"  • CAPM Beta:           {metrics['CAPM Beta']:.2f}")
        report_lines.append(f"  • Alpha t-stat:        {metrics['Alpha t-stat']:.2f}")
        report_lines.append(f"  • Avg Annual Turnover: {metrics.get('Avg Annual Turnover', 0.0):.2f}")
        report_lines.append(f"  • Final Value:         ${metrics['Final Value']:.2f}")
        report_lines.append("")

    report_lines.extend([
        "=" * 80,
        "💡 FINAL VERDICT",
        "=" * 80,
        "",
    ])

    if len(beats_both) > 0:
        report_lines.append("✅ YES! Some strategies beat BOTH the S&P 500 and Random (in this backtest).")
        report_lines.append(f"🏆 Best Strategy (Sharpe): {best_strat}")
        report_lines.append(f"   • CAGR: {best_cagr:.2%} vs Market: {sp500_cagr:.2%}")
        report_lines.append(f"   • Sharpe: {best_sharpe:.2f} (Random: {random_sharpe:.2f})")
        report_lines.append(f"   • CAPM Alpha: {best_alpha:.2%} (t-stat: {best_tstat:.2f})")
    elif len(beats_market) > 0:
        report_lines.append("⚠️ MIXED: Some strategies beat the market but NOT random (likely noise/luck).")
    elif len(beats_random) > 0:
        report_lines.append("⚠️ MIXED: Some strategies beat random but NOT the market.")
    else:
        report_lines.append("❌ NO! No strategy beat both market and random selection.")

    report_lines.extend([
        "",
        "=" * 80,
        f"All results saved to: {os.path.abspath(OUT_DIR)}",
        "=" * 80,
    ])

    report_text = "\n".join(report_lines)
    report_path = os.path.join(OUT_DIR, f"ENHANCED_REPORT_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + "=" * 80)
    print("✓✓✓ ANALYSIS COMPLETE ✓✓✓")
    print("=" * 80)
    print(f"📂 Outputs: {os.path.abspath(OUT_DIR)}")
    print(f"📝 Report: {report_path}")
    t1 = time.perf_counter()
    print(f"⏱ Total time: {t1 - t0:.2f} sec")
    print("=" * 80)


if __name__ == "__main__":
    main()
