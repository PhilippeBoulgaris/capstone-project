# main.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import time
import inspect
import random
import hashlib
from pathlib import Path
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Reproducibility (set threads BEFORE importing numpy/pandas)
# -----------------------------------------------------------------------------
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Force Python to import the local project root first (avoid "src" collisions)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
from src.config import Config
from src.paths import Paths
from src.universe import SP500_TICKERS
from src.data_loader import DataLoader
from src.fundamentals_yahoo import FundamentalFeatureEngineering
from src.dataset_builders import MLDatasetBuilder, MLDatasetBuilderPanel
from src.ml import MLTrainer, build_weights_from_meta
from src.performance import PerformanceAnalyzer, to_equity
from src.visuals import Visualizer, MLVisualizer
from src.utils import short_model_name, pick_best_model_key
from src.portfolios import (
    build_equal_weight_static,
    build_random_weights_each_month,
)
from src.constants import NON_FEATURE_COLS, SEC_FEATURE_COLS_DEFAULT
from src.validation import numeric_feature_cols
from src.sec_fundamentals import SECFundamentalsBuilder
from src.sec_style import compute_sec_value_score, build_sec_style_weights  # growth removed


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_month_end(ts: pd.Timestamp | str) -> pd.Timestamp:
    return pd.to_datetime(ts).to_period("M").to_timestamp("M")


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _slice_oos_and_rebase(s: pd.Series | None, split_dt: pd.Timestamp) -> pd.Series | None:
    """Keep only OOS window [split_dt, ...] and rebase to 1 at split."""
    if s is None:
        return None
    s2 = pd.Series(s).dropna().copy()
    if len(s2) < 2:
        return None
    s2.index = pd.to_datetime(s2.index).tz_localize(None).to_period("M").to_timestamp("M")
    s2 = s2.sort_index()
    s2 = s2.loc[s2.index >= split_dt]
    if len(s2) < 2:
        return None
    base = float(s2.iloc[0])
    if (not np.isfinite(base)) or base == 0:
        return None
    return s2 / base


def _prep_curves_oos(curves: Dict[str, pd.Series | None], split_dt: pd.Timestamp) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for k, v in curves.items():
        vv = _slice_oos_and_rebase(v, split_dt)
        if vv is not None and len(vv) >= 2:
            out[k] = vv
    return out


def _avg_annual_turnover(turnover: pd.Series | None) -> float:
    """Annualize average monthly turnover. Safe for empty / NaNs."""
    if turnover is None:
        return 0.0
    tr = pd.Series(turnover).dropna()
    if len(tr) == 0:
        return 0.0
    m = float(tr.mean())
    if not np.isfinite(m):
        return 0.0
    return 12.0 * m


def _train_universe_compat(
    trainer: MLTrainer,
    universe_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    dates_train: pd.Series | None = None,
):
    """
    Compatibility wrapper:
    - If MLTrainer.train_universe supports dates_train, pass it.
    - Otherwise call without it.
    """
    try:
        sig = inspect.signature(trainer.train_universe)
        if "dates_train" in sig.parameters:
            return trainer.train_universe(
                universe_name, X_train, y_train, X_test, y_test, dates_train=dates_train
            )
    except Exception:
        pass
    return trainer.train_universe(universe_name, X_train, y_train, X_test, y_test)


def _stable_top_percentile(scores: pd.Series, top_percentile: float) -> list[str]:
    """
    Deterministic top-percentile selection with tie-break on ticker.
    scores.index should be tickers.
    """
    s = pd.Series(scores).dropna().copy()
    if len(s) == 0:
        return []
    df = pd.DataFrame({"ticker": s.index.astype(str), "score": s.values}).dropna()
    n = max(1, int(np.floor(len(df) * float(top_percentile))))
    df = df.sort_values(["score", "ticker"], ascending=[False, True], kind="mergesort")
    return df.head(n)["ticker"].tolist()


def _cache_key(cfg: Config, tickers: list[str]) -> str:
    """
    Cache key that changes if you change core config/universe.
    """
    payload = {
        "START": getattr(cfg, "START", None),
        "INTERVAL": getattr(cfg, "INTERVAL", None),
        "BENCHMARK": getattr(cfg, "BENCHMARK", None),
        "TRAIN_TEST_SPLIT": getattr(cfg, "TRAIN_TEST_SPLIT", None),
        "TOP_PERCENTILE": getattr(cfg, "TOP_PERCENTILE", None),
        "N_TICKERS": len(tickers),
        "TICKERS_HEAD": tickers[:10],
        "TICKERS_TAIL": tickers[-10:],
    }
    raw = repr(payload).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def _load_pickle(path: Path) -> Any:
    return pd.read_pickle(path)


def _save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(obj, path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    t0 = time.perf_counter()
    cfg = Config()

    # Seeds (python + numpy)
    seed = int(getattr(cfg, "RANDOM_STATE", 42))
    _set_global_seed(seed)

    if "@" not in cfg.SEC_USER_AGENT:
        raise RuntimeError("Please set Config.SEC_USER_AGENT with your email (SEC requirement).")

    split_dt = _to_month_end(cfg.TRAIN_TEST_SPLIT)
    tickers = list(dict.fromkeys(SP500_TICKERS))

    print("=" * 80)
    print("FULL PIPELINE — Yahoo snapshot vs SEC as-of fundamentals (Value + Growth benchmark + ML)")
    print("=" * 80)
    print(f"Universe: {len(tickers)} tickers | Start: {cfg.START} | Interval: {cfg.INTERVAL}")
    print(f"Train/Test split (month-end): {split_dt.date()} | Benchmark: {cfg.BENCHMARK}")
    print(f"Seed: {seed} | Threads: OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS')} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")
    print("=" * 80)

    paths = Paths()
    out_dir = paths.ensure_out_dir()
    Path(out_dir).mkdir(parents=True, exist_ok=True)  # safety
    ts = paths.timestamp()

    # Cache (so Yahoo snapshot + prices don't change between runs)
    cache_dir = Path(out_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ck = _cache_key(cfg, tickers)

    # ------------------------------------------------------------------
    # 1) Prices / returns (cached)
    # ------------------------------------------------------------------
    print("\n[1/9] Downloading prices and computing returns...")
    loader = DataLoader(cfg)

    px_cache = cache_dir / f"px_{ck}.pkl"
    if px_cache.exists():
        px = _load_pickle(px_cache)
        print(f"  ✓ Loaded cached prices: {px_cache.name}")
    else:
        px = loader.download_prices(tickers + [cfg.BENCHMARK])
        _save_pickle(px, px_cache)
        print(f"  ✓ Saved prices cache: {px_cache.name}")

    rets = loader.compute_returns(px)
    rets.index = pd.to_datetime(rets.index).tz_localize(None).to_period("M").to_timestamp("M")
    rets = rets.sort_index()

    if cfg.BENCHMARK not in rets.columns:
        raise RuntimeError(f"Benchmark {cfg.BENCHMARK} not found in downloaded data.")

    bench_rets = rets[cfg.BENCHMARK].copy()

    asset_cols = [t for t in tickers if t in rets.columns]
    asset_rets = rets[asset_cols].copy()

    px_assets = px[asset_cols].copy()
    px_assets.index = pd.to_datetime(px_assets.index).tz_localize(None).to_period("M").to_timestamp("M")
    px_assets = px_assets.reindex(asset_rets.index)

    print(f"✓ Universe with return data: {len(asset_cols)} tickers")
    print(f"✓ Returns matrix: {asset_rets.shape[0]} months × {asset_rets.shape[1]} tickers")

    # ------------------------------------------------------------------
    # 2) Yahoo fundamentals (snapshot) + scores (cached)
    # ------------------------------------------------------------------
    print("\n[2/9] Fetching Yahoo fundamentals (snapshot) and computing scores...")
    fe = FundamentalFeatureEngineering()

    fy_cache = cache_dir / f"fund_yahoo_{ck}.pkl"
    if fy_cache.exists():
        fund_yahoo = _load_pickle(fy_cache)
        print(f"  ✓ Loaded cached Yahoo fundamentals: {fy_cache.name}")
    else:
        fund_list = []
        for i, t in enumerate(asset_cols, 1):
            fund_list.append(fe.fetch_yahoo_fundamentals(t))
            if i % 25 == 0:
                print(f"  ... {i}/{len(asset_cols)} processed")
        fund_yahoo = pd.DataFrame(fund_list).set_index("ticker")
        _save_pickle(fund_yahoo, fy_cache)
        print(f"  ✓ Saved Yahoo fundamentals cache: {fy_cache.name}")

    for c in fund_yahoo.columns:
        if c != "sector":
            fund_yahoo[c] = pd.to_numeric(fund_yahoo[c], errors="coerce")

    fund_yahoo["score_value"] = fe.compute_value_score(fund_yahoo)
    fund_yahoo["score_growth"] = fe.compute_growth_score(fund_yahoo)
    fund_yahoo["score_quality"] = fe.compute_quality_score(fund_yahoo)

    # Deterministic universe selection
    val_names = _stable_top_percentile(fund_yahoo["score_value"], cfg.TOP_PERCENTILE)
    gro_names = _stable_top_percentile(fund_yahoo["score_growth"], cfg.TOP_PERCENTILE)

    val_names = [t for t in val_names if t in asset_cols]
    gro_names = [t for t in gro_names if t in asset_cols]

    print(f"✓ Yahoo Value universe: {len(val_names)} tickers (top {int(cfg.TOP_PERCENTILE*100)}%)")
    print(f"✓ Yahoo Growth universe: {len(gro_names)} tickers (top {int(cfg.TOP_PERCENTILE*100)}%)")

    # ------------------------------------------------------------------
    # 3) SEC pipeline: annual -> monthly panel -> ratios/features
    # ------------------------------------------------------------------
    print("\n[3/9] SEC pipeline: annual FY -> monthly panel -> ratios...")
    sec_panel = pd.DataFrame()
    sec_feature_cols: list[str] = []

    try:
        sec_builder = SECFundamentalsBuilder(cfg)
        sec_annual = sec_builder.build_annual_sec_table(asset_cols, start_year=2010, end_year=2024)

        if sec_annual.empty:
            print("⚠️ SEC annual table is empty — skipping SEC pipeline.")
        else:
            sec_panel = sec_builder.annual_to_monthly_panel(sec_annual, asset_rets.index)
            sec_panel["date"] = pd.to_datetime(sec_panel["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
            sec_panel = sec_panel.dropna(subset=["date", "ticker"]).copy()
            sec_panel = sec_panel.sort_values(["date", "ticker"]).reset_index(drop=True)

            def _mcap(row: pd.Series) -> float:
                d = row.get("date")
                t = row.get("ticker")
                if (d in px_assets.index) and (t in px_assets.columns):
                    pr = float(px_assets.loc[d, t])
                else:
                    pr = np.nan
                sh = row.get("shares", np.nan)
                return float(pr * sh) if np.isfinite(pr) and np.isfinite(sh) else np.nan

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

            for c in SEC_FEATURE_COLS_DEFAULT:
                if c in sec_panel.columns:
                    sec_panel.loc[np.isinf(sec_panel[c]), c] = np.nan

            sec_feature_cols = [c for c in SEC_FEATURE_COLS_DEFAULT if c in sec_panel.columns]

            print(f"✓ SEC panel rows: {len(sec_panel):,}")
            print(f"✓ SEC feature columns available: {sec_feature_cols}")

    except Exception as e:
        print(f"⚠️ SEC pipeline failed — skipping SEC. Error: {e}")

    # ------------------------------------------------------------------
    # 4) SEC VALUE portfolio (monthly, as-of)
    # ------------------------------------------------------------------
    print("\n[4/9] Building SEC VALUE portfolio (if SEC panel available)...")
    eq_val_sec: pd.Series | None = None
    turnover_val_sec = pd.Series(dtype=float)

    if len(sec_panel) > 0:
        needed = [c for c in ["pb", "pe", "ps", "ev_ebitda"] if c in sec_panel.columns]
        if len(needed) < 2:
            print("⚠️ SEC Value: not enough columns for scoring (need at least 2 of pb/pe/ps/ev_ebitda).")
        else:
            sec_scores = sec_panel[["date", "ticker"] + needed].copy()
            sec_scores = sec_scores.sort_values(["date", "ticker"]).reset_index(drop=True)

            val_chunks = []
            for d, cs in sec_scores.groupby("date", sort=True):
                cs2 = cs.reset_index(drop=True).copy()
                cs2["score_value_sec"] = compute_sec_value_score(cs2)
                val_chunks.append(cs2[["date", "ticker", "score_value_sec"]])

            val_scores = pd.concat(val_chunks, ignore_index=True) if val_chunks else pd.DataFrame()
            sec_scored = sec_scores.merge(val_scores, on=["date", "ticker"], how="left")

            w_val_sec = build_sec_style_weights(
                sec_scored,
                "score_value_sec",
                cfg.TOP_PERCENTILE,
                allowed_tickers=asset_cols,
            )

            perf_tmp = PerformanceAnalyzer(cfg)
            logret_val_sec, turnover_val_sec = perf_tmp.backtest_portfolio(
                w_val_sec, asset_rets, costs_bps=cfg.COSTS_BPS
            )
            eq_val_sec = to_equity(logret_val_sec)

            print(f"✓ SEC Value computed for {eq_val_sec.index.nunique()} months")

    # ------------------------------------------------------------------
    # 5) Build ML datasets
    # ------------------------------------------------------------------
    print("\n[5/9] Building ML datasets...")
    builder_y = MLDatasetBuilder(cfg)
    df_ml_yahoo = builder_y.build_yahoo_dataset(asset_rets, bench_rets, fund_yahoo.reset_index())
    if len(df_ml_yahoo) > 0:
        df_ml_yahoo["date"] = pd.to_datetime(df_ml_yahoo["date"]).dt.to_period("M").dt.to_timestamp("M")
        df_ml_yahoo = df_ml_yahoo.sort_values(["date", "ticker"]).reset_index(drop=True)

    df_ml_sec = pd.DataFrame()
    if (len(sec_panel) > 0) and (len(sec_feature_cols) > 0):
        builder_s = MLDatasetBuilderPanel(cfg)
        df_ml_sec = builder_s.build_dataset_from_panel(
            asset_rets=asset_rets,
            bench_rets=bench_rets,
            feature_panel=sec_panel[["date", "ticker"] + sec_feature_cols].copy(),
            feature_cols=sec_feature_cols,
            min_non_nan_features=max(1, int(np.ceil(0.5 * len(sec_feature_cols)))),
            impute=False,
            verbose=True,
        )
        if len(df_ml_sec) > 0:
            df_ml_sec["date"] = pd.to_datetime(df_ml_sec["date"]).dt.to_period("M").dt.to_timestamp("M")
            df_ml_sec = df_ml_sec.sort_values(["date", "ticker"]).reset_index(drop=True)
    else:
        print("⚠️ SEC ML dataset skipped (no SEC panel/features).")

    if len(df_ml_sec) > 0:
        print(
            "✓ SEC ML coverage:",
            df_ml_sec["date"].min(),
            "->",
            df_ml_sec["date"].max(),
            "| train:",
            int((df_ml_sec["date"] < split_dt).sum()),
            "| test:",
            int((df_ml_sec["date"] >= split_dt).sum()),
        )
    else:
        print("✓ SEC ML coverage: empty")

    # ------------------------------------------------------------------
    # 6) ML training
    # ------------------------------------------------------------------
    print("\n[6/9] Training ML models (Yahoo)...")
    trainer = MLTrainer(cfg)

    feature_cols_y = numeric_feature_cols(df_ml_yahoo, exclude=list(NON_FEATURE_COLS))
    train_mask_y = df_ml_yahoo["date"] < split_dt

    X_train_y = df_ml_yahoo.loc[train_mask_y, feature_cols_y]
    y_train_y = df_ml_yahoo.loc[train_mask_y, "target"]
    X_test_y = df_ml_yahoo.loc[~train_mask_y, feature_cols_y]
    y_test_y = df_ml_yahoo.loc[~train_mask_y, "target"]
    meta_test_y = df_ml_yahoo.loc[~train_mask_y, ["date", "ticker", "next_month_return"]].copy()
    dates_train_y = df_ml_yahoo.loc[train_mask_y, "date"].copy()

    ml_results_y = _train_universe_compat(
        trainer, "Yahoo", X_train_y, y_train_y, X_test_y, y_test_y, dates_train=dates_train_y
    )

    ml_results_sec: Dict[str, dict] = {}
    meta_test_s = pd.DataFrame()
    y_test_s = None

    if len(df_ml_sec) > 0:
        print("\n[6bis/10] Training ML models (SEC)...")
        trainer_s = MLTrainer(cfg)

        feature_cols_s = numeric_feature_cols(df_ml_sec, exclude=list(NON_FEATURE_COLS))
        train_mask_s = df_ml_sec["date"] < split_dt

        X_train_s = df_ml_sec.loc[train_mask_s, feature_cols_s]
        y_train_s = df_ml_sec.loc[train_mask_s, "target"]
        X_test_s = df_ml_sec.loc[~train_mask_s, feature_cols_s]
        y_test_s = df_ml_sec.loc[~train_mask_s, "target"]
        meta_test_s = df_ml_sec.loc[~train_mask_s, ["date", "ticker", "next_month_return"]].copy()
        dates_train_s = df_ml_sec.loc[train_mask_s, "date"].copy()

        ml_results_sec = _train_universe_compat(
            trainer_s, "SEC", X_train_s, y_train_s, X_test_s, y_test_s, dates_train=dates_train_s
        )

    # ------------------------------------------------------------------
    # 7) Backtests
    # ------------------------------------------------------------------
    print("\n[7/9] Backtesting portfolios...")
    perf = PerformanceAnalyzer(cfg)

    w_val = build_equal_weight_static(asset_rets.index[:-1], val_names)
    logret_val, turnover_val = perf.backtest_portfolio(w_val, asset_rets, costs_bps=cfg.COSTS_BPS)
    eq_val = to_equity(logret_val) if len(logret_val) else pd.Series([1.0], index=asset_rets.index[:1])

    w_gro = build_equal_weight_static(asset_rets.index[:-1], gro_names)
    logret_gro, turnover_gro = perf.backtest_portfolio(w_gro, asset_rets, costs_bps=cfg.COSTS_BPS)
    eq_gro = to_equity(logret_gro) if len(logret_gro) else pd.Series([1.0], index=asset_rets.index[:1])

    all_names = list(asset_cols)
    w_ew = build_equal_weight_static(asset_rets.index[:-1], all_names)
    logret_ew, turnover_ew = perf.backtest_portfolio(w_ew, asset_rets, costs_bps=0)
    eq_ew = to_equity(logret_ew)

    w_rnd = build_random_weights_each_month(
        asset_rets.index[:-1], all_names, cfg.TOP_PERCENTILE, seed=cfg.RANDOM_STATE
    )
    logret_rnd, turnover_rnd = perf.backtest_portfolio(w_rnd, asset_rets, costs_bps=cfg.COSTS_BPS)
    eq_rnd = to_equity(logret_rnd)

    eq_bench = to_equity(bench_rets.dropna()) if len(bench_rets) else pd.Series(dtype=float)

    ml_backtest_logrets: Dict[str, pd.Series] = {}
    ml_backtest_turns: Dict[str, pd.Series] = {}

    for name in sorted(ml_results_y.keys()):
        res = ml_results_y[name]
        w = build_weights_from_meta(meta_test_y, res["probabilities"], cfg.TOP_PERCENTILE)
        lr, tr = perf.backtest_portfolio(w, asset_rets, costs_bps=cfg.COSTS_BPS)
        key = f"{short_model_name(name)} [Yahoo]"
        ml_backtest_logrets[key] = lr
        ml_backtest_turns[key] = tr

    for name in sorted(ml_results_sec.keys()):
        res = ml_results_sec[name]
        if len(meta_test_s) == 0:
            continue
        w = build_weights_from_meta(meta_test_s, res["probabilities"], cfg.TOP_PERCENTILE)
        lr, tr = perf.backtest_portfolio(w, asset_rets, costs_bps=cfg.COSTS_BPS)
        key = f"{short_model_name(name)} [SEC]"
        ml_backtest_logrets[key] = lr
        ml_backtest_turns[key] = tr

    eq_ml = {k: to_equity(v) for k, v in ml_backtest_logrets.items()}

    # ------------------------------------------------------------------
    # 8) Metrics table (FAIR OOS)
    # ------------------------------------------------------------------
    print("\n[8/9] Computing metrics and saving table...")

    def _metrics_oos(eq: pd.Series) -> dict:
        eq_oos = _slice_oos_and_rebase(eq, split_dt)
        if eq_oos is None or len(eq_oos) < 2:
            return perf.compute_enhanced_metrics(pd.Series(dtype=float), bench_rets)
        return perf.compute_enhanced_metrics(eq_oos, bench_rets)

    def _turnover_oos(turn: pd.Series | None) -> float:
        if turn is None:
            return 0.0
        tr = pd.Series(turn).dropna()
        if len(tr) == 0:
            return 0.0
        tr.index = pd.to_datetime(tr.index).tz_localize(None).to_period("M").to_timestamp("M")
        tr = tr.sort_index()
        tr = tr.loc[tr.index >= split_dt]
        return _avg_annual_turnover(tr)

    all_stats: Dict[str, dict] = {}

    if eq_val is not None and len(eq_val) >= 2:
        all_stats["VALUE (Yahoo snapshot)"] = _metrics_oos(eq_val)
        all_stats["VALUE (Yahoo snapshot)"]["Avg Annual Turnover"] = _turnover_oos(turnover_val)

    if eq_gro is not None and len(eq_gro) >= 2:
        all_stats["GROWTH (Yahoo snapshot)"] = _metrics_oos(eq_gro)
        all_stats["GROWTH (Yahoo snapshot)"]["Avg Annual Turnover"] = _turnover_oos(turnover_gro)

    if eq_ew is not None and len(eq_ew) >= 2:
        all_stats["EW (All Stocks)"] = _metrics_oos(eq_ew)
        all_stats["EW (All Stocks)"]["Avg Annual Turnover"] = _turnover_oos(turnover_ew)

    if eq_rnd is not None and len(eq_rnd) >= 2:
        all_stats["RANDOM (single run)"] = _metrics_oos(eq_rnd)
        all_stats["RANDOM (single run)"]["Avg Annual Turnover"] = _turnover_oos(turnover_rnd)

    if eq_val_sec is not None and len(eq_val_sec) >= 2:
        all_stats["VALUE (SEC as-of)"] = _metrics_oos(eq_val_sec)
        all_stats["VALUE (SEC as-of)"]["Avg Annual Turnover"] = _turnover_oos(turnover_val_sec)

    for strat in sorted(eq_ml.keys()):
        eq = eq_ml[strat]
        if eq is None or len(eq) < 2:
            continue
        st = _metrics_oos(eq)
        tr = ml_backtest_turns.get(strat, pd.Series(dtype=float))
        st["Avg Annual Turnover"] = _turnover_oos(tr)
        all_stats[strat] = st

    if len(all_stats) == 0:
        print("⚠️ No strategies available to compute performance table.")
    else:
        stats_df = pd.DataFrame(all_stats).T.sort_values("Sharpe", ascending=False)

        # Make CSV byte-identical across runs
        stats_df = stats_df.round(6)

        csv_path = Path(out_dir) / f"performance_table_{ts}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(csv_path, index=True, float_format="%.6f")

        print(f"✓ Performance table saved: {csv_path}")
        print("\n" + "=" * 80)
        print("PERFORMANCE TABLE (OOS, rebased @ split; sorted by Sharpe)")
        print("=" * 80)
        print(stats_df.to_string())
        print("=" * 80)

    # ------------------------------------------------------------------
    # 9) Visuals (FAIR OOS)
    # ------------------------------------------------------------------
    print("\n[9/9] Generating visuals...")
    viz = Visualizer(out_dir)
    ml_viz = MLVisualizer(out_dir)

    curves_vg_oos = _prep_curves_oos(
        {
            "VALUE (Yahoo)": eq_val,
            "GROWTH (Yahoo)": eq_gro,
            "VALUE (SEC)": eq_val_sec,
            "S&P 500 (Benchmark)": eq_bench,
            "RANDOM Portfolio": eq_rnd,
        },
        split_dt,
    )

    if len(curves_vg_oos) >= 2:
        viz.plot_equity_curves(
            curves_vg_oos,
            f"equity_VALUE_vs_GROWTH_OOS_{ts}.png",
            "Value vs Growth — Yahoo vs SEC Value (Out-of-sample, rebased @ split)",
        )
        print("✓ Value vs Growth OOS plot saved")

    best_y = pick_best_model_key(ml_results_y, "Yahoo", metric="auc") if ml_results_y else None
    best_s = pick_best_model_key(ml_results_sec, "SEC", metric="auc") if ml_results_sec else None

    curves_best_oos = _prep_curves_oos(
        {
            "EW (All Stocks)": eq_ew,
            (f"Best Yahoo (AUC): {short_model_name(best_y)}" if best_y else "Best Yahoo (AUC)"): (
                eq_ml.get(f"{short_model_name(best_y)} [Yahoo]", None) if best_y else None
            ),
            (f"Best SEC (AUC): {short_model_name(best_s)}" if best_s else "Best SEC (AUC)"): (
                eq_ml.get(f"{short_model_name(best_s)} [SEC]", None) if best_s else None
            ),
        },
        split_dt,
    )

    if len(curves_best_oos) >= 2:
        viz.plot_equity_curves(
            curves_best_oos,
            f"equity_BEST_Y_vs_S_OOS_{ts}.png",
            "Best ML — Yahoo vs SEC (Out-of-sample, rebased @ split)",
        )
        viz.plot_drawdown_curves(
            curves_best_oos,
            f"drawdown_BEST_Y_vs_S_OOS_{ts}.png",
            "Drawdowns — Best ML Yahoo vs Best ML SEC (Out-of-sample)",
        )
        print("✓ Best ML OOS equity + drawdown saved")

    if ml_results_y and ml_results_sec:
        ml_viz.plot_ml_comparison(ml_results_y, ml_results_sec, f"ml_comparison_{ts}.png")

    if ml_results_y:
        ml_viz.plot_ml_metrics_single(ml_results_y, "Yahoo", f"ml_metrics_YAHOO_{ts}.png")
    if ml_results_sec:
        ml_viz.plot_ml_metrics_single(ml_results_sec, "SEC", f"ml_metrics_SEC_{ts}.png")

    if cfg.ENABLE_ROC_AND_CM:
        if ml_results_y:
            ml_viz.plot_roc_curves(y_test_y.values, ml_results_y, "Yahoo", f"roc_YAHOO_{ts}.png")
        if ml_results_sec and y_test_s is not None:
            ml_viz.plot_roc_curves(y_test_s.values, ml_results_sec, "SEC", f"roc_SEC_{ts}.png")

    print("\n✅ DONE ✅")
    print(f"📂 Outputs: {os.path.abspath(out_dir)}")
    print(f"⏱ Total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
