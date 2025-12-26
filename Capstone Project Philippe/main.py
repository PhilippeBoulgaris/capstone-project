# main.py
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd

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
    build_top_percentile_universe_from_scores,
    build_random_weights_each_month,
)
from src.constants import NON_FEATURE_COLS, SEC_FEATURE_COLS_DEFAULT
from src.validation import numeric_feature_cols
from src.sec_fundamentals import SECFundamentalsBuilder
from src.sec_style import compute_sec_value_score, build_sec_style_weights  # growth retiré


def _slice_oos_and_rebase(s: pd.Series | None, split_dt: pd.Timestamp) -> pd.Series | None:
    """Keep only OOS window [split_dt, ...] and rebase to 1 at split."""
    if s is None:
        return None
    s2 = s.dropna().copy()
    if len(s2) < 2:
        return None
    s2 = s2.loc[s2.index >= split_dt]
    if len(s2) < 2:
        return None
    base = float(s2.iloc[0])
    if (not np.isfinite(base)) or base == 0:
        return None
    return s2 / base


def _prep_curves_oos(curves: dict[str, pd.Series | None], split_dt: pd.Timestamp) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for k, v in curves.items():
        vv = _slice_oos_and_rebase(v, split_dt)
        if vv is not None and len(vv) >= 2:
            out[k] = vv
    return out


def main():
    t0 = time.perf_counter()
    cfg = Config()

    if "@" not in cfg.SEC_USER_AGENT:
        raise RuntimeError("Please set Config.SEC_USER_AGENT with your email (SEC requirement).")

    tickers = list(dict.fromkeys(SP500_TICKERS))

    print("=" * 80)
    print("🤖 FULL PIPELINE: Yahoo vs SEC fundamentals (Value/Growth + ML)")
    print("=" * 80)
    print(f"Tickers: {len(tickers)} | Start: {cfg.START} | Interval: {cfg.INTERVAL}")
    print(f"Train/Test split: {cfg.TRAIN_TEST_SPLIT} | Benchmark: {cfg.BENCHMARK}")
    print("=" * 80)

    # output
    paths = Paths()
    out_dir = paths.ensure_out_dir()
    ts = paths.timestamp()

    # ---------------------------
    # 1) Prices / returns
    # ---------------------------
    loader = DataLoader(cfg)
    px = loader.download_prices(tickers + [cfg.BENCHMARK])
    rets = loader.compute_returns(px)

    if cfg.BENCHMARK not in rets.columns:
        raise RuntimeError(f"Benchmark {cfg.BENCHMARK} not found in downloaded data.")

    bench_rets = rets[cfg.BENCHMARK].copy()
    asset_cols = [t for t in tickers if t in rets.columns]
    asset_rets = rets[asset_cols].copy()
    px_assets = px[asset_cols].reindex(asset_rets.index)

    print(f"\n✓ Asset universe with data: {len(asset_cols)}")
    print(f"✓ Returns: {asset_rets.shape[0]} months × {asset_rets.shape[1]} tickers")

    # ---------------------------
    # 2) Yahoo fundamentals + scores
    # ---------------------------
    print("\n[2/9] Fetching Yahoo fundamentals (snapshot)...")
    fe = FundamentalFeatureEngineering()
    fund_list = []
    for i, t in enumerate(asset_cols, 1):
        fund_list.append(fe.fetch_yahoo_fundamentals(t))
        if i % 20 == 0:
            print(f"    Processed {i}/{len(asset_cols)} tickers...")

    fund_yahoo = pd.DataFrame(fund_list).set_index("ticker")
    for c in fund_yahoo.columns:
        if c != "sector":
            fund_yahoo[c] = pd.to_numeric(fund_yahoo[c], errors="coerce")

    print("  Computing Yahoo scores...")
    fund_yahoo["score_value"] = fe.compute_value_score(fund_yahoo)
    fund_yahoo["score_growth"] = fe.compute_growth_score(fund_yahoo)
    fund_yahoo["score_quality"] = fe.compute_quality_score(fund_yahoo)

    val_names = build_top_percentile_universe_from_scores(fund_yahoo["score_value"], cfg.TOP_PERCENTILE)
    gro_names = build_top_percentile_universe_from_scores(fund_yahoo["score_growth"], cfg.TOP_PERCENTILE)

    val_names = [t for t in val_names if t in asset_cols]
    gro_names = [t for t in gro_names if t in asset_cols]

    print(f"✓ Value universe size (Yahoo score): {len(val_names)} | Growth: {len(gro_names)}")

    # ---------------------------
    # 3) SEC annual -> monthly panel -> ratios/features
    # ---------------------------
    sec_panel = pd.DataFrame()
    sec_feature_cols: list[str] = []

    try:
        print("\n[3/9] SEC pipeline: annual FY -> monthly panel -> ratios...")
        sec_builder = SECFundamentalsBuilder(cfg)
        sec_annual = sec_builder.build_annual_sec_table(asset_cols, start_year=2010, end_year=2024)

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

            for c in SEC_FEATURE_COLS_DEFAULT:
                if c in sec_panel.columns:
                    sec_panel.loc[np.isinf(sec_panel[c]), c] = np.nan

            sec_feature_cols = [c for c in SEC_FEATURE_COLS_DEFAULT if c in sec_panel.columns]
            print(f"✓ SEC panel rows: {len(sec_panel):,} | SEC features: {sec_feature_cols}")
        else:
            print("⚠️ SEC annual table empty -> skipping SEC.")
    except Exception as e:
        print(f"⚠️ SEC pipeline error -> skipping SEC. Error: {e}")

    # ---------------------------
    # 4) SEC Value portfolio ONLY (monthly, as-of)
    # ---------------------------
    eq_val_sec = None
    if sec_panel is not None and len(sec_panel) > 0:
        needed = [c for c in ["pb", "pe", "ps", "ev_ebitda"] if c in sec_panel.columns]
        if len(needed) >= 2:
            print("\n[4/9] Building SEC VALUE score (monthly cross-section)...")
            sec_scores = sec_panel[["date", "ticker"] + needed].copy()

            val_chunks = []
            for d, cs in sec_scores.groupby("date"):
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
            logret_val_sec, _ = perf_tmp.backtest_portfolio(w_val_sec, asset_rets, costs_bps=cfg.COSTS_BPS)
            eq_val_sec = to_equity(logret_val_sec)

            print(f"✓ SEC Value months: {eq_val_sec.index.nunique() if eq_val_sec is not None else 0}")
        else:
            print("⚠️ SEC Value: not enough SEC columns available for scoring.")

    # ---------------------------
    # 5) Build ML datasets (Yahoo + SEC)
    # ---------------------------
    print("\n[5/9] Building ML datasets...")
    builder_y = MLDatasetBuilder(cfg)
    df_ml_yahoo = builder_y.build_yahoo_dataset(asset_rets, bench_rets, fund_yahoo.reset_index())

    df_ml_sec = pd.DataFrame()
    if len(sec_panel) > 0 and len(sec_feature_cols) > 0:
        builder_s = MLDatasetBuilderPanel(cfg)
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

    # DIAG
    if len(df_ml_sec) > 0:
        print("\n[DIAG] SEC ML coverage")
        print("  SEC ML date range:", df_ml_sec["date"].min(), "->", df_ml_sec["date"].max())
        print("  SEC train rows (< split):", int((df_ml_sec["date"] < cfg.TRAIN_TEST_SPLIT).sum()))
        print("  SEC test rows (>= split):", int((df_ml_sec["date"] >= cfg.TRAIN_TEST_SPLIT).sum()))
    else:
        print("\n[DIAG] SEC ML coverage: df_ml_sec is empty")

    # ---------------------------
    # 6) ML training (Yahoo + SEC)
    # ---------------------------
    print("\n[6/9] ML training (Yahoo fundamentals)...")
    trainer = MLTrainer(cfg)

    feature_cols_y = numeric_feature_cols(df_ml_yahoo, exclude=list(NON_FEATURE_COLS))
    train_mask_y = df_ml_yahoo["date"] < cfg.TRAIN_TEST_SPLIT

    X_train_y = df_ml_yahoo.loc[train_mask_y, feature_cols_y]
    y_train_y = df_ml_yahoo.loc[train_mask_y, "target"]
    X_test_y = df_ml_yahoo.loc[~train_mask_y, feature_cols_y]
    y_test_y = df_ml_yahoo.loc[~train_mask_y, "target"]
    meta_test_y = df_ml_yahoo.loc[~train_mask_y, ["date", "ticker", "next_month_return"]].copy()

    ml_results_y = trainer.train_universe("Yahoo", X_train_y, y_train_y, X_test_y, y_test_y)

    ml_results_sec = {}
    meta_test_s = pd.DataFrame()
    y_test_s = None

    if len(df_ml_sec) > 0:
        print("\n[6bis/9] ML training (SEC fundamentals)...")
        trainer_s = MLTrainer(cfg)

        feature_cols_s = numeric_feature_cols(df_ml_sec, exclude=list(NON_FEATURE_COLS))
        train_mask_s = df_ml_sec["date"] < cfg.TRAIN_TEST_SPLIT

        X_train_s = df_ml_sec.loc[train_mask_s, feature_cols_s]
        y_train_s = df_ml_sec.loc[train_mask_s, "target"]
        X_test_s = df_ml_sec.loc[~train_mask_s, feature_cols_s]
        y_test_s = df_ml_sec.loc[~train_mask_s, "target"]
        meta_test_s = df_ml_sec.loc[~train_mask_s, ["date", "ticker", "next_month_return"]].copy()

        ml_results_sec = trainer_s.train_universe("SEC", X_train_s, y_train_s, X_test_s, y_test_s)

    # ---------------------------
    # 7) Backtests
    # ---------------------------
    print("\n[7/9] Backtesting Value/Growth + ML portfolios...")
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

    w_rnd = build_random_weights_each_month(asset_rets.index[:-1], all_names, cfg.TOP_PERCENTILE, seed=cfg.RANDOM_STATE)
    logret_rnd, turnover_rnd = perf.backtest_portfolio(w_rnd, asset_rets, costs_bps=cfg.COSTS_BPS)
    eq_rnd = to_equity(logret_rnd)

    eq_bench = to_equity(bench_rets.dropna()) if len(bench_rets) else pd.Series(dtype=float)

    ml_backtest_logrets = {}
    ml_backtest_turns = {}

    for name, res in ml_results_y.items():
        w = build_weights_from_meta(meta_test_y, res["probabilities"], cfg.TOP_PERCENTILE)
        lr, tr = perf.backtest_portfolio(w, asset_rets, costs_bps=cfg.COSTS_BPS)
        key = f"{short_model_name(name)} [Yahoo]"
        ml_backtest_logrets[key] = lr
        ml_backtest_turns[key] = tr

    for name, res in ml_results_sec.items():
        if len(meta_test_s) == 0:
            continue
        w = build_weights_from_meta(meta_test_s, res["probabilities"], cfg.TOP_PERCENTILE)
        lr, tr = perf.backtest_portfolio(w, asset_rets, costs_bps=cfg.COSTS_BPS)
        key = f"{short_model_name(name)} [SEC]"
        ml_backtest_logrets[key] = lr
        ml_backtest_turns[key] = tr

    eq_ml = {k: to_equity(v) for k, v in ml_backtest_logrets.items()}

    # ---------------------------
    # 8) Metrics table
    # ---------------------------
    print("\n[8/9] Computing metrics + saving table...")
    all_stats = {
        "VALUE (Yahoo snapshot)": perf.compute_enhanced_metrics(eq_val, bench_rets),
        "GROWTH (Yahoo snapshot)": perf.compute_enhanced_metrics(eq_gro, bench_rets),
        "EW (All Stocks)": perf.compute_enhanced_metrics(eq_ew, bench_rets),
        "RANDOM (single run)": perf.compute_enhanced_metrics(eq_rnd, bench_rets),
    }

    if eq_val_sec is not None and len(eq_val_sec) >= 2:
        all_stats["VALUE (SEC as-of)"] = perf.compute_enhanced_metrics(eq_val_sec, bench_rets)

    all_stats["VALUE (Yahoo snapshot)"]["Avg Annual Turnover"] = float(turnover_val.mean() * 12) if len(turnover_val) else 0.0
    all_stats["GROWTH (Yahoo snapshot)"]["Avg Annual Turnover"] = float(turnover_gro.mean() * 12) if len(turnover_gro) else 0.0
    all_stats["EW (All Stocks)"]["Avg Annual Turnover"] = float(turnover_ew.mean() * 12) if len(turnover_ew) else 0.0
    all_stats["RANDOM (single run)"]["Avg Annual Turnover"] = float(turnover_rnd.mean() * 12) if len(turnover_rnd) else 0.0

    for strat, eq in eq_ml.items():
        st = perf.compute_enhanced_metrics(eq, bench_rets)
        tr = ml_backtest_turns.get(strat, pd.Series(dtype=float))
        st["Avg Annual Turnover"] = float(tr.mean() * 12) if len(tr) else 0.0
        all_stats[strat] = st

    stats_df = pd.DataFrame(all_stats).T.sort_values("Sharpe", ascending=False)
    csv_path = os.path.join(out_dir, f"performance_table_{ts}.csv")
    stats_df.to_csv(csv_path, index=True)

    print(f"✓ Performance table saved: {csv_path}")
    print("\n" + "=" * 80)
    print("⚡ PERFORMANCE TABLE (Fundamentals Only)")
    print("=" * 80)
    print(stats_df.to_string())
    print("=" * 80)

    # ---------------------------
    # 9) Visuals (FAIR OOS)
    # ---------------------------
    print("\n[9/9] Generating visuals...")
    viz = Visualizer(out_dir)
    ml_viz = MLVisualizer(out_dir)

    split_dt = pd.to_datetime(cfg.TRAIN_TEST_SPLIT)

    # (0) Keep "ALL" plot (optional / appendix)
    curves_all = {
        "VALUE (Yahoo snapshot)": eq_val,
        "GROWTH (Yahoo snapshot)": eq_gro,
        "EW (All Stocks)": eq_ew,
        "S&P 500 (Benchmark)": eq_bench,
        "RANDOM (single run)": eq_rnd,
    }
    if eq_val_sec is not None:
        curves_all["VALUE (SEC as-of)"] = eq_val_sec
    curves_all.update(eq_ml)

    viz.plot_equity_curves(curves_all, f"equity_ALL_{ts}.png",
                           "Equity Curves — All Strategies (Fundamentals Only)")
    print("✓ Equity curves (ALL) saved")

    # (A) Value vs Growth — OOS + rebased @ split (clean)
    curves_vg_oos = _prep_curves_oos({
        "VALUE (Yahoo)": eq_val,
        "GROWTH (Yahoo)": eq_gro,
        "VALUE (SEC)": eq_val_sec,
        "S&P 500 (Benchmark)": eq_bench,
        "RANDOM Portfolio": eq_rnd,
    }, split_dt)

    if len(curves_vg_oos) >= 2:
        viz.plot_equity_curves(
            curves_vg_oos,
            f"equity_VALUE_vs_GROWTH_OOS_{ts}.png",
            "Value vs Growth — Yahoo vs SEC Value (Out-of-sample, rebased @ split)"
        )
        print("✓ Value vs Growth OOS plot saved")

    # (B) Best ML — Yahoo vs SEC (OOS + rebased @ split)
    best_y = pick_best_model_key(ml_results_y, "Yahoo", metric="auc") if ml_results_y else None
    best_s = pick_best_model_key(ml_results_sec, "SEC", metric="auc") if ml_results_sec else None

    curves_best_oos = _prep_curves_oos({
        "EW (All Stocks)": eq_ew,
        (f"Best Yahoo (AUC): {short_model_name(best_y)}" if best_y else "Best Yahoo (AUC)"): (
            eq_ml.get(f"{short_model_name(best_y)} [Yahoo]", None) if best_y else None
        ),
        (f"Best SEC (AUC): {short_model_name(best_s)}" if best_s else "Best SEC (AUC)"): (
            eq_ml.get(f"{short_model_name(best_s)} [SEC]", None) if best_s else None
        ),
    }, split_dt)

    if len(curves_best_oos) >= 2:
        viz.plot_equity_curves(
            curves_best_oos,
            f"equity_BEST_Y_vs_S_OOS_{ts}.png",
            "Best ML — Yahoo vs SEC (Out-of-sample, rebased @ split)"
        )
        viz.plot_drawdown_curves(
            curves_best_oos,
            f"drawdown_BEST_Y_vs_S_OOS_{ts}.png",
            "Drawdowns — Best ML Yahoo vs Best ML SEC (Out-of-sample)"
        )
        print("✓ Best ML OOS equity + drawdown saved")

    # (C) ML metrics plots (unchanged)
    if ml_results_y and ml_results_sec:
        ml_viz.plot_ml_comparison(ml_results_y, ml_results_sec, f"ml_comparison_{ts}.png")

    if ml_results_y:
        ml_viz.plot_ml_metrics_single(ml_results_y, "Yahoo", f"ml_metrics_YAHOO_{ts}.png")
    if ml_results_sec:
        ml_viz.plot_ml_metrics_single(ml_results_sec, "SEC", f"ml_metrics_SEC_{ts}.png")

    # Optional ROC + Confusion (still OOS, based on y_test_*)
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
