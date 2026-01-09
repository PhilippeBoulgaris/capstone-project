# src/dataset_builders.py
# -*- coding: utf-8 -*-

from typing import List, Optional
import numpy as np
import pandas as pd

from .config import Config


def _to_month_end(ts) -> pd.Timestamp:
    """Normalize any timestamp-like to month-end (tz-naive)."""
    t = pd.to_datetime(ts, errors="coerce")
    if pd.isna(t):
        return pd.NaT
    t = pd.Timestamp(t).tz_localize(None)
    return t.to_period("M").to_timestamp("M")


class MLDatasetBuilder:
    """Yahoo: static fundamentals snapshot -> monthly panel by replication."""
    def __init__(self, config: Config):
        self.config = config

    def build_yahoo_dataset(
        self,
        asset_rets: pd.DataFrame,
        bench_rets: pd.Series,
        fund_data: pd.DataFrame
    ) -> pd.DataFrame:
        print("\n[5.a/9] Building ML dataset (Yahoo fundamentals only)...")

        fund_static = fund_data.copy()
        if "ticker" not in fund_static.columns:
            fund_static = fund_static.reset_index().rename(columns={"index": "ticker"})

        # ✅ Ensure monthly calendar is month-end and tz-naive
        asset_rets = asset_rets.copy()
        asset_rets.index = pd.to_datetime(asset_rets.index).tz_localize(None).to_period("M").to_timestamp("M")
        asset_rets = asset_rets.sort_index()

        bench_rets = pd.Series(bench_rets, dtype=float).copy()
        bench_rets.index = pd.to_datetime(bench_rets.index).tz_localize(None).to_period("M").to_timestamp("M")
        bench_rets = bench_rets.sort_index()

        idx = asset_rets.index
        next_map = {idx[i]: idx[i + 1] for i in range(len(idx) - 1)}

        rows = []
        num_cols = [c for c in fund_static.columns if c not in ["ticker", "sector"]]

        # Decision month = d (month-end), target uses returns at nd (next month-end)
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
                # ✅ Do NOT fill missing benchmark with 0.0 (creates artificial alpha)
                if nd not in bench_rets.index:
                    continue
                bench_ret = float(bench_rets.loc[nd])

                row = {
                    "date": d,
                    "ticker": t,
                    "target": 1 if stock_ret > bench_ret else 0,
                    "next_month_return": stock_ret,
                }
                for c in num_cols:
                    row[c] = pd.to_numeric(fr.get(c, np.nan), errors="coerce")

                rows.append(row)

        df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
        if df.empty:
            print("⚠️ Yahoo ML dataset empty.")
            return df

        # ✅ Normalize date column to month-end
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
        df = df.dropna(subset=["date", "ticker"]).copy()

        feat_cols = [c for c in df.columns if c not in ["date", "ticker", "target", "next_month_return"]]

        # Soft filter ONLY (no global imputation -> avoid leakage; impute in sklearn pipeline on train only)
        if feat_cols:
            keep_min = max(1, int(np.ceil(0.5 * len(feat_cols))))
            before = len(df)
            df = df[df[feat_cols].notna().sum(axis=1) >= keep_min].copy()
            after = len(df)
            print(f"  Soft filter (>= {keep_min} non-NaN feats): {before:,} → {after:,}")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        print(f"✓ Yahoo ML dataset: {len(df):,} samples over {df['date'].nunique():,} months | features={len(feat_cols)}")
        return df


class MLDatasetBuilderPanel:
    """SEC: as-of monthly panel -> supervised dataset (no look-ahead by design)."""
    def __init__(self, config: Config):
        self.config = config

    def build_dataset_from_panel(
        self,
        asset_rets: pd.DataFrame,
        bench_rets: pd.Series,
        feature_panel: pd.DataFrame,
        feature_cols: List[str],
        min_non_nan_features: Optional[int] = None,
        impute: bool = False,  # ✅ default False to avoid leakage
        verbose: bool = True,
    ) -> pd.DataFrame:
        print("\n[5.abis/9] Building ML dataset (SEC fundamentals panel)...")

        if feature_panel is None or feature_panel.empty:
            print("⚠️ SEC feature_panel empty -> SEC ML dataset empty.")
            return pd.DataFrame()

        # Ensure monthly calendar is month-end and tz-naive for returns + benchmark
        asset_rets = asset_rets.copy()
        asset_rets.index = pd.to_datetime(asset_rets.index).tz_localize(None).to_period("M").to_timestamp("M")
        asset_rets = asset_rets.sort_index()

        bench_rets = pd.Series(bench_rets, dtype=float).copy()
        bench_rets.index = pd.to_datetime(bench_rets.index).tz_localize(None).to_period("M").to_timestamp("M")
        bench_rets = bench_rets.sort_index()

        p = feature_panel.copy()
        p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
        p["ticker"] = p["ticker"].astype(str)
        p = p.dropna(subset=["date", "ticker"])

        # Keep only months present in returns index
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

            #  Do NOT fill missing benchmark with 0.0
            if nd not in bench_rets.index:
                continue
            bench_ret = float(bench_rets.loc[nd])
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

        #  Avoid global imputation by default; impute in sklearn pipeline (train only)
        if impute:
            med = df[feature_cols].median(numeric_only=True)
            df[feature_cols] = df[feature_cols].fillna(med)

        df = df.dropna(subset=["date", "ticker", "target", "next_month_return"]).copy()
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        print(f"✓ SEC panel ML dataset: {len(df):,} samples over {df['date'].nunique():,} months")
        return df
