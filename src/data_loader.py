# src/data_loader.py
# -*- coding: utf-8 -*-

from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from .config import Config


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

        # Extract Close
        if isinstance(data.columns, pd.MultiIndex):
            px = data["Close"].copy()
        else:
            px = data.copy()

        # Clean index + sort
        px = px.dropna(how="all")
        px.index = pd.to_datetime(px.index).tz_localize(None)
        px = px.sort_index()

        # ✅ Normalize to a consistent month-end calendar (critical for timeline coherence)
        # This avoids subtle misalignment between Yahoo monthly timestamps, SEC panel dates, and split dates.
        px = px.resample("M").last().dropna(how="all")

        print(f"✓ Prices (month-end): {px.shape[0]} months × {px.shape[1]} tickers")
        if len(px.index) > 0:
            print(f"  First month-end: {px.index.min().date()} | Last month-end: {px.index.max().date()}")

        return px

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        # Assume prices are already month-end sampled and sorted
        prices = prices.sort_index()
        rets = np.log(prices / prices.shift(1)).dropna(how="all")

        print(f"✓ Log returns computed over {len(rets)} months")
        if len(rets.index) > 0:
            print(f"  Returns span: {rets.index.min().date()} → {rets.index.max().date()}")

        return rets
