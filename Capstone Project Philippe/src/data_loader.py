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
