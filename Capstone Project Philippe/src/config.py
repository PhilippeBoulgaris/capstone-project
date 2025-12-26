# src/config.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List, Optional, Tuple


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

    # SEC (gardé déjà dans config même si on ne l’utilise pas encore)
    SEC_USER_AGENT: str = "Philippe Boulgaris boulgaris02@gmail.com - Capstone project"
    SEC_CACHE_DIR: str = "./sec_cache"
    SEC_MIN_INTERVAL_SEC: float = 0.2
    SEC_TIMEOUT_SEC: int = 30
    SEC_MAX_RETRIES: int = 8
    SEC_LAG_DAYS: int = 90

    # Visuals extras (pour plus tard)
    ENABLE_ROC_AND_CM: bool = False

    SUB_PERIODS: Optional[List[Tuple[str, str, str]]] = None

    def __post_init__(self):
        if self.SUB_PERIODS is None:
            self.SUB_PERIODS = [
                ("2010-01-01", "2015-12-31", "Period 1: 2010-2015"),
                ("2016-01-01", "2020-12-31", "Period 2: 2016-2020"),
                ("2021-01-01", "2024-12-31", "Period 3: 2021-2024"),
            ]
