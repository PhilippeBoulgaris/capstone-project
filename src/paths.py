# src/paths.py
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class Paths:
    # dossier de sortie dans la racine du projet
    base_out_dir_name: str = "backtest_fundamentals_only_results"

    def __post_init__(self):
        # .../capstone_project/src/paths.py -> .../capstone_project
        project_root = Path(__file__).resolve().parents[1]
        self.base_out_dir = str(project_root / self.base_out_dir_name)

    def ensure_out_dir(self) -> str:
        os.makedirs(self.base_out_dir, exist_ok=True)
        return self.base_out_dir

    @staticmethod
    def timestamp() -> str:
        return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
