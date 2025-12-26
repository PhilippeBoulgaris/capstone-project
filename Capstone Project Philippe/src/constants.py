# src/constants.py
# -*- coding: utf-8 -*-

# Columns that are not ML features (generic)
NON_FEATURE_COLS = {"date", "ticker", "target", "next_month_return", "sector"}

# Default SEC feature columns after you compute ratios from SEC annual panel
SEC_FEATURE_COLS_DEFAULT = ["pb", "pe", "ps", "ev_ebitda", "roe", "debt_to_equity", "net_margin"]

# Yahoo score columns (already computed in fundamentals_yahoo.py)
YAHOO_SCORE_COLS = ["score_value", "score_growth", "score_quality"]
