import numpy as np
import pandas as pd
from src.performance import to_equity, compute_drawdown

def test_to_equity_basic():
    # 0% puis +10% puis -10% en log-returns (approx)
    logret = pd.Series([0.0, np.log(1.1), np.log(0.9)])
    eq = to_equity(logret)
    assert len(eq) == 3
    assert eq.iloc[0] == 1.0  # exp(0)
    assert eq.iloc[-1] > 0.0

def test_compute_drawdown_monotonic_up():
    eq = pd.Series([1.0, 1.1, 1.2, 1.3])
    dd = compute_drawdown(eq)
    assert (dd <= 1e-12).all()   # drawdown jamais positif
    assert abs(dd.min()) < 1e-12 # jamais de drawdown

def test_compute_drawdown_has_drop():
    eq = pd.Series([1.0, 1.2, 1.1, 1.3])
    dd = compute_drawdown(eq)
    assert dd.min() < 0.0
