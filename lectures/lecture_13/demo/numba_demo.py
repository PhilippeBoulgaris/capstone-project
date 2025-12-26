#!/usr/bin/env python3
"""
Numba Finance Demo - JIT Compilation for Quant Work

Demonstrates how @njit makes financial simulations 50-150x faster.

Usage:
    pip install numba numpy  # if not installed
    python numba_demo.py
"""

import time
import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not installed. Run: pip install numba")
    print("Showing pure Python version only.\n")


# =============================================================================
# Example 1: Monte Carlo Option Pricing
# =============================================================================

def option_price_python(S, K, T, r, sigma, n_sims):
    """
    Price a European call option using Monte Carlo.
    Pure Python version (SLOW).

    S: Current stock price
    K: Strike price
    T: Time to expiry (years)
    r: Risk-free rate
    sigma: Volatility
    """
    total_payoff = 0.0
    for _ in range(n_sims):
        # Simulate stock price at expiry
        Z = np.random.randn()
        ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        # Call option payoff
        total_payoff += max(ST - K, 0)

    # Discounted expected payoff
    return np.exp(-r * T) * total_payoff / n_sims


if NUMBA_AVAILABLE:
    @njit
    def option_price_numba(S, K, T, r, sigma, n_sims):
        """Same function, but with @njit - 100x faster!"""
        total_payoff = 0.0
        for _ in range(n_sims):
            Z = np.random.randn()
            ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
            total_payoff += max(ST - K, 0)
        return np.exp(-r * T) * total_payoff / n_sims


# =============================================================================
# Example 2: Portfolio VaR Simulation
# =============================================================================

def var_python(n_sims, weights, mu, sigma):
    """
    Calculate 95% Value at Risk using Monte Carlo.
    Pure Python version (SLOW).
    """
    n_assets = len(weights)
    returns = np.empty(n_sims)

    for i in range(n_sims):
        # Simulate returns for each asset
        asset_returns = np.random.randn(n_assets) * sigma + mu
        # Portfolio return
        returns[i] = np.dot(weights, asset_returns)

    # 5th percentile = 95% VaR
    returns.sort()
    return returns[int(n_sims * 0.05)]


if NUMBA_AVAILABLE:
    @njit
    def var_numba(n_sims, weights, mu, sigma):
        """Same VaR calculation with @njit"""
        n_assets = len(weights)
        returns = np.empty(n_sims)

        for i in range(n_sims):
            asset_returns = np.random.randn(n_assets) * sigma + mu
            returns[i] = np.dot(weights, asset_returns)

        returns.sort()
        return returns[int(n_sims * 0.05)]

    @njit(parallel=True)
    def var_parallel(n_sims, weights, mu, sigma):
        """VaR with parallel=True - uses all CPU cores!"""
        n_assets = len(weights)
        returns = np.empty(n_sims)

        for i in prange(n_sims):  # prange = parallel range
            asset_returns = np.random.randn(n_assets) * sigma + mu
            returns[i] = np.dot(weights, asset_returns)

        returns.sort()
        return returns[int(n_sims * 0.05)]


def time_function(func, *args, warmup=True):
    """Time a function, with warmup for JIT compilation"""
    if warmup and NUMBA_AVAILABLE:
        # Warmup call with small input
        try:
            if 'n_sims' in func.__code__.co_varnames:
                func(*args[:1], *args[1:])
            else:
                func(*args)
        except:
            pass

    start = time.time()
    result = func(*args)
    elapsed = time.time() - start
    return elapsed, result


def main():
    print("=" * 60)
    print(" NUMBA FINANCE DEMO - JIT Compilation")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Example 1: Option Pricing
    # -----------------------------------------------------------------
    print("\n[Example 1: Monte Carlo Option Pricing]")
    print("-" * 45)
    print("Pricing AAPL $155 call, 3 months to expiry")
    print("Using 1 million Monte Carlo simulations\n")

    # Option parameters
    S, K, T, r, sigma = 150, 155, 0.25, 0.05, 0.30
    n_sims = 1_000_000

    # Python version
    print("Running pure Python...")
    t_python, price_python = time_function(
        option_price_python, S, K, T, r, sigma, n_sims, warmup=False
    )
    print(f"  Pure Python: {t_python:.2f}s  (price: ${price_python:.2f})")

    if NUMBA_AVAILABLE:
        # Numba version
        print("Running Numba @njit...")
        t_numba, price_numba = time_function(
            option_price_numba, S, K, T, r, sigma, n_sims
        )
        print(f"  Numba @njit: {t_numba:.2f}s  (price: ${price_numba:.2f})")
        print(f"  Speedup:     {t_python/t_numba:.0f}x faster!")

    # -----------------------------------------------------------------
    # Example 2: Portfolio VaR
    # -----------------------------------------------------------------
    print("\n[Example 2: Portfolio Value at Risk]")
    print("-" * 45)
    print("$1M portfolio, 10 assets, 5 million simulations\n")

    # Portfolio setup
    np.random.seed(42)
    n_assets = 10
    weights = np.random.dirichlet(np.ones(n_assets))
    mu = np.random.uniform(0.0001, 0.001, n_assets)
    sigma = np.random.uniform(0.01, 0.03, n_assets)
    n_sims = 5_000_000
    portfolio_value = 1_000_000

    # Python version
    print("Running pure Python (this will take a while)...")
    t_python, var_py = time_function(
        var_python, n_sims, weights, mu, sigma, warmup=False
    )
    loss_py = portfolio_value * abs(var_py)
    print(f"  Pure Python: {t_python:.1f}s  (95% VaR: ${loss_py:,.0f})")

    if NUMBA_AVAILABLE:
        # Numba version
        print("Running Numba @njit...")
        t_numba, var_nb = time_function(
            var_numba, n_sims, weights, mu, sigma
        )
        loss_nb = portfolio_value * abs(var_nb)
        print(f"  Numba @njit: {t_numba:.2f}s  (95% VaR: ${loss_nb:,.0f})")
        print(f"  Speedup:     {t_python/t_numba:.0f}x faster!")

        # Parallel version
        print("\nRunning Numba parallel=True...")
        t_parallel, var_par = time_function(
            var_parallel, n_sims, weights, mu, sigma
        )
        loss_par = portfolio_value * abs(var_par)
        print(f"  Parallel:    {t_parallel:.2f}s  (95% VaR: ${loss_par:,.0f})")
        print(f"  vs Python:   {t_python/t_parallel:.0f}x faster!")
        print(f"  vs Numba:    {t_numba/t_parallel:.1f}x faster!")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" KEY TAKEAWAYS")
    print("=" * 60)
    print("""
1. @njit decorator: Same Python code, 50-150x faster
2. Works great for: loops, NumPy, financial simulations
3. parallel=True + prange: automatic multi-core execution
4. First call compiles (slow), subsequent calls are fast

Use cases in finance:
- Option pricing (Monte Carlo)
- Risk calculations (VaR, CVaR)
- Portfolio optimization
- Backtesting trading strategies
""")


if __name__ == "__main__":
    main()
