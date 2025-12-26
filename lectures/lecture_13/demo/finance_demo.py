#!/usr/bin/env python3
"""
Finance Parallelism Demo - Monte Carlo VaR

This demonstrates how parallelism speeds up financial simulations.
Watch btop/htop to see your cores working!

Usage:
    python finance_demo.py
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import time

# Portfolio setup: 10 assets with random weights
np.random.seed(42)
N_ASSETS = 10
WEIGHTS = np.random.dirichlet(np.ones(N_ASSETS))  # Random weights summing to 1
MU = np.random.uniform(0.0001, 0.001, N_ASSETS)   # Expected daily returns
SIGMA = np.random.uniform(0.01, 0.03, N_ASSETS)   # Daily volatilities
PORTFOLIO_VALUE = 1_000_000  # $1M portfolio


def simulate_returns_chunk(n_sims):
    """Simulate portfolio returns for a chunk of simulations"""
    returns = np.empty(n_sims)
    for i in range(n_sims):
        # Simulate daily returns for each asset
        asset_returns = np.random.randn(N_ASSETS) * SIGMA + MU
        # Portfolio return = weighted sum
        returns[i] = np.dot(WEIGHTS, asset_returns)
    return returns


def calculate_var_sequential(n_sims):
    """Calculate VaR using a single core"""
    returns = simulate_returns_chunk(n_sims)
    returns.sort()
    var_95 = returns[int(n_sims * 0.05)]
    return var_95


def calculate_var_parallel(n_sims, n_workers):
    """Calculate VaR using multiple cores"""
    chunk_size = n_sims // n_workers
    chunks = [chunk_size] * n_workers

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(simulate_returns_chunk, chunks))

    # Combine all simulations
    all_returns = np.concatenate(results)
    all_returns.sort()
    var_95 = all_returns[int(len(all_returns) * 0.05)]
    return var_95


def main():
    n_sims = 5_000_000  # 5 million simulations
    n_cores = multiprocessing.cpu_count()
    n_workers = min(4, n_cores)

    print("\n" + "="*60)
    print(" MONTE CARLO VALUE AT RISK (VaR) DEMO")
    print("="*60)
    print(f"\nPortfolio: ${PORTFOLIO_VALUE:,} across {N_ASSETS} assets")
    print(f"Simulations: {n_sims:,}")
    print(f"Your system has {n_cores} CPU cores")
    print("\nTIP: Open btop/htop to watch CPU usage!")

    # Sequential
    input("\nPress Enter to run SEQUENTIAL (1 core)...")
    print("\nRunning sequential simulation...")
    start = time.time()
    var_seq = calculate_var_sequential(n_sims)
    time_seq = time.time() - start
    loss_seq = PORTFOLIO_VALUE * abs(var_seq)
    print(f"  Time: {time_seq:.1f}s")
    print(f"  95% VaR: {var_seq:.4%} = ${loss_seq:,.0f} potential daily loss")

    # Parallel
    input(f"\nPress Enter to run PARALLEL ({n_workers} cores)...")
    print(f"\nRunning parallel simulation across {n_workers} cores...")
    start = time.time()
    var_par = calculate_var_parallel(n_sims, n_workers)
    time_par = time.time() - start
    loss_par = PORTFOLIO_VALUE * abs(var_par)
    print(f"  Time: {time_par:.1f}s")
    print(f"  95% VaR: {var_par:.4%} = ${loss_par:,.0f} potential daily loss")

    # Summary
    speedup = time_seq / time_par
    print("\n" + "="*60)
    print(" RESULTS")
    print("="*60)
    print(f"\n  Sequential: {time_seq:.1f}s")
    print(f"  Parallel:   {time_par:.1f}s")
    print(f"  Speedup:    {speedup:.1f}x faster!")
    print(f"\n  Both methods found similar VaR (as expected)")
    print("\n" + "="*60)
    print("\nThis is how banks calculate risk for regulatory requirements!")
    print("Basel III requires banks to run these simulations daily.\n")


if __name__ == "__main__":
    main()
