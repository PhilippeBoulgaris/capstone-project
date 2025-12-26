"""
Session 13: Live Coding Exercises
Run each section, then code the solution together.
"""

import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# =============================================================================
# EXERCISE 1: Slow Downloads (I/O-bound)
# =============================================================================

def download_stock(ticker):
    """Simulate downloading — takes 1 second"""
    time.sleep(1)
    return f"{ticker}: ${np.random.uniform(100, 500):.2f}"

tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA"]

# Sequential version — run this first
print("=== Exercise 1: Sequential ===")
start = time.time()
for t in tickers:
    print(download_stock(t))
print(f"Time: {time.time() - start:.1f}s")

# CHALLENGE: Make it faster with ThreadPoolExecutor
# print("\n=== Exercise 1: Parallel ===")
# start = time.time()
#
# ... your code here ...
#
# print(f"Time: {time.time() - start:.1f}s")


# =============================================================================
# EXERCISE 2: CPU-Bound Work
# =============================================================================

def slow_sum(n):
    """Sum of squares — pure CPU work"""
    total = 0
    for i in range(n):
        total += i * i
    return total

# Sequential version
print("\n=== Exercise 2: Sequential ===")
start = time.time()
results = [slow_sum(5_000_000) for _ in range(4)]
print(f"Time: {time.time() - start:.1f}s")

# CHALLENGE 1: Try ThreadPoolExecutor — does it help?
# print("\n=== Exercise 2: Threaded ===")
# start = time.time()
#
# ... your code here ...
#
# print(f"Time: {time.time() - start:.1f}s")

# CHALLENGE 2: Try ProcessPoolExecutor
# Note: need wrapper function for ProcessPoolExecutor
# print("\n=== Exercise 2: Multiprocessing ===")
# start = time.time()
#
# ... your code here ...
#
# print(f"Time: {time.time() - start:.1f}s")


# =============================================================================
# EXERCISE 3: Numba
# =============================================================================

# Uncomment if numba is installed:
# from numba import njit

# CHALLENGE: Add @njit decorator
def fast_sum(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

# print("\n=== Exercise 3: Numba ===")
# fast_sum(100)  # Warmup / compile
# start = time.time()
# result = fast_sum(50_000_000)  # 10x more!
# print(f"Time: {time.time() - start:.3f}s")


# =============================================================================
# EXERCISE 4: Monte Carlo Pi
# =============================================================================

def estimate_pi(n):
    inside = 0
    for _ in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y < 1:
            inside += 1
    return 4 * inside / n

print("\n=== Exercise 4: Slow Pi ===")
start = time.time()
pi = estimate_pi(1_000_000)
print(f"π ≈ {pi:.6f}")
print(f"Time: {time.time() - start:.2f}s")

# CHALLENGE: Speed it up with @njit
# print("\n=== Exercise 4: Fast Pi ===")
#
# ... your code here ...


# =============================================================================
# EXERCISE 5: Option Pricing (finance example)
# =============================================================================

def price_option(args):
    """Monte Carlo option pricing"""
    S, K, T, r, sigma, n_sims = args
    total = 0.0
    for _ in range(n_sims):
        Z = np.random.randn()
        ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        total += max(ST - K, 0)
    return np.exp(-r * T) * total / n_sims

# Price one option
print("\n=== Exercise 5: Single Option ===")
start = time.time()
price = price_option((100, 100, 1.0, 0.05, 0.2, 1_000_000))
print(f"Option price: ${price:.2f}")
print(f"Time: {time.time() - start:.2f}s")

# CHALLENGE: Price 8 options with different strikes in parallel
# strikes = [80, 85, 90, 95, 100, 105, 110, 115]
# ... your code here ...


# =============================================================================
# EXERCISE 6: Numba Parallel (prange)
# =============================================================================

# from numba import njit, prange

# CHALLENGE: Make estimate_pi use all cores with prange
# Hint: @njit(parallel=True) and change range() to prange()

# @njit(parallel=True)
# def estimate_pi_parallel(n):
#     ... your code here ...


# =============================================================================
# BONUS: Max out all cores (watch in btop)
# =============================================================================

# def burn_cpu(x):
#     total = 0
#     for i in range(50_000_000):
#         total += i * i % 1000
#     return total
#
# print("\n=== Bonus: Watch btop ===")
# with ProcessPoolExecutor(max_workers=4) as pool:
#     list(pool.map(burn_cpu, range(4)))
# print("Done!")
