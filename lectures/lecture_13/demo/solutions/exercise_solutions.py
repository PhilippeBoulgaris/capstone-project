"""
Parallel Python Workshop - Exercise Solutions
==============================================

This file contains solutions to all exercises from the workshop notebooks.
"""

import numpy as np
import pandas as pd
import time
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from scipy.stats import norm
from itertools import product

n_cores = os.cpu_count()

# =============================================================================
# NOTEBOOK 01: MOTIVATION - EXERCISE SOLUTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Exercise 1: European Put Options
# -----------------------------------------------------------------------------

def price_european_put_sequential(S0, K, T, r, sigma, n_simulations):
    """
    Price a European put option using Monte Carlo simulation.

    The payoff of a put option is max(K - S_T, 0)
    """
    # Generate random normal samples
    Z = np.random.standard_normal(n_simulations)

    # Simulate terminal stock prices
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Calculate PUT payoffs (difference from call: K - ST instead of ST - K)
    payoffs = np.maximum(K - ST, 0)

    # Discounted expected payoff
    option_price = np.exp(-r * T) * np.mean(payoffs)

    return option_price


def price_put_option(args):
    """Wrapper for parallel execution."""
    S0, K, T, r, sigma, n_sims = args
    return price_european_put_sequential(S0, K, T, r, sigma, n_sims)


def demo_put_option_pricing():
    """Demonstrate parallel put option pricing."""
    # Parameters
    S0, T, r, sigma = 100, 1.0, 0.05, 0.2
    n_sims = 1_000_000
    strikes = np.linspace(80, 120, 20)

    # Prepare arguments
    args_list = [(S0, K, T, r, sigma, n_sims) for K in strikes]

    # Sequential
    start = time.time()
    seq_prices = [price_put_option(args) for args in args_list]
    seq_time = time.time() - start

    # Parallel
    start = time.time()
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        par_prices = list(executor.map(price_put_option, args_list))
    par_time = time.time() - start

    print("Exercise 1: European Put Options")
    print(f"  Sequential: {seq_time:.2f}s")
    print(f"  Parallel: {par_time:.2f}s")
    print(f"  Speedup: {seq_time/par_time:.2f}x")
    print(f"  Sample prices: K=80: ${par_prices[0]:.4f}, K=100: ${par_prices[10]:.4f}, K=120: ${par_prices[-1]:.4f}")


# -----------------------------------------------------------------------------
# Exercise 2: Varying Volatility
# -----------------------------------------------------------------------------

def price_call_with_vol(args):
    """Price a call option with given volatility."""
    S0, K, T, r, sigma, n_sims = args
    Z = np.random.standard_normal(n_sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoffs)


def demo_volatility_sensitivity():
    """Demonstrate parallel volatility sensitivity analysis."""
    # Parameters
    S0, K, T, r = 100, 100, 1.0, 0.05
    n_sims = 500_000
    volatilities = np.linspace(0.10, 0.50, 20)  # 10% to 50%

    # Prepare arguments
    args_list = [(S0, K, T, r, sigma, n_sims) for sigma in volatilities]

    # Sequential
    start = time.time()
    seq_prices = [price_call_with_vol(args) for args in args_list]
    seq_time = time.time() - start

    # Parallel
    start = time.time()
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        par_prices = list(executor.map(price_call_with_vol, args_list))
    par_time = time.time() - start

    print("\nExercise 2: Volatility Sensitivity")
    print(f"  Sequential: {seq_time:.2f}s")
    print(f"  Parallel: {par_time:.2f}s")
    print(f"  Speedup: {seq_time/par_time:.2f}x")
    print(f"  Price at 10% vol: ${par_prices[0]:.4f}")
    print(f"  Price at 30% vol: ${par_prices[10]:.4f}")
    print(f"  Price at 50% vol: ${par_prices[-1]:.4f}")


# -----------------------------------------------------------------------------
# Exercise 3: Amdahl's Law Calculator
# -----------------------------------------------------------------------------

def amdahl_speedup(p, n):
    """Calculate theoretical speedup using Amdahl's Law."""
    return 1 / ((1 - p) + p / n)


def demo_amdahl_law():
    """Demonstrate Amdahl's Law calculations."""
    p = 0.80  # 80% parallelizable

    print("\nExercise 3: Amdahl's Law")
    print(f"  If 80% of code is parallelizable:")
    print(f"  - Speedup with 4 cores: {amdahl_speedup(p, 4):.2f}x")
    print(f"  - Speedup with 8 cores: {amdahl_speedup(p, 8):.2f}x")
    print(f"  - Maximum speedup (infinite cores): {1/(1-p):.2f}x")


# =============================================================================
# NOTEBOOK 02: THREADING - EXERCISE SOLUTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Exercise 1: Extended Data Fetching
# -----------------------------------------------------------------------------

def fetch_extended_data(ticker):
    """Fetch extended stock data including 52-week high/low."""
    # Simulate variable network latency
    delay = 0.1 + 0.4 * np.random.random()
    time.sleep(delay)

    # Generate fake stock data
    np.random.seed(hash(ticker) % 2**32)
    current_price = np.random.uniform(10, 500)

    return {
        'ticker': ticker,
        'price': round(current_price, 2),
        'change': round(np.random.uniform(-5, 5), 2),
        'volume': np.random.randint(100000, 10000000),
        '52w_high': round(current_price * np.random.uniform(1.1, 1.5), 2),
        '52w_low': round(current_price * np.random.uniform(0.5, 0.9), 2),
        'fetch_time': round(delay, 3)
    }


def demo_extended_fetch():
    """Demonstrate extended data fetching."""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META',
               'NVDA', 'TSLA', 'JPM', 'BAC', 'WMT',
               'PG', 'JNJ', 'UNH', 'HD', 'V', 'MA',
               'DIS', 'NFLX', 'PYPL', 'INTC']

    print("\nExercise 1: Extended Data Fetching (20 tickers)")

    # Sequential
    start = time.time()
    seq_results = [fetch_extended_data(t) for t in tickers]
    seq_time = time.time() - start
    print(f"  Sequential: {seq_time:.2f}s")

    # Parallel with threads
    start = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        par_results = list(executor.map(fetch_extended_data, tickers))
    par_time = time.time() - start
    print(f"  Parallel: {par_time:.2f}s")
    print(f"  Speedup: {seq_time/par_time:.1f}x")


# -----------------------------------------------------------------------------
# Exercise 2: Retry Logic
# -----------------------------------------------------------------------------

def unreliable_fetch(ticker):
    """Simulate an unreliable API."""
    time.sleep(0.1)

    # 20% chance of failure
    if np.random.random() < 0.2:
        raise ConnectionError(f"Network timeout for {ticker}")

    np.random.seed(hash(ticker) % 2**32)
    return {'ticker': ticker, 'price': round(np.random.uniform(10, 500), 2)}


def fetch_with_retry(ticker, max_retries=3):
    """Fetch data with automatic retry on failure."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return unreliable_fetch(ticker)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff

    # All retries failed
    return {'ticker': ticker, 'price': None, 'error': str(last_exception)}


def demo_retry_logic():
    """Demonstrate retry logic."""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']

    print("\nExercise 2: Retry Logic")

    np.random.seed(42)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(fetch_with_retry, tickers))

    successful = [r for r in results if r.get('price') is not None]
    failed = [r for r in results if r.get('price') is None]

    print(f"  Successful: {len(successful)}/{len(tickers)}")
    print(f"  Failed after retries: {len(failed)}")


# -----------------------------------------------------------------------------
# Exercise 3: Progress Tracking with tqdm
# -----------------------------------------------------------------------------

def demo_progress_tracking():
    """Demonstrate progress tracking (requires tqdm)."""
    print("\nExercise 3: Progress Tracking")
    print("  Code example:")
    print("""
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tickers = ['AAPL', 'GOOGL', 'MSFT', ...]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_data, t): t for t in tickers}

        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching"):
            result = future.result()
            results.append(result)
    """)


# =============================================================================
# NOTEBOOK 03: MULTIPROCESSING - EXERCISE SOLUTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Exercise 1: European Put Option (Monte Carlo)
# -----------------------------------------------------------------------------

def monte_carlo_put_batch(args):
    """Price a European put option using Monte Carlo simulation."""
    S0, K, T, r, sigma, n_paths, seed = args

    np.random.seed(seed)

    # Generate random paths
    Z = np.random.standard_normal(n_paths)

    # Simulate terminal stock prices
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # PUT payoffs: max(K - ST, 0)
    payoffs = np.maximum(K - ST, 0)

    return np.exp(-r * T) * np.mean(payoffs)


def demo_mc_put():
    """Demonstrate Monte Carlo put pricing."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    total_paths = 4_000_000
    n_batches = 8
    paths_per_batch = total_paths // n_batches

    # Prepare batch arguments
    batch_args = [
        (S0, K, T, r, sigma, paths_per_batch, seed)
        for seed in range(n_batches)
    ]

    print("\nExercise 1: Monte Carlo Put Option")

    # Parallel execution
    start = time.time()
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        prices = list(executor.map(monte_carlo_put_batch, batch_args))
    par_time = time.time() - start

    mc_price = np.mean(prices)

    # Analytical Black-Scholes put price for comparison
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_put = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

    print(f"  Monte Carlo price: ${mc_price:.4f}")
    print(f"  Black-Scholes price: ${bs_put:.4f}")
    print(f"  Error: ${abs(mc_price - bs_put):.4f}")
    print(f"  Time: {par_time:.2f}s")


# -----------------------------------------------------------------------------
# Exercise 2: Optimal Chunk Size
# -----------------------------------------------------------------------------

def small_task(x):
    """A small task for chunking experiment."""
    return sum(i * i for i in range(1000))


def process_chunk(chunk):
    """Process a chunk of items."""
    return [small_task(x) for x in chunk]


def demo_optimal_chunk_size():
    """Find optimal chunk size."""
    n_items = 1000
    items = list(range(n_items))
    chunk_sizes = [10, 50, 100, 250, 500, 1000]

    print("\nExercise 2: Optimal Chunk Size")
    print(f"  Testing {n_items} items with various chunk sizes\n")

    # Sequential baseline
    start = time.time()
    _ = [small_task(x) for x in items]
    seq_time = time.time() - start
    print(f"  Sequential: {seq_time:.3f}s")

    for chunk_size in chunk_sizes:
        chunks = [items[i:i+chunk_size] for i in range(0, n_items, chunk_size)]

        start = time.time()
        with ProcessPoolExecutor(max_workers=4) as executor:
            chunk_results = list(executor.map(process_chunk, chunks))
        par_time = time.time() - start

        speedup = seq_time / par_time
        print(f"  Chunk size {chunk_size:4d}: {par_time:.3f}s (speedup: {speedup:.2f}x)")


# -----------------------------------------------------------------------------
# Exercise 3: Asian Option Pricing
# -----------------------------------------------------------------------------

def price_asian_call_batch(args):
    """
    Price an Asian call option using Monte Carlo simulation.

    Asian options have payoff based on average price over the life of the option.
    Payoff = max(average(S) - K, 0)
    """
    S0, K, T, r, sigma, n_paths, n_steps, seed = args

    np.random.seed(seed)

    dt = T / n_steps

    # Initialize
    payoffs = np.zeros(n_paths)

    for path in range(n_paths):
        S = S0
        price_sum = S

        for step in range(n_steps):
            Z = np.random.standard_normal()
            S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            price_sum += S

        average_price = price_sum / (n_steps + 1)
        payoffs[path] = max(average_price - K, 0)

    return np.exp(-r * T) * np.mean(payoffs)


def demo_asian_option():
    """Demonstrate Asian option pricing."""
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    n_steps = 252  # Daily observations
    total_paths = 100_000
    n_batches = 8
    paths_per_batch = total_paths // n_batches

    batch_args = [
        (S0, K, T, r, sigma, paths_per_batch, n_steps, seed)
        for seed in range(n_batches)
    ]

    print("\nExercise 3: Asian Option Pricing")

    start = time.time()
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        prices = list(executor.map(price_asian_call_batch, batch_args))
    par_time = time.time() - start

    asian_price = np.mean(prices)

    # Note: Asian options are worth less than European options
    # because averaging reduces volatility
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    european_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    print(f"  Asian call price: ${asian_price:.4f}")
    print(f"  European call price: ${european_price:.4f}")
    print(f"  Discount: {(1 - asian_price/european_price)*100:.1f}%")
    print(f"  Time: {par_time:.2f}s")


# =============================================================================
# NOTEBOOK 05: PROJECT SOLUTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Project A: Parallel Portfolio Optimizer
# -----------------------------------------------------------------------------

def calculate_portfolio_metrics(weights, expected_returns, cov_matrix):
    """Calculate portfolio return, volatility, and Sharpe ratio."""
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
    return portfolio_return, portfolio_volatility, sharpe_ratio


def generate_random_portfolios_batch(args):
    """Generate batch of random portfolios."""
    n_portfolios, n_assets, expected_returns, cov_matrix, seed = args
    np.random.seed(seed)

    results = []
    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights = weights / weights.sum()

        ret, vol, sharpe = calculate_portfolio_metrics(weights, expected_returns, cov_matrix)

        results.append({
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe,
            'weights': weights.copy()
        })

    return results


def demo_portfolio_optimizer():
    """Demonstrate parallel portfolio optimization."""
    # Setup
    assets = ['Stocks', 'Bonds', 'Gold', 'Real Estate', 'Commodities']
    expected_returns = np.array([0.10, 0.04, 0.05, 0.08, 0.06])
    volatilities = np.array([0.18, 0.06, 0.15, 0.12, 0.20])
    correlations = np.array([
        [1.00, 0.20, 0.05, 0.60, 0.40],
        [0.20, 1.00, 0.30, 0.10, 0.05],
        [0.05, 0.30, 1.00, 0.10, 0.25],
        [0.60, 0.10, 0.10, 1.00, 0.30],
        [0.40, 0.05, 0.25, 0.30, 1.00]
    ])
    cov_matrix = np.outer(volatilities, volatilities) * correlations

    # Parameters
    total_portfolios = 50_000
    n_batches = 8
    portfolios_per_batch = total_portfolios // n_batches
    n_assets = len(assets)

    print("\nProject A: Portfolio Optimizer")
    print(f"  Generating {total_portfolios:,} random portfolios...")

    # Create batch arguments
    batch_args = [
        (portfolios_per_batch, n_assets, expected_returns, cov_matrix, seed)
        for seed in range(n_batches)
    ]

    # Sequential
    start = time.time()
    seq_results = [generate_random_portfolios_batch(args) for args in batch_args]
    seq_time = time.time() - start

    # Parallel
    start = time.time()
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        par_results = list(executor.map(generate_random_portfolios_batch, batch_args))
    par_time = time.time() - start

    # Flatten results
    all_portfolios = [p for batch in par_results for p in batch]
    portfolios_df = pd.DataFrame(all_portfolios)

    # Find best portfolios
    max_sharpe_idx = portfolios_df['sharpe'].idxmax()
    min_vol_idx = portfolios_df['volatility'].idxmin()

    print(f"  Sequential: {seq_time:.2f}s")
    print(f"  Parallel: {par_time:.2f}s")
    print(f"  Speedup: {seq_time/par_time:.2f}x")

    print(f"\n  Max Sharpe Portfolio:")
    print(f"    Return: {portfolios_df.loc[max_sharpe_idx, 'return']*100:.2f}%")
    print(f"    Volatility: {portfolios_df.loc[max_sharpe_idx, 'volatility']*100:.2f}%")
    print(f"    Sharpe: {portfolios_df.loc[max_sharpe_idx, 'sharpe']:.3f}")

    print(f"\n  Min Volatility Portfolio:")
    print(f"    Return: {portfolios_df.loc[min_vol_idx, 'return']*100:.2f}%")
    print(f"    Volatility: {portfolios_df.loc[min_vol_idx, 'volatility']*100:.2f}%")
    print(f"    Sharpe: {portfolios_df.loc[min_vol_idx, 'sharpe']:.3f}")


# -----------------------------------------------------------------------------
# Project B: Parameter Sensitivity Analysis
# -----------------------------------------------------------------------------

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def calculate_price_for_params(args):
    """Calculate option price for parameters."""
    S, K, T, r, sigma = args
    price = black_scholes_call(S, K, T, r, sigma)
    return {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'price': price}


def demo_sensitivity_analysis():
    """Demonstrate parameter sensitivity analysis."""
    S0, K = 100, 100

    sigma_range = np.linspace(0.10, 0.50, 21)
    r_range = np.linspace(0.00, 0.10, 21)
    T_range = np.linspace(0.1, 2.0, 20)

    param_grid = list(product([S0], [K], T_range, r_range, sigma_range))

    print("\nProject B: Sensitivity Analysis")
    print(f"  Parameter combinations: {len(param_grid)}")

    start = time.time()
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(calculate_price_for_params, param_grid))
    par_time = time.time() - start

    results_df = pd.DataFrame(results)

    print(f"  Time: {par_time:.2f}s")
    print(f"\n  Price ranges:")
    print(f"    Min: ${results_df['price'].min():.2f}")
    print(f"    Max: ${results_df['price'].max():.2f}")
    print(f"    Mean: ${results_df['price'].mean():.2f}")


# -----------------------------------------------------------------------------
# Project C: Multi-Strategy Backtester
# -----------------------------------------------------------------------------

def calculate_performance_metrics(strategy_name, param_value, returns):
    """Calculate strategy performance metrics."""
    if len(returns) == 0:
        return None

    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'strategy': strategy_name,
        'param': param_value,
        'total_return': total_return * 100,
        'annual_return': annual_return * 100,
        'volatility': volatility * 100,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown * 100
    }


def backtest_momentum(prices, params):
    """Momentum strategy."""
    lookback = params['lookback']
    returns = prices.pct_change(lookback)
    signal = (returns > 0).astype(int).shift(1)
    daily_returns = prices.pct_change()
    strategy_returns = signal * daily_returns
    strategy_returns = strategy_returns.dropna()
    return calculate_performance_metrics('momentum', lookback, strategy_returns)


def backtest_mean_reversion(prices, params):
    """Mean reversion strategy."""
    window = params['window']
    ma = prices.rolling(window).mean()
    signal = (prices < ma).astype(int).shift(1)
    daily_returns = prices.pct_change()
    strategy_returns = signal * daily_returns
    strategy_returns = strategy_returns.dropna()
    return calculate_performance_metrics('mean_reversion', window, strategy_returns)


def backtest_volatility_breakout(prices, params):
    """Volatility breakout strategy."""
    lookback = params['lookback']
    rolling_high = prices.rolling(lookback).max().shift(1)
    signal = (prices > rolling_high).astype(int).shift(1)
    daily_returns = prices.pct_change()
    strategy_returns = signal * daily_returns
    strategy_returns = strategy_returns.dropna()
    return calculate_performance_metrics('volatility_breakout', lookback, strategy_returns)


def backtest_strategy(args):
    """Dispatch to appropriate strategy."""
    strategy_name, prices, params = args
    if strategy_name == 'momentum':
        return backtest_momentum(prices, params)
    elif strategy_name == 'mean_reversion':
        return backtest_mean_reversion(prices, params)
    elif strategy_name == 'volatility_breakout':
        return backtest_volatility_breakout(prices, params)
    return None


def demo_multi_strategy():
    """Demonstrate multi-strategy backtester."""
    # Generate price data
    np.random.seed(42)
    n_days = 2520
    returns = np.random.standard_t(df=5, size=n_days) * 0.015 + 0.0003
    prices = pd.Series(100 * np.cumprod(1 + returns))

    # Strategy parameters
    momentum_params = [{'lookback': lb} for lb in range(5, 61, 5)]
    mean_rev_params = [{'window': w} for w in range(10, 101, 10)]
    vol_breakout_params = [{'lookback': lb} for lb in range(5, 41, 5)]

    # Create all args
    all_args = []
    all_args.extend([('momentum', prices, p) for p in momentum_params])
    all_args.extend([('mean_reversion', prices, p) for p in mean_rev_params])
    all_args.extend([('volatility_breakout', prices, p) for p in vol_breakout_params])

    print("\nProject C: Multi-Strategy Backtester")
    print(f"  Strategy variants to test: {len(all_args)}")

    start = time.time()
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(executor.map(backtest_strategy, all_args))
    par_time = time.time() - start

    results = [r for r in results if r is not None]
    results_df = pd.DataFrame(results)

    print(f"  Time: {par_time:.2f}s")

    # Best by strategy
    print("\n  Best parameters by strategy:")
    for strategy in ['momentum', 'mean_reversion', 'volatility_breakout']:
        subset = results_df[results_df['strategy'] == strategy]
        best = subset.loc[subset['sharpe'].idxmax()]
        print(f"    {strategy}: param={int(best['param'])}, Sharpe={best['sharpe']:.3f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PARALLEL PYTHON WORKSHOP - EXERCISE SOLUTIONS")
    print("=" * 60)

    # Notebook 01 exercises
    print("\n" + "=" * 60)
    print("NOTEBOOK 01: MOTIVATION")
    print("=" * 60)
    demo_put_option_pricing()
    demo_volatility_sensitivity()
    demo_amdahl_law()

    # Notebook 02 exercises
    print("\n" + "=" * 60)
    print("NOTEBOOK 02: THREADING")
    print("=" * 60)
    demo_extended_fetch()
    demo_retry_logic()
    demo_progress_tracking()

    # Notebook 03 exercises
    print("\n" + "=" * 60)
    print("NOTEBOOK 03: MULTIPROCESSING")
    print("=" * 60)
    demo_mc_put()
    demo_optimal_chunk_size()
    demo_asian_option()

    # Notebook 05 projects
    print("\n" + "=" * 60)
    print("NOTEBOOK 05: PROJECTS")
    print("=" * 60)
    demo_portfolio_optimizer()
    demo_sensitivity_analysis()
    demo_multi_strategy()

    print("\n" + "=" * 60)
    print("ALL SOLUTIONS COMPLETED!")
    print("=" * 60)
