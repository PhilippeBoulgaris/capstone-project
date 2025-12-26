#!/usr/bin/env python3
"""
CPU Parallelism Demo - Watch your cores light up!

Run this script while monitoring CPU usage with btop, htop, or Activity Monitor.
You'll see the difference between sequential (1 core) and parallel (all cores).

Usage:
    python cpu_demo.py

In another terminal, run:
    btop    # or htop, or open Activity Monitor on Mac
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import sys


def cpu_intensive_work(worker_id):
    """
    Burn CPU cycles for a few seconds.
    This simulates heavy computation like ML training or data processing.
    """
    total = 0
    # 50 million iterations takes ~3-5 seconds per core
    for i in range(50_000_000):
        total += i * i % 1000
    return worker_id, total


def run_sequential(n_tasks):
    """Run tasks one after another (uses 1 core)"""
    print(f"\n{'='*60}")
    print("SEQUENTIAL EXECUTION")
    print(f"Running {n_tasks} tasks one at a time...")
    print("Watch your CPU monitor: Only 1 core should be at 100%")
    print(f"{'='*60}\n")

    start = time.time()
    results = []
    for i in range(n_tasks):
        print(f"  Task {i+1}/{n_tasks} running...")
        result = cpu_intensive_work(i)
        results.append(result)
        print(f"  Task {i+1}/{n_tasks} done")

    elapsed = time.time() - start
    print(f"\nSequential time: {elapsed:.1f} seconds")
    return elapsed


def run_parallel(n_tasks, n_workers):
    """Run tasks in parallel (uses multiple cores)"""
    print(f"\n{'='*60}")
    print("PARALLEL EXECUTION")
    print(f"Running {n_tasks} tasks across {n_workers} workers...")
    print(f"Watch your CPU monitor: {n_workers} cores should be at 100%")
    print(f"{'='*60}\n")

    start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(cpu_intensive_work, i) for i in range(n_tasks)]

        # Wait for completion
        for i, future in enumerate(futures):
            worker_id, result = future.result()
            print(f"  Task {i+1}/{n_tasks} done (worker {worker_id})")

    elapsed = time.time() - start
    print(f"\nParallel time: {elapsed:.1f} seconds")
    return elapsed


def main():
    n_cores = multiprocessing.cpu_count()
    n_tasks = min(4, n_cores)  # Use 4 tasks or fewer if limited cores

    print("\n" + "="*60)
    print(" CPU PARALLELISM DEMO")
    print("="*60)
    print(f"\nYour system has {n_cores} CPU cores")
    print(f"We'll run {n_tasks} CPU-intensive tasks\n")
    print("TIP: Open btop/htop in another terminal to watch CPU usage!")
    print("     On Mac: open Activity Monitor > CPU")

    input("\nPress Enter to start SEQUENTIAL execution...")
    seq_time = run_sequential(n_tasks)

    input("\nPress Enter to start PARALLEL execution...")
    par_time = run_parallel(n_tasks, n_tasks)

    # Summary
    speedup = seq_time / par_time
    print("\n" + "="*60)
    print(" RESULTS")
    print("="*60)
    print(f"\n  Sequential: {seq_time:.1f}s (1 core)")
    print(f"  Parallel:   {par_time:.1f}s ({n_tasks} cores)")
    print(f"  Speedup:    {speedup:.1f}x faster!")
    print(f"\n  Theoretical max speedup: {n_tasks}x")
    print(f"  Efficiency: {(speedup/n_tasks)*100:.0f}%")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
