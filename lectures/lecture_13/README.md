# Parallel Programming in Python: A crash course for Economics & Finance Students

## Overview

This 4-hour lecture introduces parallel programming concepts in Python, specifically tailored for economics and finance students. All examples use financial applications such as Monte Carlo simulations, portfolio analysis, and data processing.

**Target Audience:** 4th year Economics/Finance students with basic Python knowledge

**Duration:** 4 hours (including breaks)

**Format:** Interactive Jupyter notebooks with supporting lecture slides

---

## Workshop Schedule

| Session | Duration | Topic | Slides | Notebook |
|---------|----------|-------|--------|----------|
| **1** | 45 min | Why Parallelism? Motivation & Core Concepts | Topic 1 | `01_motivation.ipynb` |
| **2** | 50 min | Threading & I/O-Bound Tasks | Topic 2 | `02_threading_io_bound.ipynb` |
| *Break* | 15 min | - | - | - |
| **3** | 55 min | Multiprocessing & CPU-Bound Tasks | Topic 3 | `03_multiprocessing_cpu_bound.ipynb` |
| **4** | 45 min | Real-World Finance Applications | - | `04_finance_applications.ipynb` |
| *Break* | 10 min | - | - | - |
| **5** | 40 min | Hands-On Project & Best Practices | Topic 4 | `05_project_exercises.ipynb` |

**All slides are in a single file:** `slides/parallel_python_workshop.tex`

---

## Learning Objectives

By the end of this workshop, students will be able to:

1. **Understand** why parallel programming matters in modern computing
2. **Distinguish** between I/O-bound and CPU-bound tasks
3. **Apply** threading for concurrent I/O operations (e.g., data fetching)
4. **Implement** multiprocessing for CPU-intensive computations (e.g., Monte Carlo simulations)
5. **Use** `concurrent.futures` for clean, Pythonic parallel code
6. **Recognize** when parallelization is beneficial and when it adds unnecessary complexity
7. **Debug** common issues in parallel Python programs

---

## Topic Details

### Topic 1: Why Parallelism Matters (45 min)

**Concepts Covered:**
- The end of Moore's Law and the multi-core era
- Sequential vs. parallel execution
- Amdahl's Law (intuitive explanation)
- Real bottlenecks in finance: simulations, backtesting, data processing

**Finance Context:**
- Why pricing 10,000 options takes too long sequentially
- The need for speed in quantitative finance

---

### Topic 2: Threading & I/O-Bound Tasks (50 min)

**Concepts Covered:**
- What is a thread?
- The Global Interpreter Lock (GIL) - simplified explanation
- I/O-bound vs. CPU-bound tasks
- `ThreadPoolExecutor` from `concurrent.futures`

**Finance Examples:**
- Simulated API data fetching for multiple stocks
- Parallel file reading operations
- When threading helps vs. when it doesn't

---

### Topic 3: Multiprocessing & CPU-Bound Tasks (55 min)

**Concepts Covered:**
- Why threads don't help CPU-bound work
- Processes vs. threads
- `ProcessPoolExecutor` from `concurrent.futures`
- The `multiprocessing` module
- Overhead and when parallelization pays off

**Finance Examples:**
- Monte Carlo option pricing (Black-Scholes simulation)
- Portfolio Value-at-Risk (VaR) calculation
- Bootstrap confidence intervals for Sharpe ratios

---

### Topic 4: Best Practices & Advanced Topics (40 min)

**Best Practices Covered:**
- Common pitfalls (shared state, pickling issues, overhead)
- When NOT to parallelize
- Debugging parallel code
- Brief intro to advanced tools: `joblib`, `dask`, `ray`

**Mini-Projects:**
- Portfolio optimization with parallel objective evaluation
- Parallel bootstrap for statistical inference
- Parameter sweep for trading strategies

---

## Repository Structure

```
lecture_13/
├── README.md                           # This file
├── slides/
│   └── Advanced_Programming_2025_lecture_13.pdf    # All lecture slides (Topics 1-4)
├── demo/
│   ├── 01_motivation.ipynb
│   ├── 02_threading_io_bound.ipynb
│   ├── 03_multiprocessing_cpu_bound.ipynb
│   ├── 04_finance_applications.ipynb
│   └── 05_project_exercises.ipynb
├── demo/solutions/
│   └── exercise_solutions.py        # Solutions to all exercises
└── data/
    └── (generated within notebooks - self-contained)
```

---

## Key Takeaways

| Task Type | Solution | Python Tool | Finance Example |
|-----------|----------|-------------|-----------------|
| I/O-Bound | Threading | `ThreadPoolExecutor` | Fetching market data |
| CPU-Bound | Multiprocessing | `ProcessPoolExecutor` | Monte Carlo simulation |
| Mixed | Combine both | Nested executors | Fetch data, then analyze |

---

## Prerequisites

Students should be familiar with:
- Basic Python programming (functions, loops, lists, dictionaries)
- NumPy and Pandas basics
- Elementary finance concepts (stocks, returns, options basics)
