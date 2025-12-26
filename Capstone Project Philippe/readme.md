# DSAP Project — Yahoo vs SEC Fundamentals (Value/Growth + ML)

This project compares two end-to-end pipelines for stock selection using **fundamentals only**:

- **Yahoo Finance fundamentals (snapshot)**: easy to access but not point-in-time (risk of timing bias).
- **SEC XBRL fundamentals (as-of)**: constructed from filings with an explicit availability lag to reduce look-ahead bias.

The pipeline:
1) Download monthly prices for an S&P 500 ticker subset + benchmark  
2) Build Yahoo snapshot features and scores (Value/Growth/Quality)  
3) Build SEC annual fundamentals → convert to monthly as-of panel → compute ratios/features  
4) Train ML classifiers to predict next-month outperformance vs benchmark  
5) Backtest portfolios (traditional Value/Growth, ML portfolios, EW, Random)  
6) Save metrics table + equity curve figures to `results/`

---


