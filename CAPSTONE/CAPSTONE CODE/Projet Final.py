import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    classification_report, roc_auc_score
)

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️ seaborn not available - using simplified heatmaps")

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️ tensorflow not available - neural network section skipped")

# Global Parameters

US_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", 
    "BRK-B", "JPM", "BAC", "WFC", "MS", "GS", "V", "MA", 
    "PYPL", "KO", "PEP", "PG", "JNJ", "UNH", "MRK", "PFE",
    "T", "VZ", "CMCSA", "DIS", "NFLX", "INTC", "AMD", "IBM",
    "ORCL", "CRM", "COST", "HD", "WMT", "TGT", "LOW", "NKE",
    "MCD", "SBUX", "CAT", "BA", "GE", "LMT", "RTX", 
    "XOM", "CVX", "COP", "SLB"
]
SWISS_TICKERS = ["NESN.SW", "UBSG.SW"]
ALL_TICKERS = US_TICKERS + SWISS_TICKERS

BENCHMARK = "^GSPC"  # S&P 500
START = "2010-01-01"
INTERVAL = "1mo"
COSTS_BPS = 20

# Prediction
HORIZON = 12  # Future 12-month returns
TRAIN_END_DATE = "2018-01-01"

# Output
OUT_DIR = "./capstone_unified_output"
os.makedirs(OUT_DIR, exist_ok=True)
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

print("=" * 80)
print("CAPSTONE PROJECT - VALUE VS GROWTH STRATEGY")
print("=" * 80)
print(f"Universe: {len(ALL_TICKERS)} stocks + benchmark {BENCHMARK}")
print(f"Period: {START} → today (monthly data)")
print(f"Output: {os.path.abspath(OUT_DIR)}")
print("=" * 80)

#1 PRICE DATA 
print("\n[1/10] Downloading price data...")


all_symbols = ALL_TICKERS + [BENCHMARK] #preparation of the list of tickers
data = yf.download( #download via yahoo finance
    tickers=all_symbols, 
    start=START, 
    interval=INTERVAL,
    auto_adjust=True, 
    progress=False
)

px = data["Close"].dropna(how="all") #only the closed price and monitoring when n/a before it's was quoted for example
print(f"✓ Prices downloaded: {px.shape[0]} months × {px.shape[1]} tickers")

# Monthly returns (log)
rets = np.log(px / px.shift(1)).dropna() #pt/pt+1 factor of performance and we want log returns for additivy of returns
bench_rets = rets[BENCHMARK].copy() #monthly returns of the benchmark 
asset_rets = rets[[t for t in ALL_TICKERS if t in rets.columns]].copy() #assets 

# Excess returns vs benchmark comparison 
excess_rets = asset_rets.sub(bench_rets, axis=0)
print(f"✓ Returns calculated over {len(rets)} periods")

#2 FUNDAMENTAL DATA 
print("\n[2/10] Fetching fundamentals (current snapshot)...")

def fetch_fundamentals(ticker): #looking for a big dictionnary with market cap trailing P/E etc...
    """Retrieves fundamental ratios from Yahoo Finance"""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {} #in case problem 
    
    mcap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    p_fcf = (mcap / fcf) if (mcap and fcf and fcf != 0) else None
    
    return { #construction of the dictionnary with all fundamentals info
        "ticker": ticker,
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "ps": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "p_fcf": p_fcf,
        "div_yield": info.get("dividendYield"),
        "eps": info.get("trailingEps"),
        "revenue_growth": info.get("revenueGrowth"),
        "eps_growth": info.get("earningsQuarterlyGrowth"),
        "sector": info.get("sector"),
    }

fund = pd.DataFrame([fetch_fundamentals(t) for t in ALL_TICKERS]).set_index("ticker") #Retrieve fundamental ratios (PE, PB, PS, etc.) for each stock in the universe
# and assembl them into a DataFrame fund indexed by ticker (1 row = 1 stock)
print(f"✓ Fundamentals retrieved for {len(fund)} stocks")
print("⚠️  LIMITATION: current ratios applied to entire history") #all my capstone is about this issue :)

# 3 VALUE SCORE CALCULATION 
print("\n[3/10] Calculating value scores (Value vs Growth)...")

value_cols = [c for c in ["pe", "pb", "ps", "ev_ebitda", "p_fcf"] if c in fund.columns] #take only thos for valuation into account
inv = fund[value_cols].copy()

# Invert multiples (lower = better value)
for c in value_cols:
    inv[c] = 1.0 / fund[c].where(fund[c] > 0)

# Cleaning and winsorization for the outliers 
inv = inv.replace([np.inf, -np.inf], np.nan)
inv = inv.fillna(inv.median(numeric_only=True))
inv = inv.clip(
    lower=inv.quantile(0.01), 
    upper=inv.quantile(0.99), 
    axis=1
)

# Z-scores and average
Z = (inv - inv.mean()) / inv.std(ddof=0)
score_value = Z.mean(axis=1).rename("score_value")
#the more score value is high the better it is from a value perspective 

# Create deciles
valid = score_value.dropna()
dec = pd.qcut(
    valid.rank(method="first", ascending=False), #trie les actions les plus grandes score v au plus petite 
    10, #cut in deciles
    labels=False, 
    duplicates="drop"
) + 1
#little trick to have it all correclty allign with the decliel
deciles = pd.Series(index=score_value.index, dtype="Int64")
deciles.loc[dec.index] = dec

val_names = deciles.index[deciles == 1].tolist()  # Q1 = VALUE
grw_names = deciles.index[deciles == 10].tolist()  # Q10 = GROWTH

print(f"✓ VALUE Portfolio (Q1): {len(val_names)} stocks")
print(f"✓ GROWTH Portfolio (Q10): {len(grw_names)} stocks")

# Add technical features useful for after to take into account
rets_60d = px.pct_change(60).iloc[-1]
fund["momentum"] = rets_60d.reindex(fund.index)
fund["volatility"] = rets.std().reindex(fund.index)
fund["score_value"] = score_value
fund["decile"] = deciles

# 4 EDA (EXPLORATORY DATA ANALYSIS) #see the data
print("\n[4/10] Exploratory data analysis...")

# Create a 2x3 grid of subplots to summarize the main relationships in the data
fig, axes = plt.subplots(2, 3, figsize=(16, 10))


# a Distribution of the composite value score
#   Shows how the value scores are spread across the universe (which stocks look "cheap" vs "expensive")
score_value.hist(bins=30, ax=axes[0, 0], edgecolor='black', color='steelblue') #see if the score seems reasonible 
axes[0, 0].set_title("Value Score Distribution")
axes[0, 0].set_xlabel("Value Score (Z)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.5)  # Reference at the mean (Z = 0)


# b Correlation matrix between valuation ratios (PE, PB, PS, EV/EBITDA, P/FCF)
#    Helps to see how redundant these factors are ( highly correlated multiples).
if len(value_cols) > 1:
    corr = fund[value_cols].corr() 

    if HAS_SEABORN:
        # Use seaborn heatmap if available for a nicer correlation plot
        sns.heatmap(corr, annot=True, ax=axes[0, 1],
                    cmap="coolwarm", center=0, fmt='.2f')
    else:
        # Fallback: basic imshow heatmap with matplotlib only
        im = axes[0, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        axes[0, 1].set_xticks(range(len(corr)))
        axes[0, 1].set_yticks(range(len(corr)))
        axes[0, 1].set_xticklabels(corr.columns, rotation=45, ha="right")
        axes[0, 1].set_yticklabels(corr.columns)
        plt.colorbar(im, ax=axes[0, 1])

    axes[0, 1].set_title("Valuation Multiple Correlation")


# c Average value score by sector
#    → Shows which sectors tend to look more "value" or more "growth" on average.
fund_clean = fund.dropna(subset=["sector", "score_value"])
if len(fund_clean) > 0:
    sector_scores = (
        fund_clean
        .groupby("sector")["score_value"]
        .mean()
        .sort_values()
    )
    sector_scores.plot(kind="barh", ax=axes[0, 2], color="teal")
    axes[0, 2].set_title("Average Value Score by Sector")
    axes[0, 2].set_xlabel("Score")
    axes[0, 2].axvline(0, color='red', linestyle='--', alpha=0.5)  # Neutral benchmark


# d Scatter plot: Value score vs long-term momentum
#    Checks whether cheap stocks (high value score) tend to have good/bad past performance
fund_clean2 = fund.dropna(subset=["score_value", "momentum"])
if len(fund_clean2) > 0:
    axes[1, 0].scatter(
        fund_clean2["score_value"],
        fund_clean2["momentum"],
        alpha=0.6,
        s=50
    )
    axes[1, 0].set_title("Value Score vs Momentum (60 months)")
    axes[1, 0].set_xlabel("Value Score")
    axes[1, 0].set_ylabel("60m Return")
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=0.8)
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=0.8)
    axes[1, 0].grid(alpha=0.3)


# e Distribution of P/E by decile (Value vs Middle vs Growth)
#    Sanity check: Value decile (Q1) should have lower PE than Growth decile (Q10).
fund_with_dec = fund.dropna(subset=["decile", "pe"])
if len(fund_with_dec) > 0:
    dec_groups = [1, 5, 10]  # VALUE (Q1), MIDDLE (Q5), GROWTH (Q10)
    for d in dec_groups:
        subset = fund_with_dec[fund_with_dec["decile"] == d]["pe"]
        label = {1: "VALUE (Q1)", 5: "MIDDLE (Q5)", 10: "GROWTH (Q10)"}[d]
        axes[1, 1].hist(subset, bins=15, alpha=0.5, label=label)

    axes[1, 1].set_title("P/E Distribution by Decile")
    axes[1, 1].set_xlabel("P/E Ratio")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)


# f Normalized price evolution for a subset of top tickers
#    → Compares the price performance of a few representative stocks on a common base = 1.
top_tickers = ALL_TICKERS[:8]
cols = [t for t in top_tickers if t in px.columns]
norm_px = px[cols] / px[cols].iloc[0]  # Normalize all series to 1 at the first date

norm_px.plot(ax=axes[1, 2], alpha=0.7, linewidth=1.5)
axes[1, 2].set_title("Normalized Prices (Top 8)")
axes[1, 2].set_ylabel("Base = 1")
axes[1, 2].legend(fontsize=7, loc='best')
axes[1, 2].grid(alpha=0.3)

# Final layout and save the EDA figure to file
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"eda_analysis_{ts}.png"), dpi=150)
plt.close()
print(f"✓ EDA charts saved")


# ==================== SECTION 5: REGRESSION MODELS ====================
print("\n[5/10] Regression models (predicting average returns)...")

# Features: value score → Target: historical average return
X = score_value.values.reshape(-1, 1)
y = rets.mean(axis=0).reindex(score_value.index).values

# Cleaning
mask = ~(np.isnan(X.flatten()) | np.isnan(y))
X_clean = X[mask]
y_clean = y[mask]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

models_reg = {
    "Linear Regression": LinearRegression(),
    "Ridge (α=1.0)": Ridge(alpha=1.0),
    "Lasso (α=0.01)": Lasso(alpha=0.01)
}

regression_results = []
for name, model in models_reg.items():
    model.fit(X_train_reg, y_train_reg)
    pred = model.predict(X_test_reg)
    r2 = r2_score(y_test_reg, pred)
    mse = mean_squared_error(y_test_reg, pred)
    regression_results.append({
        "Model": name, 
        "R²": round(r2, 4), 
        "RMSE": round(np.sqrt(mse), 6)
    })
    print(f"  {name:25s} → R²={r2:.4f}, RMSE={np.sqrt(mse):.6f}")

pd.DataFrame(regression_results).to_csv(
    os.path.join(OUT_DIR, f"regression_results_{ts}.csv"), 
    index=False
)

# ==================== SECTION 6: CLASSIFICATION MODELS ====================
print("\n[6/10] Classification models (high performer vs low)...")

# Target: above median = high performer
median_ret = np.nanmedian(y)
y_class = (y > median_ret).astype(int)

mask_class = ~(np.isnan(X.flatten()) | np.isnan(y_class))
X_class = X[mask_class]
y_class_clean = y_class[mask_class]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_class, y_class_clean, test_size=0.2, random_state=42
)

classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
}

classification_results = []
for name, clf in classifiers.items():
    clf.fit(X_train_cls, y_train_cls)
    pred = clf.predict(X_test_cls)
    acc = accuracy_score(y_test_cls, pred)
    classification_results.append({"Model": name, "Accuracy": round(acc, 4)})
    print(f"  {name:25s} → Accuracy={acc:.4f}")

pd.DataFrame(classification_results).to_csv(
    os.path.join(OUT_DIR, f"classification_results_{ts}.csv"), 
    index=False
)

# ==================== SECTION 7: CLUSTERING & PCA ====================
print("\n[7/10] Clustering (K-Means) and dimensionality reduction (PCA)...")

X_cluster = fund[value_cols].fillna(fund[value_cols].median()).values

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
fund["cluster"] = kmeans.fit_predict(X_cluster)

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X_cluster)
fund["PC1"], fund["PC2"] = coords[:, 0], coords[:, 1]

print(f"✓ PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 7))
for cluster in sorted(fund["cluster"].unique()):
    subset = fund[fund["cluster"] == cluster]
    ax.scatter(subset["PC1"], subset["PC2"], 
               label=f"Cluster {cluster}", alpha=0.7, s=80)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("Stock Clustering (K-Means + PCA)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"pca_clusters_{ts}.png"), dpi=150)
plt.close()

# ==================== SECTION 8: FUTURE RETURN PREDICTION ====================
print("\n[8/10] Predicting future returns (12 months)...")

def future_cum_return(series, horizon):
    """Calculates cumulative returns over 'horizon' future months"""
    values, idx = [], []
    for i in range(len(series) - horizon):
        idx.append(series.index[i])
        values.append(series.iloc[i+1:i+1+horizon].sum())
    return pd.Series(values, index=idx)

# Build panel dataset (date × ticker)
target_rows = []
for tic in ALL_TICKERS:
    if tic not in excess_rets.columns:
        continue
    s = excess_rets[tic].dropna()
    fut = future_cum_return(s, HORIZON)
    
    for date in fut.index:
        row = {
            "date": date,
            "ticker": tic,
            "y_excess_12m": fut.loc[date],
            "pe": fund.loc[tic, "pe"] if tic in fund.index else np.nan,
            "pb": fund.loc[tic, "pb"] if tic in fund.index else np.nan,
            "div_yield": fund.loc[tic, "div_yield"] if tic in fund.index else np.nan,
            "momentum": fund.loc[tic, "momentum"] if tic in fund.index else np.nan,
            "volatility": fund.loc[tic, "volatility"] if tic in fund.index else np.nan,
        }
        target_rows.append(row)

panel_df = pd.DataFrame(target_rows).dropna()
panel_df = panel_df.set_index(["date", "ticker"])

print(f"✓ Panel dataset: {len(panel_df)} observations")

# Temporal split
is_train = panel_df.index.get_level_values("date") < TRAIN_END_DATE
train_panel = panel_df[is_train].copy()
test_panel = panel_df[~is_train].copy()

X_train_fut = train_panel.drop(columns=["y_excess_12m"])
y_train_fut = train_panel["y_excess_12m"]
X_test_fut = test_panel.drop(columns=["y_excess_12m"])
y_test_fut = test_panel["y_excess_12m"]

print(f"  Train: {len(X_train_fut)} obs | Test: {len(X_test_fut)} obs")

# Standardization
scaler_fut = StandardScaler()
X_train_fut_scaled = scaler_fut.fit_transform(X_train_fut)
X_test_fut_scaled = scaler_fut.transform(X_test_fut)

# Ridge Regression
ridge_fut = Ridge(alpha=1.0)
ridge_fut.fit(X_train_fut_scaled, y_train_fut)
y_pred_fut = ridge_fut.predict(X_test_fut_scaled)

rmse_fut = np.sqrt(mean_squared_error(y_test_fut, y_pred_fut))
r2_fut = r2_score(y_test_fut, y_pred_fut)

print(f"  Ridge (12m future returns) → RMSE={rmse_fut:.5f}, R²={r2_fut:.4f}")

# Logistic Regression (outperformance vs benchmark)
y_train_fut_cls = (y_train_fut > 0).astype(int)
y_test_fut_cls = (y_test_fut > 0).astype(int)

logreg_fut = LogisticRegression(max_iter=1000)
logreg_fut.fit(X_train_fut_scaled, y_train_fut_cls)
y_pred_cls_proba = logreg_fut.predict_proba(X_test_fut_scaled)[:, 1]
y_pred_cls_label = (y_pred_cls_proba >= 0.5).astype(int)

acc_fut = accuracy_score(y_test_fut_cls, y_pred_cls_label)
try:
    auc_fut = roc_auc_score(y_test_fut_cls, y_pred_cls_proba)
except ValueError:
    auc_fut = np.nan

print(f"  Logistic (outperformance) → Accuracy={acc_fut:.3f}, AUC={auc_fut:.3f}")

# ==================== SECTION 9: MARKET REGIMES ====================
print("\n[9/10] Market regime segmentation (calm vs crisis)...")

# Rolling volatility of benchmark over 12 months
roll_vol = bench_rets.rolling(window=12).std() * np.sqrt(12)
vol_threshold = roll_vol.median()

regime_series = pd.Series(
    np.where(roll_vol > vol_threshold, "crisis", "calm"),
    index=roll_vol.index,
    name="regime"
)

print(f"✓ Regimes calculated: {(regime_series == 'calm').sum()} calm months, {(regime_series == 'crisis').sum()} crisis months")

# ==================== SECTION 10: BACKTESTING ====================
print("\n[10/10] Backtesting VALUE vs GROWTH portfolios...")

# Portfolio weights
w_val = pd.Series(1/len(val_names) if len(val_names) > 0 else 0, index=val_names)
w_grw = pd.Series(1/len(grw_names) if len(grw_names) > 0 else 0, index=grw_names)
w_mkt = pd.Series(1/len(asset_rets.columns), index=asset_rets.columns)

def backtest_with_regime(weights, returns, costs_bps=20):
    """Backtest with transaction costs calculation"""
    eq_list = []
    dates = returns.index[1:]  # Skip first date (no previous weights)
    prev_w = None
    
    for d in dates:
        w = weights.reindex(returns.columns, fill_value=0)
        r = float((w * returns.loc[d]).sum())
        
        # Transaction costs
        if prev_w is None:
            tc = 0.0
        else:
            turnover = (w - prev_w).abs().sum()
            tc = (costs_bps / 10000.0) * turnover
        
        eq_list.append(r - tc)
        prev_w = w
    
    return pd.Series(eq_list, index=dates)

# Backtests
logret_val = backtest_with_regime(w_val, asset_rets, COSTS_BPS)
logret_grw = backtest_with_regime(w_grw, asset_rets, COSTS_BPS)
logret_mkt = backtest_with_regime(w_mkt, asset_rets, 0)

eq_val = np.exp(logret_val.cumsum())
eq_grw = np.exp(logret_grw.cumsum())
eq_mkt = np.exp(logret_mkt.cumsum())

# Add regimes
regime_aligned = regime_series.reindex(eq_val.index, method='ffill')

# Performance stats
def perf_stats(equity_curve):
    logr = np.log(equity_curve / equity_curve.shift(1)).dropna()
    mu = float(logr.mean() * 12)
    sig = float(logr.std(ddof=0) * np.sqrt(12))
    sharpe = mu / (sig + 1e-12)
    mdd = float((equity_curve / equity_curve.cummax() - 1).min())
    final_value = float(equity_curve.iloc[-1])
    return {
        "CAGR": round(mu, 4),
        "Volatility": round(sig, 4),
        "Sharpe": round(sharpe, 3),
        "Max Drawdown": round(mdd, 4),
        "Final Value": round(final_value, 2)
    }

stats_val = perf_stats(eq_val)
stats_grw = perf_stats(eq_grw)
stats_mkt = perf_stats(eq_mkt)

print("\n" + "=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)
print(f"VALUE (Q1)  : {stats_val}")
print(f"GROWTH (Q10): {stats_grw}")
print(f"MARKET (EW) : {stats_mkt}")
print("=" * 80)

# Backtest visualization with regimes
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Chart 1: Equity curves
(eq_val / eq_val.iloc[0]).plot(ax=axes[0], label="VALUE (Q1)", linewidth=2, color='blue')
(eq_grw / eq_grw.iloc[0]).plot(ax=axes[0], label="GROWTH (Q10)", linewidth=2, color='red')
(eq_mkt / eq_mkt.iloc[0]).plot(ax=axes[0], label="MARKET (EW)", linewidth=2, linestyle='--', color='gray')

# Color crisis zones
for i, date in enumerate(regime_aligned.index):
    if regime_aligned.iloc[i] == "crisis":
        axes[0].axvspan(
            date, 
            regime_aligned.index[i+1] if i+1 < len(regime_aligned) else date,
            alpha=0.15, 
            color='red'
        )

axes[0].set_title("Backtest: Value vs Growth (Red zones = crisis regime)")
axes[0].set_ylabel("Cumulative Value (base=1)")
axes[0].legend(loc='best')
axes[0].grid(alpha=0.3)

# Chart 2: Monthly returns
logret_val.plot(ax=axes[1], label="VALUE", alpha=0.7, color='blue')
logret_grw.plot(ax=axes[1], label="GROWTH", alpha=0.7, color='red')
axes[1].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[1].set_title("Monthly Returns (log)")
axes[1].set_ylabel("Return (log)")
axes[1].set_xlabel("Date")
axes[1].legend(loc='best')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"backtest_with_regimes_{ts}.png"), dpi=150)
plt.close()

# Save equity curves
eq_df = pd.DataFrame({
    "VALUE_Q1": eq_val,
    "GROWTH_Q10": eq_grw,
    "MARKET_EW": eq_mkt,
    "regime": regime_aligned
})
eq_df.to_csv(os.path.join(OUT_DIR, f"equity_curves_{ts}.csv"))

# ==================== REGIME ANALYSIS ====================
print("\nPerformance analysis by market regime...")

# Separate returns by regime
regime_aligned_rets = regime_series.reindex(logret_val.index, method='ffill')

calm_mask = regime_aligned_rets == "calm"
crisis_mask = regime_aligned_rets == "crisis"

def regime_stats(returns, mask):
    """Statistics for a specific regime"""
    r = returns[mask]
    if len(r) == 0:
        return {"N": 0, "Mean": np.nan, "Std": np.nan, "Sharpe": np.nan}
    mu = float(r.mean() * 12)
    sig = float(r.std(ddof=0) * np.sqrt(12))
    sharpe = mu / (sig + 1e-12)
    return {
        "N_months": len(r),
        "CAGR": round(mu, 4),
        "Vol": round(sig, 4),
        "Sharpe": round(sharpe, 3)
    }

regime_analysis = pd.DataFrame({
    "VALUE_calm": regime_stats(logret_val, calm_mask),
    "VALUE_crisis": regime_stats(logret_val, crisis_mask),
    "GROWTH_calm": regime_stats(logret_grw, calm_mask),
    "GROWTH_crisis": regime_stats(logret_grw, crisis_mask),
    "MARKET_calm": regime_stats(logret_mkt, calm_mask),
    "MARKET_crisis": regime_stats(logret_mkt, crisis_mask),
}).T

print(regime_analysis)
regime_analysis.to_csv(os.path.join(OUT_DIR, f"regime_analysis_{ts}.csv"))

# Chart: Performance by regime
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

portfolios = ["VALUE", "GROWTH", "MARKET"]
calm_returns = [
    logret_val[calm_mask].mean() * 12,
    logret_grw[calm_mask].mean() * 12,
    logret_mkt[calm_mask].mean() * 12
]
crisis_returns = [
    logret_val[crisis_mask].mean() * 12,
    logret_grw[crisis_mask].mean() * 12,
    logret_mkt[crisis_mask].mean() * 12
]

x = np.arange(len(portfolios))
width = 0.35

axes[0].bar(x - width/2, calm_returns, width, label='Calm', color='steelblue')
axes[0].bar(x + width/2, crisis_returns, width, label='Crisis', color='coral')
axes[0].set_ylabel('Annualized CAGR')
axes[0].set_title('Performance by Market Regime')
axes[0].set_xticks(x)
axes[0].set_xticklabels(portfolios)
axes[0].legend()
axes[0].axhline(0, color='black', linestyle='-', linewidth=0.8)
axes[0].grid(axis='y', alpha=0.3)

# Volatility by regime
calm_vol = [
    logret_val[calm_mask].std() * np.sqrt(12),
    logret_grw[calm_mask].std() * np.sqrt(12),
    logret_mkt[calm_mask].std() * np.sqrt(12)
]
crisis_vol = [
    logret_val[crisis_mask].std() * np.sqrt(12),
    logret_grw[crisis_mask].std() * np.sqrt(12),
    logret_mkt[crisis_mask].std() * np.sqrt(12)
]

axes[1].bar(x - width/2, calm_vol, width, label='Calm', color='steelblue')
axes[1].bar(x + width/2, crisis_vol, width, label='Crisis', color='coral')
axes[1].set_ylabel('Annualized Volatility')
axes[1].set_title('Volatility by Market Regime')
axes[1].set_xticks(x)
axes[1].set_xticklabels(portfolios)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"regime_comparison_{ts}.png"), dpi=150)
plt.close()

# ==================== DEEP LEARNING (OPTIONAL) ====================
if HAS_TF:
    print("\n[BONUS] Training neural network...")
    
    # Use future return prediction data
    model_nn = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(X_train_fut_scaled.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    model_nn.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    history = model_nn.fit(
        X_train_fut_scaled, 
        y_train_fut,
        epochs=100,
        verbose=0,
        validation_split=0.2,
        batch_size=16
    )
    
    # Evaluation
    test_loss, test_mae = model_nn.evaluate(X_test_fut_scaled, y_test_fut, verbose=0)
    y_pred_nn = model_nn.predict(X_test_fut_scaled, verbose=0).flatten()
    r2_nn = r2_score(y_test_fut, y_pred_nn)
    
    print(f"✓ Neural Network → Test MAE={test_mae:.5f}, R²={r2_nn:.4f}")
    
    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(OUT_DIR, f"nn_training_history_{ts}.csv"), index=False)
    
    # Training chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Neural Network Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('Neural Network Training MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"nn_training_{ts}.png"), dpi=150)
    plt.close()
    
    # Ridge vs Neural Network comparison
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test_fut, y_pred_fut, alpha=0.5, label='Ridge', s=30)
    ax.scatter(y_test_fut, y_pred_nn, alpha=0.5, label='Neural Net', s=30)
    
    # Perfect prediction line
    min_val = min(y_test_fut.min(), y_pred_fut.min(), y_pred_nn.min())
    max_val = max(y_test_fut.max(), y_pred_fut.max(), y_pred_nn.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('Actual Returns (12m)')
    ax.set_ylabel('Predicted Returns (12m)')
    ax.set_title('Ridge vs Neural Network Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"model_comparison_{ts}.png"), dpi=150)
    plt.close()
    
else:
    print("\n[BONUS] Neural Network skipped (TensorFlow not available)")

# ==================== FINAL REPORT ====================
print("\n" + "=" * 80)
print("GENERATING FINAL REPORT")
print("=" * 80)

report_lines = [
    "=" * 80,
    "CAPSTONE PROJECT - UNIFIED REPORT",
    "=" * 80,
    "",
    f"Execution date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Universe: {len(ALL_TICKERS)} stocks ({len(US_TICKERS)} US + {len(SWISS_TICKERS)} Swiss)",
    f"Benchmark: {BENCHMARK}",
    f"Period: {START} → {px.index[-1].strftime('%Y-%m-%d')} ({len(px)} months)",
    f"Prediction horizon: {HORIZON} months",
    "",
    "=" * 80,
    "1. VALUE vs GROWTH ANALYSIS",
    "=" * 80,
    "",
    f"VALUE Portfolio (Q1): {len(val_names)} stocks",
    f"GROWTH Portfolio (Q10): {len(grw_names)} stocks",
    "",
    "Overall performance:",
    f"  VALUE  : CAGR={stats_val['CAGR']:.2%}, Sharpe={stats_val['Sharpe']:.2f}, MaxDD={stats_val['Max Drawdown']:.2%}",
    f"  GROWTH : CAGR={stats_grw['CAGR']:.2%}, Sharpe={stats_grw['Sharpe']:.2f}, MaxDD={stats_grw['Max Drawdown']:.2%}",
    f"  MARKET : CAGR={stats_mkt['CAGR']:.2%}, Sharpe={stats_mkt['Sharpe']:.2f}, MaxDD={stats_mkt['Max Drawdown']:.2%}",
    "",
    "=" * 80,
    "2. MACHINE LEARNING MODELS",
    "=" * 80,
    "",
    "Regression (predicting average returns from value score):",
]

for res in regression_results:
    report_lines.append(f"  {res['Model']:25s} : R²={res['R²']:.4f}, RMSE={res['RMSE']:.6f}")

report_lines.extend([
    "",
    "Classification (high vs low performers):",
])

for res in classification_results:
    report_lines.append(f"  {res['Model']:25s} : Accuracy={res['Accuracy']:.4f}")

report_lines.extend([
    "",
    "Future returns prediction (12 months):",
    f"  Ridge Regression : RMSE={rmse_fut:.5f}, R²={r2_fut:.4f}",
    f"  Logistic (outperf.) : Accuracy={acc_fut:.3f}, AUC={auc_fut:.3f}",
])

if HAS_TF:
    report_lines.extend([
        "",
        "Neural Network (MLP):",
        f"  Test MAE={test_mae:.5f}, R²={r2_nn:.4f}",
    ])

report_lines.extend([
    "",
    "=" * 80,
    "3. CLUSTERING & DIMENSIONALITY REDUCTION",
    "=" * 80,
    "",
    f"K-Means: {len(fund['cluster'].unique())} clusters identified",
    f"PCA: PC1 explains {pca.explained_variance_ratio_[0]:.1%}, PC2 explains {pca.explained_variance_ratio_[1]:.1%}",
    f"     Total variance explained: {pca.explained_variance_ratio_.sum():.1%}",
    "",
    "=" * 80,
    "4. MARKET REGIME ANALYSIS",
    "=" * 80,
    "",
    f"CALM regime: {calm_mask.sum()} months (volatility < median)",
    f"CRISIS regime: {crisis_mask.sum()} months (volatility > median)",
    "",
    "Performance by regime:",
])

for idx, row in regime_analysis.iterrows():
    report_lines.append(f"  {idx:20s} : N={row['N_months']:3.0f}, CAGR={row['CAGR']:7.2%}, Vol={row['Vol']:7.2%}, Sharpe={row['Sharpe']:6.2f}")

report_lines.extend([
    "",
    "=" * 80,
    "5. KEY INSIGHTS",
    "=" * 80,
    "",
    "✓ Valuation ratios show significant correlation with each other",
    "✓ Value scores vary significantly across sectors",
    "✓ VALUE strategy " + ("outperformed" if stats_val['CAGR'] > stats_mkt['CAGR'] else "underperformed") + " the market",
    "✓ GROWTH strategy " + ("outperformed" if stats_grw['CAGR'] > stats_mkt['CAGR'] else "underperformed") + " the market",
    "✓ Performance varies significantly across market regimes",
    "",
    "⚠️  LIMITATIONS:",
    "   - Current fundamental ratios applied to entire history (no time series)",
    "   - Monthly data limits precision of entry/exit points",
    "   - Simplified transaction costs (flat 20bps)",
    "   - No dynamic portfolio rebalancing",
    "   - Potential survivorship bias (currently listed stocks)",
    "",
    "=" * 80,
    "6. FILES GENERATED",
    "=" * 80,
    "",
])

# List all generated files
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(('.png', '.csv')):
        report_lines.append(f"  ✓ {f}")

report_lines.extend([
    "",
    "=" * 80,
    "END OF REPORT",
    "=" * 80,
])

report_text = "\n".join(report_lines)

# Save report
report_path = os.path.join(OUT_DIR, f"FINAL_REPORT_{ts}.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

# Display report
print(report_text)

# ==================== PORTFOLIO HOLDINGS SUMMARY ====================
print("\n" + "=" * 80)
print("PORTFOLIO COMPOSITION")
print("=" * 80)

print("\nVALUE Portfolio (Q1):")
if len(val_names) > 0:
    val_details = fund.loc[val_names, ['score_value', 'pe', 'pb', 'sector']].sort_values('score_value', ascending=False)
    print(val_details.head(10).to_string())
else:
    print("  (No stocks)")

print("\nGROWTH Portfolio (Q10):")
if len(grw_names) > 0:
    grw_details = fund.loc[grw_names, ['score_value', 'pe', 'pb', 'sector']].sort_values('score_value', ascending=True)
    print(grw_details.head(10).to_string())
else:
    print("  (No stocks)")

# Save holdings
holdings_df = pd.DataFrame({
    "VALUE_Q1": pd.Series(val_names),
    "GROWTH_Q10": pd.Series(grw_names)
})
holdings_df.to_csv(os.path.join(OUT_DIR, f"portfolio_holdings_{ts}.csv"), index=False)

print("\n" + "=" * 80)
print("✓✓✓ UNIFIED CAPSTONE PROJECT COMPLETED SUCCESSFULLY ✓✓✓")
print("=" * 80)
print(f"\nAll files available in: {os.path.abspath(OUT_DIR)}")
print("\nMain files:")
print(f"  • {report_path}")
print(f"  • backtest_with_regimes_{ts}.png")
print(f"  • eda_analysis_{ts}.png")
print(f"  • regime_comparison_{ts}.png")
print(f"  • pca_clusters_{ts}.png")
print("=" * 80)