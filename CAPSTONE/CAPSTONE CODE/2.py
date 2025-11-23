import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Optional imports - will skip sections if not available A VERIFIER 
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn not found - skipping correlation heatmap")

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  tensorflow not found - skipping neural network section")

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Choice of tickers
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","BAC","WFC","MS","GS",
           "V","MA","PYPL","KO","PEP","PG","JNJ","UNH","MRK","PFE","T","VZ","CMCSA","DIS","NFLX",
           "INTC","AMD","IBM","ORCL","CRM","COST","HD","WMT","TGT","LOW","NKE","MCD","SBUX","CAT",
           "BA","GE","LMT","RTX","XOM","CVX","COP","SLB"]
START = "2014-01-01"
INTERVAL = "1mo" # daily would include noise 
COSTS_BPS = 20

# put the output in a folder
out_dir = "./capstone_output"
os.makedirs(out_dir, exist_ok=True)
ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

print("=" * 80)
print("CAPSTONE PROJECT: VALUE vs GROWTH with DATA SCIENCE TECHNIQUES")
print("=" * 80)

# 1 DATA ACQUISITION from the ticker choice
print("\n[1/8] Downloading price data...")
px = yf.download(TICKERS, start=START, interval=INTERVAL, auto_adjust=True, progress=False)["Close"]
px = px.dropna(how="all")
rets = np.log(px/px.shift(1)).dropna()

print(f"✓ Downloaded {len(px)} months of data for {len(px.columns)} tickers")

# 2: FUNDAMENTAL part (Data)
print("\n[2/8] Fetching fundamental data from Yahoo Finance...")

def fetch_info(ticker):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    mcap = info.get("marketCap")
    fcf  = info.get("freeCashflow")
    p_fcf = (mcap / fcf) if (mcap and fcf and fcf != 0) else None
    return {
        "ticker": ticker,
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "ps": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "p_fcf": p_fcf,
        "revenue_growth": info.get("revenueGrowth"),
        "eps_growth": info.get("earningsQuarterlyGrowth"),
        "sector": info.get("sector"),
    }

fund = pd.DataFrame([fetch_info(t) for t in TICKERS]).set_index("ticker")
print(f"✓ Fetched fundamentals for {len(fund)} stocks")

# Start of the data analyis
# 3 VALUE SCORE CALCULATION for stocks
print("\n[3/8] Computing value scores...")

value_cols = [c for c in ["pe","pb","ps","ev_ebitda","p_fcf"] if c in fund.columns]
inv = fund[value_cols].copy()

# Invert multiples otherwise it doesn't make any sense ! (higher P/E for example is not good for value)
for c in value_cols:
    inv[c] = 1.0 / fund[c].where(fund[c] > 0)

# Clean winsorise for the outliers that we don't want to reduce bias
inv = inv.replace([np.inf, -np.inf], np.nan)
inv = inv.fillna(inv.median(numeric_only=True))
inv = inv.clip(lower=inv.quantile(0.01), upper=inv.quantile(0.99), axis=1)

# Z-score, average
Z = (inv - inv.mean())/inv.std(ddof=0)
score_value = Z.mean(axis=1).rename("score_value")

# Create deciles 
valid = score_value.dropna()
dec = pd.qcut(valid.rank(method="first", ascending=False), 10, labels=False, duplicates="drop") + 1
deciles = pd.Series(index=score_value.index, dtype="Int64")
deciles.loc[dec.index] = dec

val_names = deciles.index[deciles == 1]     # VALUE = Q1 (highest scores)
grw_names = deciles.index[deciles == 10]    # GROWTH = Q10 (lowest scores)

print(f"✓ Value portfolio: {len(val_names)} stocks")
print(f"✓ Growth portfolio: {len(grw_names)} stocks")

# 4 EXPLORATORY DATA ANALYSIS 
print("\n[4/8] Performing EDA and feature engineering ...")

# Add momentum and volatility features to enrich the set of explanatory variables and capture risk and trend effects
rets_60d = px.pct_change(60).iloc[-1]
fund["momentum"] = rets_60d.reindex(fund.index)
fund["volatility"] = rets.std().reindex(fund.index)
fund["score_value"] = score_value

# EDA plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Distribution of value scores
score_value.hist(bins=20, ax=axes[0,0], edgecolor='black')
axes[0,0].set_title("Distribution of Value Scores")
axes[0,0].set_xlabel("Value Score")
axes[0,0].set_ylabel("Frequency")

# Correlation heatmap
if len(value_cols) > 1: #otherwise it's complicated...
    if HAS_SEABORN:
        sns.heatmap(fund[value_cols].corr(), annot=True, ax=axes[0,1], cmap="coolwarm", center=0)
    else:
        corr = fund[value_cols].corr()
        im = axes[0,1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        axes[0,1].set_xticks(range(len(corr.columns)))
        axes[0,1].set_yticks(range(len(corr.columns)))
        axes[0,1].set_xticklabels(corr.columns, rotation=45, ha="right")
        axes[0,1].set_yticklabels(corr.columns)
        plt.colorbar(im, ax=axes[0,1])
    axes[0,1].set_title("Correlation of Valuation Ratios")

# Value score by sector
fund_clean = fund.dropna(subset=["sector", "score_value"])
if len(fund_clean) > 0:
    sector_scores = fund_clean.groupby("sector")["score_value"].mean().sort_values()
    sector_scores.plot(kind="barh", ax=axes[1,0], color="steelblue")
    axes[1,0].set_title("Average Value Score by Sector")
    axes[1,0].set_xlabel("Value Score")

# Momentum vs Value
fund_clean2 = fund.dropna(subset=["score_value", "momentum"])
if len(fund_clean2) > 0:
    axes[1,1].scatter(fund_clean2["score_value"], fund_clean2["momentum"], alpha=0.6)
    axes[1,1].set_title("Value Score vs Momentum")
    axes[1,1].set_xlabel("Value Score")
    axes[1,1].set_ylabel("60-Month Return")
    axes[1,1].axhline(0, color='red', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"eda_{ts}.png"), dpi=150)
plt.close()
print(f"✓ Saved EDA plots")

# ==================== SECTION 5: LINEAR REGRESSION (Week 7) ====================
print("\n[5/8] Building regression models (Week 7)...")

# Prepare data: predict average return from value score
X = score_value.values.reshape(-1, 1)
y = rets.mean(axis=0).reindex(score_value.index).values

# Remove NaN
mask = ~(np.isnan(X.flatten()) | np.isnan(y))
X_clean = X[mask]
y_clean = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

models = {
    "Linear": LinearRegression(),
    "Ridge (α=1.0)": Ridge(alpha=1.0),
    "Lasso (α=0.01)": Lasso(alpha=0.01)
}

regression_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    regression_results.append({"Model": name, "R²": round(r2, 4), "MSE": round(mse, 6)})
    print(f"  {name:20s} → R²: {r2:.4f}, MSE: {mse:.6f}")

reg_df = pd.DataFrame(regression_results)
reg_df.to_csv(os.path.join(out_dir, f"regression_results_{ts}.csv"), index=False)

# ==================== SECTION 6: CLASSIFICATION (Week 8) ====================
print("\n[6/8] Building classification models (Week 8)...")

# Predict if stock is "high performer" (above median return)
median_ret = np.nanmedian(y)
y_class = (y > median_ret).astype(int)

mask_class = ~(np.isnan(X.flatten()) | np.isnan(y_class))
X_class = X[mask_class]
y_class_clean = y_class[mask_class]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class_clean, test_size=0.2, random_state=42
)

classifiers = {
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
}

classification_results = []
for name, clf in classifiers.items():
    clf.fit(X_train_c, y_train_c)
    pred = clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, pred)
    classification_results.append({"Model": name, "Accuracy": round(acc, 4)})
    print(f"  {name:20s} → Accuracy: {acc:.4f}")

clf_df = pd.DataFrame(classification_results)
clf_df.to_csv(os.path.join(out_dir, f"classification_results_{ts}.csv"), index=False)

# ==================== SECTION 7: CLUSTERING & PCA (Week 9) ====================
print("\n[7/8] Performing clustering and PCA (Week 9)...")

X_cluster = fund[value_cols].fillna(fund[value_cols].median()).values

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
fund["cluster"] = kmeans.fit_predict(X_cluster)

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X_cluster)
fund["PC1"], fund["PC2"] = coords[:, 0], coords[:, 1]

print(f"✓ PCA explained variance: {pca.explained_variance_ratio_}")

# Plot clusters
fig, ax = plt.subplots(figsize=(10, 6))
for cluster in fund["cluster"].unique():
    subset = fund[fund["cluster"] == cluster]
    ax.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}", alpha=0.7, s=80)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("Stock Clusters (K-Means + PCA)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"pca_clusters_{ts}.png"), dpi=150)
plt.close()
print(f"✓ Saved clustering plot")

# ==================== SECTION 8: DEEP LEARNING (Week 10) ====================
if HAS_TF:
    print("\n[8/8] Training neural network (Week 10)...")

    # Simple MLP for return prediction
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(1,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Suppress verbose output
    history = model.fit(X_train, y_train, epochs=100, verbose=0, validation_split=0.2, batch_size=8)

    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"✓ Neural Network → Test MAE: {test_mae:.6f}, Test Loss: {test_loss:.6f}")

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(out_dir, f"nn_training_{ts}.csv"), index=False)
else:
    print("\n[8/8] Skipping neural network (tensorflow not installed)")

# ==================== SECTION 9: BACKTESTING ====================
print("\n[Bonus] Running backtests...")

# Weight panels
w_val = pd.Series(1/len(val_names), index=val_names) if len(val_names) else pd.Series(dtype=float)
w_grw = pd.Series(1/len(grw_names), index=grw_names) if len(grw_names) else pd.Series(dtype=float)
w_mkt = pd.Series(1/len(rets.columns), index=rets.columns)

weights_val, weights_grw, weights_mkt = {}, {}, {}
dates = list(rets.index)
for i in range(1, len(dates)):
    d = dates[i]
    weights_val[d] = w_val
    weights_grw[d] = w_grw
    weights_mkt[d] = w_mkt

def backtest(weights_panel, returns, costs_bps=20.0):
    dates_bt = sorted(set(weights_panel.keys()).intersection(returns.index))
    eq = []
    prev_w = None
    for d in dates_bt:
        w = weights_panel[d].reindex(returns.columns).fillna(0.0)
        r = float((w * returns.loc[d]).sum())
        if prev_w is None:
            tc = 0.0
        else:
            turnover = (w - prev_w).abs().sum()
            tc = (costs_bps/10000.0) * turnover
        eq.append(r - tc)
        prev_w = w
    logret = pd.Series(eq, index=dates_bt)
    return np.exp(logret.cumsum())

eq_val = backtest(weights_val, rets, COSTS_BPS)
eq_grw = backtest(weights_grw, rets, COSTS_BPS)
eq_mkt = backtest(weights_mkt, rets, 0.0)

# Plot
def _base1(s): return s / s.iloc[0] if len(s) else s

fig, ax = plt.subplots(figsize=(10, 6))
_base1(eq_val).rename("VALUE (Q1)").plot(ax=ax, linewidth=2)
_base1(eq_grw).rename("GROWTH (Q10)").plot(ax=ax, linewidth=2)
_base1(eq_mkt).rename("MARKET (EW)").plot(ax=ax, linewidth=2, linestyle="--")
ax.set_title(f"Value vs Growth vs Market ({ts})")
ax.set_ylabel("Cumulative Value (base=1)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"backtest_{ts}.png"), dpi=150)
plt.close()

# Save equity curves
eq_df = pd.concat([
    eq_val.rename("VALUE_Q1"),
    eq_grw.rename("GROWTH_Q10"),
    eq_mkt.rename("MARKET_EW")
], axis=1).dropna()
eq_df.to_csv(os.path.join(out_dir, f"equity_curves_{ts}.csv"))

# Performance stats
def perf_stats(eq):
    logr = np.log(eq/eq.shift(1)).dropna()
    mu = float(logr.mean()*12)
    sig = float(logr.std(ddof=0)*(12**0.5))
    sharpe = mu/(sig + 1e-12)
    mdd = float((eq/eq.cummax()-1).min())
    return {"CAGR": round(mu, 4), "Sharpe": round(sharpe, 3), "MaxDD": round(mdd, 4)}

print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(f"VALUE (Q1):    {perf_stats(eq_val)}")
print(f"GROWTH (Q10):  {perf_stats(eq_grw)}")
print(f"MARKET (EW):   {perf_stats(eq_mkt)}")

print("\n" + "=" * 80)
print("ALL FILES SAVED TO:", os.path.abspath(out_dir))
print("=" * 80)
print("✓ Project complete!")