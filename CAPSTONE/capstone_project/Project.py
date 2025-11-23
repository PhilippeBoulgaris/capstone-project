import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------- PARAMS --------------------
TICKERS = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","JPM","BAC","WFC","MS","GS",
           "V","MA","PYPL","KO","PEP","PG","JNJ","UNH","MRK","PFE","T","VZ","CMCSA","DIS","NFLX",
           "INTC","AMD","IBM","ORCL","CRM","COST","HD","WMT","TGT","LOW","NKE","MCD","SBUX","CAT",
           "BA","GE","LMT","RTX","XOM","CVX","COP","SLB"]
START = "2014-01-01"      # ~10 ans d'historique
INTERVAL = "1mo"          # mensuel pour calmer le bruit
COSTS_BPS = 20            # 20 bps par rééquilibrage (0.20%)

# -------------------- PRIX --------------------
px = yf.download(TICKERS, start=START, interval=INTERVAL, auto_adjust=True, progress=False)["Close"]
px = px.dropna(how="all")
rets = np.log(px/px.shift(1)).dropna()   # log-returns mensuels

# -------------------- RATIOS (snapshot Yahoo) --------------------
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

# -------------------- SCORE VALUE --------------------
value_cols = [c for c in ["pe","pb","ps","ev_ebitda","p_fcf"] if c in fund.columns]
inv = fund[value_cols].copy()

# inverser UNIQUEMENT si le multiple > 0 (sinon NaN)
for c in value_cols:
    inv[c] = 1.0 / fund[c].where(fund[c] > 0)

# nettoyage + winsorisation légère
inv = inv.replace([np.inf, -np.inf], np.nan)
inv = inv.fillna(inv.median(numeric_only=True))
inv = inv.clip(lower=inv.quantile(0.01), upper=inv.quantile(0.99), axis=1)

# z-score cross-section, moyenne => score_value (plus grand = plus "value")
Z = (inv - inv.mean())/inv.std(ddof=0)
score_value = Z.mean(axis=1).rename("score_value")

# -------------------- DÉCILES (corrigé) --------------------
# Q1 = vrai top Value (scores les + élevés)
valid = score_value.dropna()
dec = pd.qcut(
    valid.rank(method="first", ascending=False),   # tri décroissant !
    10, labels=False, duplicates="drop"
) + 1
deciles = pd.Series(index=score_value.index, dtype="Int64")
deciles.loc[dec.index] = dec

# membres fixes (snapshot)
val_names = deciles.index[deciles == 1]     # VALUE = Q1
grw_names = deciles.index[deciles == 10]    # GROWTH proxy = Q10

# -------------------- PANNEAU DE POIDS (1 mois de décalage) --------------------
w_val = pd.Series(1/len(val_names), index=val_names) if len(val_names) else pd.Series(dtype=float)
w_grw = pd.Series(1/len(grw_names), index=grw_names) if len(grw_names) else pd.Series(dtype=float)
w_mkt = pd.Series(1/len(rets.columns), index=rets.columns)

weights_val, weights_grw, weights_mkt = {}, {}, {}
dates = list(rets.index)
for i in range(1, len(dates)):  # on “trade” à t avec l’info dispo à t-1
    d = dates[i]
    weights_val[d] = w_val
    weights_grw[d] = w_grw
    weights_mkt[d] = w_mkt

# -------------------- BACKTEST --------------------
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

# -------------------- PLOT + SAVE + CSV + STATS (horodatés) --------------------
# Dossier de sortie
out_dir = "/files/capstone_project/fig"   # ou "./fig"
os.makedirs(out_dir, exist_ok=True)

ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
plot_path = os.path.join(out_dir, f"value_vs_growth_equity_{ts}.png")
csv_path  = os.path.join(out_dir, f"equity_curves_{ts}.csv")

def _base1(s: pd.Series):
    return s / s.iloc[0] if len(s) else s

# Figure
fig, ax = plt.subplots(figsize=(9,5))
_base1(eq_val).rename("VALUE (Q1)").plot(ax=ax)
_base1(eq_grw).rename("GROWTH (Q10)").plot(ax=ax)
_base1(eq_mkt).rename("MARKET (EW)").plot(ax=ax)
ax.set_title(f"Value vs Growth vs Market — snapshot (run {ts})")
ax.set_ylabel("Valeur cumulée (base=1)")
ax.set_xlabel("Date")
ax.legend()
fig.tight_layout()
fig.savefig(plot_path, dpi=150)
plt.close(fig)

# CSV
eq_df = pd.concat([
    eq_val.rename("VALUE_Q1"),
    eq_grw.rename("GROWTH_Q10"),
    eq_mkt.rename("MARKET_EW")
], axis=1).dropna()
eq_df.to_csv(csv_path)

# Stats
def perf_stats_from_equity(eq):
    logr = np.log(eq/eq.shift(1)).dropna()
    mu = float(logr.mean()*12)                     # CAGR (log)
    sig = float(logr.std(ddof=0)*(12**0.5))        # vol annualisée
    sharpe = mu/(sig + 1e-12)
    mdd = float((eq/eq.cummax()-1).min())
    return {"CAGR_log": round(mu, 4), "Sharpe": round(sharpe, 3), "MaxDrawdown": round(mdd, 4)}

print("[OK] Fichiers créés :")
print(" - Figure :", os.path.abspath(plot_path))
print(" - CSV    :", os.path.abspath(csv_path))
print("Stats VALUE :", perf_stats_from_equity(eq_val))
print("Stats GROWTH:", perf_stats_from_equity(eq_grw))
print("Stats MARKET:", perf_stats_from_equity(eq_mkt))
