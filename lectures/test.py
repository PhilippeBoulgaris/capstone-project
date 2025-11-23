import yfinance as yf
import pandas as pd

# Télécharger les données de 2330.TW sur 5 ans en daily
ticker = yf.Ticker("2330.TW")
data = ticker.history(period="5y", interval="1d")

# Choisir la bonne colonne de prix (Adj Close si dispo, sinon Close)
if "Adj Close" in data.columns:
    price_col = "Adj Close"
else:
    price_col = "Close"

# Calcul des rendements journaliers
data["daily_return"] = data[price_col].pct_change()

# On enlève la première ligne (NaN) et on garde uniquement la colonne des rendements
daily_returns = data[["daily_return"]].dropna()

# Sauvegarde dans un CSV nommé "daily.csv"
daily_returns.to_csv("daily.csv", index=True)  # index = Date

print("Fichier 'daily.csv' créé avec les rendements journaliers.")
