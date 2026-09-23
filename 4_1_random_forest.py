"""
5_random_forest.py

Entraîne un modèle Random Forest et le compare à la régression linéaire naïve.
Vise à vérifier si un algorithme non-linéaire basé sur des arbres peut extraire
un signal prédictif là où l'approche linéaire échoue.
"""

import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

DB_PATH = "fnspid.duckdb"
LOOKBACK = 5
TEST_FRACTION = 0.15

def load_panel() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH)
    prices = con.execute("SELECT ticker, date, adj_close AS close FROM prices ORDER BY ticker, date").df()
    news = con.execute("SELECT ticker, date, sentiment_score FROM news WHERE sentiment_score IS NOT NULL").df()
    con.close()

    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["return"] = prices.groupby("ticker")["close"].pct_change(fill_method=None)
    prices = prices.dropna(subset=["return"])

    market = prices.groupby("date")["return"].mean().rename("market_return").reset_index()

    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    daily_sentiment = news.groupby(["ticker", "date"])["sentiment_score"].mean().reset_index()

    panel = prices.merge(market, on="date", how="left")
    panel = panel.merge(daily_sentiment, on=["ticker", "date"], how="left")
    panel["sentiment_score"] = panel["sentiment_score"].fillna(0)

    return panel[["ticker", "date", "return", "market_return", "sentiment_score"]]

def build_sequences(panel: pd.DataFrame, lookback: int):
    X, y, dates = [], [], []
    features = ["return", "market_return", "sentiment_score"]

    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date")
        values = group[features].values
        target_dates = group["date"].values

        for i in range(lookback, len(group)):
            X.append(values[i - lookback:i])
            y.append(values[i][0])
            dates.append(target_dates[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(dates)

def chronological_split(X, y, dates, test_fraction: float):
    cutoff = np.quantile(dates.astype("datetime64[ns]").astype(np.int64), 1 - test_fraction)
    is_test = dates.astype("datetime64[ns]").astype(np.int64) >= cutoff
    return X[~is_test], y[~is_test], X[is_test], y[is_test]

def main():
    print("Reconstruction du panel...")
    panel = load_panel()

    print(f"Construction des fenêtres de {LOOKBACK} jours...")
    X, y, dates = build_sequences(panel, LOOKBACK)
    
    # Aplatissement des données (Random Forest prend un format tabulaire classique : 5 jours * 3 features = 15 colonnes)
    X_flat = X.reshape(len(X), -1)

    X_train, y_train, X_test, y_test = chronological_split(X_flat, y, dates, TEST_FRACTION)
    print(f"Train : {len(X_train)} | Test : {len(X_test)}")

    # --- 1. Régression Linéaire (Baseline) ---
    print("\nEntraînement de la régression linéaire...")
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)

    # --- 2. Random Forest ---
    print("Entraînement du Random Forest (peut prendre 1 à 2 minutes)...")
    # n_estimators=100 (100 arbres), max_depth limité pour éviter un surapprentissage massif
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # --- Évaluation ---
    print("\n--- RÉSULTATS SUR LE TEST SET ---")
    
    mse_naive = mean_squared_error(y_test, np.zeros_like(y_test))
    r2_naive = r2_score(y_test, np.zeros_like(y_test))
    print(f"Prédiction naïve (0) : MSE = {mse_naive:.6f} | R² = {r2_naive:.4f}")

    for name, y_pred in [("Régression Linéaire", y_pred_lin), ("Random Forest", y_pred_rf)]:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name:<20} : MSE = {mse:.6f} | R² = {r2:.4f}")

if __name__ == "__main__":
    main()