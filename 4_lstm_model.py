"""
4_lstm_model.py

Compare un modèle Deep Learning (LSTM) à une régression linéaire simple,
sur une tâche de PRÉDICTION (contrairement à la régression APT qui était
contemporaine) : prédire le rendement du jour t à partir des 5 jours
précédents (rendement, rendement du marché, sentiment).

Découpage chronologique strict train/test (pas de mélange temporel).

Prérequis :
    pip install duckdb pandas torch scikit-learn
"""

import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

DB_PATH = "fnspid.duckdb"
LOOKBACK = 5           # nombre de jours passés utilisés pour prédire le jour suivant
TEST_FRACTION = 0.15   # 15% des dates les plus récentes -> jeu de test
HIDDEN_SIZE = 32
EPOCHS = 30
BATCH_SIZE = 64


def load_panel() -> pd.DataFrame:
    """Reconstruit le même panel (rendement, marché, sentiment) que pour l'APT."""
    con = duckdb.connect(DB_PATH)
    prices = con.execute("SELECT ticker, date, adj_close AS close FROM prices ORDER BY ticker, date").df()
    news = con.execute("""
        SELECT ticker, date, sentiment_score FROM news WHERE sentiment_score IS NOT NULL
    """).df()
    con.close()

    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["return"] = prices.groupby("ticker")["close"].pct_change(fill_method=None)
    prices = prices.dropna(subset=["return"])

    market = prices.groupby("date")["return"].mean().rename("market_return").reset_index()

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    daily_sentiment = news.groupby(["ticker", "date"])["sentiment_score"].mean().reset_index()

    panel = prices.merge(market, on="date", how="left")
    panel = panel.merge(daily_sentiment, on=["ticker", "date"], how="left")
    panel["sentiment_score"] = panel["sentiment_score"].fillna(0)

    return panel[["ticker", "date", "return", "market_return", "sentiment_score"]]


def build_sequences(panel: pd.DataFrame, lookback: int):
    """
    Pour chaque entreprise, construit des fenêtres glissantes :
    X = les `lookback` jours précédents (return, market_return, sentiment)
    y = le rendement du jour suivant (jamais vu par le modèle en entrée)
    On garde aussi la date cible, pour pouvoir faire un split chronologique propre.
    """
    X, y, dates = [], [], []
    features = ["return", "market_return", "sentiment_score"]

    for ticker, group in panel.groupby("ticker"):
        group = group.sort_values("date")
        values = group[features].values
        target_dates = group["date"].values

        for i in range(lookback, len(group)):
            X.append(values[i - lookback:i])
            y.append(values[i][0])  # "return" est la 1re colonne = rendement du jour cible
            dates.append(target_dates[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(dates)


def chronological_split(X, y, dates, test_fraction: float):
    """Split par date (pas aléatoire) : le test set est toujours dans le futur du train set."""
    cutoff = np.quantile(dates.astype("datetime64[ns]").astype(np.int64), 1 - test_fraction)
    is_test = dates.astype("datetime64[ns]").astype(np.int64) >= cutoff
    return X[~is_test], y[~is_test], X[is_test], y[is_test]


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)      # h_n : dernier état caché
        return self.head(h_n[-1]).squeeze(-1)


def train_lstm(X_train, y_train, X_test, y_test, n_features: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(n_features, HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train).to(device)
    y_train_t = torch.tensor(y_train).to(device)
    X_test_t = torch.tensor(X_test).to(device)

    n = len(X_train_t)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            pred = model(X_train_t[idx])
            loss = loss_fn(pred, y_train_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS} - loss (train) : {total_loss / n:.6f}")

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_t).cpu().numpy()

    return y_pred_test


def main():
    print("Reconstruction du panel (rendement, marché, sentiment)...")
    panel = load_panel()

    print(f"Construction des séquences (fenêtre de {LOOKBACK} jours)...")
    X, y, dates = build_sequences(panel, LOOKBACK)
    print(f"  {len(X)} séquences construites au total.")

    X_train, y_train, X_test, y_test = chronological_split(X, y, dates, TEST_FRACTION)
    print(f"  Train : {len(X_train)} séquences | Test : {len(X_test)} séquences (les plus récentes)")

    # --- Modèle 1 : LSTM ---
    print("\nEntraînement du LSTM...")
    y_pred_lstm = train_lstm(X_train, y_train, X_test, y_test, n_features=X.shape[2])

    # --- Modèle 2 : régression linéaire simple (baseline) ---
    # On aplatit la fenêtre (5 jours x 3 features = 15 variables) pour la régression linéaire
    print("\nEntraînement de la régression linéaire (baseline)...")
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    lin_model = LinearRegression()
    lin_model.fit(X_train_flat, y_train)
    y_pred_lin = lin_model.predict(X_test_flat)

    # --- Comparaison ---
    print("\n--- Résultats sur le jeu de TEST (données jamais vues, les plus récentes) ---")
    for name, y_pred in [("LSTM", y_pred_lstm), ("Régression linéaire", y_pred_lin)]:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  {name:<22} MSE = {mse:.6f} | R² = {r2:.4f}")

    # Baseline naïve : toujours prédire 0 (aucune information) -> sert de repère
    mse_naive = mean_squared_error(y_test, np.zeros_like(y_test))
    print(f"  {'Prédiction naïve (0)':<22} MSE = {mse_naive:.6f} | R² = {r2_score(y_test, np.zeros_like(y_test)):.4f}")

    pd.DataFrame({
        "y_true": y_test, "y_pred_lstm": y_pred_lstm, "y_pred_linear": y_pred_lin
    }).to_csv("lstm_vs_linear_predictions.csv", index=False)
    print("\nPrédictions sauvegardées dans : lstm_vs_linear_predictions.csv")


if __name__ == "__main__":
    main()
