import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif : jamais de fenêtre, jamais de blocage sur plt.show()
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

DB_PATH = "fnspid.duckdb"
FRED_DGS10_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
FRED_VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def _load_fred_series(url: str, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = ["date", value_name]
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna().sort_values("date")
    df[f"delta_{value_name}"] = df[value_name].diff()
    return df.dropna()[["date", f"delta_{value_name}"]]


def load_and_prepare_data():
    print("  Connexion à la base DuckDB...", flush=True)
    con = duckdb.connect(DB_PATH)
    prices = con.execute("SELECT ticker, date, adj_close AS close FROM prices").df()
    news = con.execute("SELECT ticker, date, sentiment_score FROM news WHERE sentiment_score IS NOT NULL").df()
    con.close()
    print(f"  {len(prices)} lignes de prix, {len(news)} lignes de sentiment chargées.", flush=True)

    prices["date"] = pd.to_datetime(prices["date"], utc=True).dt.tz_localize(None).dt.normalize()
    news["date"] = pd.to_datetime(news["date"], utc=True).dt.tz_localize(None).dt.normalize()

    prices = prices.sort_values(["ticker", "date"])
    prices["return"] = prices.groupby("ticker")["close"].pct_change()
    prices["target_next_day"] = prices.groupby("ticker")["return"].shift(-1)

    market = prices.groupby("date")["return"].mean().reset_index().rename(columns={"return": "market_return"})
    daily_sentiment = news.groupby(["ticker", "date"])["sentiment_score"].mean().reset_index()

    print("  Téléchargement des séries FRED (taux 10 ans, VIX)...", flush=True)
    taux = _load_fred_series(FRED_DGS10_URL, "taux")
    vix = _load_fred_series(FRED_VIX_URL, "vix")
    macro = taux.merge(vix, on="date", how="inner")
    print(f"  {len(macro)} jours avec facteurs macro disponibles.", flush=True)

    df = prices.dropna(subset=["return", "target_next_day"]).merge(market, on="date", how="left")
    df = df.merge(daily_sentiment, on=["ticker", "date"], how="left").fillna({"sentiment_score": 0})
    df = df.merge(macro, on="date", how="inner")

    result = df.dropna()
    print(f"  {len(result)} lignes après fusion complète et nettoyage.", flush=True)
    return result


def main():
    print("Préparation des données pour Elastic Net...", flush=True)
    df = load_and_prepare_data()

    if len(df) == 0:
        print("ERREUR : le DataFrame final est vide après les jointures -- "
              "vérifiez que les dates de prices/news et macro se recoupent bien. Arrêt.")
        return

    features = ["return", "market_return", "sentiment_score", "delta_taux", "delta_vix"]

    df_sorted = df.sort_values("date")
    split_idx = int(len(df_sorted) * 0.8)

    X_train, X_test = df_sorted[features].iloc[:split_idx], df_sorted[features].iloc[split_idx:]
    y_train, y_test = df_sorted["target_next_day"].iloc[:split_idx], df_sorted["target_next_day"].iloc[split_idx:]

    print(f"Entraînement sur {len(X_train)} jours, Test sur {len(X_test)} jours.", flush=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Recherche des meilleurs hyperparamètres (Validation Croisée)...", flush=True)
    elastic_net = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99, 1.0],
        cv=5,
        random_state=42,
        max_iter=10000,
    )
    elastic_net.fit(X_train_scaled, y_train)

    y_pred = elastic_net.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    hit_ratio = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100

    print(f"\n--- Résultats Elastic Net ---")
    print(f"Meilleur ratio L1 trouvé : {elastic_net.l1_ratio_} (1.0 = Lasso strict)")
    print(f"Meilleur Alpha (pénalité)  : {elastic_net.alpha_:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"Hit Ratio directionnel : {hit_ratio:.2f}%")

    coefficients = elastic_net.coef_

    plt.figure(figsize=(8, 5))
    colors = ["#1B6B55" if c > 0 else "#A33228" for c in coefficients]
    plt.bar(features, coefficients, color=colors)
    plt.axhline(0, color="black", linewidth=1)

    for i, coef in enumerate(coefficients):
        plt.text(i, coef + (np.sign(coef) * 0.0001), f"{coef:.5f}", ha='center', va='bottom' if coef > 0 else 'top', fontsize=9)

    plt.title(f"Elastic Net : Poids des variables (Coefficients Standardisés)\nAlpha: {elastic_net.alpha_:.5f} | L1 Ratio: {elastic_net.l1_ratio_}")
    plt.ylabel("Impact sur le rendement de demain")
    plt.xticks(rotation=15)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)  # évite le FileNotFoundError si le dossier n'existe pas
    out_img = "figures/11_elastic_net_coefs.png"
    plt.savefig(out_img, dpi=200)
    plt.close()  # ferme la figure au lieu de plt.show() -- pas de fenêtre, pas de blocage
    print(f"\nGraphique sauvegardé : {out_img}")


if __name__ == "__main__":
    main()