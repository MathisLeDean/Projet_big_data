"""
3_apt_regression.py

Construit le facteur marché et le sentiment agrégé, puis lance une régression
APT (multi-facteurs) par entreprise :

    Rendement(i,t) = alpha(i) + beta_marche(i) * Rendement_marche(t)
                              + beta_sentiment(i) * Sentiment(i,t) + erreur

beta_sentiment(i) est l'indicateur clé : plus il est élevé (en valeur absolue),
plus l'entreprise i sur-réagit au climat médiatique par rapport à la moyenne.

"""

import duckdb
import pandas as pd
import statsmodels.api as sm

DB_PATH = "fnspid.duckdb"
MIN_SENTIMENT_OBS = 30  # nombre minimum de jours avec sentiment pour qu'une régression soit fiable
FRED_DGS10_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
FRED_VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"


def _load_fred_series(url: str, value_name: str) -> pd.DataFrame:
    """Télécharge une série FRED et calcule sa variation quotidienne.
    FRED marque les jours sans valeur (fériés...) avec un "." -> traités comme NaN."""
    df = pd.read_csv(url)
    df.columns = ["date", value_name]
    df["date"] = pd.to_datetime(df["date"])
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=[value_name]).sort_values("date")

    delta_col = f"delta_{value_name}"
    df[delta_col] = df[value_name].diff()
    return df.dropna(subset=[delta_col])[["date", delta_col]]


def load_macro_factors() -> pd.DataFrame:
    """
    Facteur "conjoncture", en deux dimensions complémentaires :
    - delta_taux : variation du taux 10 ans US (DGS10) -> anticipations de
      politique monétaire / coût du crédit
    - delta_vix : variation du VIX (VIXCLS) -> stress / incertitude de marché,
      une dimension différente de la conjoncture, pas juste les taux

    Les deux séries sont fusionnées sur la date ; une date absente de l'une
    des deux (jour férié différent entre les deux séries, rare) est retirée.
    """
    print("Téléchargement du taux 10 ans (FRED, DGS10)...")
    taux = _load_fred_series(FRED_DGS10_URL, "taux")

    print("Téléchargement du VIX (FRED, VIXCLS)...")
    vix = _load_fred_series(FRED_VIX_URL, "vix")

    macro = taux.merge(vix, on="date", how="inner")
    print(f"  {len(macro)} jours avec les deux facteurs macro disponibles "
          f"({macro['date'].min().date()} -> {macro['date'].max().date()})")
    return macro


def load_data():
    con = duckdb.connect(DB_PATH)
    prices = con.execute("SELECT ticker, date, adj_close AS close FROM prices ORDER BY ticker, date").df()
    news = con.execute("""
        SELECT ticker, date, sentiment_score
        FROM news
        WHERE sentiment_score IS NOT NULL
    """).df()
    con.close()
    return prices, news


def build_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcule le rendement quotidien (variation relative du close) par ticker."""
    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["return"] = prices.groupby("ticker")["close"].pct_change(fill_method=None)
    return prices.dropna(subset=["return"])


def build_market_factor(returns: pd.DataFrame) -> pd.DataFrame:
    """Facteur marché = moyenne des rendements de toutes les entreprises, par jour."""
    market = returns.groupby("date")["return"].mean().reset_index()
    market = market.rename(columns={"return": "market_return"})
    return market


def build_daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Sentiment moyen par jour et par entreprise (une news peut être publiée
    à toute heure -> on agrège à la journée)."""
    news = news.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    daily = news.groupby(["ticker", "date"])["sentiment_score"].mean().reset_index()
    return daily


def run_apt_regressions(returns: pd.DataFrame, market: pd.DataFrame, sentiment: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    results = []
    excluded = []

    panel = returns.merge(market, on="date", how="left")
    panel = panel.merge(macro, on="date", how="inner")  # inner : on ne garde que les jours où les facteurs macro existent

    FACTOR_COLS = ["market_return", "sentiment_score", "delta_taux", "delta_vix"]

    for ticker, group in panel.groupby("ticker"):
        group = group.merge(
            sentiment[sentiment["ticker"] == ticker][["date", "sentiment_score"]],
            on="date", how="left"
        )
        n_sentiment_obs = group["sentiment_score"].notna().sum()

        if n_sentiment_obs < MIN_SENTIMENT_OBS:
            excluded.append((ticker, n_sentiment_obs))
            continue

        # Jour sans article = sentiment neutre (0), plutôt que d'exclure la ligne
        group["sentiment_score"] = group["sentiment_score"].fillna(0)

        X = sm.add_constant(group[FACTOR_COLS])
        y = group["return"]
        model = sm.OLS(y, X, missing="drop").fit()

        results.append({
            "ticker": ticker,
            "alpha": model.params["const"],
            "beta_marche": model.params["market_return"],
            "pval_marche": model.pvalues["market_return"],
            "beta_sentiment": model.params["sentiment_score"],
            "pval_sentiment": model.pvalues["sentiment_score"],
            "beta_taux": model.params["delta_taux"],
            "pval_taux": model.pvalues["delta_taux"],
            "beta_vix": model.params["delta_vix"],
            "pval_vix": model.pvalues["delta_vix"],
            "r_squared": model.rsquared,
            "n_obs": int(model.nobs),
            "n_articles": int(n_sentiment_obs),
        })

    if excluded:
        print(f"\nTickers exclus (moins de {MIN_SENTIMENT_OBS} jours avec sentiment) : {len(excluded)}")
        for t, n in excluded:
            print(f"  {t} : {n} jours avec sentiment")

    return pd.DataFrame(results)


def main():
    print("Chargement des données depuis DuckDB...")
    prices, news = load_data()

    print("Calcul des rendements quotidiens...")
    returns = build_returns(prices)

    print("Construction du facteur marché...")
    market = build_market_factor(returns)

    print("Agrégation du sentiment quotidien par entreprise...")
    daily_sentiment = build_daily_sentiment(news)

    print("Chargement des facteurs macro (conjoncture)...")
    macro = load_macro_factors()

    print("\nLancement des régressions APT par entreprise...")
    results = run_apt_regressions(returns, market, daily_sentiment, macro)

    results = results.sort_values("beta_sentiment", ascending=False)

    print(f"\n{len(results)} entreprises avec une régression valide.\n")
    print("--- Classement par beta_sentiment (sur-réaction au sentiment médiatique) ---")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(results[[
        "ticker", "beta_sentiment", "pval_sentiment",
        "beta_marche", "pval_marche",
        "beta_taux", "pval_taux", "beta_vix", "pval_vix",
        "r_squared", "n_articles"
    ]])

    output_path = "apt_regression_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nRésultats sauvegardés dans : {output_path}")

    # Petits résumés interprétatifs automatiques
    sig_sentiment = results[results["pval_sentiment"] < 0.05]
    sig_taux = results[results["pval_taux"] < 0.05]
    sig_vix = results[results["pval_vix"] < 0.05]
    print(f"\nEntreprises avec beta_sentiment significatif (p < 0.05) : {len(sig_sentiment)}/{len(results)}")
    print(f"Entreprises avec beta_taux significatif (p < 0.05)      : {len(sig_taux)}/{len(results)}")
    print(f"Entreprises avec beta_vix significatif (p < 0.05)       : {len(sig_vix)}/{len(results)}")


if __name__ == "__main__":
    main()
