"""
fix_prices_yfinance.py

Remplace la table `prices` de la base (issue de FNSPID, dont l'ajustement
des splits s'est révélé défaillant -- close et adj_close identiques,
provoquant un faux "krach" sur les tickers ayant splitté, ex: TSLA) par des
prix correctement ajustés (splits + dividendes) téléchargés directement
depuis Yahoo Finance.

Les news/sentiments de FNSPID ne sont pas concernés et restent inchangés --
seule la table `prices` est reconstruite.

Prérequis :
    pip install yfinance duckdb pandas
"""

import duckdb
import pandas as pd
import yfinance as yf

DB_PATH = "fnspid.duckdb"
START_DATE = "2018-01-01"
END_DATE = "2020-06-30"


def main():
    con = duckdb.connect(DB_PATH)
    tickers = con.execute("SELECT ticker FROM companies ORDER BY ticker").df()["ticker"].tolist()
    print(f"{len(tickers)} tickers à retélécharger : {tickers}")

    print("\nTéléchargement des prix ajustés (Yahoo Finance)...")
    # auto_adjust=True : open/high/low/close sont directement corrigés des
    # splits ET des dividendes -- plus besoin de colonne "adj_close" séparée.
    raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, group_by="ticker", progress=True)

    frames = []
    missing = []
    for ticker in tickers:
        try:
            df = raw[ticker].copy()
        except KeyError:
            missing.append(ticker)
            continue
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        df["adj_close"] = df["close"]  # identique par construction (auto_adjust=True)
        frames.append(df[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]])

    if missing:
        print(f"\nTickers sans donnée récupérée ({len(missing)}) : {missing}")

    new_prices = pd.concat(frames, ignore_index=True)
    new_prices = new_prices.dropna(subset=["close"])
    print(f"\n{len(new_prices)} lignes de prix récupérées au total.")

    print("\nRemplacement de la table 'prices' dans DuckDB...")
    con.execute("DROP TABLE IF EXISTS prices")
    con.execute("""
        CREATE TABLE prices AS
        SELECT ticker, date::DATE AS date, open, high, low, close, adj_close, volume
        FROM new_prices
    """)

    n = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    print(f"Table 'prices' reconstruite : {n:,} lignes.")

    con.close()
    print("\nTerminé. Relancez maintenant 3_apt_regression.py, 4_lstm_model.py puis 5_dataviz.py.")


if __name__ == "__main__":
    main()
