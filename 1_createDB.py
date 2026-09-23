"""
1_createDB.py

Création de la base de données à partir des CSV générés par explore_fnspid.py.

DuckDB : base de données analytique EMBARQUÉE. Pas de serveur, pas de
conteneur Docker à lancer -- le fichier fnspid.duckdb EST la base, comme
avec SQLite, mais le moteur est optimisé pour les requêtes analytiques
(jointures, agrégations, GROUP BY) qu'on fera en partie 2 et 3, ce que
SQLite fait plus lentement.

Prérequis :
    pip install duckdb
"""

import duckdb
import os

DATA_DIR = "fnspid_data"
DB_PATH = "fnspid.duckdb"


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # on repart d'une base propre à chaque exécution

    con = duckdb.connect(DB_PATH)

    # --- Table prices ---
    # read_csv_auto détecte automatiquement les types de colonnes.
    con.execute(f"""
        CREATE TABLE prices AS
        SELECT
            ticker,
            date::DATE AS date,
            open, high, low, close,
            "adj close" AS adj_close,
            volume
        FROM read_csv_auto('{DATA_DIR}/prices_sample.csv')
    """)

    # --- Table news ---
    # sentiment_score est vide pour l'instant : rempli en partie 2 (BERT/FinBERT).
    con.execute(f"""
        CREATE TABLE news AS
        SELECT
            "Stock_symbol" AS ticker,
            "Date"::TIMESTAMP AS date,
            text_for_sentiment,
            CAST(NULL AS DOUBLE) AS sentiment_score
        FROM read_csv_auto('{DATA_DIR}/news_sample.csv')
    """)

    # --- Table companies ---
    # Table de référence : un ticker par entreprise présente dans prices ET/OU news.
    con.execute("""
        CREATE TABLE companies AS
        SELECT DISTINCT ticker FROM (
            SELECT ticker FROM prices
            UNION
            SELECT ticker FROM news
        )
    """)

    n_prices = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    n_news = con.execute("SELECT COUNT(*) FROM news").fetchone()[0]
    n_companies = con.execute("SELECT COUNT(*) FROM companies").fetchone()[0]

    print(f"Base créée : {DB_PATH}")
    print(f"  companies : {n_companies} entreprises")
    print(f"  prices    : {n_prices:,} lignes")
    print(f"  news      : {n_news:,} articles")

    con.close()


if __name__ == "__main__":
    main()
