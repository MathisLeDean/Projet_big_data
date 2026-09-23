"""
Exploration et filtrage initial du dataset FNSPID.

Objectif : télécharger l'historique des prix pour un petit échantillon
d'entreprises S&P 500, et filtrer les articles de presse correspondants
sans avoir à rapatrier l'intégralité des fichiers (plusieurs Go).

Prérequis :
    pip install huggingface_hub pandas
"""

import pandas as pd
from huggingface_hub import hf_hub_download
import zipfile
import os

# --- Les tickers du S&P 500 avec, à la fois, un historique de prix ET des
# articles de news dans FNSPID/Benzinga sur 2018-2023 ---
# Liste de départ : top 100 S&P 500 par capitalisation (classement figé
# manuellement, source : stockanalysis.com/list/sp-500-stocks), dont on retire :
# - les 22 tickers sans AUCUN article de news (diagnose_missing_tickers.py) :
#   ABT, BRK-B, CB, COST, CRWD, CVX, DELL, GE, GEV, GLW, GS, ISRG, LIN, MSFT,
#   PLD, PLTR, PM, RTX, SNDK, SPGI, UBER, WELL
# - 3 tickers avec des news mais sans fichier de prix dans l'archive
#   (diagnose_missing_prices.py) : STX, UNH, VZ
# On garde META et BKNG (articles + prix récupérés via leur ancien ticker,
# FB / PCLN — voir TICKER_ALIASES) et on prend les 47 suivants dans l'ordre
# de capitalisation.
TICKER_ALIASES = {
    "FB": "META",
    "PCLN": "BKNG",
}

TICKERS = [
    "NVDA", "AAPL", "GOOGL", "GOOG", "AMZN", "AVGO", "META", "TSLA", "MU", "LLY",
    "JPM", "WMT", "AMD", "XOM", "V", "JNJ", "INTC", "MA", "ABBV", "ORCL",
    "BAC", "CSCO", "KO", "LRCX", "CAT", "AMAT", "MRK", "UNH", "MS", "PG",
    "NFLX", "HD", "PANW", "WFC", "ANET", "TXN", "C", "KLAC", "TMO", "IBM",
    "AXP", "VZ", "AMGN", "CRM", "MRVL", "APH", "STX", "TMUS", "QCOM", "PEP",
]
TICKERS = [t for t in TICKERS if t not in ("STX", "UNH", "VZ")]
print(f"{len(TICKERS)} tickers retenus (filtrés pour couverture prix + news).")

START_DATE = "2018-01-01"
END_DATE = "2023-12-31"

REPO_ID = "Zihan1004/FNSPID"
REPO_TYPE = "dataset"

DATA_DIR = "fnspid_data"
os.makedirs(DATA_DIR, exist_ok=True)


def download_and_extract_prices():
    """Télécharge et dézippe l'historique des prix (Stock_price/full_history.zip)."""
    print("Téléchargement de l'historique des prix...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename="Stock_price/full_history.zip",
        local_dir=DATA_DIR,
    )
    extract_dir = os.path.join(DATA_DIR, "Stock_price_extracted")
    # Insensible à la casse (certains fichiers de l'archive sont en minuscules,
    # ex: "nvda.csv") + on inclut aussi les anciens tickers (ex: "FB" pour
    # récupérer l'historique de prix de META, publié sous FB avant juin 2022).
    SYMBOLS_TO_MATCH = set(TICKERS) | set(TICKER_ALIASES.keys())
    with zipfile.ZipFile(zip_path, "r") as zf:
        # On ne liste que les fichiers correspondant à nos tickers, pour ne pas
        # tout extraire (l'archive contient l'historique de milliers d'entreprises).
        # Correspondance EXACTE sur le nom de fichier (sans extension), pas une
        # recherche de sous-chaîne : "ticker in m" matchait par exemple tous les
        # fichiers contenant juste la lettre "V" ou "T" quelque part dans leur nom.
        members = [
            m for m in zf.namelist()
            if os.path.splitext(os.path.basename(m))[0].upper() in SYMBOLS_TO_MATCH
        ]
        print(f"Fichiers correspondant à nos tickers trouvés dans l'archive : {len(members)}")
        zf.extractall(extract_dir, members=members)
    return extract_dir


def load_price_sample(extract_dir):
    """Charge les prix filtrés sur la période choisie pour chaque ticker."""
    frames = []
    tickers_trouves = set()
    TICKERS_SET = set(TICKERS)
    for root, _, files in os.walk(extract_dir):
        for f in files:
            raw_name = os.path.splitext(f)[0].upper()
            if raw_name in TICKERS_SET:
                canonical_ticker = raw_name
            elif raw_name in TICKER_ALIASES:
                canonical_ticker = TICKER_ALIASES[raw_name]
            else:
                continue
            df = pd.read_csv(os.path.join(root, f))
            df["ticker"] = canonical_ticker
            frames.append(df)
            tickers_trouves.add(canonical_ticker)

    tickers_sans_prix = sorted(set(TICKERS) - tickers_trouves)
    if tickers_sans_prix:
        print(f"Tickers SANS fichier de prix dans l'archive ({len(tickers_sans_prix)}) : {tickers_sans_prix}")

    if not frames:
        print("Aucun fichier de prix trouvé pour ces tickers — vérifier le format des noms de fichiers dans l'archive.")
        return pd.DataFrame()

    prices = pd.concat(frames, ignore_index=True)
    # Le nom de la colonne date peut varier ; à ajuster après un premier print(prices.columns)
    date_col = "Date" if "Date" in prices.columns else prices.columns[0]
    prices[date_col] = pd.to_datetime(prices[date_col])
    prices = prices[(prices[date_col] >= START_DATE) & (prices[date_col] <= END_DATE)]
    return prices


def filter_news_sample():
    """
    Parcourt le fichier de news par morceaux (chunks) et ne garde que les lignes
    concernant nos tickers, sans jamais charger tout le fichier en mémoire.

    Colonnes confirmées dans FNSPID : Date, Article_title, Stock_symbol, Url,
    Publisher, Author, Article (texte brut), Lsa_summary, Luhn_summary,
    Textrank_summary, Lexrank_summary.
    """
    print("\nTéléchargement du fichier de news (cela peut prendre du temps, plusieurs Go)...")
    news_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename="Stock_news/All_external.csv",
        local_dir=DATA_DIR,
    )

    USECOLS = ["Date", "Article_title", "Stock_symbol"]
    # On filtre à la fois sur les tickers actuels et sur leurs anciens noms,
    # pour ne pas perdre les articles publiés avant un changement de ticker.
    SYMBOLS_TO_MATCH = set(TICKERS) | set(TICKER_ALIASES.keys())
    filtered_chunks = []
    rows_scanned = 0
    chunksize = 100_000

    for chunk in pd.read_csv(news_path, chunksize=chunksize, usecols=USECOLS, on_bad_lines="skip"):
        match = chunk[chunk["Stock_symbol"].isin(SYMBOLS_TO_MATCH)]
        if not match.empty:
            filtered_chunks.append(match)

        rows_scanned += len(chunk)
        if rows_scanned % 1_000_000 == 0:
            print(f"  ... {rows_scanned:,} lignes scannées, {sum(len(c) for c in filtered_chunks)} articles trouvés jusqu'ici")

    print(f"Scan terminé : {rows_scanned:,} lignes au total.")

    if filtered_chunks:
        news_sample = pd.concat(filtered_chunks, ignore_index=True)
        news_sample["Date"] = pd.to_datetime(news_sample["Date"], errors="coerce")
        news_sample = news_sample[
            (news_sample["Date"] >= START_DATE) & (news_sample["Date"] <= END_DATE)
        ]
        news_sample = news_sample.rename(columns={"Article_title": "text_for_sentiment"})
        # On réattribue les articles publiés sous l'ancien ticker (FB, PCLN...)
        # au ticker actuel (META, BKNG...) pour ne pas les compter à part.
        news_sample["Stock_symbol"] = news_sample["Stock_symbol"].replace(TICKER_ALIASES)

        print(f"\nArticles trouvés pour nos tickers (sur la période 2018-2023) : {len(news_sample)}")
        print(news_sample["Stock_symbol"].value_counts())

        tickers_sans_article = sorted(set(TICKERS) - set(news_sample["Stock_symbol"].unique()))
        print(f"\nTickers sans AUCUN article trouvé ({len(tickers_sans_article)}) : {tickers_sans_article}")

        return news_sample
    else:
        print("\nAucun article trouvé.")
        return pd.DataFrame()


def main():
    extract_dir = download_and_extract_prices()
    prices = load_price_sample(extract_dir)
    print(f"\nLignes de prix chargées : {len(prices)}")
    if not prices.empty:
        print(prices.head())
        prices.to_csv(os.path.join(DATA_DIR, "prices_sample.csv"), index=False)

    news_sample = filter_news_sample()
    if not news_sample.empty:
        news_sample.to_csv(os.path.join(DATA_DIR, "news_sample.csv"), index=False)
        print(news_sample.head())


if __name__ == "__main__":
    main()
