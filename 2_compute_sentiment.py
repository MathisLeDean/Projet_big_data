"""
2_compute_sentiment.py

Calcule un score de sentiment pour chaque titre d'article (table `news`)
avec FinBERT (modèle BERT pré-entraîné sur du texte financier anglais),
et écrit le résultat dans la colonne `sentiment_score` de la base DuckDB.

FinBERT classe chaque texte en positive / negative / neutral avec une
probabilité pour chacune des 3 classes. On convertit ça en un score unique
et continu, plus facile à utiliser dans une régression ensuite :

    sentiment_score = P(positive) - P(negative)   (compris entre -1 et +1)

"""

import duckdb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

DB_PATH = "fnspid.duckdb"
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé : {device}")
    if device == "cpu":
        print("Pas de GPU détecté -- le traitement sera plus lent (potentiellement 30-60 min "
              "selon le volume et votre machine). C'est normal, laissez tourner.")

    print(f"\nChargement du modèle {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Labels du modèle FinBERT, dans l'ordre de ses indices de sortie
    id2label = model.config.id2label  # normalement {0: 'positive', 1: 'negative', 2: 'neutral'}
    print(f"Labels du modèle : {id2label}")

    con = duckdb.connect(DB_PATH)

    # On ne récupère que les lignes pas encore traitées (utile si le script
    # est interrompu et relancé : on ne recalcule pas ce qui est déjà fait)
    rows = con.execute("""
        SELECT rowid, text_for_sentiment
        FROM news
        WHERE sentiment_score IS NULL
          AND text_for_sentiment IS NOT NULL
    """).fetchall()

    print(f"\n{len(rows)} articles à traiter.")

    if not rows:
        print("Rien à faire -- tous les articles ont déjà un score de sentiment.")
        con.close()
        return

    results = []  # liste de (rowid, sentiment_score)

    for i in tqdm(range(0, len(rows), BATCH_SIZE)):
        batch = rows[i:i + BATCH_SIZE]
        rowids = [r[0] for r in batch]
        texts = [str(r[1]) for r in batch]

        inputs = tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Repère dynamiquement les index positive/negative selon id2label,
        # pour ne pas dépendre d'un ordre supposé
        pos_idx = [k for k, v in id2label.items() if v.lower() == "positive"][0]
        neg_idx = [k for k, v in id2label.items() if v.lower() == "negative"][0]

        for rowid, prob_row in zip(rowids, probs):
            score = float(prob_row[pos_idx] - prob_row[neg_idx])
            results.append((score, rowid))

    print("\nÉcriture des scores dans la base...")
    con.executemany("UPDATE news SET sentiment_score = ? WHERE rowid = ?", results)

    # Petit résumé pour vérifier que ça a l'air cohérent
    stats = con.execute("""
        SELECT
            COUNT(*) AS n,
            ROUND(AVG(sentiment_score), 3) AS moyenne,
            ROUND(MIN(sentiment_score), 3) AS min,
            ROUND(MAX(sentiment_score), 3) AS max
        FROM news
        WHERE sentiment_score IS NOT NULL
    """).fetchone()
    print(f"Articles avec score : {stats[0]:,} | moyenne : {stats[1]} | min : {stats[2]} | max : {stats[3]}")

    print("\nAperçu de quelques résultats :")
    sample = con.execute("""
        SELECT text_for_sentiment, sentiment_score
        FROM news
        WHERE sentiment_score IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 8
    """).fetchall()
    for text, score in sample:
        print(f"  [{score:+.3f}] {text[:80]}")

    con.close()
    print("\nTerminé. La colonne sentiment_score de la table 'news' est remplie.")


if __name__ == "__main__":
    main()
