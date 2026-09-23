"""
5_dataviz.py

Génère les 5 visualisations finales du projet, en PNG haute résolution,
prêtes à être insérées dans le support de soutenance.

Prérequis :
    pip install duckdb pandas matplotlib
"""

import os
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DB_PATH = "fnspid.duckdb"
APT_CSV = "apt_regression_results.csv"
LSTM_CSV = "lstm_vs_linear_predictions.csv"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

POS = "#1B6B55"
NEG = "#A33228"
NEUTRAL = "#4C5A68"


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# --- 1. Classement des beta_sentiment ---
def fig_beta_ranking(apt: pd.DataFrame):
    print("1. Classement des beta_sentiment...")
    data = apt.sort_values("beta_sentiment", ascending=True)
    colors = [POS if v >= 0 else NEG for v in data["beta_sentiment"]]

    fig, ax = plt.subplots(figsize=(9, 12))
    ax.barh(data["ticker"], data["beta_sentiment"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Beta sentiment (sur-réaction au climat médiatique)")
    ax.set_title("Classement des entreprises par sensibilité au sentiment de presse")
    fig.tight_layout()
    savefig(fig, "1_classement_beta_sentiment.png")


# --- 2. Sentiment vs prix, Tesla ---
def fig_tesla_sentiment_vs_price(con):
    print("2. Sentiment vs prix -- Tesla...")
    prices = con.execute("""
        SELECT date, adj_close AS close FROM prices WHERE ticker = 'TSLA' ORDER BY date
    """).df()
    news = con.execute("""
        SELECT date, sentiment_score FROM news
        WHERE ticker = 'TSLA' AND sentiment_score IS NOT NULL
    """).df()

    prices["date"] = pd.to_datetime(prices["date"])
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    daily_sentiment = news.groupby("date")["sentiment_score"].mean().reset_index()

    # Moyenne mobile 20 jours pour lisser le sentiment (bruité au jour le jour)
    daily_sentiment = daily_sentiment.sort_values("date")
    daily_sentiment["sentiment_smoothed"] = daily_sentiment["sentiment_score"].rolling(20, min_periods=5).mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(prices["date"], prices["close"], color=NEUTRAL, linewidth=1.2, label="Cours de clôture (TSLA)")
    ax1.set_ylabel("Cours de clôture ($)", color=NEUTRAL)
    ax1.tick_params(axis="y", labelcolor=NEUTRAL)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = ax1.twinx()
    ax2.plot(daily_sentiment["date"], daily_sentiment["sentiment_smoothed"], color=POS, linewidth=1.3,
              label="Sentiment (moyenne mobile 20j)")
    ax2.axhline(0, color=POS, linewidth=0.6, linestyle="--", alpha=0.5)
    ax2.set_ylabel("Score de sentiment (lissé)", color=POS)
    ax2.tick_params(axis="y", labelcolor=POS)

    fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.92))
    ax1.set_title("Tesla : cours de clôture et sentiment médiatique (2018-2020)")
    fig.tight_layout()
    savefig(fig, "2_tesla_sentiment_vs_prix.png")


# --- 3. Heatmap sentiment / taux / VIX ---
def fig_factor_heatmap(apt: pd.DataFrame):
    print("3. Carte de chaleur des trois facteurs...")
    data = apt.sort_values("beta_sentiment", ascending=False).copy()

    def signed_log_p(beta, pval):
        pval = np.clip(pval, 1e-300, 1)  # évite log(0)
        return np.sign(beta) * -np.log10(pval)

    matrix = np.column_stack([
        signed_log_p(data["beta_sentiment"], data["pval_sentiment"]),
        signed_log_p(data["beta_taux"], data["pval_taux"]),
        signed_log_p(data["beta_vix"], data["pval_vix"]),
    ])
    # On plafonne l'échelle pour ne pas laisser un seul cas extrême écraser tout le reste
    vmax = np.nanpercentile(np.abs(matrix), 98)

    fig, ax = plt.subplots(figsize=(6, 12))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Sentiment", "Taux (10y)", "VIX"])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["ticker"], fontsize=8)
    ax.set_title("Direction et significativité\npar facteur et par entreprise", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("<- effet négatif   |   effet positif ->\n(intensité = significativité)", fontsize=8)
    fig.tight_layout()
    savefig(fig, "3_heatmap_facteurs.png")


# --- 4. Distribution des scores FinBERT ---
def fig_sentiment_distribution(con):
    print("4. Distribution des scores de sentiment FinBERT...")
    scores = con.execute("""
        SELECT sentiment_score FROM news WHERE sentiment_score IS NOT NULL
    """).df()["sentiment_score"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=60, color=NEUTRAL, alpha=0.85)
    ax.axvline(scores.mean(), color=POS, linewidth=1.5, label=f"Moyenne = {scores.mean():.3f}")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Score de sentiment (P(positif) - P(négatif))")
    ax.set_ylabel("Nombre de titres")
    ax.set_title(f"Distribution du sentiment FinBERT sur {len(scores):,} titres")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "4_distribution_sentiment.png")


# --- 5. LSTM / linéaire vs réalité ---
def fig_prediction_scatter(lstm_df: pd.DataFrame):
    print("5. Prédictions LSTM/linéaire vs réalité...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)

    lims = (lstm_df[["y_true", "y_pred_lstm", "y_pred_linear"]].quantile(0.01).min(),
            lstm_df[["y_true", "y_pred_lstm", "y_pred_linear"]].quantile(0.99).max())

    for ax, col, title in zip(axes, ["y_pred_linear", "y_pred_lstm"], ["Régression linéaire", "LSTM"]):
        ax.scatter(lstm_df["y_true"], lstm_df[col], s=6, alpha=0.25, color=NEUTRAL)
        ax.plot(lims, lims, color=NEG, linewidth=1, linestyle="--", label="Prédiction parfaite")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Rendement réel")
        ax.set_title(title)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Rendement prédit")
    fig.suptitle("Prédiction du rendement du lendemain : aucun modèle ne bat la diagonale attendue par hasard")
    fig.tight_layout()
    savefig(fig, "5_predictions_vs_realite.png")


def main():
    print(f"Sauvegarde des figures dans ./{OUT_DIR}/\n")

    apt = pd.read_csv(APT_CSV)
    lstm_df = pd.read_csv(LSTM_CSV)
    con = duckdb.connect(DB_PATH)

    fig_beta_ranking(apt)
    fig_tesla_sentiment_vs_price(con)
    fig_factor_heatmap(apt)
    fig_sentiment_distribution(con)
    fig_prediction_scatter(lstm_df)

    con.close()
    print("\nTerminé. 5 figures générées dans le dossier 'figures/'.")


if __name__ == "__main__":
    main()
