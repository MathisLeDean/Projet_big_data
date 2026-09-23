"""
Diagnostic : pourquoi META, NVDA, STX, TMO, UNH, VZ n'ont-ils pas de fichier
de prix dans full_history.zip alors qu'ils ont bien des articles de news ?

On recherche, dans la liste des fichiers de l'archive (déjà en cache, donc
pas de nouveau téléchargement), tout nom de fichier qui CONTIENT (substring,
insensible à la casse) l'un de ces tickers -- même si ce n'est pas une
correspondance exacte -- pour repérer un éventuel suffixe, une casse
différente, ou un ancien ticker (ex: FB pour META).
"""

from huggingface_hub import hf_hub_download
import zipfile
import os

REPO_ID = "Zihan1004/FNSPID"
REPO_TYPE = "dataset"
DATA_DIR = "fnspid_data"

MISSING_PRICE_TICKERS = ["META", "NVDA", "STX", "TMO", "UNH", "VZ"]
# Alias connu (comme pour les news) à vérifier en priorité
KNOWN_ALIASES = {"META": "FB"}


def main():
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename="Stock_price/full_history.zip",
        local_dir=DATA_DIR,
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = zf.namelist()
        print(f"Nombre total de fichiers dans l'archive : {len(all_names)}\n")

        for ticker in MISSING_PRICE_TICKERS:
            matches = [
                m for m in all_names
                if ticker.upper() in os.path.basename(m).upper()
            ]
            if matches:
                print(f"{ticker} : {len(matches)} fichier(s) approchant(s) trouvé(s) -> {matches[:10]}")
            else:
                print(f"{ticker} : AUCUN fichier approchant -> absent de l'archive sous ce nom")

        print()
        for new, old in KNOWN_ALIASES.items():
            alias_matches = [m for m in all_names if os.path.splitext(os.path.basename(m))[0] == old]
            print(f"Alias {new} -> ancien ticker '{old}' : {'trouvé -> ' + str(alias_matches) if alias_matches else 'non trouvé'}")


if __name__ == "__main__":
    main()
