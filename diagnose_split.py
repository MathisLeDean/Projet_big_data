"""
diagnose_split.py

Vérifie si la colonne adj_close de la base est réellement ajustée du split
Tesla du 28 août 2020 (5-pour-1), en comparant close et adj_close juste
avant et juste après cette date.

Si adj_close est correctement ajusté : le ratio close/adj_close doit être
proche de 1 avant le split, et proche de 5 après (ou l'inverse selon la
convention). Si close et adj_close sont quasiment identiques des deux
côtés du split, alors adj_close N'EST PAS ajusté dans ce dataset.
"""

import duckdb

con = duckdb.connect("fnspid.duckdb")

df = con.execute("""
    SELECT date, close, adj_close
    FROM prices
    WHERE ticker = 'TSLA' AND date BETWEEN '2020-08-20' AND '2020-09-05'
    ORDER BY date
""").df()

print(df.to_string(index=False))

con.close()
