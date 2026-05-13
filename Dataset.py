import pandas as pd
from Agente.PPO import Path


DATABASE_DIR = Path(__file__).parent.resolve()
DATASET_PATH = str(DATABASE_DIR / "Data" / "database.dat")



df = pd.read_csv(DATASET_PATH)

nome_colonna = 'OF_CSI_OP_01'
valore_limite = 0.016535

# Applica il filtro
righe_filtrate = df[df[nome_colonna] <= valore_limite]

print(f"\n--- RISULTATI DELLA RICERCA ---")
if not righe_filtrate.empty:
    print(f"Trovate {len(righe_filtrate)} righe in cui '{nome_colonna}' è < {valore_limite}:")
    # Stampa le righe trovate allineate
    print(righe_filtrate.to_string())
else:
    print(f"Nessun valore trovato nella colonna '{nome_colonna}' minore di {valore_limite}.")