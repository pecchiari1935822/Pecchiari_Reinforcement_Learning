import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


DATABASE_DIR = Path(__file__).parent.resolve()
DATASET_PATH = str(DATABASE_DIR / "Data" / "database.dat")

df = pd.read_csv(DATASET_PATH)

nome_colonna = 'OF_alfa_ex_OP_01'
nome_beta1 = 'DOF_BETA1_GEOM_'
nome_beta2 = 'DOF_BETA2_GEOM_'
alfa_ex = df[nome_colonna]
beta1 = df[nome_beta1]
beta2 = df[nome_beta2]

for indice, (a_ex, b1, b2) in enumerate(zip(alfa_ex, beta1, beta2)):

    deflessione_alfa = 10 - a_ex
    deflessione_beta = b1 - b2
    print(f"\nRiga {indice}:")
    print(f"deflessione alfa: {deflessione_alfa:.3f}")
    print(f"deflessione beta: {deflessione_beta:.3f}")