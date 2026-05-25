import pandas as pd
import numpy as np
from Agente.PPO import Path


DATABASE_DIR = Path(__file__).resolve()
data = str(DATABASE_DIR / "Data" / "database.dat")
df = pd.DataFrame(data)

for idx, row in df.iterrows():
    alpha_0 = np.radians(10)
    tan_alpha_0 = np.tan(alpha_0)

    alpha_2 = np.radians(row['OF_alfa_ex_OP_01'])
    tan_alpha_2 = np.tan(alpha_2)

    beta_1 = np.radians(row['DOF_BETA1_GEOM_'])
    tan_beta_1 = np.tan(beta_1)

    beta_2 = np.radians(row['DOF_BETA2_GEOM_'])
    tan_beta_2 = np.tan(beta_2)

    alpha_1 = np.arctan(row['OF_psi_OP_01']/row['OF_phi_OP_01']-tan_alpha_0)
    tan_alpha_1 = np.tan(alpha_1)

    R = 1 - (row['OF_phi_OP_01']**2 / 2*row['OF_psi_OP_01'])*(tan_alpha_2**2-tan_alpha_0**2)
    R_new = row['OF_phi_OP_01']/2*(tan_beta_2-tan_beta_1)
    print(f"Riga {idx}: Grado di reazione = {R:.4f}, R = {R_new:.4f}")


alpha_1 = np.radians(10)
tan_alpha_1 = np.tan(alpha_1)

alpha_2 = np.radians(59.8939)
tan_alpha_2 = np.tan(alpha_2)

beta_1 = np.radians(13.966)
tan_beta_1 = np.tan(beta_1)

beta_2= np.radians(-63.7692)
tan_beta_2 = np.tan(beta_2)

R = 1 - (0.6443**2 / 2*1.2667)*(tan_alpha_2**2-tan_alpha_1**2)
R_new = 0.6443/2*(tan_beta_2-tan_beta_1)

print(f"Gradi di reazione profilo ottimizzato: {R:.4f}, R = {R_new:.4f}")