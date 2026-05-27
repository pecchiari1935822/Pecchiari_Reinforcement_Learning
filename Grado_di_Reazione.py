import pandas as pd
import numpy as np
from Agente.PPO import Path



DATABASE_DIR = Path(__file__).resolve().parent
data = str(DATABASE_DIR / "Data" / "database.dat")
df = pd.read_csv(data)

for idx, row in df.iterrows():
    alpha_0 = np.radians(10)
    tan_alpha_0 = np.tan(alpha_0)

    alpha_2 = np.radians(row['OF_alfa_ex_OP_01'])
    alpha_2_deg = np.degrees(alpha_2)
    tan_alpha_2 = np.tan(alpha_2)

    beta_1 = np.radians(row['DOF_BETA1_GEOM_'])
    beta_1_deg = np.degrees(beta_1)
    tan_beta_1 = np.tan(beta_1)

    beta_2 = np.radians(row['DOF_BETA2_GEOM_'])
    beta_2_deg = np.degrees(beta_2)
    tan_beta_2 = np.tan(beta_2)

    alpha_1 = np.arctan(row['OF_psi_OP_01']/row['OF_phi_OP_01']+tan_alpha_0)
    alpha_1_deg = np.degrees(alpha_1)
    tan_alpha_1 = np.tan(alpha_1)

    R = 1 - (row['OF_phi_OP_01']**2 / (2*row['OF_psi_OP_01']))*(tan_beta_2**2-tan_beta_1**2)

    print(f"\nRiga {idx}: Grado di reazione = {R:.4f}")
    print(f"aplha1 = {alpha_1_deg:.4f}, aplha2 = {alpha_2_deg:.4f}, beta1 = {beta_1_deg:.4f}, beta2 = {beta_2_deg:.4f} ")


alpha_0 = np.radians(10)
tan_alpha_0 = np.tan(alpha_0)

alpha_2 = np.radians(59.8939)
tan_alpha_2 = np.tan(alpha_2)

beta_1 = np.radians(13.966)
tan_beta_1 = np.tan(beta_1)

beta_2= np.radians(-63.7692)
tan_beta_2 = np.tan(beta_2)

R = 1 - (0.6443**2 / 2*1.2667)*(tan_alpha_2**2-tan_alpha_1**2)
R_new = 0.6443/2*(tan_beta_2-tan_beta_1)

print(f"Gradi di reazione profilo ottimizzato: {R:.4f}, R = {R_new:.4f}")