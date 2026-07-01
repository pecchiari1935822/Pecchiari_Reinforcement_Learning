import pandas as pd
from pathlib import Path
import numpy as np


# =====================================================================
# 0. CONFIGURAZIONE DEL DATASET CORRENTE
# =====================================================================

USE_MULTIMODEL = True # per il momento sto utilizzando il medello unico che ho addestrato io per il nuovo datast che

data_dir = Path(__file__).parent.parent.resolve()

if USE_MULTIMODEL == None:
    modello_rete = str(data_dir / "Data" / "STEP_999" / "models" / "best_model.keras")
    scaler_rete = str(data_dir / "Data" / "STEP_999" / "scalers" / "scalers.joblib")
    dataset = str(data_dir / "Data" / "STEP_999" / "database.dat")

else:
    nuovi_modelli_dir = data_dir / "Data" / "STEP_028"

    modello_rete = {
        "ALFA_EX": str(nuovi_modelli_dir / "models" / "model_alfa_ex_FL.keras"),
        "CPT": str(nuovi_modelli_dir / "models" / "model_Cpt_FL.keras"),
        "CSI": str(nuovi_modelli_dir / "models" / "model_CSI_FL.keras"),
        "PSI": str(nuovi_modelli_dir / "models" / "model_PSI_FL.keras"),
        "PHI": str(nuovi_modelli_dir / "models" / "model_PHI_FL.keras"),
        "ZWC": str(nuovi_modelli_dir / "models" / "model_Zwc_FL.keras"),
        "DFSS_MIS": str(nuovi_modelli_dir / "models" / "model_DFss_Mis_LO.keras"),
        "DS_CP": str(nuovi_modelli_dir / "models" / "model_Ds_cp_LO.keras"),
        "TMAX": str(nuovi_modelli_dir / "models" / "model_Tmax_GM.keras"),
        "X_TMAX": str(nuovi_modelli_dir / "models" / "model_X_Tmax_GM.keras"),
        "WEDGE_TE": str(nuovi_modelli_dir / "models" / "model_Wedge_TE_GM.keras"),
        "UGT": str(nuovi_modelli_dir / "models" / "model_UGT_GM.keras"),
        "Area": str(nuovi_modelli_dir / "models" / "model_Area_GM.keras")

    }

    scaler_rete = {
        "X_GLOBAL": str(nuovi_modelli_dir / "scalers" / "scaler_DOF.joblib"),
        "ALFA_EX" : str(nuovi_modelli_dir / "scalers" / "scaler_OF_alfa_ex_FL.joblib"),
        "CPT": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Cpt_FL.joblib"),
        "CSI": str(nuovi_modelli_dir / "scalers" / "scaler_OF_CSI_FL.joblib"),
        "PSI": str(nuovi_modelli_dir / "scalers" / "scaler_OF_PSI_FL.joblib"),
        "PHI": str(nuovi_modelli_dir / "scalers" / "scaler_OF_PHI_FL.joblib"),
        "ZWC": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Zwc_FL.joblib"),
        "DFSS_MIS": str(nuovi_modelli_dir / "scalers" / "scaler_OF_DFss_Mis_LO.joblib"),
        "DS_CP": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Ds_cp_LO.joblib"),
        "TMAX": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Tmax_GM.joblib"),
        "X_TMAX": str(nuovi_modelli_dir / "scalers" / "scaler_OF_X_Tmax_GM.joblib"),
        "WEDGE_TE": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Wedge_TE_GM.joblib"),
        "UGT": str(nuovi_modelli_dir / "scalers" / "scaler_OF_UGT_GM.joblib"),
        "Area": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Area_GM.joblib")
    }

    dataset = str(nuovi_modelli_dir / "database.dat")

    print(modello_rete)
    print("\n", scaler_rete)


# Carica il dataset (usa pd.read_excel se i tuoi file sono in formato Excel)
df = pd.read_csv(dataset)
df.columns = df.columns.str.replace("_OP_01", "", regex=False)
df.columns = df.columns.str.replace("_GEOM_", "", regex=False)
df.columns = df.columns.str.replace("_BC_", "", regex=False)
df.columns = df.columns.str.replace("_G0_", "", regex=False)
df.columns = df.columns.str.replace("_G1_", "", regex=False)
df.columns = df.columns.str.replace("_G2_", "", regex=False)
df.columns = df.columns.str.replace("_FL_", "", regex=False)
df.columns = df.columns.str.replace("_LO_", "", regex=False)
df.columns = df.columns.str.replace("_GM_", "", regex=False)


# =====================================================================
# 1. PARAMETRI DI TRAINING (Rimangono invariati)
# =====================================================================
TOTAL_TIMESTEPS = 100_000
learning_rate = [0.00003]
n_steps = [500]
early_stopping = None
ACTION_SCALE = 0.01

PPO_PARAMS = dict(
    n_epochs        = 30,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
)

ROW_INDEX = [3]  # Modificalo se la riga di partenza cambia nel nuovo dataset


# =====================================================================
# 2. ESTRAZIONE DINAMICA AUTOMATICA (La parte magica ✨)
# =====================================================================

# Trova automaticamente tutte le colonne che iniziano con "DOF" o "OF"
DOF_NAMES_ALL = [col for col in df.columns if col.upper().startswith("DOF")]
OF_NAMES      = [col for col in df.columns if col.upper().startswith("OF")]

print("OF_names", OF_NAMES)
print("DOF_NAMES", DOF_NAMES_ALL)

n_dof_totali = len(DOF_NAMES_ALL)
n_of_totali  = len(OF_NAMES)

# Calcola i BOUNDS (min e max) estraendoli direttamente dai valori del dataset
# Crea una lista di tuple (min, max) per ogni DOF trovato
DOF_BOUNDS_ALL = [(float(df[col].min()), float(df[col].max())) for col in DOF_NAMES_ALL]
OF_BOUNDS_ALL = [(float(df[col].min()), float(df[col].max())) for col in OF_NAMES]

DATASET_DOF_ALL = df[DOF_NAMES_ALL].to_numpy(dtype=np.float32)
N_DATASET_PROFILES = len(DATASET_DOF_ALL)

print("DOF_BOUNDS_ALL", DOF_BOUNDS_ALL)

# =====================================================================
# 3. ASSEGNAZIONE DELLE COLONNE TARGET
# =====================================================================
# Queste variabili cercano la colonna in cui è contenuto il nome così da poter essere usata
# come argomento posizionale in seguito
TARGET_CSI = next((col for col in OF_NAMES if "CSI" in col.upper()), None)
TARGET_PSI = next((col for col in OF_NAMES if "PSI" in col.upper()), None)
TARGET_PHI = next((col for col in OF_NAMES if "PHI" in col.upper()), None)

# Se voglio che phi e psi siano fissate ad un determinato valore
target_phi = None
target_psi = None
perturbazione_dof_attivi = None
# tolleranza rispetto a phi e psi del profilo di partenza
tolleranza_profilo_partenza=0.005
# tolleranza rispetto a phi e psi imposti arbitrariamente
tolleranza_phi_psi_imposti = 0.005

# =====================================================================
# 4. GESTIONE DEI DOF ATTIVI E COMBINAZIONI
# =====================================================================
# Di default, attiva TUTTI i DOF trovati nel dataset
ACTIVE_DOF_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

print("ACTIVE_DOF_INDICES", ACTIVE_DOF_INDICES)

# Coppie custom (manteniamo le tue, ma con un controllo di sicurezza)
# Se il nuovo dataset ha meno DOF, evita che il codice vada in errore (IndexError)
COPPIE_CUSTOM_INPUT = [[1, 3], [2, 4], [5, 6]]
COPPIE_CUSTOM = [coppia for coppia in COPPIE_CUSTOM_INPUT if max(coppia) < n_dof_totali]

combinazioni_da_testare = []

# A. Aggiungi ogni DOF singolarmente (opzionale, scommenta se serve)
'''for idx in ACTIVE_DOF_INDICES:
    combinazioni_da_testare.append([idx])'''

# B. Aggiungi le coppie custom filtrate
'''for coppia in COPPIE_CUSTOM:
    combinazioni_da_testare.append(coppia)'''

# C. Aggiungi tutti i DOF insieme
if len(ACTIVE_DOF_INDICES) > 1:
    combinazioni_da_testare.append(ACTIVE_DOF_INDICES)