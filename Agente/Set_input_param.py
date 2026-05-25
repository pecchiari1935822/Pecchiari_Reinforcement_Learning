# Parametri che descrivono lunghezza episodio e lunghezza del training
TOTAL_TIMESTEPS = 200_000
learning_rate = [ 0.00003]
n_steps = [200, 1024]

# Azione che viene fatta dall'attore che interagisce con l'ambiente
ACTION_SCALE   = 0.05

# Parametri che decrivono come è costruito il PPO
PPO_PARAMS = dict(
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
)

# Cosa si vuole ottimizzare (quali DOF) e in quale riga del file di input
ROW_INDEX = [62]  # INSERISCI L'INDICE DELLA RIGA CHE VUOI OTTIMIZZARE

DOF_BOUNDS_ALL = [
    (0.084,   0.140),    # DOF_PITCH_GEOM
    (0.034,  19.966),    # DOF_BETA1_GEOM
    (-69.996, -60.121),  # DOF_BETA2_GEOM_
    (0.0,     0.745),    # DOF_W1_GEOM
    (0.001,   0.999),    # DOF_W2_GEOM
    (-0.149,  0.199),    # DOF_TMOVXU_GEOM_
    (-0.150,  0.200),    # DOF_TMOVXL_GEOM_
]

DOF_NAMES_ALL = [
    "DOF_PITCH", "DOF_BETA1", "DOF_BETA2",
    "DOF_W1", "DOF_W2", "DOF_TMOVXU", "DOF_TMOVXL"
]

OF_NAMES = [
    "OF_alfa_ex", "OF_Cpt",      "OF_CSI",
    "OF_phi",     "OF_psi",      "OF_Zwi",
    "OF_Zwc",     "OF_DFss_mis", "OF_DFss_cp",
    "OF_Mis_peak","OF_s_peak",   "OF_s_diff_dim",
    "OF_s_tot_SS","OF_Tmax",     "OF_X_Tmax"
]

n_dof_totali = len(DOF_NAMES_ALL)
n_of_totali = len(OF_NAMES)

# Colonne target usate nella funzione di reward e nei controlli
TARGET_CSI = "OF_CSI"
TARGET_PSI = "OF_psi"
TARGET_PHI = "OF_phi"

ACTIVE_DOF_INDICES = [0,1,2,3,4,5,6]  # INSERISCI GLI INDICI DEI DOF CHE VUOI OTTIMIZZARE (0-5)

COPPIE_CUSTOM = [[1, 3], [2, 4], [5, 6]]

combinazioni_da_testare = []

# A. Aggiungi ogni DOF singolarmente
'''for idx in ACTIVE_DOF_INDICES:
    combinazioni_da_testare.append([idx])'''

# B. Aggiungi le coppie custom definite sopra
'''for coppia in COPPIE_CUSTOM:
    combinazioni_da_testare.append(coppia)'''

# C. Aggiungi tutti i DOF insieme (solo se sono più di 1, per evitare doppioni)
if len(ACTIVE_DOF_INDICES) > 1:
    combinazioni_da_testare.append(ACTIVE_DOF_INDICES)