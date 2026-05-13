# ============================================================
# TRAINING PARAMETERS
# ============================================================

# PPO Training
TOTAL_TIMESTEPS = 100_000
EPISODE_LENGTH = 20
LEARNING_RATES = [0.0003, 0.00003]
N_STEPS = [50, 200]
ACTION_SCALE = 0.05

# PPO Architecture and hyperparameters
PPO_CONFIG = {
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# Early stopping
PATIENCE_STEPS = 4000

# ============================================================
# DATASET PARAMETERS
# ============================================================

# Riga del dataset da ottimizzare
TARGET_ROWS = [3]

# DOF (Degrees of Freedom) configuration
DOF_NAMES = [
    "DOF_PITCH_GEOM",
    "DOF_BETA1_GEOM",
    "DOF_BETA2_GEOM",
    "DOF_W1_GEOM",
    "DOF_W2_GEOM",
    "DOF_TMOVXU_GEOM",
    "DOF_TMOVXL_GEOM",
]

# Limiti fisici per ogni DOF [min, max]
DOF_BOUNDS = [
    (0.084, 0.140),  # DOF_PITCH_GEOM
    (0.034, 19.966),  # DOF_BETA1_GEOM
    (-69.996, -60.121),  # DOF_BETA2_GEOM
    (0.0, 0.745),  # DOF_W1_GEOM
    (0.001, 0.999),  # DOF_W2_GEOM
    (-0.149, 0.199),  # DOF_TMOVXU_GEOM
    (-0.150, 0.200),  # DOF_TMOVXL_GEOM
]

# Objectives (OF) configuration
OF_NAMES = [
    "OF_alfa_ex",
    "OF_Cpt",
    "OF_CSI",
    "OF_phi",
    "OF_psi",
    "OF_Zwi",
    "OF_Zwc",
    "OF_DFss_mis",
    "OF_DFss_cp",
    "OF_Mis_peak",
    "OF_s_peak",
    "OF_s_diff_dim",
    "OF_s_tot_SS",
    "OF_Tmax",
    "OF_X_Tmax",
]

# Indici per OF specifici
IDX_CSI = OF_NAMES.index("OF_CSI")
IDX_CPT = OF_NAMES.index("OF_Cpt")
IDX_PSI = OF_NAMES.index("OF_psi")
IDX_PHI = OF_NAMES.index("OF_phi")

# ============================================================
# DOF SELECTION (quale sottinsieme di DOF ottimizzare)
# ============================================================

# Seleziona quale/quali DOF l'agente PPO può modificare
ACTIVE_DOF_INDICES = [0]  # Solo PITCH_GEOM

# Coppie custom (per test multipli)
CUSTOM_DOF_COMBINATIONS = [
    [1, 3],  # BETA1 + W1
    [2, 4],  # BETA2 + W2
    [5, 6],  # TMOVXU + TMOVXL
]

# Combinazioni da testare (costruito dinamicamente)
DOF_COMBINATIONS_TO_TEST = []

# A. Aggiungi ogni DOF singolarmente
for idx in ACTIVE_DOF_INDICES:
    DOF_COMBINATIONS_TO_TEST.append([idx])

# B. Aggiungi le coppie custom
for pair in CUSTOM_DOF_COMBINATIONS:
    DOF_COMBINATIONS_TO_TEST.append(pair)

# C. Aggiungi tutti insieme
if len(ACTIVE_DOF_INDICES) > 1:
    DOF_COMBINATIONS_TO_TEST.append(list(ACTIVE_DOF_INDICES))

# Se nessuna combinazione specificata, usa ACTIVE_DOF_INDICES come default
if not DOF_COMBINATIONS_TO_TEST:
    DOF_COMBINATIONS_TO_TEST = [ACTIVE_DOF_INDICES]

# ============================================================
# OUTPUT PARAMETERS
# ============================================================

# Presentazione PowerPoint
PRESENTATION_TEMPLATE = "Template.pptx"
PRESENTATION_OUTPUT = "Report_Simulazioni_PPO.pptx"

# Checkpoint PPO
CHECKPOINT_FREQUENCY = 10000

# ============================================================
# LOGGING AND DEBUG
# ============================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
VERBOSE = True
PRINT_INTERVAL = 100  # Stampa progress ogni N step

# ============================================================
# VALIDATION THRESHOLDS
# ============================================================

# Soglie per valutare la qualità dei risultati
MIN_R2_EXCELLENT = 0.85
MIN_R2_GOOD = 0.70
MAX_RMSE_PERCENTAGE = 10.0

# ============================================================
# NEURAL NETWORK (DATA PREPARATION)
# ============================================================

# Train/Val/Test split (dopo divisione iniziale)
TEST_SIZE_FIRST = 0.30  # 70% train, 30% temp
VAL_TEST_SPLIT = 0.50  # Dividi temp in 50/50 val/test → 15% ciascuno

# Outlier removal
IQR_MULTIPLIER = 1.5  # Valori: Q1 - 1.5*IQR e Q3 + 1.5*IQR

# Scaler type
SCALER_TYPE = "StandardScaler"  # MinMaxScaler o StandardScaler

# Network architecture (Keras)
NETWORK_ARCHITECTURE = {
    "layers": [
        {"units": 64, "activation": "relu"},
        {"units": 32, "activation": "relu"},
    ],
    "input_layer": {"activation": None},
    "output_layer": {"activation": "linear"},
    "dropout": 0.2,
}

TRAINING_PARAMS = {
    "epochs": 500,
    "batch_size": 32,
    "early_stopping_patience": 25,
    "validation_split": 0.15,
}

# ============================================================
# FILE I/O
# ============================================================

# Temporanei da eliminare dopo training
TEMPORARY_FILES = [
    "plot_results.png",
    "plot_metrics_actor.png",
    "plot_metrics_critic.png",
    "plot_dof_evolution_barre.png",
    "plot_dof_evolution.png"
]

TEMPORARY_DIRS_PATTERNS = [
    "ppo_blade_*_checkpoints_*",
    "ppo_blade_*_logs_*",
    "ppo_blade_checkpoints",
]