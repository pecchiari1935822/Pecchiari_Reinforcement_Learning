import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from Config.Set_input_param import (ACTIVE_DOF_INDICES, ACTION_SCALE, DOF_BOUNDS_ALL, OF_NAMES, TARGET_CSI,
                                    TARGET_PHI, TARGET_PSI
                                    )
# ============================================================
# CONFIGURAZIONE
# ============================================================

# Definisci il percorso assoluto della cartella Ambiente
AMBIENTE_DIR = Path(__file__).parent.parent.resolve()

# Percorsi assoluti
SURROGATE_MODEL_PATH = str(AMBIENTE_DIR / "Data" / "models" / "best_model.keras")
SCALER_PATH = str(AMBIENTE_DIR / "Data" / "models" / "scalers.joblib")

print(f"DEBUG: SURROGATE_MODEL_PATH = {SURROGATE_MODEL_PATH}")
print(f"DEBUG: File esiste? {os.path.exists(SURROGATE_MODEL_PATH)}")
print(f"DEBUG: SCALER_PATH = {SCALER_PATH}")
print(f"DEBUG: File esiste? {os.path.exists(SCALER_PATH)}")


# -------------------------------------------------------
# SELEZIONE DOF ATTIVI
# Cambia questa lista per aggiungere DOF man mano
# Es: [0]       → solo PITCH
#     [0, 1]    → PITCH + BETA1
#     list(range(7)) → tutti e 7
# -------------------------------------------------------

# Applica la selezione
DOF_BOUNDS = [DOF_BOUNDS_ALL[i] for i in ACTIVE_DOF_INDICES]


# Indice CSI — usato nella reward attuale
IDX_CSI = OF_NAMES.index(TARGET_CSI)
IDX_PSI = OF_NAMES.index(TARGET_PSI)
IDX_PHI = OF_NAMES.index(TARGET_PHI)

# Indici commentati — da attivare quando aggiungi compute_efficiency
"""IDX_CPT = OF_NAMES.index("OF_Cpt")"""


# ============================================================
# CARICAMENTO SURROGATE KERAS
# ============================================================

def load_surrogate(model_path=SURROGATE_MODEL_PATH,
                   scaler_path=SCALER_PATH):
    import tensorflow as tf
    import joblib

    # CArica il modello surrogato (la rete neurale che ho addestrato)
    print(f"\n  Caricamento surrogate: {model_path}")
    keras_model = tf.keras.models.load_model(model_path)

    # Carica il dizionario joblib con entrambi gli scaler
    print(f"  Caricamento scaler: {scaler_path}")
    scalers  = joblib.load(scaler_path)
    scaler_X = scalers['scaler_X']   # per scalare i DOF in input
    scaler_y = scalers['scaler_y']   # per de-scalare gli OF in output

    # Converte i parametri dello scaler_X in costanti numpy
    # per evitare overhead sklearn ad ogni chiamata
    scaler_type = type(scaler_X).__name__
    print(f"  Tipo scaler_X: {scaler_type}")
    print(f"  Tipo scaler_y: {type(scaler_y).__name__}")

    # In base allo scaler che ho utilizzato, definisco funzioni di scaling e inverse scaling
    if scaler_type == "MinMaxScaler":
        # transform(x) = (x - data_min_) / data_range_
        X_offset_ = scaler_X.data_min_.astype(np.float32)
        X_scale_ = scaler_X.data_range_.astype(np.float32)

        def _scale_X(x):
            return (x - X_offset_) / (X_scale_ + 1e-8)

    elif scaler_type == "StandardScaler":
        # transform(x) = (x - mean_) / scale_
        X_offset_ = scaler_X.mean_.astype(np.float32)
        X_scale_ = scaler_X.scale_.astype(np.float32)

        def _scale_X(x):
            return (x - X_offset_) / (X_scale_ + 1e-8)

    else:
        # Fallback generico: usa sklearn direttamente
        print(f"  Scaler non riconosciuto ({scaler_type}), uso sklearn.transform()")

        def _scale_X(x):
            return scaler_X.transform(x.reshape(1, -1))[0].astype(np.float32)

    # Stessa logica per scaler_y (inverse transform)
    scaler_y_type = type(scaler_y).__name__
    if scaler_y_type == "MinMaxScaler":
        y_offset_ = scaler_y.data_min_.astype(np.float32)
        y_scale_ = scaler_y.data_range_.astype(np.float32)

        def _inverse_scale_y(y):
            return y * y_scale_ + y_offset_

    elif scaler_y_type == "StandardScaler":
        y_offset_ = scaler_y.mean_.astype(np.float32)
        y_scale_ = scaler_y.scale_.astype(np.float32)

        def _inverse_scale_y(y):
            return y * y_scale_ + y_offset_

    else:
        def _inverse_scale_y(y):
            return scaler_y.inverse_transform(y.reshape(1, -1))[0].astype(np.float32)

    # Compila il modello come funzione TF statica —
    # la prima chiamata è lenta (compilazione), le successive veloci
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, keras_model.input_shape[-1]], dtype=tf.float32)
    ])
    def fast_infer(x):
        return keras_model(x, training=False)

    def predict(dof_raw):
        """
        dof_raw: np.array shape (7,) — DOF in unità fisiche reali
        Restituisce: np.array shape (15,) — OF in unità fisiche reali

        Pipeline:
            DOF grezzi
              → normalizzazione (numpy, veloce)
              → inferenza rete (tf.function, veloce)
              → de-normalizzazione OF (numpy, veloce)
              → OF in unità fisiche reali
        """
        # 1. Scala i DOF con i parametri estratti
        x_scaled = _scale_X(dof_raw.astype(np.float32))
        x_tensor = x_scaled.reshape(1, -1).astype(np.float32)

        # 2. Inferenza veloce con tf.function
        of_scaled = fast_infer(tf.constant(x_tensor)).numpy()

        # 3. De-scala gli OF → valori fisici reali
        of_real = _inverse_scale_y(of_scaled[0])

        return of_real.astype(np.float32)

    # Warm-up: prima chiamata lenta (compilazione grafo TF)
    print("  Warm-up tf.function (prima chiamata)...")
    dummy_low = np.array([b[0] for b in DOF_BOUNDS_ALL], dtype=np.float32)
    dummy_high = np.array([b[1] for b in DOF_BOUNDS_ALL], dtype=np.float32)
    dummy = np.random.uniform(dummy_low, dummy_high).astype(np.float32)
    predict(dummy)
    print("  Surrogate pronta.\n")

    return predict

# ============================================================
# FUNZIONE DI REWARD
# ============================================================

def compute_reward(of_current, of_previous, of_start, tolleranza=None):
    """
    La reward è POSITIVA quando CSI diminuisce (meno perdite).
    La reward è NEGATIVA quando CSI aumenta (più perdite).

    TODO: sostituire con compute_efficiency() quando si vuole
    ottimizzare l'efficienza completa psi/(1+Cpt).
    """
    # Controllo base su valori non validi
    if np.isnan(of_current).any() or np.isinf(of_current).any():
        return -10.0   # surrogate fuori distribuzione

    # Variabili per ottimizzare solamente le perdite CSI
    csi_curr = of_current[IDX_CSI]
    csi_prev = of_previous[IDX_CSI]

    psi_curr = of_current[IDX_PSI]
    psi_prev = of_previous[IDX_PSI]


    # Variabili per ottimizzare l'efficienza
    eta_curr = psi_curr / (1+csi_curr)
    eta_prev = psi_prev / (1+csi_prev)

    # Variabili per la penalizzazione se phi o psi cambiano più di un tot %
    psi_start = of_start[IDX_PSI]
    phi_start = of_start[IDX_PHI]
    phi_curr = of_current[IDX_PHI]

    errore_psi = abs(psi_curr - psi_start) / abs((psi_start) + 1e-8)


    errore_phi = abs(phi_curr - phi_start) / abs((phi_start) + 1e-8)

    penalty = 0.0
    if errore_psi > tolleranza:
        # Moltiplicatore da calibrare (es. 10.0) in modo che la penalità
        # superi il guadagno del delta CSI
        penalty += 10.0 * (errore_psi - tolleranza)

    if errore_phi > tolleranza:
        penalty += 10.0 * (errore_phi - tolleranza)

    reward_csi = float(csi_prev - csi_curr)

    # Rcompensa per l'ottimizzazione dell'efficienza
    # return float (eta_curr - eta_prev)

    # Ricompensa per l'ottimizzazione delle perdite con la penalità
    return reward_csi - penalty

def compute_reward_target(
    of_current,
    of_previous,
    phi_target,
    psi_target,
    tol_rel=0.01,
    penalty_k=10.0,
    invalid_penalty=-10.0
):
    """
    Stessa struttura di compute_reward:
      - reward base: miglioramento CSI (csi_prev - csi_curr)
      - penalità SOLO se phi/psi superano la tolleranza relativa rispetto ai target
    """
    # Controllo base su valori non validi
    if np.isnan(of_current).any() or np.isinf(of_current).any():
        return float(invalid_penalty)

    csi_curr = float(of_current[IDX_CSI])
    csi_prev = float(of_previous[IDX_CSI])

    phi_curr = float(of_current[IDX_PHI])
    psi_curr = float(of_current[IDX_PSI])

    # errori relativi rispetto ai TARGET
    e_phi = abs(phi_curr - phi_target) / (abs(phi_target) + 1e-8)
    e_psi = abs(psi_curr - psi_target) / (abs(psi_target) + 1e-8)

    penalty = 0.0
    if e_phi > tol_rel:
        penalty += penalty_k * (e_phi - tol_rel)
    if e_psi > tol_rel:
        penalty += penalty_k * (e_psi - tol_rel)

    reward_csi = csi_prev - csi_curr
    return float(reward_csi - penalty)

# ============================================================
# AMBIENTE GYMNASIUM CUSTOM
# ============================================================

class BladeOptimEnv(gym.Env):
    """
    Ambiente Gymnasium per l'ottimizzazione del profilo palare.

    STRUTTURA ATTUALE (1 DOF attivo):
      Stato  : [DOF_PITCH normalizzato (1)] + [15 OF] = 16 valori
      Azione : [delta PITCH] = 1 valore in [-1, +1]
      Reward : CSI_prev - CSI_curr  (minimizza perdite comprimibili)"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, surrogate_fn, start_dof=None,
                 action_scale=ACTION_SCALE, use_delta=False, episode_length=None,
                 target_phi=None, target_psi=None, ref_of=None):
        super().__init__()

        self.use_delta = use_delta


        self.predict      = surrogate_fn
        self.start_dof    = start_dof
        self.ref_of = ref_of
        self.ep_length    = episode_length
        self.action_scale = action_scale

        self.target_phi = target_phi
        self.target_psi = target_psi

        n_active_dof = len(DOF_BOUNDS)          # DOF che l'agente può modificare
        n_of         = len(OF_NAMES)             # 15 OF prodotti dalla surrogate

        # Bounds solo per i DOF attivi
        self.dof_low   = np.array([b[0] for b in DOF_BOUNDS], dtype=np.float32)
        self.dof_high  = np.array([b[1] for b in DOF_BOUNDS], dtype=np.float32)
        self.dof_range = self.dof_high - self.dof_low

        # Bounds per tutti e 7 i DOF (per campionare i DOF fissi all'inizio)
        self.dof_low_all  = np.array([b[0] for b in DOF_BOUNDS_ALL], dtype=np.float32)
        self.dof_high_all = np.array([b[1] for b in DOF_BOUNDS_ALL], dtype=np.float32)

        # --- SPAZIO AZIONI ---
        # Solo i DOF attivi: ogni valore in [-1, +1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_active_dof,),
            dtype=np.float32
        )

        # --- SPAZIO OSSERVAZIONI ---
        # DOF attivi normalizzati [0,1] + tutti i 15 OF
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_active_dof + n_of+2,),
            dtype=np.float32
        )

        # Stato interno
        self.current_dof_active = None   # solo i DOF che l'agente modifica
        self.current_dof_full   = None   # tutti e 7 i DOF (per la surrogate)
        self.current_of         = None
        self.start_of           = None
        self.step_count         = 0

    def _build_obs(self, dof_active, of_vals):
        """
        Osservazione = DOF attivi normalizzati in [0,1] + OF grezzi.
        La normalizzazione aiuta la rete PPO a lavorare su scale comparabili.
        """
        dof_norm = (dof_active - self.dof_low) / (self.dof_range + 1e-8)
        if self.start_of is not None:
            target_psi = self.start_of[IDX_PSI]
            target_phi = self.start_of[IDX_PHI]
        else:
            target_psi = 0.0
            target_phi = 0.0

        target_array = np.array([target_psi, target_phi], dtype=np.float32)

        return np.concatenate([dof_norm, of_vals, target_array]).astype(np.float32)

    def _get_observation(self):

        # Costruisci l'osservazione: DOF normalizzati + OF
        return self._build_obs(self.current_dof_active, self.current_of)

    def reset(self, seed=None, options=None):
        """
        Reset Gymnasium-compliant con parametri seed e options.
        """
        super().reset(seed=seed)

        # Inizializza DOF e calcola OF
        if self.start_dof is not None:
            self.current_dof_full = np.array(self.start_dof, dtype=np.float32)
        else:
            self.current_dof_full = self.np_random.uniform(
                self.dof_low_all, self.dof_high_all
            ).astype(np.float32)

        self.current_dof_active = self.current_dof_full[ACTIVE_DOF_INDICES].copy()
        self.current_of = self.predict(self.current_dof_full)
        self.start_of = self.current_of.copy()
        self.step_count = 0

        if self.ref_of is None:
            self.ref_of = self.start_of.copy()

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """
        Applica l'azione dell'agente:
        1. Modifica solo i DOF attivi
        2. I DOF non attivi restano invariati
        3. Valuta il profilo completo con la surrogate
        4. Reward = CSI_prev - CSI_curr
        """
        self.step_count += 1

        prev_of = self.current_of.copy()

        if self.use_delta == True:

            # Calcola delta solo per i DOF attivi
            delta = action * self.action_scale * self.dof_range



            # Aggiorna i DOF attivi nel vettore completo (7 DOF)
            new_dof_full = self.current_dof_full.copy()
            new_dof_active = self.current_dof_active + delta
            new_dof_active = np.clip(new_dof_active, self.dof_low, self.dof_high)

        else:
            new_dof_active = self.dof_low + (action + 1.0) / 2.0 * self.dof_range


            # Aggiorna il profilo completo
            new_dof_full = self.current_dof_full.copy()

        # Scrivi i DOF aggiornati nel vettore completo
        for i, idx in enumerate(ACTIVE_DOF_INDICES):
            new_dof_full[idx] = new_dof_active[i]



        # Valuta il nuovo profilo (surrogate riceve sempre tutti e 7 i DOF)
        new_of = self.predict(new_dof_full).astype(np.float32)

        tolleranza_target = 0.02  # 1% (scegli tu)

        if (self.target_phi is not None) and (self.target_psi is not None):
            reward = compute_reward_target(
                new_of, prev_of,
                phi_target=self.target_phi,
                psi_target=self.target_psi,
                tol_rel=tolleranza_target
            )

            errore_psi = abs(new_of[IDX_PSI] - self.target_psi) / (abs(self.target_psi) + 1e-8)
            errore_phi = abs(new_of[IDX_PHI] - self.target_phi) / (abs(self.target_phi) + 1e-8)

            is_valid = bool((errore_psi <= tolleranza_target) and (errore_phi <= tolleranza_target))

        else:
            # fallback: la tua reward attuale con vincoli su start_of
            tolleranza_max = 0.05
            reward_val = compute_reward(new_of, prev_of, self.ref_of, tolleranza=tolleranza_max)
            reward = float(np.squeeze(reward_val))

            errore_psi = abs(new_of[IDX_PSI] - self.ref_of[IDX_PSI]) / (abs(self.ref_of[IDX_PSI]) + 1e-8)
            errore_phi = abs(new_of[IDX_PHI] - self.ref_of[IDX_PHI]) / (abs(self.ref_of[IDX_PHI]) + 1e-8)

            # True se ENTRAMBI gli errori sono sotto il 3%
            is_valid = bool(errore_psi <= tolleranza_max and errore_phi <= tolleranza_max)

        # Aggiorna stato interno
        self.current_dof_active = new_dof_active
        self.current_dof_full = new_dof_full
        self.current_of = new_of

        terminated = False
        truncated = self.step_count >= self.ep_length

        obs = self._build_obs(self.current_dof_active, self.current_of)


        info = {
            "efficiency": None,
            "csi": float(new_of[IDX_CSI]),
            "psi": float(new_of[IDX_PSI]),
            "dof_active": self.current_dof_active.copy(),
            "dof_full": self.current_dof_full.copy(),
            "of": self.current_of.copy(),
            "is_valid": is_valid,"err_phi_rel": float(errore_phi),

        }


        return obs, reward, terminated, truncated, info

