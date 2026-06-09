import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from Config.Set_input_param import (ACTIVE_DOF_INDICES, ACTION_SCALE, DOF_BOUNDS_ALL, OF_NAMES, TARGET_CSI,
                                    TARGET_PHI, TARGET_PSI, modello_rete, scaler_rete
                                    )

# ============================================================
# CONFIGURAZIONE
# ============================================================

# Definisci il percorso assoluto della cartella Ambiente
AMBIENTE_DIR = Path(__file__).parent.parent.resolve()

# Percorsi assoluti (possono essere stringhe o dizionari)
SURROGATE_MODEL_PATH = modello_rete
SCALER_PATH = scaler_rete

print(f"DEBUG: SURROGATE_MODEL_PATH = {SURROGATE_MODEL_PATH}")
if isinstance(SURROGATE_MODEL_PATH, dict):
    for of_k, path_v in SURROGATE_MODEL_PATH.items():
        print(f"  - Modello per {of_k} esiste? {os.path.exists(path_v)}")
else:
    print(f"DEBUG: File modello esiste? {os.path.exists(SURROGATE_MODEL_PATH)}")

print(f"DEBUG: SCALER_PATH = {SCALER_PATH}")
if isinstance(SCALER_PATH, dict):
    for of_k, path_v in SCALER_PATH.items():
        print(f"  - Scaler per {of_k} esiste? {os.path.exists(path_v)}")
else:
    print(f"DEBUG: File scaler esiste? {os.path.exists(SCALER_PATH)}")

# -------------------------------------------------------
# SELEZIONE DOF ATTIVI
# -------------------------------------------------------
DOF_BOUNDS = [DOF_BOUNDS_ALL[i] for i in ACTIVE_DOF_INDICES]

# Indici delle Objective Functions
IDX_CSI = OF_NAMES.index(TARGET_CSI)
IDX_PSI = OF_NAMES.index(TARGET_PSI)
IDX_PHI = OF_NAMES.index(TARGET_PHI)


# ============================================================
# CARICAMENTO SURROGATE KERAS (Supporta Singolo e Multi-Modello)
# ============================================================

def load_surrogate(model_path=SURROGATE_MODEL_PATH, scaler_path=SCALER_PATH):
    import tensorflow as tf
    import joblib

    # CONTROLLO ARCHITETTURA: Struttura Singola o Multi-Modello?
    if not isinstance(model_path, dict):
        # ============================================================
        # STRUTTURA 1: VECCHIO MODO (Modello Unico)
        # ============================================================
        print(f"\n[INFO] Caricamento surrogate singolo standard: {model_path}")
        keras_model = tf.keras.models.load_model(model_path)

        print(f"  Caricamento scaler unico: {scaler_path}")
        scalers = joblib.load(scaler_path)
        scaler_X = scalers['scaler_X']
        scaler_y = scalers['scaler_y']

        # Ottimizzazione numpy dei parametri dello scaler X
        scaler_type = type(scaler_X).__name__
        if scaler_type == "MinMaxScaler":
            X_offset_ = scaler_X.data_min_.astype(np.float32)
            X_scale_ = scaler_X.data_range_.astype(np.float32)
            _scale_X = lambda x: (x - X_offset_) / (X_scale_ + 1e-8)
        elif scaler_type == "StandardScaler":
            X_offset_ = scaler_X.mean_.astype(np.float32)
            X_scale_ = scaler_X.scale_.astype(np.float32)
            _scale_X = lambda x: (x - X_offset_) / (X_scale_ + 1e-8)
        else:
            _scale_X = lambda x: scaler_X.transform(x.reshape(1, -1))[0].astype(np.float32)

        # Ottimizzazione numpy dei parametri dello scaler y
        scaler_y_type = type(scaler_y).__name__
        if scaler_y_type == "MinMaxScaler":
            y_offset_ = scaler_y.data_min_.astype(np.float32)
            y_scale_ = scaler_y.data_range_.astype(np.float32)
            _inverse_scale_y = lambda y: y * y_scale_ + y_offset_
        elif scaler_y_type == "StandardScaler":
            y_offset_ = scaler_y.mean_.astype(np.float32)
            y_scale_ = scaler_y.scale_.astype(np.float32)
            _inverse_scale_y = lambda y: y * y_scale_ + y_offset_
        else:
            _inverse_scale_y = lambda y: scaler_y.inverse_transform(y.reshape(1, -1))[0].astype(np.float32)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, keras_model.input_shape[-1]], dtype=tf.float32)
        ])
        def fast_infer(x):
            return keras_model(x, training=False)

        def predict(dof_raw):
            x_scaled = _scale_X(dof_raw.astype(np.float32))
            x_tensor = x_scaled.reshape(1, -1).astype(np.float32)
            of_scaled = fast_infer(tf.constant(x_tensor)).numpy()
            of_real = _inverse_scale_y(of_scaled[0])
            return of_real.astype(np.float32)

        # Warm-up grafo TF
        print("  Warm-up tf.function (Metamodello Singolo)...")
        dummy = np.random.uniform(
            np.array([b[0] for b in DOF_BOUNDS_ALL]),
            np.array([b[1] for b in DOF_BOUNDS_ALL])
        ).astype(np.float32)
        predict(dummy)
        return predict



    else:
        # ============================================================
        # STRUTTURA 2: NUOVO MODO (Multi-Modello Separato per OF)
        # ============================================================

        print(f"\n[INFO] Rilevato assetto MULTI-MODELLO con scaler DOF unico. Inizializzazione...")

        import tensorflow as tf
        import joblib
        import pandas as pd  # Importiamo pandas per risolvere il problema dei nomi delle colonne
        from Config.Set_input_param import DOF_NAMES_ALL  # Importiamo i nomi reali delle colonne dei DOF

        # 1. Carichiamo lo scaler GLOBALE per i DOF (Input X)
        if "X_GLOBAL" in scaler_path:
            print(f"  -> Caricamento Scaler di Input Globale: {scaler_path['X_GLOBAL']}")
            scaler_X_global = joblib.load(scaler_path["X_GLOBAL"])

            # Estraiamo automaticamente i nomi esatti delle colonne attesi dallo scaler
            if hasattr(scaler_X_global, "feature_names_in_"):
                colonne_attese_scaler = scaler_X_global.feature_names_in_
                print(
                    f"  [INFO] Nomi colonne estratti dallo scaler con successo ({len(colonne_attese_scaler)} feature).")
            else:
                # Fallback protettivo se la proprietà non esistesse (ma nel tuo ColumnTransformer c'è sicuramente)
                from Config.Set_input_param import DOF_NAMES_ALL
                colonne_attese_scaler = DOF_NAMES_ALL

            # Funzione di scaling che impacchetta l'input con i nomi corretti richiesti dallo scaler
            def _scale_x_with_df(x_numpy):
                x_2d = x_numpy.reshape(1, -1)
                # Creiamo il DataFrame usando i nomi esatti che lo scaler pretende di vedere
                df_temp = pd.DataFrame(x_2d, columns=colonne_attese_scaler)
                return scaler_X_global.transform(df_temp)[0].astype(np.float32)

            _scale_x_global = _scale_x_with_df
        else:
            raise KeyError("Errore: Manca la chiave 'X_GLOBAL' in scaler_rete per caricare lo scaler_DOF.joblib")

        of_predictors = {}

        # Funzione helper per buildare la pipeline di ogni singola OF

        def build_single_of_pipeline(m_file, s_file):
            # Usiamo tf.keras.models.load_model (standard di tensorflow)
            k_model = tf.keras.models.load_model(m_file)
            scaler_y = joblib.load(s_file)  # Questo è lo scaler specifico per l'output (es. scaler_CSI.joblib)

            # Ottimizzazione numpy per l'inversione dell'output della singola OF
            s_y_type = type(scaler_y).__name__
            if s_y_type == "MinMaxScaler":
                off_y = scaler_y.data_min_.astype(np.float32)
                sc_y = scaler_y.data_range_.astype(np.float32)
                _inv_scale_y_loc = lambda y: y * sc_y + off_y
            elif s_y_type == "StandardScaler":
                off_y = scaler_y.mean_.astype(np.float32)
                sc_y = scaler_y.scale_.astype(np.float32)
                _inv_scale_y_loc = lambda y: y * sc_y + off_y
            else:
                _inv_scale_y_loc = lambda y: scaler_y.inverse_transform(y.reshape(1, -1))[0].astype(np.float32)
            @tf.function(input_signature=[
                tf.TensorSpec(shape=[1, k_model.input_shape[-1]], dtype=tf.float32)
            ])
            def fast_infer_loc(x):

                return k_model(x, training=False)

            def predict_single_of(x_scaled_vector):
                # Riceve il vettore X già scalato globalmente e lo passa alla rete
                x_tensor = x_scaled_vector.reshape(1, -1).astype(np.float32)
                pred_scaled = fast_infer_loc(tf.constant(x_tensor)).numpy()

                # Applica l'inverse transform specifico per questa OF
                pred_real = _inv_scale_y_loc(pred_scaled[0])
                return pred_real.flatten()[0] if isinstance(pred_real, np.ndarray) else pred_real

            return predict_single_of

        # Istanziamo i modelli mappandoli sulle chiavi corrispondenti
        for of_name in OF_NAMES:
            # Rende il matching flessibile: es. se of_name è "OF_phi" o "PHI", cercherà "PHI" nel dizionario
            matching_key = next(
                (k for k in model_path.keys() if k.upper() in of_name.upper() or of_name.upper() in k.upper()), None)

            if matching_key:
                print(f"  -> Pipeline OF generata per {of_name} (Associata a chiave modello: {matching_key})")
                of_predictors[of_name] = build_single_of_pipeline(model_path[matching_key], scaler_path[matching_key])
            else:
                # Se non trova una corrispondenza perfetta, proviamo a mappare PHI, PSI, CSI ovunque siano contenute
                fallback_key = None
                if "CSI" in of_name.upper():
                    fallback_key = "CSI"
                elif "CPT" in of_name.upper():
                    fallback_key = "CPT"
                elif "PSI" in of_name.upper():
                    fallback_key = "PSI"
                elif "PHI" in of_name.upper():
                    fallback_key = "PHI"
                elif "ALFA_EX" in of_name.upper():
                    fallback_key = "ALFA_EX"
                elif "AREA" in of_name.upper():
                    fallback_key = "AREA"
                elif "DFSS_MISS" in of_name.upper():
                    fallback_key = "DFSS_MISS"
                elif "DS_CP" in of_name.upper():
                    fallback_key = "DS_CP"
                elif "TMAX" in of_name.upper():
                    fallback_key = "TMAX"
                elif "XTMAX" in of_name.upper():
                    fallback_key = "X_TMAX"
                elif "UGT" in of_name.upper():
                    fallback_key = "UGT"
                elif "WEDGE" in of_name.upper():
                    fallback_key = "WEDGE"
                elif "ZWC" in of_name.upper():
                    fallback_key = "ZWC"

                if fallback_key and fallback_key in model_path:
                    print(f"  -> [FALLBACK SUCCESSO] Pipeline OF generata per {of_name} usando modello {fallback_key}")
                    of_predictors[of_name] = build_single_of_pipeline(model_path[fallback_key],
                                                                      scaler_path[fallback_key])
                else:
                    print(f"  [AVVISO] Nessun modello trovato per {of_name}. Verrà restituito 0.0 di default.")

        # Funzione di aggregazione finale eseguita a ogni step dell'ambiente RL
        def predict_multi(dof_raw):
            # 1. Scaliamo l'input una volta sola usando lo scaler globale DOF
            x_input = dof_raw.astype(np.float32)
            x_scaled = _scale_x_global(x_input)

            # 2. Otteniamo le predizioni ciclando sulle OF
            of_real_all = np.zeros(len(OF_NAMES), dtype=np.float32)
            for i, of_name in enumerate(OF_NAMES):

                if of_name in of_predictors:
                    # Passiamo il vettore già scalato alla pipeline dell'OF
                    of_real_all[i] = of_predictors[of_name](x_scaled)
                else:
                    of_real_all[i] = 0.0

            return of_real_all

        print("  Warm-up del grafo Multi-Modello strutturato...")
        dummy = np.random.uniform(
            np.array([b[0] for b in DOF_BOUNDS_ALL]),
            np.array([b[1] for b in DOF_BOUNDS_ALL])
        ).astype(np.float32)

        predict_multi(dummy)
        print("  Sistema Multi-Modello configurato con successo.\n")

        return predict_multi


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
        # Penalità quadratica: se sgarri di poco, la penalità è minima. Se sgarri di tanto, esplode.
        penalty += 50.0 * ((errore_psi - tolleranza) ** 2)

    if errore_phi > tolleranza:
        penalty += 50.0 * ((errore_phi - tolleranza) ** 2)

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
        penalty += 50.0 * ((e_phi - tol_rel)**2)
    if e_psi > tol_rel:
        penalty += 50.0 * ((e_psi - tol_rel)**2)

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
        self.surrogate = load_surrogate()

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
            shape=(n_active_dof + n_of+4,),
            dtype=np.float32
        )

        # Stato interno
        self.current_dof_active = None   # solo i DOF che l'agente modifica
        self.current_dof_full   = None   # tutti e 7 i DOF (per la surrogate)
        self.current_of         = None
        self.start_of           = None
        self.step_count         = 0

    def _build_obs(self, dof_active, of_vals):
        dof_norm = (dof_active - self.dof_low) / (self.dof_range + 1e-8)

        if (self.target_phi is not None) and (self.target_psi is not None):
            target_phi_val = self.target_phi
            target_psi_val = self.target_psi
        else:
            # Usiamo self.ref_of così l'agente vede i vincoli del profilo corrente dell'episodio
            target_phi_val = self.ref_of[IDX_PHI]
            target_psi_val = self.ref_of[IDX_PSI]

        target_array = np.array([target_psi_val, target_phi_val], dtype=np.float32)

        phi_curr = of_vals[IDX_PHI]
        psi_curr = of_vals[IDX_PSI]

        errore_psi = abs(psi_curr - target_psi_val) / (abs(target_psi_val) + 1e-8)
        errore_phi = abs(phi_curr - target_phi_val) / (abs(target_phi_val) + 1e-8)

        error_array = np.array([errore_psi, errore_phi], dtype=np.float32)

        return np.concatenate([dof_norm, of_vals, target_array, error_array]).astype(np.float32)

    def _get_observation(self):

        # Costruisci l'osservazione: DOF normalizzati + OF
        return self._build_obs(self.current_dof_active, self.current_of)

    def reset(self, seed=None, options=None):
        """
        Reset Gymnasium-compliant con parametri seed e options.
        Gestisce correttamente l'ancoraggio dei vincoli sia per profili fissi che casuali.
        """
        super().reset(seed=seed)

        # 1. Inizializza i DOF (Grandi di Libertà)
        if self.start_dof is not None:
            # Task 2: Parte dal profilo specifico richiesto
            self.current_dof_full = np.array(self.start_dof, dtype=np.float32)
        else:
            # Task 1: Genera un profilo iniziale completamente casuale entro i bound fisici
            self.current_dof_full = self.np_random.uniform(
                self.dof_low_all, self.dof_high_all
            ).astype(np.float32)

        # Estrae solo i DOF attivi che l'agente può effettivamente modificare
        self.current_dof_active = self.current_dof_full[ACTIVE_DOF_INDICES].copy()

        # Valuta le performance (OF) di questo profilo iniziale tramite il metamodello Keras
        self.current_of = self.surrogate(self.current_dof_full)
        self.start_of = self.current_of.copy()
        self.step_count = 0

        # 2. GESTIONE ANCORAGGIO VINCOLI (Risoluzione Bug Task 1)
        if self.start_dof is None:
            # TASK 1: Poiché la pala cambia ad ogni episodio, il punto di riferimento
            # per i vincoli del 2% deve forzatamente resettarsi e ancorarsi alla nuova pala corrente.
            self.ref_of = self.start_of.copy()
        else:
            # TASK 2: Il profilo di partenza è fisso da database. Il vincolo del 2% si fissa
            # una volta sola all'inizio e rimane lo stesso per tutti i cicli di addestramento.
            if self.ref_of is None:
                self.ref_of = self.start_of.copy()

        # 3. Costruisce lo stato iniziale da passare all'agente (inclusi i nuovi errori relativi)
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
        new_of = self.surrogate(new_dof_full)

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
            tolleranza_max = 0.005
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

