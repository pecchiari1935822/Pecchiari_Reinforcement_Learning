import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from Config.Set_input_param import (ACTIVE_DOF_INDICES, ACTION_SCALE, DOF_BOUNDS_ALL, OF_NAMES, TARGET_CSI,
                                    TARGET_PHI, TARGET_PSI, modello_rete, scaler_rete, tolleranza_profilo_partenza, tolleranza_phi_psi_imposti
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
            scaler_type = type(scaler_X_global).__name__

            if scaler_type == "MinMaxScaler":
                X_offset_ = scaler_X_global.data_min_.astype(np.float32)
                X_scale_ = scaler_X_global.data_range_.astype(np.float32)
                _scale_x_global = lambda x: (x.astype(np.float32) - X_offset_) / (X_scale_ + 1e-8)

            elif scaler_type == "StandardScaler":
                X_offset_ = scaler_X_global.mean_.astype(np.float32)
                X_scale_ = scaler_X_global.scale_.astype(np.float32)
                _scale_x_global = lambda x: (x.astype(np.float32) - X_offset_) / (X_scale_ + 1e-8)

            elif scaler_type == "ColumnTransformer":
                # Ogni step è una Pipeline con un MinMaxScaler/StandardScaler interno.
                # Ricostruiamo i vettori offset e scale nell'ordine dei transformers.
                offsets = []
                scales = []
                for step_name, pipeline, cols in scaler_X_global.transformers_:
                    if step_name == "remainder":
                        continue
                    # Estrai il sotto-scaler dalla Pipeline (l'ultimo step)
                    inner = pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline
                    inner_type = type(inner).__name__
                    if inner_type == "MinMaxScaler":
                        offsets.append(inner.data_min_[0])
                        scales.append(inner.data_range_[0])
                    elif inner_type == "StandardScaler":
                        offsets.append(inner.mean_[0])
                        scales.append(inner.scale_[0])
                    else:
                        raise ValueError(f"Scaler interno non supportato: {inner_type} nello step '{step_name}'")

                X_offset_ = np.array(offsets, dtype=np.float32)
                X_scale_ = np.array(scales, dtype=np.float32)
                _scale_x_global = lambda x: (x.astype(np.float32) - X_offset_) / (X_scale_ + 1e-8)

            else:
                # Fallback con DataFrame (lento ma funziona)
                col_names = list(colonne_attese_scaler)
                _scale_x_global = lambda x: scaler_X_global.transform(
                    pd.DataFrame(x.reshape(1, -1), columns=col_names)
                )[0].astype(np.float32)
        else:
            raise KeyError("Errore: Manca la chiave 'X_GLOBAL' in scaler_rete per caricare lo scaler_DOF.joblib")

        # Raccoglie modelli e parametri scaler nell'ordine di OF_NAMES
        keras_models_list = []
        inv_off_list = []
        inv_sc_list = []

        for of_name in OF_NAMES:
            matching_key = next(
                (k for k in model_path.keys() if k.upper() in of_name.upper() or of_name.upper() in k.upper()),
                None)

            if not matching_key:
                for kw, mk in [("CSI", "CSI"), ("CPT", "CPT"), ("PSI", "PSI"), ("PHI", "PHI"),
                               ("ALFA_EX", "ALFA_EX"), ("AREA", "Area"), ("DFSS_MIS", "DFSS_MIS"),
                               ("DS_CP", "DS_CP"), ("TMAX", "TMAX"), ("X_TMAX", "X_TMAX"),
                               ("WEDGE", "WEDGE_TE"), ("UGT", "UGT"), ("ZWC", "ZWC")]:
                    if kw in of_name.upper() and mk in model_path:
                        matching_key = mk
                        break

            if matching_key:
                print(f"  -> Caricamento modello per {of_name} (chiave: {matching_key})")
                k_model = tf.keras.models.load_model(model_path[matching_key])
                scaler_y = joblib.load(scaler_path[matching_key])

                s_y_type = type(scaler_y).__name__
                if s_y_type == "Pipeline":
                    inner_scaler_y = scaler_y.steps[-1][1]
                    s_y_type = type(inner_scaler_y).__name__
                else:
                    inner_scaler_y = scaler_y

                if s_y_type == "MinMaxScaler":
                    off_y = float(inner_scaler_y.data_min_[0])
                    sc_y = float(inner_scaler_y.data_range_[0])
                elif s_y_type == "StandardScaler":
                    off_y = float(inner_scaler_y.mean_[0])
                    sc_y = float(inner_scaler_y.scale_[0])
                else:
                    off_y, sc_y = 0.0, 1.0

                keras_models_list.append(k_model)
                inv_off_list.append(off_y)
                inv_sc_list.append(sc_y)
            else:
                print(f"  [AVVISO] Nessun modello per {of_name}, output = 0.0")
                keras_models_list.append(None)
                inv_off_list.append(0.0)
                inv_sc_list.append(1.0)

        # Tensori costanti per l'inverse scaling
        inv_off_array = np.array(inv_off_list, dtype=np.float32)
        inv_sc_array = np.array(inv_sc_list, dtype=np.float32)

        # Indici e modelli validi (quelli che hanno un modello associato)
        valid_indices = [i for i, m in enumerate(keras_models_list) if m is not None]
        valid_models = [keras_models_list[i] for i in valid_indices]

        # Unico grafo TF che esegue tutti i modelli in sequenza con una sola chiamata
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, len(DOF_BOUNDS_ALL)], dtype=tf.float32)
        ])
        def fast_infer_all(x):
            results = tf.zeros([len(OF_NAMES)], dtype=tf.float32)
            for i, model in zip(valid_indices, valid_models):
                pred = model(x, training=False)
                results = tf.tensor_scatter_nd_update(results, [[i]], tf.reshape(pred, [1]))
            return results

        def predict_multi(dof_raw):
            x_scaled = _scale_x_global(dof_raw.astype(np.float32))
            x_tensor = tf.constant(x_scaled.reshape(1, -1), dtype=tf.float32)
            of_scaled = fast_infer_all(x_tensor).numpy()
            of_real = of_scaled * inv_sc_array + inv_off_array
            return of_real.astype(np.float32)

        # Warm-up
        print("  Warm-up del grafo Multi-Modello unificato...")
        dummy = np.random.uniform(
            np.array([b[0] for b in DOF_BOUNDS_ALL]),
            np.array([b[1] for b in DOF_BOUNDS_ALL])
        ).astype(np.float32)
        predict_multi(dummy)
        print("  Sistema Multi-Modello configurato con successo.\n")

        return predict_multi

surrogate = load_surrogate(SURROGATE_MODEL_PATH, SCALER_PATH)
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

    reward_csi = float(csi_prev - csi_curr)

    return reward_csi

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


    reward_csi = csi_prev - csi_curr
    return reward_csi

# ============================================================
# AMBIENTE GYMNASIUM CUSTOM
# ============================================================

# ============================================================
# AMBIENTE GYMNASIUM CUSTOM CON SUPPORTO SAFE RL (OMNISAFE)
# ============================================================

class BladeOptimEnv(gym.Env):
    """
    Ambiente Gymnasium per l'ottimizzazione del profilo palare adattato per Safe RL.
    Mantiene il tracciamento globale del miglior profilo tramite attributi di classe.
    """
    metadata = {"render_modes": ["human"]}

    # Variabili di classe globali per estrarre il profilo migliore a fine run
    best_csi = np.inf
    best_dof = None
    best_of = None

    def __init__(self, start_dof=None,
                 action_scale=ACTION_SCALE, use_delta=False, episode_length=None,
                 target_phi=None, target_psi=None, ref_of=None):
        super().__init__()

        self.use_delta = use_delta
        self.surrogate = surrogate

        self.start_dof = start_dof
        self.ref_of = ref_of
        self.ep_length = episode_length
        self.action_scale = action_scale

        self.target_phi = target_phi
        self.target_psi = target_psi

        n_active_dof = len(DOF_BOUNDS)
        n_of = len(OF_NAMES)

        self.dof_low = np.array([b[0] for b in DOF_BOUNDS], dtype=np.float32)
        self.dof_high = np.array([b[1] for b in DOF_BOUNDS], dtype=np.float32)
        self.dof_range = self.dof_high - self.dof_low

        self.dof_low_all = np.array([b[0] for b in DOF_BOUNDS_ALL], dtype=np.float32)
        self.dof_high_all = np.array([b[1] for b in DOF_BOUNDS_ALL], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_active_dof,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_active_dof + n_of + 4,),
            dtype=np.float32
        )

        self.current_dof_active = None
        self.current_dof_full = None
        self.current_of = None
        self.start_of = None
        self.step_count = 0

    def _build_obs(self, dof_active, of_vals):
        dof_norm = (dof_active - self.dof_low) / (self.dof_range + 1e-8)

        if (self.target_phi is not None) and (self.target_psi is not None):
            target_phi_val = self.target_phi
            target_psi_val = self.target_psi
        else:
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
        return self._build_obs(self.current_dof_active, self.current_of)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.start_dof is not None:
            self.current_dof_full = np.array(self.start_dof, dtype=np.float32)
        else:
            self.current_dof_full = self.np_random.uniform(
                self.dof_low_all, self.dof_high_all
            ).astype(np.float32)

        self.current_dof_active = self.current_dof_full[ACTIVE_DOF_INDICES].copy()
        self.current_of = self.surrogate(self.current_dof_full)
        self.start_of = self.current_of.copy()
        self.step_count = 0

        if self.start_dof is None:
            self.ref_of = self.start_of.copy()
        else:
            if self.ref_of is None:
                self.ref_of = self.start_of.copy()

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.step_count += 1
        prev_of = self.current_of.copy()

        if self.use_delta:
            delta = action * self.action_scale * self.dof_range
            new_dof_full = self.current_dof_full.copy()
            new_dof_active = self.current_dof_active + delta
            new_dof_active = np.clip(new_dof_active, self.dof_low, self.dof_high)
        else:
            new_dof_active = self.dof_low + (action + 1.0) / 2.0 * self.dof_range
            new_dof_full = self.current_dof_full.copy()

        for i, idx in enumerate(ACTIVE_DOF_INDICES):
            new_dof_full[idx] = new_dof_active[i]

        new_of = self.surrogate(new_dof_full)

        # REWARD PURA & CALCOLO COSTI NATIVI PER OMNISAFE
        reward = float(prev_of[IDX_CSI] - new_of[IDX_CSI])  # L'obiettivo è massimizzare la riduzione delle perdite

        if (self.target_phi is not None) and (self.target_psi is not None):
            tolleranza_target = tolleranza_phi_psi_imposti
            errore_psi = abs(new_of[IDX_PSI] - self.target_psi) / (abs(self.target_psi) + 1e-8)
            errore_phi = abs(new_of[IDX_PHI] - self.target_phi) / (abs(self.target_phi) + 1e-8)
            is_valid = bool((errore_psi <= tolleranza_target) and (errore_phi <= tolleranza_target))

            # Costo proporzionale allo sforamento rispetto al target
            cost_psi = max(0.0, errore_psi - tolleranza_target)
            cost_phi = max(0.0, errore_phi - tolleranza_target)
        else:
            tolleranza_max = tolleranza_profilo_partenza
            errore_psi = abs(new_of[IDX_PSI] - self.ref_of[IDX_PSI]) / (abs(self.ref_of[IDX_PSI]) + 1e-8)
            errore_phi = abs(new_of[IDX_PHI] - self.ref_of[IDX_PHI]) / (abs(self.ref_of[IDX_PHI]) + 1e-8)
            is_valid = bool(errore_psi <= tolleranza_max and errore_phi <= tolleranza_max)

            # Costo proporzionale allo sforamento rispetto al profilo iniziale
            cost_psi = max(0.0, errore_psi - tolleranza_max)
            cost_phi = max(0.0, errore_phi - tolleranza_max)

        # Il costo totale inviato a OmniSafe è la somma delle violazioni dei vincoli cinematici
        cost = float(cost_psi + cost_phi)

        # Aggiornamento delle variabili di classe globali per il miglior profilo valido trovato
        if is_valid and float(new_of[IDX_CSI]) < BladeOptimEnv.best_csi:
            BladeOptimEnv.best_csi = float(new_of[IDX_CSI])
            BladeOptimEnv.best_dof = new_dof_full.copy()
            BladeOptimEnv.best_of = new_of.copy()

        self.current_dof_active = new_dof_active
        self.current_dof_full = new_dof_full
        self.current_of = new_of

        terminated = False
        truncated = self.step_count >= self.ep_length
        obs = self._build_obs(self.current_dof_active, self.current_of)

        info = {
            "cost": cost,  # <--- CHIAVE CRITICA: OmniSafe leggerà automaticamente questo valore come costo dello step
            "csi": float(new_of[IDX_CSI]),
            "psi": float(new_of[IDX_PSI]),
            "phi": float(new_of[IDX_PHI]),
            "dof_active": self.current_dof_active.copy(),
            "dof_full": self.current_dof_full.copy(),
            "of": self.current_of.copy(),
            "is_valid": is_valid,
            "err_phi_rel": float(errore_phi),
            "err_psi_rel": float(errore_psi)
        }

        return obs, reward, terminated, truncated, info

