import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from Config.Set_input_param import (ACTIVE_DOF_INDICES, ACTION_SCALE, DOF_BOUNDS_ALL, OF_NAMES, TARGET_CSI,
                                    TARGET_PHI, TARGET_PSI
                                    )

# ============================================================
# CONFIGURAZIONE PERCORSI
# ============================================================

AMBIENTE_DIR = Path(__file__).parent.parent.resolve()

# ── Surrogato VECCHIO (un solo modello → tutti gli OF insieme) ───────────────
SURROGATE_MODEL_PATH = str(AMBIENTE_DIR / "Data" / "models" / "best_model.keras")
SCALER_PATH          = str(AMBIENTE_DIR / "Data" / "models" / "scalers.joblib")

# ── Surrogato NUOVO (un modello per OF + uno scaler DOF condiviso) ───────────
# Cartella che contiene:
#   - scaler_DOF.joblib              → scaler condiviso per i DOF in ingresso
#   - model_NOME.keras               → un modello per ogni OF
#   - scaler_OF_NOME.joblib          → uno scaler per ogni OF
#
# La convenzione di naming deve essere rispettata:
#   model_{NOME}.keras  ↔  scaler_OF_{NOME}.joblib
# dove NOME è il suffisso che identifica ogni OF (es. "CSI_FL", "phi_FL", ...).
#
# La mappatura tra i nomi degli OF nel codice (OF_NAMES, es. "OF_CSI")
# e i nomi dei file (es. "CSI_FL") è definita nella variabile
# OF_NAME_TO_FILE_KEY in Set_input_param.py (vedi sotto).
MULTI_SURROGATE_DIR  = str(AMBIENTE_DIR / "Data" / "models" / "multi")

# ── Selezione DOF attivi ─────────────────────────────────────────────────────
DOF_BOUNDS = [DOF_BOUNDS_ALL[i] for i in ACTIVE_DOF_INDICES]

# Indici OF usati nella reward e nei vincoli
IDX_CSI = OF_NAMES.index(TARGET_CSI)
IDX_PSI = OF_NAMES.index(TARGET_PSI)
IDX_PHI = OF_NAMES.index(TARGET_PHI)


# ============================================================
# CARICAMENTO SURROGATO — MODALITÀ SINGOLA (vecchio)
# ============================================================

def load_surrogate(model_path=SURROGATE_MODEL_PATH,
                   scaler_path=SCALER_PATH):
    """
    Carica il surrogato SINGOLO (vecchia modalità):
      - un solo file .keras che predice tutti gli OF insieme
      - un file .joblib con due chiavi: 'scaler_X' e 'scaler_y'

    Restituisce una funzione predict(dof_raw) -> np.array(n_of,)

    Parametri
    ----------
    model_path  : percorso al file .keras
    scaler_path : percorso al file .joblib contenente scaler_X e scaler_y
    """
    import tensorflow as tf
    import joblib

    print(f"\n  [Surrogato SINGOLO] Caricamento modello: {model_path}")
    keras_model = tf.keras.models.load_model(model_path)

    print(f"  [Surrogato SINGOLO] Caricamento scaler: {scaler_path}")
    scalers  = joblib.load(scaler_path)
    scaler_X = scalers['scaler_X']
    scaler_y = scalers['scaler_y']

    # Estrai parametri degli scaler come costanti numpy per evitare overhead sklearn
    scaler_type = type(scaler_X).__name__

    if scaler_type == "MinMaxScaler":
        X_offset = scaler_X.data_min_.astype(np.float32)
        X_scale  = scaler_X.data_range_.astype(np.float32)
        def _scale_X(x):
            return (x - X_offset) / (X_scale + 1e-8)

    elif scaler_type == "StandardScaler":
        X_offset = scaler_X.mean_.astype(np.float32)
        X_scale  = scaler_X.scale_.astype(np.float32)
        def _scale_X(x):
            return (x - X_offset) / (X_scale + 1e-8)

    else:
        # Fallback generico: sklearn puro
        print(f"  Scaler_X non riconosciuto ({scaler_type}), uso sklearn.transform()")
        def _scale_X(x):
            return scaler_X.transform(x.reshape(1, -1))[0].astype(np.float32)

    scaler_y_type = type(scaler_y).__name__

    if scaler_y_type == "MinMaxScaler":
        y_offset = scaler_y.data_min_.astype(np.float32)
        y_scale  = scaler_y.data_range_.astype(np.float32)
        def _inverse_scale_y(y):
            return y * y_scale + y_offset

    elif scaler_y_type == "StandardScaler":
        y_offset = scaler_y.mean_.astype(np.float32)
        y_scale  = scaler_y.scale_.astype(np.float32)
        def _inverse_scale_y(y):
            return y * y_scale + y_offset

    else:
        def _inverse_scale_y(y):
            return scaler_y.inverse_transform(y.reshape(1, -1))[0].astype(np.float32)

    # Compila il modello come grafo TF statico per massima velocità
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, keras_model.input_shape[-1]], dtype=tf.float32)
    ])
    def _fast_infer(x):
        return keras_model(x, training=False)

    def predict(dof_raw):
        """
        dof_raw : np.array shape (n_dof_totali,) — DOF in unità fisiche reali
        return  : np.array shape (n_of_totali,)  — OF in unità fisiche reali
        """
        x_scaled = _scale_X(dof_raw.astype(np.float32))
        x_tensor = tf.constant(x_scaled.reshape(1, -1).astype(np.float32))
        of_scaled = _fast_infer(x_tensor).numpy()
        return _inverse_scale_y(of_scaled[0]).astype(np.float32)

    # Warm-up
    print("  Warm-up surrogato singolo...")
    dummy = np.random.uniform(
        [b[0] for b in DOF_BOUNDS_ALL],
        [b[1] for b in DOF_BOUNDS_ALL]
    ).astype(np.float32)
    predict(dummy)
    print("  Surrogato singolo pronto.\n")

    return predict


# ============================================================
# CARICAMENTO SURROGATO — MODALITÀ MULTI-MODELLO (nuovo)
# ============================================================

def load_multi_surrogate(models_dir=MULTI_SURROGATE_DIR,
                         of_name_to_file_key=None):
    """
    Carica il surrogato MULTI-MODELLO (nuova modalità):
      - uno scaler DOF condiviso:  scaler_DOF.joblib  (ColumnTransformer, vuole DataFrame)
      - per ogni OF:
            model_{FILE_KEY}.keras        →  predice quell'OF
            scaler_OF_{FILE_KEY}.joblib   →  de-normalizza l'output

    La mappatura tra i nomi interni degli OF (OF_NAMES, es. "OF_CSI") e
    i nomi dei file (es. "CSI_FL") deve essere fornita tramite il parametro
    of_name_to_file_key, oppure viene letta automaticamente da Set_input_param.py.

    Esempio di of_name_to_file_key:
        {
            "OF_CSI"     : "CSI_FL",
            "OF_phi"     : "phi_FL",
            "OF_psi"     : "psi_FL",
            "OF_alfa_ex" : "alfa_ex_FL",
            ...
        }

    Gli OF per cui non esiste una entry nella mappatura vengono restituiti
    come NaN (non predicibili con questo surrogato).

    Restituisce una funzione predict(dof_raw) -> np.array(n_of_totali,)

    Parametri
    ----------
    models_dir         : cartella contenente i file .keras e .joblib
    of_name_to_file_key: dict {nome_OF_interno -> chiave_file}
                         Se None, viene importato da Set_input_param.py
    """
    import tensorflow as tf
    import joblib
    import pandas as pd

    # Leggi la mappatura da Set_input_param se non fornita esplicitamente
    if of_name_to_file_key is None:
        try:
            from Config.Set_input_param import OF_NAME_TO_FILE_KEY
            of_name_to_file_key = OF_NAME_TO_FILE_KEY
        except ImportError:
            raise ValueError(
                "of_name_to_file_key non fornita e OF_NAME_TO_FILE_KEY non trovata "
                "in Set_input_param.py. Definisci la mappatura prima di usare "
                "load_multi_surrogate()."
            )

    models_dir = Path(models_dir)
    print(f"\n  [Surrogato MULTI] Cartella modelli: {models_dir}")

    # ── 1. Carica lo scaler DOF condiviso ────────────────────────────────────
    scaler_dof_path = models_dir / "scaler_DOF.joblib"
    print(f"  Caricamento scaler DOF: {scaler_dof_path}")
    scaler_dof = joblib.load(str(scaler_dof_path))
    dof_feature_names = list(scaler_dof.feature_names_in_)
    print(f"  Features DOF attese ({len(dof_feature_names)}): {dof_feature_names}")

    # ── 2. Carica un modello + scaler per ogni OF mappato ────────────────────
    # Struttura interna: lista parallela a OF_NAMES
    #   _models[i]  = tf.function compilata per OF_NAMES[i], oppure None se non mappato
    #   _scalers[i] = scaler inverse_transform per OF_NAMES[i], oppure None

    _models  = []   # tf.function per ogni OF
    _scalers = []   # scaler output per ogni OF

    for of_name in OF_NAMES:
        file_key = of_name_to_file_key.get(of_name, None)

        if file_key is None:
            # OF non mappato: questo surrogato non lo predice
            print(f"  ⚠  {of_name:<20} → non mappato, verrà restituito come NaN")
            _models.append(None)
            _scalers.append(None)
            continue

        model_path  = models_dir / f"model_{file_key}.keras"
        scaler_path = models_dir / f"scaler_OF_{file_key}.joblib"

        if not model_path.exists():
            print(f"  ⚠  {of_name:<20} → modello non trovato: {model_path}, verrà restituito come NaN")
            _models.append(None)
            _scalers.append(None)
            continue

        if not scaler_path.exists():
            print(f"  ⚠  {of_name:<20} → scaler non trovato: {scaler_path}, verrà restituito come NaN")
            _models.append(None)
            _scalers.append(None)
            continue

        # Carica modello e compilalo come tf.function
        keras_model = tf.keras.models.load_model(str(model_path))

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, keras_model.input_shape[-1]], dtype=tf.float32)
        ])
        def _make_infer(m=keras_model):
            # Closure esplicita per catturare il modello corretto
            def _infer(x, model=m):
                return model(x, training=False)
            return _infer

        fast_infer = _make_infer()

        # Carica scaler output (Pipeline sklearn)
        scaler_of = joblib.load(str(scaler_path))

        print(f"  ✓  {of_name:<20} → {model_path.name} + {scaler_path.name}")
        _models.append(fast_infer)
        _scalers.append(scaler_of)

    # ── 3. Definisce la funzione predict ─────────────────────────────────────

    def predict(dof_raw):
        """
        dof_raw : np.array shape (n_dof_totali,) — DOF in unità fisiche reali
                  I valori devono corrispondere alle features attese da scaler_DOF
                  nell'ordine definito da DOF_NAMES_ALL in Set_input_param.py.

        return  : np.array shape (n_of_totali,)  — OF in unità fisiche reali
                  Gli OF non mappati vengono restituiti come np.nan.

        Pipeline per ogni OF:
            DOF grezzi (array numpy)
              → DataFrame con nomi colonne (richiesto da ColumnTransformer)
              → scaler_DOF.transform()   → DOF normalizzati (condiviso)
              → model_OF_i.predict()     → OF_i normalizzato
              → scaler_OF_i.inverse_transform() → OF_i in unità fisiche
        """
        import pandas as pd

        # Costruisce il DataFrame che il ColumnTransformer si aspetta
        dof_df = pd.DataFrame(
            dof_raw.reshape(1, -1).astype(np.float64),
            columns=dof_feature_names
        )

        # Normalizza i DOF (una sola volta per tutti i modelli)
        dof_scaled = scaler_dof.transform(dof_df).astype(np.float32)
        x_tensor   = tf.constant(dof_scaled.reshape(1, -1).astype(np.float32))

        of_values = np.full(len(OF_NAMES), np.nan, dtype=np.float32)

        for i, (infer_fn, scaler_of) in enumerate(zip(_models, _scalers)):
            if infer_fn is None or scaler_of is None:
                continue  # OF non mappato → lascia NaN
            pred_scaled = infer_fn(x_tensor).numpy()                # shape (1,1)
            pred_real   = scaler_of.inverse_transform(pred_scaled)  # shape (1,1)
            of_values[i] = float(pred_real[0, 0])

        return of_values

    # ── 4. Warm-up ───────────────────────────────────────────────────────────
    print("  Warm-up surrogato multi-modello...")
    dummy = np.random.uniform(
        [b[0] for b in DOF_BOUNDS_ALL],
        [b[1] for b in DOF_BOUNDS_ALL]
    ).astype(np.float32)
    predict(dummy)
    print("  Surrogato multi-modello pronto.\n")

    return predict


# ============================================================
# FUNZIONE DI REWARD
# ============================================================

def compute_reward(of_current, of_previous, of_start, tolleranza=None):
    """
    Reward positiva se CSI diminuisce, negativa se aumenta.
    Penalità quadratica progressiva se φ o ψ escono dalla tolleranza.
    """
    if np.isnan(of_current).any() or np.isinf(of_current).any():
        return -10.0

    csi_curr = of_current[IDX_CSI]
    csi_prev = of_previous[IDX_CSI]

    psi_curr  = of_current[IDX_PSI]
    phi_curr  = of_current[IDX_PHI]
    psi_start = of_start[IDX_PSI]
    phi_start = of_start[IDX_PHI]

    errore_psi = abs(psi_curr - psi_start) / (abs(psi_start) + 1e-8)
    errore_phi = abs(phi_curr - phi_start) / (abs(phi_start) + 1e-8)

    penalty = 0.0
    if errore_psi > tolleranza:
        penalty += 50.0 * ((errore_psi - tolleranza) ** 2)
    if errore_phi > tolleranza:
        penalty += 50.0 * ((errore_phi - tolleranza) ** 2)

    return float(csi_prev - csi_curr) - penalty


def compute_reward_target(of_current, of_previous, phi_target, psi_target, tol_rel=0.02):
    """
    Variante della reward con target assoluti di φ e ψ (anziché relativi al profilo di partenza).
    """
    if np.isnan(of_current).any() or np.isinf(of_current).any():
        return -10.0

    csi_curr = of_current[IDX_CSI]
    csi_prev = of_previous[IDX_CSI]

    errore_psi = abs(of_current[IDX_PSI] - psi_target) / (abs(psi_target) + 1e-8)
    errore_phi = abs(of_current[IDX_PHI] - phi_target) / (abs(phi_target) + 1e-8)

    penalty = 0.0
    if errore_psi > tol_rel:
        penalty += 50.0 * ((errore_psi - tol_rel) ** 2)
    if errore_phi > tol_rel:
        penalty += 50.0 * ((errore_phi - tol_rel) ** 2)

    return float(csi_prev - csi_curr) - penalty


# ============================================================
# AMBIENTE GYMNASIUM
# ============================================================

class BladeOptimEnv(gym.Env):
    """
    Ambiente Gymnasium per l'ottimizzazione del profilo di pala.

    Compatibile con entrambe le modalità di surrogato:
      - predict() restituita da load_surrogate()        (surrogato singolo)
      - predict() restituita da load_multi_surrogate()  (surrogato multi-modello)

    L'interfaccia è identica in entrambi i casi: la funzione predict riceve
    un np.array (n_dof_totali,) e restituisce un np.array (n_of_totali,).
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 surrogate_fn,
                 start_dof=None,
                 use_delta=True,
                 episode_length=60,
                 action_scale=ACTION_SCALE,
                 ref_of=None,
                 target_phi=None,
                 target_psi=None):

        super().__init__()

        self.predict      = surrogate_fn
        self.start_dof    = start_dof
        self.ref_of       = ref_of
        self.ep_length    = episode_length
        self.action_scale = action_scale
        self.use_delta    = use_delta
        self.target_phi   = target_phi
        self.target_psi   = target_psi

        n_active_dof = len(DOF_BOUNDS)
        n_of         = len(OF_NAMES)

        self.dof_low   = np.array([b[0] for b in DOF_BOUNDS], dtype=np.float32)
        self.dof_high  = np.array([b[1] for b in DOF_BOUNDS], dtype=np.float32)
        self.dof_range = self.dof_high - self.dof_low

        self.dof_low_all  = np.array([b[0] for b in DOF_BOUNDS_ALL], dtype=np.float32)
        self.dof_high_all = np.array([b[1] for b in DOF_BOUNDS_ALL], dtype=np.float32)

        # Spazio azioni: un valore in [-1, +1] per ogni DOF attivo
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_active_dof,),
            dtype=np.float32
        )

        # Spazio osservazioni: DOF attivi normalizzati + tutti gli OF + target + errori
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_active_dof + n_of + 4,),
            dtype=np.float32
        )

        self.current_dof_active = None
        self.current_dof_full   = None
        self.current_of         = None
        self.start_of           = None
        self.step_count         = 0

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
            # Task 2: profilo fisso dal database
            self.current_dof_full = np.array(self.start_dof, dtype=np.float32)
        else:
            # Task 1: profilo casuale
            self.current_dof_full = self.np_random.uniform(
                self.dof_low_all, self.dof_high_all
            ).astype(np.float32)

        self.current_dof_active = self.current_dof_full[ACTIVE_DOF_INDICES].copy()
        self.current_of = self.predict(self.current_dof_full)
        self.start_of   = self.current_of.copy()
        self.step_count = 0

        # Ancoraggio vincoli
        if self.start_dof is None:
            # Task 1: il riferimento si riancoraall'episodio corrente
            self.ref_of = self.start_of.copy()
        else:
            # Task 2: il riferimento è fisso (impostato una sola volta)
            if self.ref_of is None:
                self.ref_of = self.start_of.copy()

        return self._get_observation(), {}

    def step(self, action):
        self.step_count += 1
        prev_of = self.current_of.copy()

        if self.use_delta:
            delta          = action * self.action_scale * self.dof_range
            new_dof_active = np.clip(self.current_dof_active + delta, self.dof_low, self.dof_high)
        else:
            new_dof_active = self.dof_low + (action + 1.0) / 2.0 * self.dof_range

        new_dof_full = self.current_dof_full.copy()
        for i, idx in enumerate(ACTIVE_DOF_INDICES):
            new_dof_full[idx] = new_dof_active[i]

        new_of = self.predict(new_dof_full).astype(np.float32)

        tolleranza_max = 0.005

        if (self.target_phi is not None) and (self.target_psi is not None):
            reward = compute_reward_target(
                new_of, prev_of,
                phi_target=self.target_phi,
                psi_target=self.target_psi,
                tol_rel=tolleranza_max
            )
            errore_psi = abs(new_of[IDX_PSI] - self.target_psi) / (abs(self.target_psi) + 1e-8)
            errore_phi = abs(new_of[IDX_PHI] - self.target_phi) / (abs(self.target_phi) + 1e-8)
        else:
            reward     = float(np.squeeze(compute_reward(new_of, prev_of, self.ref_of, tolleranza=tolleranza_max)))
            errore_psi = abs(new_of[IDX_PSI] - self.ref_of[IDX_PSI]) / (abs(self.ref_of[IDX_PSI]) + 1e-8)
            errore_phi = abs(new_of[IDX_PHI] - self.ref_of[IDX_PHI]) / (abs(self.ref_of[IDX_PHI]) + 1e-8)

        is_valid = bool(errore_psi <= tolleranza_max and errore_phi <= tolleranza_max)

        self.current_dof_active = new_dof_active
        self.current_dof_full   = new_dof_full
        self.current_of         = new_of

        terminated = False
        truncated  = self.step_count >= self.ep_length

        obs  = self._build_obs(self.current_dof_active, self.current_of)
        info = {
            "csi":        float(new_of[IDX_CSI]),
            "psi":        float(new_of[IDX_PSI]),
            "dof_active": self.current_dof_active.copy(),
            "dof_full":   self.current_dof_full.copy(),
            "of":         self.current_of.copy(),
            "is_valid":   is_valid,
            "err_phi_rel": float(errore_phi),
            "err_psi_rel": float(errore_psi),
        }

        return obs, reward, terminated, truncated, info