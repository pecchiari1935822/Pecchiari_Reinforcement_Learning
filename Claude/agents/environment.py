import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from pathlib import Path

from config.settings import (
    DOF_NAMES, DOF_BOUNDS, OF_NAMES, IDX_CSI,
    EPISODE_LENGTH, ACTION_SCALE, TARGET_ROWS
)
from config.paths import DATASET_PATH, validate_required_files
from models.surrogate import get_surrogate
from utils.logger import logger


class BladeOptimizationEnv(gym.Env):
    """
    Ambiente Gymnasium per l'ottimizzazione del profilo palare.

    AZIONE:
      - Vettore di N valori in [-1, +1], uno per ogni DOF attivo
      - Mappa direttamente a [dof_low, dof_high]

    STATO:
      - DOF attivi normalizzati [0, 1]
      - 15 OF grezzi (non normalizzati)
      - Totale: N_active_dof + 15 valori

    REWARD:
      - Positiva se CSI scende (meno perdite)
      - Formula: CSI_precedente - CSI_attuale

    TERMINAZIONE:
      - truncated=True dopo EPISODE_LENGTH step
      - terminated=False (no early stopping per episodio)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 surrogate_fn=None,
                 start_dof=None,
                 active_dof_indices=None,
                 episode_length=EPISODE_LENGTH,
                 action_scale=ACTION_SCALE):
        """
        Inizializza l'ambiente.

        Args:
            surrogate_fn: Funzione predict(dof) → of. Se None, carica automaticamente
            start_dof: Profilo iniziale (array 7). Se None, random
            active_dof_indices: Quali DOF l'agente può modificare
            episode_length: Max step per episodio
            action_scale: Scala delle azioni (deprecato, non usato)
        """
        super().__init__()

        # Valida configurazione
        validate_required_files()

        # Surrogate model
        if surrogate_fn is None:
            self.surrogate = get_surrogate()
        else:
            self.surrogate = surrogate_fn

        # Configurazione DOF
        self.start_dof = start_dof.copy() if start_dof is not None else None
        self.active_dof_indices = active_dof_indices or list(range(len(DOF_NAMES)))
        self.n_active_dof = len(self.active_dof_indices)

        # Bounds
        self.dof_bounds_all = np.array(DOF_BOUNDS, dtype=np.float32)
        self.dof_low_all = self.dof_bounds_all[:, 0]
        self.dof_high_all = self.dof_bounds_all[:, 1]

        # Bounds per DOF attivi
        self.dof_low_active = self.dof_low_all[self.active_dof_indices]
        self.dof_high_active = self.dof_high_all[self.active_dof_indices]
        self.dof_range_active = self.dof_high_active - self.dof_low_active

        # Episode
        self.episode_length = episode_length
        self.step_count = 0

        # Stato interno
        self.current_dof_full = None  # Tutti i 7 DOF
        self.current_dof_active = None  # Solo i DOF attivi
        self.current_of = None  # 15 OF

        # ─── SPAZI (Gymnasium) ───
        # Azioni: N DOF attivi, ognuno in [-1, +1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_active_dof,),
            dtype=np.float32
        )

        # Osservazioni: DOF normalizzati [0,1] + OF
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_active_dof + len(OF_NAMES),),
            dtype=np.float32
        )

        logger.info(f"BladeOptimizationEnv initialized:")
        logger.info(f"  Active DOF: {[DOF_NAMES[i] for i in self.active_dof_indices]}")
        logger.info(f"  Action space: {self.action_space}")
        logger.info(f"  Observation space: {self.observation_space}")

    def _get_observation(self):
        """
        Costruisce osservazione: DOF normalizzati [0,1] + OF grezzi.

        Returns:
            array shape (n_active_dof + 15,)
        """
        # Estrai DOF attivi dal vettore completo
        self.current_dof_active = self.current_dof_full[self.active_dof_indices].copy()

        # Normalizza DOF attivi in [0, 1]
        dof_normalized = (self.current_dof_active - self.dof_low_active) / (self.dof_range_active + 1e-8)

        # Concatena: DOF normalizzati + OF grezzi
        obs = np.concatenate([dof_normalized, self.current_of]).astype(np.float32)

        return obs

    def reset(self, seed=None, options=None):
        """
        Reset conforme a Gymnasium API v26.

        Returns:
            (observation, info_dict)
        """
        super().reset(seed=seed)

        # Inizializza DOF
        if self.start_dof is not None:
            self.current_dof_full = np.array(self.start_dof, dtype=np.float32).copy()
        else:
            # Random all'interno dei bounds
            self.current_dof_full = self.np_random.uniform(
                self.dof_low_all, self.dof_high_all
            ).astype(np.float32)

        # Valuta surrogate
        try:
            self.current_of = self.surrogate.predict(self.current_dof_full)
        except Exception as e:
            logger.error(f"Surrogate prediction failed during reset: {e}")
            raise

        self.step_count = 0

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """
        Applica azione dell'agente.

        Args:
            action: array shape (n_active_dof,) in [-1, +1]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1

        # Valida azione
        if not self.action_space.contains(action):
            logger.warning(f"Action out of bounds: {action}")
            action = np.clip(action, -1.0, 1.0)

        # OF precedente (per reward)
        prev_of = self.current_of.copy()

        # ─── MAPPING AZIONE → DOF ───
        # Azione in [-1, 1] mappa direttamente a [dof_low, dof_high]
        # -1.0 → dof_low, 0.0 → midpoint, +1.0 → dof_high
        new_dof_active = self.dof_low_active + (action + 1.0) / 2.0 * self.dof_range_active

        # Aggiorna profilo completo (mantieni i DOF non attivi)
        new_dof_full = self.current_dof_full.copy()
        for i, idx in enumerate(self.active_dof_indices):
            new_dof_full[idx] = new_dof_active[i]

        # Valuta surrogate
        try:
            new_of = self.surrogate.predict(new_dof_full)
        except Exception as e:
            logger.error(f"Surrogate prediction failed during step: {e}")
            new_of = prev_of.copy()  # Fallback

        # ─── REWARD: minimizza CSI ───
        csi_curr = new_of[IDX_CSI]
        csi_prev = prev_of[IDX_CSI]
        reward = float(csi_prev - csi_curr)

        # Aggiorna stato
        self.current_dof_full = new_dof_full
        self.current_of = new_of

        # ─── TERMINAZIONE ───
        terminated = False
        truncated = self.step_count >= self.episode_length

        # Osservazione
        obs = self._get_observation()

        # Info per monitor/callback
        info = {
            "step": self.step_count,
            "csi": float(csi_curr),
            "csi_delta": float(reward),
            "dof_full": self.current_dof_full.copy(),
            "dof_active": self.current_dof_active.copy(),
            "of": self.current_of.copy(),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Rendering (non implementato per questo env)."""
        pass

    def close(self):
        """Cleanup."""
        super().close()


# ============================================================
# Utility functions
# ============================================================

def make_env(active_dof_indices=None, start_dof=None, seed=None):
    """
    Factory function per creare environment.

    Uso:
      >>> env = make_env(active_dof_indices=[0], seed=42)
      >>> obs, info = env.reset()
    """
    env = BladeOptimizationEnv(
        active_dof_indices=active_dof_indices,
        start_dof=start_dof
    )

    if seed is not None:
        env.reset(seed=seed)

    return env


if __name__ == "__main__":
    # Test rapido
    logger.info("Testing BladeOptimizationEnv...")

    env = make_env(active_dof_indices=[0])
    obs, info = env.reset()

    logger.info(f"Observation shape: {obs.shape}")
    logger.info(f"First few obs values: {obs[:5]}")

    # Esegui alcuni step
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"Step {i}: reward={reward:.4f}, csi={info['csi']:.6f}")

    env.close()
    logger.info("Test completed!")