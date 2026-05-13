"""
agents/callbacks.py
===================
Custom callbacks per PPO training.

Classe BladeCallback:
  - Traccia CSI e Score per episodio
  - Implementa early stopping
  - Raccoglie metriche PPO interne
  - Identifica miglior profilo trovato
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from utils.logger import logger


class BladeCallback(BaseCallback):
    """
    Custom callback per monitorare training PPO dell'ottimizzazione blade.

    Traccia:
      - CSI finale per episodio
      - Score (reward cumulativa) per episodio
      - Migliore CSI globale e relativo DOF/OF
      - Metriche PPO (entropy, KL divergence, value loss, etc.)

    Early stopping:
      - Se il CSI non migliora per PATIENCE step, ferma il training
    """

    def __init__(self, patience: int = 4000, verbose: int = 0):
        """
        Args:
            patience: Step senza miglioramento CSI prima di fermare
            verbose: Verbosità logging
        """
        super().__init__(verbose)

        # ─── EARLY STOPPING ───
        self.patience = patience
        self.steps_senza_miglioramenti = 0

        # ─── METRICHE EPISODIO ───
        self.episode_csi = []  # CSI migliore in ogni episodio
        self.episode_scores = []  # Reward cumulativa per episodio
        self.episode_best_dofs = []  # Profilo che ha dato miglior CSI nell'episodio
        self._current_score = 0.0  # Accumula reward nello step corrente

        # ─── BEST GLOBALE ───
        self.best_csi = np.inf
        self.best_dof = None
        self.best_of = None

        # ─── BEST EPISODIO (variabile temporanea) ───
        self.best_csi_ep = np.inf
        self.best_dof_ep = None

        # ─── METRICHE PPO ───
        # Questi sono i valori interni che PPO calcola durante training
        self.metrics = {
            "explained_variance": [],  # Quanto la value function spiega i returns
            "entropy_loss": [],  # Entropia della policy (esplorazione)
            "std": [],  # Standard deviation azioni
            "approx_kl": [],  # KL divergence approssimato
            "clip_fraction": [],  # % di azioni clippate
            "value_loss": [],  # Errore value function
            "policy_gradient_loss": [],  # Loss della policy
            "ep_rew_mean": [],  # Media rewards ultimi episodi
        }

        # Timestamp per sincronizzare metriche
        self.n_episodes = 0
        self.metrics_episodes = []  # Numero episodio quando è stata registrata metrica
        self.metrics_timesteps = []  # Timestep quando è stata registrata metrica

        logger.info(f"BladeCallback initialized (patience={patience})")

    def _on_step(self) -> bool:
        """
        Chiamato ad OGNI STEP del training.

        Returns:
            False se deve fermare il training, True altrimenti
        """
        # Incrementa contatore step senza miglioramento
        self.steps_senza_miglioramenti += 1

        # Estrai reward e info dello step corrente
        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # ─── TRACCIA REWARD MEDIO ───
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            if len(ep_rewards) > 0:
                ep_rew_mean = np.mean(ep_rewards)
                self.ep_rewards.append(ep_rew_mean)
                self.timesteps.append(self.num_timesteps)

                if self.num_timesteps % 100 == 0:
                    logger.debug(f"Step {self.num_timesteps}: ep_rew_mean={ep_rew_mean:.4f}")

        # ─── PROCESSA OGNI STEP ───
        for reward, info, done in zip(rewards, infos, dones):
            # Estrai CSI e DOF dallo step
            csi_step = info.get("csi", None)
            dof_step = info.get("dof_full", None)

            if csi_step is not None:
                # Aggiorna best episodio (meglio in questo episodio)
                self.best_csi_ep = min(self.best_csi_ep, csi_step)
                if dof_step is not None:
                    self.best_dof_ep = dof_step.copy()

            # Accumula reward dello step
            self._current_score += reward

            # ─── EPISODIO TERMINATO ───
            if done:
                # Salva metriche episodio
                self.episode_csi.append(self.best_csi_ep)
                self.episode_scores.append(self._current_score)
                self.episode_best_dofs.append(self.best_dof_ep)

                self.n_episodes += 1
                self._current_score = 0.0

                # Log a TensorBoard
                self.logger.record("custom/CSI", self.best_csi_ep)
                self.logger.record("custom/Score", self.episode_scores[-1])

                # ─── AGGIORNA BEST GLOBALE ───
                if self.best_csi_ep < self.best_csi:
                    self.best_csi = self.best_csi_ep
                    self.steps_senza_miglioramenti = 0  # RESETTA contatore
                    self.best_dof = info.get("dof_full", None)
                    self.best_of = info.get("of", None)
                    self.logger.record("custom/best_CSI", self.best_csi)

                    logger.info(f"🎯 New best CSI: {self.best_csi:.6f} (ep {self.n_episodes})")

                # Reset best episodio per il prossimo
                self.best_csi_ep = np.inf
                self.best_dof_ep = None

        # ─── EARLY STOPPING ───
        if self.steps_senza_miglioramenti >= self.patience:
            logger.warning(
                f"\n⚠️  EARLY STOPPING: No CSI improvement for {self.patience} steps. "
                f"Best CSI: {self.best_csi:.6f}"
            )
            return False  # STOP training

        return True  # CONTINUE training

    def _on_rollout_end(self) -> None:
        """
        Chiamato DOPO OGNI ROLLOUT (raccolta di n_steps).

        Estrae e salva le metriche PPO interne da SB3.
        Queste sono generate durante l'update della policy.
        """
        # Accedi al logger interno di SB3
        log = self.model.logger.name_to_value

        # Mappa: chiave_SB3 → chiave_nostra
        chiavi_sb3 = {
            "train/explained_variance": "explained_variance",
            "train/entropy_loss": "entropy_loss",
            "train/std": "std",
            "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction",
            "train/value_loss": "value_loss",
            "train/policy_gradient_loss": "policy_gradient_loss",
        }

        # Estrai solo le metriche che SB3 ha registrato
        valori = {
            k_nostra: log[k_sb3]
            for k_sb3, k_nostra in chiavi_sb3.items()
            if k_sb3 in log
        }

        # Aggiungi ep_rew_mean (media ultimi episodi)
        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            if ep_rewards:
                valori["ep_rew_mean"] = float(np.mean(ep_rewards))

        # Salva tutte le metriche
        if valori:
            for chiave, valore in valori.items():
                self.metrics[chiave].append(float(valore))

            self.metrics_timesteps.append(self.num_timesteps)
            self.metrics_episodes.append(self.n_episodes)

        return True


# ============================================================
# Utility per callback
# ============================================================

def get_best_result_from_callback(callback: BladeCallback) -> dict:
    """
    Estrae il risultato migliore dal callback.

    Args:
        callback: BladeCallback dell'allenamento

    Returns:
        dict con: best_csi, best_dof, best_of, n_episodes
    """
    return {
        "best_csi": callback.best_csi,
        "best_dof": callback.best_dof,
        "best_of": callback.best_of,
        "n_episodes": callback.n_episodes,
        "total_episodes": len(callback.episode_csi),
    }


if __name__ == "__main__":
    # Test instantiation
    cb = BladeCallback(patience=4000)
    print(f"BladeCallback initialized with {len(cb.metrics)} metrics")