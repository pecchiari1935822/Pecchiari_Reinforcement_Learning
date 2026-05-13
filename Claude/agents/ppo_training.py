"""
agents/ppo_training.py
=====================
Funzione principale per addestrare il modello PPO.

Responsabilità:
  1. Setup dell'ambiente
  2. Configurazione e creazione del modello PPO
  3. Esecuzione del training
  4. Generazione grafici di metriche
  5. Salvataggio del modello
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from config.settings import (
    TOTAL_TIMESTEPS, DOF_NAMES, ACTIVE_DOF_INDICES, PPO_CONFIG, OF_NAMES
)
from config.paths import (
    get_checkpoint_dir, get_log_dir, get_model_save_path,
    CHECKPOINTS_DIR
)
from agents.environment import BladeOptimizationEnv
from agents.callbacks import BladeCallback
from models.surrogate import get_surrogate
from visualization.plots import plot_results, plot_metrics_actor, plot_metrics_critic
from utils.logger import logger


def train_ppo(
        active_dof_indices: list = None,
        start_dof: np.ndarray = None,
        learning_rate: float = 0.0003,
        n_steps: int = 50,
        batch_size: int = 32,
        total_timesteps: int = TOTAL_TIMESTEPS,
        ppo_config: dict = None,
) -> Tuple[object, np.ndarray, np.ndarray, float, str]:
    """
    Addestra il modello PPO per ottimizzazione blade.

    Args:
        active_dof_indices: List di indici DOF attivi (es. [0] per solo PITCH)
        start_dof: Array (7,) con profilo iniziale. Se None, random
        learning_rate: Learning rate PPO
        n_steps: Number di steps prima di update
        batch_size: Batch size per update
        total_timesteps: Total training timesteps
        ppo_config: Dict con parametri PPO aggiuntivi

    Returns:
        Tuple: (model, best_dof, best_of, best_csi, model_path)

    Raises:
        FileNotFoundError: Se surrogate model non trovato
        ValueError: Se configurazione non valida
    """

    # ─── VALIDAZIONE PARAMETRI ───
    if active_dof_indices is None:
        active_dof_indices = ACTIVE_DOF_INDICES

    if ppo_config is None:
        ppo_config = PPO_CONFIG

    logger.info("=" * 70)
    logger.info("🚀 PPO BLADE OPTIMIZATION TRAINING")
    logger.info("=" * 70)
    logger.info(f"Active DOF: {[DOF_NAMES[i] for i in active_dof_indices]}")
    logger.info(f"Start profile: {'Random' if start_dof is None else 'Custom'}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"N_steps: {n_steps} | Batch size: {batch_size}")
    logger.info(f"Total timesteps: {total_timesteps:,}")

    # ─── CARICA SURROGATE ───
    logger.info("\nLoading surrogate model...")
    try:
        surrogate = get_surrogate()
        logger.info("✅ Surrogate loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load surrogate: {e}")
        raise

    # ─── CREA AMBIENTE ───
    logger.info("\nCreating environment...")
    try:
        env_raw = BladeOptimizationEnv(
            surrogate_fn=surrogate,
            start_dof=start_dof,
            active_dof_indices=active_dof_indices
        )

        # Validazione ambiente (SB3)
        logger.info("Validating environment...")
        check_env(env_raw, warn=True)
        logger.info("✅ Environment OK")
    except Exception as e:
        logger.error(f"❌ Environment creation failed: {e}")
        raise

    # ─── WRAPPA CON MONITOR ───
    logger.info("Wrapping with Monitor...")
    env = Monitor(env_raw)

    # ─── SETUP CHECKPOINT E LOG ───
    checkpoint_dir = get_checkpoint_dir(
        [DOF_NAMES[i] for i in active_dof_indices],
        suffix="training"
    )
    log_dir = get_log_dir(
        [DOF_NAMES[i] for i in active_dof_indices],
        suffix="training"
    )
    model_path = get_model_save_path(
        [DOF_NAMES[i] for i in active_dof_indices],
        learning_rate=learning_rate,
        n_steps=n_steps,
        suffix="training"
    )

    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Log dir: {log_dir}")
    logger.info(f"Model will be saved to: {model_path}.zip")

    # ─── CREA CALLBACKS ───
    logger.info("\nSetting up callbacks...")

    # Blade callback (traccia CSI, early stopping)
    cb_blade = BladeCallback(
        patience=4000,
        verbose=0
    )

    # Checkpoint callback (salva modello ogni N timestep)
    cb_checkpoint = CheckpointCallback(
        save_freq=10000,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_blade",
        verbose=1
    )

    # ─── CREA MODELLO PPO ───
    logger.info("\nCreating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
        tensorboard_log=str(log_dir),
        seed=42,
        verbose=1,
        **ppo_config
    )

    logger.info(f"✅ PPO model created")

    # ─── ADDESTRAMENTO ───
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)
    logger.info(f"Tensorboard: tensorboard --logdir {log_dir}")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[cb_blade, cb_checkpoint],
            progress_bar=True,
            log_interval=10
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Training duration: {duration:.1f}s ({duration / 60:.1f}m)")

    # ─── SALVA MODELLO ───
    logger.info("\nSaving model...")
    try:
        model.save(model_path)
        logger.info(f"✅ Model saved to: {model_path}.zip")
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")

    # ─── CHIUDI AMBIENTE ───
    env.close()

    # ─── ESTRAI RISULTATI ───
    best_dof = cb_blade.best_dof
    best_of = cb_blade.best_of
    best_csi = cb_blade.best_csi

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Best CSI found: {best_csi:.6f}")
    logger.info(f"Total episodes: {cb_blade.n_episodes}")
    logger.info(f"Total steps: {cb_blade.num_timesteps:,}")
    logger.info("=" * 70)

    # ─── GENERA GRAFICI ───
    logger.info("\nGenerating plots...")
    try:
        plot_results(
            cb_blade,
            learning_rate=learning_rate,
            n_steps=n_steps,
            training_time=duration
        )
        logger.info("✅ Results plot generated")

        plot_metrics_actor(
            cb_blade,
            learning_rate=learning_rate,
            n_steps=n_steps
        )
        logger.info("✅ Actor metrics plot generated")

        plot_metrics_critic(
            cb_blade,
            learning_rate=learning_rate,
            n_steps=n_steps
        )
        logger.info("✅ Critic metrics plot generated")
    except Exception as e:
        logger.error(f"⚠️  Failed to generate plots: {e}")

    return model, best_dof, best_of, best_csi, model_path


def print_training_results(best_dof, best_of, best_csi):
    """
    Stampa risultati training in formato leggibile.

    Args:
        best_dof: Array (7,) con DOF ottimali
        best_of: Array (15,) con OF corrispondenti
        best_csi: Best CSI value
    """
    logger.info("\n🎯 BEST FOUND CONFIGURATION")
    logger.info("=" * 70)

    if best_dof is not None:
        logger.info("DOF Ottimali (* = modificati):")
        for i, (name, val) in enumerate(zip(DOF_NAMES, best_dof)):
            marker = " *" if i in ACTIVE_DOF_INDICES else "  "
            logger.info(f"  {name:<25}{marker}: {val:.6f}")

    if best_of is not None:
        logger.info("\nOF Risultanti:")
        for name, val in zip(OF_NAMES, best_of):
            logger.info(f"  {name:<25}: {val:.6f}")

    logger.info(f"\nBest CSI: {best_csi:.6f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    # Test
    logger.info("Testing PPO training...")

    model, best_dof, best_of, best_csi, model_path = train_ppo(
        active_dof_indices=[0],
        learning_rate=0.0003,
        n_steps=50,
        batch_size=32,
        total_timesteps=50_000  # Short for testing
    )

    print_training_results(best_dof, best_of, best_csi)