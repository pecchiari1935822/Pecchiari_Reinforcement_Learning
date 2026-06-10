import numpy as np
import omnisafe
import gymnasium as gym
import os
import time
import shutil
import glob
import pandas as pd
from Ambiente.FOCOPS_Ambiente import BladeOptimEnv
from Config.Set_input_param import PPO_PARAMS, \
    TOTAL_TIMESTEPS, n_dof_totali, target_phi, target_psi, DOF_NAMES_ALL, OF_NAMES, ACTIVE_DOF_INDICES, early_stopping
from Report.Plot import _plot_results, _plot_training_metrics_actor, _plot_training_metrics_critic, _plot_dof_evolution

# Registrazione dell'ambiente custom in Gymnasium per permettere ad OmniSafe di istanziarlo
try:
    gym.register(
        id='BladeOptimEnv-v0',
        entry_point='Ambiente.Ambiente:BladeOptimEnv',
    )
except gym.error.Error:
    pass  # Già registrato in precedenza


# ============================================================
# LOG ADAPTER — Traduce i log di OmniSafe per la suite di Plot
# ============================================================
class OmniSafeLogAdapter:
    """
    Simula la struttura del vecchio Callback di SB3 leggendo il file progress.csv di OmniSafe.
    Consente alle funzioni di plot esistenti di girare senza modifiche.
    """

    def __init__(self, log_dir, best_csi, best_dof, best_of):
        self.best_csi = best_csi
        self.best_dof = best_dof
        self.best_of = best_of

        # Cerca dinamicamente il file csv generato da OmniSafe
        csv_files = glob.glob(os.path.join(log_dir, "**", "progress.csv"), recursive=True)
        if not csv_files:
            print(f"⚠️ Attenzione: progress.csv non trovato in {log_dir}. Grafici vuoti.")
            metrics_df = pd.DataFrame()
        else:
            metrics_df = pd.read_csv(csv_files[-1])

        # Mappatura delle colonne da OmniSafe a nomi compatibili con i tuoi script di plot
        length = len(metrics_df) if not metrics_df.empty else 0

        self.metrics = {
            "explained_variance": metrics_df[
                "Loss/ExplainedVariance"].tolist() if "Loss/ExplainedVariance" in metrics_df else [0.0] * length,
            "entropy_loss": (
                -metrics_df["Train/Entropy"]).tolist() if "Train/Entropy" in metrics_df else [0.0] * length,
            # Invertito perché SB3 usa la loss negativa
            "std": [0.0] * length,
            "approx_kl": metrics_df["Train/KL"].tolist() if "Train/KL" in metrics_df else [0.0] * length,
            "clip_fraction": [0.0] * length,
            "value_loss": metrics_df["Loss/Value"].tolist() if "Loss/Value" in metrics_df else metrics_df[
                "Loss/Loss_v"].tolist() if "Loss/Loss_v" in metrics_df else [0.0] * length,
            "policy_gradient_loss": metrics_df["Loss/Pi"].tolist() if "Loss/Pi" in metrics_df else metrics_df[
                "Loss/Loss_pi"].tolist() if "Loss/Loss_pi" in metrics_df else [0.0] * length,
            "ep_rew_mean": metrics_df["Metrics/EpRet"].tolist() if "Metrics/EpRet" in metrics_df else [0.0] * length,
            "ep_cost_mean": metrics_df["Metrics/EpCost"].tolist() if "Metrics/EpCost" in metrics_df else [0.0] * length
        }

        self.timesteps = metrics_df["TotalSteps"].tolist() if "TotalSteps" in metrics_df else list(range(length))
        self.num_timesteps = self.timesteps[-1] if length > 0 else 0


# ============================================================
# TRAINING CON FOCOPS (OMNISAFE)
# ============================================================

def train(start_dof=None,
          learning_rate=None, n_steps=None, batch_size=None, ROW_INDEX=None, use_delta=True, episode_length=None,
          ref_of=None,
          task1=None
          ):
    print("=" * 60)
    print("  FOCOPS Blade Optimization — OmniSafe Framework")
    print(f"  DOF attivi: {[DOF_NAMES_ALL[i] for i in ACTIVE_DOF_INDICES]}")
    print(f"  Reward: Massimizza riduzione CSI | Costo: Vincoli su Phi/Psi")
    print("=" * 60)

    # Reset totale delle variabili statiche di tracciamento prima di iniziare il run
    BladeOptimEnv.best_csi = np.inf
    BladeOptimEnv.best_dof = None
    BladeOptimEnv.best_of = None

    # Configurazione percorsi logs nativi OmniSafe
    if start_dof is None:
        log_dir = f"./focops_blade_task1_logs"
        model_basename = f"focops_task1_lr{learning_rate}_nsteps{n_steps}"
    else:
        log_dir = f"./focops_blade_task2_logs_riga{ROW_INDEX}"
        model_basename = f"focops_task2_lr{learning_rate}_nsteps{n_steps}_riga{ROW_INDEX}"

    # Dizionario delle configurazioni custom per OmniSafe
    custom_cfgs = {
        'env_cfgs': {
            'start_dof': start_dof,
            'use_delta': use_delta,
            'episode_length': episode_length,
            'target_phi': target_phi,
            'target_psi': target_psi,
            'ref_of': ref_of,
        },
        'train_cfgs': {
            'total_steps': TOTAL_TIMESTEPS,
            'vector_env_nums': 1,
            # Manteniamo 1 per garantire l'allineamento dei profili ottimi nel processo principale
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': n_steps if n_steps is not None else 2048,
            'update_iters': 10,
            'batch_size': batch_size if batch_size is not None else 64,
            'cost_limit': 0.0,  # FOCOPS forzerà l'agente a riportare il costo (sforamento) a 0.0 (cioè profili validi)
        },
        'model_cfgs': {
            'actor': {
                'hidden_sizes': [256, 256] if task1 else [128, 128],
            },
            'critic': {
                'hidden_sizes': [256, 256] if task1 else [128, 128],
            },
            'actor_lr': learning_rate if learning_rate is not None else 3e-4,
            'critic_lr': learning_rate if learning_rate is not None else 3e-4,
        },
        'logger_cfgs': {
            'use_tensorboard': True,
            'log_dir': log_dir,
            'save_model_freq': 10,
        }
    }

    start_time = time.time()

    # Inizializzazione del Learner di OmniSafe impostando l'algoritmo FOCOPS
    agent = omnisafe.Learner(
        algo='FOCOPS',
        env_id='BladeOptimEnv-v0',
        custom_cfgs=custom_cfgs,
    )

    # Avvio dell'addestramento nativo
    agent.learn()

    end_time = time.time()
    training_time = end_time - start_time

    # Salvataggio del modello finale in formato standard torch
    model_save_path = os.path.join("Risultati", "Modelli", model_basename)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    # OmniSafe salva automaticamente in log_dir/torch_save/model.pt, creiamo un link simbolico o copia se serve

    # Recuperiamo il profilo ottimale dalle variabili di classe statiche valorizzate durante gli step validi
    best_dof = BladeOptimEnv.best_dof
    best_of = BladeOptimEnv.best_of
    best_csi = BladeOptimEnv.best_csi

    # Creazione dell'adapter per far girare i vecchi plot trasparentemente
    adapter_cb = OmniSafeLogAdapter(log_dir, best_csi, best_dof, best_of)

    # Esecuzione dei tuoi vecchi script di grafici e metriche
    _print_results(adapter_cb)
    _plot_results(adapter_cb, learning_rate, n_steps, training_time=training_time)
    _plot_training_metrics_actor(adapter_cb, learning_rate, n_steps)
    _plot_training_metrics_critic(adapter_cb, learning_rate, n_steps)
    _plot_dof_evolution(adapter_cb, learning_rate, n_steps, start_dof=start_dof)

    return agent, best_dof, best_of, best_csi, log_dir


def _print_results(cb: OmniSafeLogAdapter):
    print(f"\n  Miglior CSI valido trovato durante FOCOPS training: {cb.best_csi:.6f}")
    if cb.best_dof is not None:
        print("\n  DOF ottimali che mi danno il miglior CSI (* = modificato):")
        for i, (name, val) in enumerate(zip(DOF_NAMES_ALL, cb.best_dof)):
            m = " *" if i in ACTIVE_DOF_INDICES else "  "
            print(f"    {name:<28}{m}: {val:.6f}")
        print("\n  OF corrispondenti ai DOF ottimali:")
        for name, val in zip(OF_NAMES, cb.best_of):
            print(f"    {name:<30}: {val:.6f}")
    else:
        print("  ❌ Nessun profilo valido (entro le tolleranze cinemetiche) è stato esplorato in questo run.")


def pulisci_file_temporanei(task_1=False):
    """Elimina immagini temporanee mantenendo intatti i log OmniSafe strutturati."""
    print("\n" + "=" * 60)
    print("  PULIZIA FILE TEMPORANEI IMPOSTATA")
    print("=" * 60)

    immagini = ["plot_results.png", "plot_metrics_actor.png", "plot_metrics_critic.png", "plot_dof_evolution_barre.png",
                "plot_dof_evolution.png", "smith_diagram_action_assiale.png",
                "smith_diagram_action_total_to_total.png",
                "smith_diagram_reaction_total_to_total.png", "smith_diagram_reaction_ZOOM.png",
                "plot_dof_evolution_1.png",
                "plot_dof_evolution_2.png", "plot_dof_evolution_3.png"
                ]
    for img in immagini:
        if os.path.exists(img):
            os.remove(img)
            print(f"  [X] Eliminato grafico locale: {img}")

    if task_1:
        if os.path.exists("task1_results.csv"):
            os.remove("task1_results.csv")

    print("  Pulizia completata! I log strutturati di OmniSafe sono conservati per l'analisi.")
    print("=" * 60)