import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import time
import shutil
import glob
from Ambiente.Ambiente import BladeOptimEnv
from Config.Set_input_param import PPO_PARAMS, \
    TOTAL_TIMESTEPS, n_dof_totali, target_phi, target_psi, DOF_NAMES_ALL, OF_NAMES, ACTIVE_DOF_INDICES
from Report.Plot import _plot_results, _plot_training_metrics_actor, _plot_training_metrics_critic, _plot_dof_evolution, _plot_dof_evolution_barre


# ============================================================
# CALLBACK — traccia CSI e Score per episodio
# ============================================================

class BladeCallback(BaseCallback):
    """
    Traccia per ogni episodio:
      - CSI finale (ultimo step dell'episodio)
      - Score = reward cumulativa dell'episodio (come nel paper)
      - DOF e OF del miglior risultato trovato in tutto il training
    """

    def __init__(self, verbose=0, start_dof=None, episode_length=None):
        super().__init__(verbose)
        self.ep_rewards = []
        self.ep_lengths = []
        self.timesteps = []
        self.start_dof = start_dof

        # Early stopping
        if episode_length == 20:
            self.patience = 4000  # numero di step senza miglioramento prima di fermare
        if episode_length == 40:
                self.patience = 12000
        self.steps_senza_miglioramenti = 0

        # Metriche episodio
        self.episode_csi = []
        self.episode_scores = []
        self._current_score = 0.0
        self.best_csi_ep = np.inf
        self.best_csi = np.inf
        self.best_dof = None
        self.best_of = None

        self.best_dof_ep = None
        self.episode_best_dofs = []

        # Metriche PPO per aggiornamento
        self.metrics = {
            "explained_variance": [],
            "entropy_loss": [],
            "std": [],
            "approx_kl": [],
            "clip_fraction": [],
            "value_loss": [],
            "policy_gradient_loss": [],
            "ep_rew_mean":[],
        }
        self.n_episodes = 0
        self.metrics_episodes = []
        self.metrics_timesteps = []

    # smma le ricompense ottenute in ogni step (20) per ottenere lo score totale dell'episodio
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        self.steps_senza_miglioramenti += 1

        if len(self.model.ep_info_buffer) > 0:
            # ep_info_buffer contiene info sugli episodi completati

            # Estrai ep_rew_mean (media ultimi 100 episodi)
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            if len(ep_rewards) > 0:
                ep_rew_mean = np.mean(ep_rewards)

                # Salva
                self.ep_rewards.append(ep_rew_mean)
                self.timesteps.append(self.num_timesteps)

                # Debug
                if self.num_timesteps % 100 == 0:
                    print(f"Step {self.num_timesteps}: ep_rew_mean = {ep_rew_mean:.4f}")

        for reward, info, done in zip(rewards, infos, dones):
            csi_step = info.get("csi", None)
            dof_step = info.get("dof_full", None)
            of_step = info.get("of", None)
            is_valid = info.get("is_valid", True)  # <--- Leggiamo se il profilo ha superato i vincoli

            # Entriamo nell'aggiornamento del "Migliore" SOLO SE il profilo è valido (entro il 3%)
            if csi_step is not None and is_valid:

                if csi_step < self.best_csi_ep:
                    self.best_csi_ep = csi_step
                    self.best_dof_ep = dof_step.copy() if dof_step is not None else None

                if csi_step < self.best_csi:
                    self.best_csi = csi_step
                    if dof_step is not None:
                        self.best_dof = dof_step.copy()
                    if of_step is not None:
                        self.best_of = of_step.copy()
                    self.steps_senza_miglioramenti = 0
                    self.logger.record("custom/best_CSI", self.best_csi)

            self._current_score += reward



            if done:
                if self.best_dof_ep is None:
                    self.best_dof_ep = np.full(n_dof_totali, np.nan)
                    # ---------------------------------------------------

                self.episode_csi.append(self.best_csi_ep)
                self.episode_scores.append(self._current_score)
                self.episode_best_dofs.append(self.best_dof_ep)

                self._current_score = 0.0
                self.n_episodes += 1

                self.logger.record("custom/CSI", self.best_csi_ep)
                self.logger.record("custom/Score", self.episode_scores[-1])

                if self.best_csi_ep < self.best_csi:
                    self.best_csi = self.best_csi_ep
                    self.steps_senza_miglioramenti = 0
                    # Qui non c'è bisogno del copy() perché in questo if ci entriamo
                    # solo se csi_step è stato aggiornato con dati validi
                    tmp_dof = info.get("dof_full", None)
                    tmp_of = info.get("of", None)
                    if tmp_dof is not None:
                        self.best_dof = tmp_dof
                    if tmp_of is not None:
                        self.best_of = tmp_of
                    self.logger.record("custom/best_CSI", self.best_csi)

                self.best_csi_ep = np.inf  # Reset per il prossimo episodio
                self.best_dof_ep = None


        '''if self.start_dof is not None:
            if self.steps_senza_miglioramenti >= self.patience:
                    print(f"\n[Early Stopping] Nessun miglioramento del CSI per {self.patience} step. Interruzione.")
                    return False  # Questo ferma il PPO!'''

        return True

    def _on_rollout_end(self) -> None:
        """
        Chiamato dopo ogni raccolta di n_steps.
        Legge le metriche PPO dal logger interno di SB3.
        """
        log = self.model.logger.name_to_value

        chiavi_sb3 = {
            "train/explained_variance": "explained_variance",
            "train/entropy_loss": "entropy_loss",
            "train/std": "std",
            "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction",
            "train/value_loss": "value_loss",
            "train/policy_gradient_loss": "policy_gradient_loss",

        }

        valori = {
            k_nostra: log[k_sb3]
            for k_sb3, k_nostra in chiavi_sb3.items()
            if k_sb3 in log
        }

        if len(self.model.ep_info_buffer) > 0:
            ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            if ep_rewards:
                valori["ep_rew_mean"] = float(np.mean(ep_rewards))

        if valori:
            for chiave, valore in valori.items():
                self.metrics[chiave].append(float(valore))
            self.metrics_timesteps.append(self.num_timesteps)
            self.metrics_episodes.append(self.n_episodes)

        return True


# ============================================================
# TRAINING
# ============================================================

def train(surrogate_fn,
          start_dof=None,
          learning_rate=None, n_steps=None, batch_size=None, ROW_INDEX=None, use_delta = True, episode_length=None, ref_of=None):
    print("=" * 60)
    print("  PPO Blade Optimization — Stable Baselines3")
    print(f"  DOF attivi: {[DOF_NAMES_ALL[i] for i in ACTIVE_DOF_INDICES]}")
    print(f"  Reward: minimizza CSI (OF_CSI_OP_01)")
    print("=" * 60)

    # check_env PRIMA di Monitor (evita bug NoneType)
    print("\n  Verifica ambiente...")
    env_raw = BladeOptimEnv(surrogate_fn, start_dof=start_dof, use_delta =use_delta, episode_length= episode_length,
                            target_phi=target_phi, target_psi=target_psi, ref_of=ref_of)
    check_env(env_raw, warn=True)
    print("  Ambiente OK.\n")

    if start_dof is None:
        import re
        active_names = [DOF_NAMES_ALL[i] for i in ACTIVE_DOF_INDICES]
        safe_names = [re.sub(r'[^0-9A-Za-z]+', '', n) for n in active_names]
        active_tag = "_".join(safe_names) if safe_names else "ALL"

        # --- percorsi dinamici per checkpoint, log e monitor ---
        checkpoint_dir = f"./ppo_blade_generale_checkpoints_{active_tag}/"
        log_dir = f"./ppo_blade_generale_logs_{active_tag}/"
        monitor_file = f"./ppo_blade_generale_monitor_{active_tag}"
    else:
        import re
        active_names = [DOF_NAMES_ALL[i] for i in ACTIVE_DOF_INDICES]
        safe_names = [re.sub(r'[^0-9A-Za-z]+', '', n) for n in active_names]
        active_tag = "_".join(safe_names) if safe_names else "ALL"

        # --- percorsi dinamici per checkpoint, log e monitor ---
        checkpoint_dir = f"./ppo_blade_start_profile_checkpoints_{active_tag}/"
        log_dir = f"./ppo_blade_start_profile_logs_{active_tag}/"
        monitor_file = f"./ppo_blade_start_profile_monitor_{active_tag}"

        # Crea le directory se non esistono
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)


    env = Monitor(env_raw, filename="./ppo_blade_monitor")

    cb_blade = BladeCallback(verbose=0, start_dof=start_dof)
    cb_ckpt = CheckpointCallback(
        save_freq=10000, save_path="./ppo_blade_checkpoints/",
        name_prefix="ppo_blade", verbose=1,
    )

    model = PPO(
        policy="MlpPolicy", env=env,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        tensorboard_log=log_dir,
        seed=42,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        **PPO_PARAMS
    )

    print(f"  Addestramento: {TOTAL_TIMESTEPS:,} step")
    print("  (tensorboard --logdir ./ppo_blade_logs/)\n")

    # --- costruisci tag identificativo basato sui nomi DOF attivi ---
    if start_dof is None:
        active_names = [DOF_NAMES_ALL[i] for i in ACTIVE_DOF_INDICES]
        safe_names = [re.sub(r'[^0-9A-Za-z]+', '', n) for n in active_names]
        active_tag = "_".join(safe_names) if safe_names else "ALL"
        if use_delta == True:
            model_basename = f"ppo_task1_con_phi_psi_uguali_{active_tag}_lr{learning_rate}_nsteps{n_steps}_con_delta"
        else:
            model_basename = f"ppo_task1_con_phi_psi_uguali_{active_tag}_lr{learning_rate}_nsteps{n_steps}_senza_delta"
        model_path = model_basename  # SB3 aggiunge .zip se serve

        print(f"  Addestramento: {TOTAL_TIMESTEPS:,} step")
        print(f"  Modello di output: {model_path}.zip")
        print("  (tensorboard --logdir ./ppo_blade_logs/)\n")

        start_time = time.time()

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[cb_blade, cb_ckpt],
            progress_bar=True,
        )

        model.save(model_path)
        print(f"\n  Modello salvato: {model_path}.zip")

    else:
        active_names = [DOF_NAMES_ALL[i] for i in ACTIVE_DOF_INDICES]
        safe_names = [re.sub(r'[^0-9A-Za-z]+', '', n) for n in active_names]
        active_tag = "_".join(safe_names) if safe_names else "ALL"
        if use_delta == True:
            model_basename = f"ppo_task2_use_delta_con_phi_psi_uguali_{active_tag}_lr{learning_rate}_nsteps{n_steps}_riga{ROW_INDEX}"
        else:
            model_basename = f"ppo_task2_no_use_delta_con_phi_psi_uguali_{active_tag}_lr{learning_rate}_nsteps{n_steps}_riga{ROW_INDEX}"
        model_path = model_basename  # SB3 aggiunge .zip se serve

        print(f"  Addestramento: {TOTAL_TIMESTEPS:,} step")
        print(f"  Modello di output: {model_path}.zip")
        print("  (tensorboard --logdir ./ppo_blade_logs/)\n")

        start_time = time.time()

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[cb_blade, cb_ckpt],
            progress_bar=True,
        )

        model.save(model_path)
        print(f"\n  Modello salvato: {model_path}.zip")

    end_time = time.time()
    training_time = end_time - start_time

    _print_results(cb_blade)
    _plot_results(cb_blade, learning_rate, n_steps, training_time=training_time)
    _plot_training_metrics_actor(cb_blade, learning_rate, n_steps)
    _plot_training_metrics_critic(cb_blade, learning_rate, n_steps)
    _plot_dof_evolution(cb_blade, learning_rate, n_steps, start_dof=start_dof)
    _plot_dof_evolution_barre(cb_blade, learning_rate, n_steps, start_dof=start_dof)

    env.close()

    return model, cb_blade.best_dof, cb_blade.best_of, cb_blade.best_csi, model_path

# ============================================================
# UTILITY
# ============================================================

def _print_results(cb: BladeCallback):
    print(f"\n  Miglior CSI durante training: {cb.best_csi:.6f}")
    if cb.best_dof is not None:
        print("\n  DOF ottimali che mi danno il miglior CSI (* = modificato):")
        for i, (name, val) in enumerate(zip(DOF_NAMES_ALL, cb.best_dof)):
            m = " *" if i in ACTIVE_DOF_INDICES else "  "
            print(f"    {name:<28}{m}: {val:.6f}")
        print("\n  OF corrispondenti ai DOF ottimali:")
        for name, val in zip(OF_NAMES, cb.best_of):
            print(f"    {name:<30}: {val:.6f}")

    print("\n  Questo è il profilo che ha il miglior CSI Durante il training")



def pulisci_file_temporanei():
    """Elimina log, immagini temporanee e checkpoint, mantenendo solo Modelli e PPTX."""
    print("\n" + "=" * 60)
    print("  PULIZIA FILE TEMPORANEI")
    print("=" * 60)

    # 1. Elimina le immagini dei grafici
    immagini = ["plot_results.png", "plot_metrics_actor.png", "plot_metrics_critic.png","plot_dof_evolution_barre.png",
                "plot_dof_evolution.png", "smith_diagram_action_assiale.png",
                "smith_diagram_action_total_to_total.png",
                "smith_diagram_reaction_total_to_total.png"
                ]
    for img in immagini:
        if os.path.exists(img):
            os.remove(img)
            print(f"  [X] Eliminato: {img}")

    # 2. Elimina i file CSV del Monitor di Gym
    for monitor_file in glob.glob("*monitor*.csv"):
        os.remove(monitor_file)
        print(f"  [X] Eliminato: {monitor_file}")

    os.remove("task1_results.csv")

    # 3. Elimina le cartelle di Checkpoint e Log di TensorBoard
    # Trova tutte le cartelle che corrispondono a questi pattern
    cartelle_da_eliminare = [
        "ppo_blade_checkpoints",  # Quella hardcoded di base
    ]
    cartelle_da_eliminare.extend(glob.glob("ppo_blade_*_checkpoints_*"))
    cartelle_da_eliminare.extend(glob.glob("ppo_blade_*_logs_*"))

    for cartella in cartelle_da_eliminare:
        if os.path.exists(cartella) and os.path.isdir(cartella):
            shutil.rmtree(cartella)  # rmtree cancella la cartella e tutto il suo contenuto
            print(f"  [X] Eliminata cartella: {cartella}")

    print("=" * 60)
    print("  Pulizia completata! Conservati solo i file .zip finali e la presentazione.")
    print("=" * 60)




