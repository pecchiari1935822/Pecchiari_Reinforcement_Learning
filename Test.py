import re
import numpy as np
import pandas as pd


from stable_baselines3 import PPO
from Ambiente.Ambiente_claude_senza_keras import BladeOptimEnv, load_surrogate, DOF_BOUNDS_ALL
from Agente.PPO import SURROGATE_MODEL_PATH, SCALER_PATH, Path

import matplotlib.pyplot as plt
import numpy as np


def plot_inferenza_results(steps_csi, best_csi, csi_originale, ep_length, modello_name, riga_idx=None):
    """
    Plotta i risultati dell'inferenza e mostra in alto il modello e la riga del dataset.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico 1: CSI per step
    steps = np.arange(1, len(steps_csi) + 1)
    axes[0].plot(steps, steps_csi, 'b-o', linewidth=2, markersize=10, label='CSI per step')
    axes[0].axhline(y=best_csi, color='g', linestyle='--', linewidth=2, label=f'Best CSI: {best_csi:.6f}')
    axes[0].axhline(y=csi_originale, color='r', linestyle='--', linewidth=2,
                    label=f'CSI originale: {csi_originale:.6f}')
    axes[0].set_xlabel('Step', fontsize=15)
    axes[0].set_ylabel('CSI', fontsize=15)
    axes[0].set_title(f'CSI Evolution', fontsize=15, fontweight='bold')
    axes[0].legend(fontsize=13)
    axes[0].tick_params(axis='both', labelsize=14)
    axes[0].grid(True, alpha=0.3)

    # Grafico 2: Miglioramento percentuale
    miglioramento = csi_originale - best_csi
    percentuale = (miglioramento / csi_originale * 100) if csi_originale > 0 else 0

    categories = ['CSI Originale', 'Best CSI PPO']
    values = [csi_originale, best_csi]
    colors = ['red', 'green']

    bars = axes[1].bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('CSI Value', fontsize=15)
    axes[1].set_title(f'Comparison - Improvement: {miglioramento:+.6f} ({percentuale:+.2f}%)',
                      fontsize=15, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=14)

    # Aggiungi valori sopra le barre
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.6f}',
                     ha='center', va='bottom', fontsize=15, fontweight='bold')

    axes[1].grid(True, alpha=0.3, axis='y')

    # Titolo principale con modello e riga dataset
    suptitle_text = modello_name
    if riga_idx is not None:
        suptitle_text += f'  |  riga dataset: {riga_idx}'
    fig.suptitle(suptitle_text, fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



if __name__ == "__main__":
    from stable_baselines3 import PPO

    import Ambiente.Ambiente_claude_senza_keras as env_module

    active_dof_per_test = [0, 1, 2, 3, 4, 5, 6]
    env_module.ACTIVE_DOF_INDICES = active_dof_per_test
    env_module.DOF_BOUNDS = [DOF_BOUNDS_ALL[i] for i in active_dof_per_test]

    DATABASE_DIR = Path(__file__).parent.resolve()
    DATASET_PATH = str(DATABASE_DIR / "Data" / "database.dat")

    df = pd.read_csv(DATASET_PATH)
    riga_nuova_idx = 3
    riga_nuova = df.iloc[riga_nuova_idx].values
    DOF_profilo = riga_nuova[2:9].astype(np.float32).copy()
    OF_profilo = riga_nuova[9:24].astype(np.float32).copy()
    csi_nuovo_originale = float(riga_nuova[11])

    #modello_salvato = "ppo_task2_use_delta_con_phi_psi_uguali_DOFPITCH_DOFBETA1_DOFBETA2_DOFW1_DOFW2_DOFTMOVXU_DOFTMOVXL_lr3e-05_nsteps200_riga[710]"
    modello_salvato = "ppo_task1_con_phi_psi_uguali_DOFPITCH_DOFBETA1_DOFBETA2_DOFW1_DOFW2_DOFTMOVXU_DOFTMOVXL_lr3e-05_nsteps200_con_delta"

    print(f"\nProfilo da ottimizzare (riga {riga_nuova_idx}):")
    print(f"\nDOF {DOF_profilo}")
    print(f"\nOF {OF_profilo}")
    print(f"\nCSI di partenza: {csi_nuovo_originale:.6f}")

    # ===== MODIFICATO: Traccia i CSI per ogni step =====
    model = PPO.load(modello_salvato)
    surrogate = load_surrogate(SURROGATE_MODEL_PATH, SCALER_PATH)
    env = BladeOptimEnv(surrogate, start_dof=DOF_profilo, episode_length=40)

    obs, info = env.reset()
    best_csi = np.inf
    best_dof = None
    best_of = None
    steps_csi = []  # ← NUOVO: Salva CSI di ogni step

    print("\n  Inizio episodi di inferenza...")
    for step in range(40):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        current_csi = info["csi"]
        steps_csi.append(current_csi)  # ← NUOVO: Aggiungi alla lista
        print(f"    Step {step + 1:02d} | CSI: {current_csi:.6f}")

        if current_csi < best_csi:
            best_csi = current_csi
            best_dof = info["dof_full"]
            best_of = info["of"]

        if terminated or truncated:
            break

    print(f"\n  Ottimizzazione completata. Miglior CSI trovato: {best_csi:.6f}")
    print(f"\n  Modello: {modello_salvato}")

    print(f"\n DOF ottimizzati: {best_dof}")
    print(f"\n OF ottimizzati: {best_of}")


    miglioramento = csi_nuovo_originale - best_csi
    print("\n" + "=" * 60)
    print("  RISULTATO INFERENZA PPO vs DATASET")
    print("=" * 60)
    print(f"  CSI Originale (riga {riga_nuova_idx}) : {csi_nuovo_originale:.6f}")
    print(f"  CSI Migliore PPO           : {best_csi:.6f}")
    print(f"  Differenza (Delta CSI)     : {miglioramento:+.6f}")
    print("=" * 60)



    # Determina l'etichetta da mostrare
    m = re.search(r'riga\[\d+\]$', modello_salvato)
    if m:
        display_name = f"task 2 con {m.group(0)}"
    elif 'senza_delta' in modello_salvato:
        display_name = "task1"
    else:
        display_name = modello_salvato.split('/')[-1][:30]

    # ===== NUOVO: Genera il grafico =====
    plot_inferenza_results(
        steps_csi=steps_csi,
        best_csi=best_csi,
        csi_originale=csi_nuovo_originale,
        ep_length=40,
        modello_name=display_name,
        riga_idx=riga_nuova_idx
    )