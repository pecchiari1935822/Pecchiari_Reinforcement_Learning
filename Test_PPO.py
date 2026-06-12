import re
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO

import Ambiente.PPO_Ambiente as env_module
from Ambiente.PPO_Ambiente import (
    BladeOptimEnv, surrogate,
    DOF_BOUNDS_ALL, IDX_CSI, IDX_PHI, IDX_PSI
)
from Config.PPO_Set_input_param import (
    ACTIVE_DOF_INDICES, DOF_NAMES_ALL, OF_NAMES,
    TARGET_CSI, TARGET_PHI, TARGET_PSI,
    tolleranza_profilo_partenza, df
)

# ============================================================
# CONFIGURAZIONE — modifica solo questa sezione
# ============================================================

ROW_INDEX    = 564        # riga del dataset da ottimizzare
EP_LENGTH    = 40       # numero di step per episodio

# Task2 con modello vecchio
MODELLO_NOME = "PPO_task2_vecchio_dataset_lr3e-05_nsteps200_riga[3].zip"

# Task1 con modello vecchio
#MODELLO_NOME = "PPO_task1_vecchio_dataset_lr3e-05_nsteps200_con_delta.zip"

# Task2 con modello nuovo
#MODELLO_NOME = "PPO_task2_nuovo_dataset_lr3e-05_nsteps200_riga[3].zip"

# Task1 con modello nuovo
#MODELLO_NOME = "PPO_task1_nuovo_dataset_lr3e-05_nsteps200_con_delta.zip"

# ============================================================


def plot_inferenza_results(steps_csi, best_csi, csi_originale, ep_length, modello_name, riga_idx=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = np.arange(1, len(steps_csi) + 1)
    axes[0].plot(steps, steps_csi, 'b-o', linewidth=2, markersize=6, label='CSI per step')
    axes[0].axhline(y=best_csi, color='g', linestyle='--', linewidth=2, label=f'Best CSI: {best_csi:.6f}')
    axes[0].axhline(y=csi_originale, color='r', linestyle='--', linewidth=2,
                    label=f'CSI originale: {csi_originale:.6f}')
    axes[0].set_xlabel('Step', fontsize=15)
    axes[0].set_ylabel('CSI', fontsize=15)
    axes[0].set_title('CSI Evolution', fontsize=17, fontweight='bold')
    axes[0].legend(fontsize=15)
    axes[0].tick_params(axis='both', labelsize=14)
    axes[0].grid(True, alpha=0.3)

    miglioramento = csi_originale - best_csi
    percentuale = (miglioramento / abs(csi_originale) * 100) if csi_originale != 0 else 0

    categories = ['Originale', 'Min esplorato', 'Migliore valido']
    values = [csi_originale, min(steps_csi), best_csi]
    colors = ['#d9534f', '#f0ad4e', '#5cb85c']
    bars = axes[1].bar(categories, values, color=colors, width=0.4, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('CSI', fontsize=15)
    axes[1].set_title(f'Confronto  delle prestazioni',
                      fontsize=17, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f'{height:.6f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=11, fontweight='bold')

    riga_str = f'  |  riga dataset: {riga_idx}' if riga_idx is not None else ''
    plt.suptitle(f'{modello_name}{riga_str}', fontsize=17, fontweight='bold', color='navy')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs('Risultati_Inferenza', exist_ok=True)
    nome_pulito = re.sub(r'[^\w]', '_', modello_name)
    save_path = f'Risultati_Inferenza/Plot_Inferenza_PPO_{nome_pulito}_riga_{riga_idx}.png'
    plt.savefig(save_path, dpi=150)
    print(f'  📊 Grafico salvato in: {save_path}')
    plt.show()


if __name__ == '__main__':
    print('=' * 60)
    print('  INIZIO INFERENZA PPO')
    print('=' * 60)

    # ----------------------------------------------------------
    # 1. Configura env_module prima di costruire l'ambiente
    # ----------------------------------------------------------
    env_module.ACTIVE_DOF_INDICES = ACTIVE_DOF_INDICES
    env_module.DOF_BOUNDS = [DOF_BOUNDS_ALL[i] for i in ACTIVE_DOF_INDICES]

    # ----------------------------------------------------------
    # 2. Leggi il profilo di partenza usando i nomi colonna
    #    (flessibile rispetto al dataset)
    # ----------------------------------------------------------
    riga = df.iloc[ROW_INDEX]
    start_dof = np.array([riga[col] for col in DOF_NAMES_ALL], dtype=np.float32)
    of_dataset = np.array([riga[col] for col in OF_NAMES], dtype=np.float32)

    csi_dataset = float(riga[TARGET_CSI])  # CSI reale/simulato scritto nel dataset

    # CSI predetto dal surrogate sullo stesso profilo di partenza
    of_surrogato = surrogate(start_dof)
    csi_surrogato = float(of_surrogato[IDX_CSI])

    target_phi_val = float(riga[TARGET_PHI])
    target_psi_val = float(riga[TARGET_PSI])
    tol = tolleranza_profilo_partenza

    print(f'\n  Profilo di partenza: riga {ROW_INDEX} del dataset')
    print(f'\nPPO:')
    print(f'  CSI nel dataset (reale/simulato): {csi_dataset:.6f}')
    print(f'  CSI predetto dal surrogate:       {csi_surrogato:.6f}')
    print(f'  Differenza surrogate vs dataset:  {csi_surrogato - csi_dataset:+.6f}')
    print(f'  target phi: {target_phi_val:.6f}')
    print(f'  target psi: {target_psi_val:.6f}')
    print(f'  tolleranza: {tol}')
    print('-' * 60)

    # ----------------------------------------------------------
    # 3. Carica modello e costruisci ambiente
    # ----------------------------------------------------------
    modello_salvato = os.path.join('Risultati', 'Modelli', MODELLO_NOME)
    print(f'  Caricamento modello: {modello_salvato}')

    model = PPO.load(modello_salvato)
    env = BladeOptimEnv(start_dof=start_dof, episode_length=EP_LENGTH)
    obs, info = env.reset()

    print(f'  Avvio loop di ottimizzazione per {EP_LENGTH} passi...\n')

    # ----------------------------------------------------------
    # 4. Loop di inferenza
    # ----------------------------------------------------------
    steps_csi = []
    best_csi = float('inf')
    best_dof = None
    best_of = None
    best_step = -1

    for step in range(EP_LENGTH):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        current_csi = float(info['csi'])
        is_valid = bool(info.get('is_valid', False))
        steps_csi.append(current_csi)

        # Calcolo errori phi e psi (stessa logica di Test.py)
        current_of = info['of']
        phi_val = float(current_of[IDX_PHI])
        psi_val = float(current_of[IDX_PSI])
        errore_phi = abs(phi_val - target_phi_val) / (abs(target_phi_val) + 1e-8)
        errore_psi = abs(psi_val - target_psi_val) / (abs(target_psi_val) + 1e-8)

        valid_tag = '[✓ VALIDO]' if is_valid else '[x NON VALIDO]'
        print(
            f'\n  Step {step + 1:02d} | CSI: {current_csi:.6f}  phi_err: {errore_phi:.4f}  psi_err: {errore_psi:.4f}  {valid_tag}')
        print(f'    phi: {phi_val:.6f}  (target: {target_phi_val:.6f})')
        print(f'    psi: {psi_val:.6f}  (target: {target_psi_val:.6f})')

        if is_valid and current_csi < best_csi:
            best_csi = current_csi
            best_dof = info['dof_full'].copy()
            best_of = info['of'].copy()
            best_step = step + 1

        if terminated or truncated:
            print(f'  Episodio terminato al passo {step + 1}')
            break

    # ----------------------------------------------------------
    # 5. Risultati finali
    # ----------------------------------------------------------
    print('\n' + '=' * 60)
    print('  RISULTATI FINALI INFERENZA PPO')
    print('=' * 60)

    modello_display = os.path.basename(modello_salvato)

    if best_dof is not None:
        miglioramento = best_csi - csi_surrogato
        percentuale = (miglioramento / abs(csi_surrogato)) * 100

        print(f'  🏆 Ottimizzazione riuscita al passo {best_step}/{EP_LENGTH}!')
        print(f'  Modello: {modello_display}')
        print(f'\n  CSI nel dataset (reale)   : {csi_dataset:.6f}')
        print(f'  CSI surrogate (partenza)  : {csi_surrogato:.6f}')
        print(f'  CSI Migliore PPO (valido) : {best_csi:.6f}')
        print(f'  Delta PPO vs surrogate    : {miglioramento:+.6f} ({percentuale:+.2f}%)')
        print(f'  Delta PPO vs dataset      : {best_csi - csi_dataset:+.6f}')
        print('-' * 60)
        print('  DOF Ottimizzati (* = attivo):')
        for i, (name, val_ott, val_orig) in enumerate(zip(DOF_NAMES_ALL, best_dof, start_dof)):
            marker = ' *' if i in ACTIVE_DOF_INDICES else '  '
            print(f'    {name:<28}{marker}: {val_ott:.6f}  (originale: {val_orig:+.6f})')
        print('\n  OF Associate:')
        for name, val_ott, val_orig in zip(OF_NAMES, best_of, of_surrogato):
            print(f'    {name:<30}: {val_ott:.6f}  (originale: {val_orig:+.6f})')
    else:
        print('  ❌ L\'agente non ha trovato nessun profilo valido.')
        print('  Tutti i profili generati hanno violato le tolleranze su phi/psi.')
        print(f'\n  CSI nel dataset (reale)   : {csi_dataset:.6f}')
        print(f'  CSI surrogate (partenza)  : {csi_surrogato:.6f}')
        best_csi = csi_surrogato  # fallback per il plot

    print('=' * 60)

    # ----------------------------------------------------------
    # 6. Plot
    # ----------------------------------------------------------
    plot_inferenza_results(
        steps_csi=steps_csi,
        best_csi=best_csi,
        csi_originale=csi_surrogato,  # confronto contro il CSI predetto dal surrogate
        ep_length=EP_LENGTH,
        modello_name=modello_display,
        riga_idx=ROW_INDEX,
    )