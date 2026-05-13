"""
visualization/plots.py
======================
Generazione grafici per il training PPO.

Contiene:
  - plot_results(): CSI, Score, Minimo progressivo
  - plot_metrics_actor(): Metriche della policy
  - plot_metrics_critic(): Metriche della value function
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from config.settings import DOF_NAMES, OF_NAMES, TOTAL_TIMESTEPS
from config.paths import OUTPUT_DIR
from utils.logger import logger


def plot_results(callback, learning_rate: float = None, n_steps: int = None,
                 training_time: float = None, save_path: str = None):
    """
    Genera grafico principale dei risultati: CSI, Score, Minimo progressivo.

    Args:
        callback: BladeCallback con i dati di training
        learning_rate: LR usato (per titolo)
        n_steps: N_steps usato (per titolo)
        training_time: Tempo totale training (secondi)
        save_path: Path dove salvare. Default: output/plot_results.png
    """

    if not callback.episode_csi:
        logger.warning("No data to plot (episode_csi is empty)")
        return None

    if save_path is None:
        save_path = str(OUTPUT_DIR / "plot_results.png")

    # Estrai dati
    csi_arr = np.array(callback.episode_csi)
    score_arr = np.array(callback.episode_scores)
    n_ep = len(csi_arr)
    ep_axis = np.arange(n_ep)

    # Media mobile per leggibilità
    W = min(40, max(n_ep // 5, 2))

    def moving_avg(arr, w):
        if len(arr) < w:
            return ep_axis, arr
        ma = np.convolve(arr, np.ones(w) / w, mode='valid')
        return np.arange(w - 1, len(arr)), ma

    # ─── SETUP FIGURE ───
    fig = plt.figure(figsize=(16, 10))

    title_str = (
        f"PPO Blade Optimization — DOF attivi: {[DOF_NAMES[i] for i in range(min(3, len(DOF_NAMES)))]}\n"
        f"Total steps={TOTAL_TIMESTEPS:,}"
    )
    if learning_rate is not None:
        title_str += f"  LR={learning_rate}"
    if n_steps is not None:
        title_str += f"  n_steps={n_steps}"
    if training_time is not None:
        mins = int(training_time // 60)
        secs = int(training_time % 60)
        title_str += f"  Duration={mins}m {secs}s"

    fig.suptitle(title_str, fontsize=16, fontweight="bold")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ─── 1: CSI per episodio ───
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(ep_axis, csi_arr, s=8, alpha=0.4, color='orange', label='CSI')
    x_ma, ma = moving_avg(csi_arr, W)
    ax1.plot(x_ma, ma, color='steelblue', lw=2, label=f'<CSI> ({W}ep)', alpha=0.7)
    ax1.plot(ep_axis, np.minimum.accumulate(csi_arr), color='green', lw=2, label='Min CSI')
    ax1.axhline(callback.best_csi, color='green', ls='--', lw=1, alpha=0.5)
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('CSI', fontsize=11)
    ax1.set_title('CSI per episodio', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=10)

    # ─── 2: Score per episodio ───
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(ep_axis, score_arr, s=8, alpha=0.4, color='orange', label='Score')
    x_ma2, ma2 = moving_avg(score_arr, W)
    ax2.plot(x_ma2, ma2, color='steelblue', lw=2, label=f'<Score> ({W}ep)', alpha=0.7)
    ax2.plot(ep_axis, np.maximum.accumulate(score_arr), color='green', lw=2, label='Max Score')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Reward cumulativa', fontsize=11)
    ax2.set_title('Score per episodio', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=10)

    # ─── 3: CSI minimo progressivo ───
    ax3 = fig.add_subplot(gs[0, 2])
    min_csi = np.minimum.accumulate(csi_arr)
    ax3.plot(ep_axis, min_csi, color='green', lw=2.5)
    ax3.fill_between(ep_axis, min_csi, alpha=0.2, color='green')
    ax3.annotate(
        f'Min CSI = {callback.best_csi:.5f}',
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=11, color='darkgreen', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', ec='green', alpha=0.8)
    )
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Miglior CSI trovato', fontsize=11)
    ax3.set_title('CSI minimo progressivo', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=10)

    # ─── 4: Distribuzione CSI ───
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(csi_arr, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(callback.best_csi, color='green', lw=2, label=f'Best={callback.best_csi:.5f}')
    ax4.axvline(np.mean(csi_arr), color='orange', lw=2, ls='--', label=f'Mean={np.mean(csi_arr):.5f}')
    ax4.set_xlabel('CSI value', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Distribuzione CSI', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3, axis='y')
    ax4.tick_params(labelsize=10)

    # ─── 5: Distribuzione Score ───
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(score_arr, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax5.axvline(np.max(score_arr), color='green', lw=2, label=f'Max={np.max(score_arr):.2f}')
    ax5.axvline(np.mean(score_arr), color='orange', lw=2, ls='--', label=f'Mean={np.mean(score_arr):.2f}')
    ax5.set_xlabel('Score value', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Distribuzione Score', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3, axis='y')
    ax5.tick_params(labelsize=10)

    # ─── 6: Statistiche ───
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    stats_text = f"""
    STATISTICHE TRAINING
    {'=' * 35}

    Episodi totali: {len(csi_arr)}

    CSI - Best: {callback.best_csi:.6f}
    CSI - Mean: {np.mean(csi_arr):.6f}
    CSI - Std:  {np.std(csi_arr):.6f}

    Score - Max:  {np.max(score_arr):.2f}
    Score - Mean: {np.mean(score_arr):.2f}
    Score - Std:  {np.std(score_arr):.2f}

    Miglioramento (ultimi 10 ep):
      ΔCSIi ≈ {csi_arr[-10:].mean() - csi_arr[-1]:.6f}
    """

    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Results plot saved: {save_path}")
    return save_path

def _plot_dof_evolution(callback, lr, n_step, save_path="plot_dof_evolution.png"):
    """
    Crea un grafico per ogni DOF mostrando l'andamento del miglior valore trovato
    in ogni episodio per tutta la durata dell'addestramento, colorando i punti in
    base al numero dell'episodio.
    """
    if not callback.episode_best_dofs:
        print("  Nessun dato dei DOF da plottare.")
        return

    # Trasforma la lista di array in una matrice (n_episodi, 7_dof)
    dof_data = np.array(callback.episode_best_dofs)
    n_episodes = dof_data.shape[0]
    ep_axis = np.arange(n_episodes)

    best_ep_idx = int(np.argmin(callback.episode_csi))

    np.random.seed(42)  # Fissa il seed così i colori non cambiano a ogni run
    colori_per_episodio = np.random.uniform(0.1,0.75,size=(n_episodes, 3))

    # Creiamo una griglia 4x2 standard (senza sharex)
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle(
        f"Evoluzione dei DOF (Miglior profilo per Episodio)\n"
        f"Total steps={TOTAL_TIMESTEPS:,}  n_steps={n_step}  Learning_Rate={lr}",
        fontsize=16
    )
    axes_flat = axes.flatten()

    for i in range(7):
        ax = axes_flat[i]
        y_vals = dof_data[:, i]

        if i in ACTIVE_DOF_INDICES:
            title_suffix = " (Attivo)"
            ma_color = "black"

            # Sfumatura di colore: c=ep_axis mappa il colore sul numero dell'episodio
            # cmap='viridis' va da viola scuro (inizio), a verde (centro), a giallo (fine)
            ax.scatter(ep_axis, y_vals, c=colori_per_episodio, s=15, alpha=0.8)

            # Elemento "fantasma" per mostrare l'etichetta nella legenda
            # (altrimenti scatter con c= array non crea un'etichetta semplice)
            ax.plot([], [], 'o', color='mediumseagreen', markersize=5, label="Miglior DOF/Ep")
        else:
            title_suffix = " (Fisso)"
            ma_color = "black"
            ax.scatter(ep_axis, y_vals, color='gray', s=15, alpha=0.5, label="Fisso")

        # Aggiungi una media mobile per vedere meglio il trend
        '''if n_episodes > 10:
            w = max(5, n_episodes // 20)
            ma = np.convolve(y_vals, np.ones(w) / w, mode='valid')
            ax.plot(np.arange(w - 1, n_episodes), ma, color=ma_color, lw=2, label=f"Media mobile ({w} ep)")'''

        if callback.best_dof is not None:
            best_val = callback.best_dof[i]
            # Usiamo zorder=5 per assicurarci che la X venga disegnata SOPRA le barre e le linee
            ax.plot(best_ep_idx, best_val, marker='X', color='red', markeredgecolor='black',
                    markersize=12, linestyle='None', zorder=5, label="Miglior Assoluto")

        # Disegna i limiti fisici (bounds)
        dof_min, dof_max = DOF_BOUNDS_ALL[i]
        ax.axhline(dof_min, color='red', ls='--', lw=1, alpha=0.5)
        ax.axhline(dof_max, color='red', ls='--', lw=1, alpha=0.5)

        ax.set_title(DOF_NAMES_ALL[i] + title_suffix, fontsize=12)
        ax.set_xlabel("Episodio", fontsize=10)
        ax.set_ylabel("Valore", fontsize=10)
        ax.grid(alpha=0.3)

        if i in ACTIVE_DOF_INDICES:
            ax.legend(fontsize=8, loc='best')

    # Nascondi l'ottavo grafico vuoto
    axes_flat[7].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    return save_path


def _plot_dof_evolution_barre(callback, lr, n_step, save_path="plot_dof_evolution_barre.png"):
    """
    Crea un grafico per ogni DOF mostrando l'andamento del miglior valore trovato
    in ogni episodio. Usa barre verticali colorate casualmente per ogni episodio.
    """
    if not callback.episode_best_dofs:
        print("  Nessun dato dei DOF da plottare.")
        return

    # Trasforma la lista di array in una matrice (n_episodi, 7_dof)
    dof_data = np.array(callback.episode_best_dofs)
    n_episodes = dof_data.shape[0]
    ep_axis = np.arange(n_episodes)

    best_ep_idx = int(np.argmin(callback.episode_csi))

    # Generiamo un colore RGB casuale per OGNI episodio
    np.random.seed(42)
    colori_per_episodio = np.random.uniform(0.1,0.75,size=(n_episodes, 3))

    # Creiamo una griglia 4x2 standard
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig.suptitle(
        f"Evoluzione dei DOF (Miglior profilo per Episodio)\n"
        f"Total steps={TOTAL_TIMESTEPS:,}  n_steps={n_step}  Learning_Rate={lr}",
        fontsize=16
    )
    axes_flat = axes.flatten()

    for i in range(7):
        ax = axes_flat[i]
        y_vals = dof_data[:, i]

        if i in ACTIVE_DOF_INDICES:
            title_suffix = " (Attivo)"
            ma_color = "black"

            # Disegna le BARRE colorate
            ax.bar(ep_axis, y_vals, color="dodgerblue", width=1.0, alpha=0.9)

            # Elemento fantasma per la legenda (barra vuota)
            ax.bar([-10], [0], color='gray', label="Miglior DOF/Ep")
        else:
            title_suffix = " (Fisso)"
            ma_color = "black"
            ax.bar(ep_axis, y_vals, color='gray', width=1.0, alpha=0.5, label="Fisso")

        # Aggiungi una media mobile per vedere meglio il trend
        '''if n_episodes > 10:
            w = max(5, n_episodes // 20)
            ma = np.convolve(y_vals, np.ones(w) / w, mode='valid')
            ax.plot(np.arange(w - 1, n_episodes), ma, color=ma_color, lw=2, label=f"Media mobile ({w} ep)")'''

        if callback.best_dof is not None:
            best_val = callback.best_dof[i]
            # Usiamo zorder=5 per assicurarci che la X venga disegnata SOPRA le barre e le linee
            ax.plot(best_ep_idx, best_val, marker='X', color='red', markeredgecolor='black',
                    markersize=12, linestyle='None', zorder=5, label="Miglior Assoluto")

        # Disegna i limiti fisici (bounds)
        dof_min, dof_max = DOF_BOUNDS_ALL[i]
        ax.axhline(dof_min, color='red', ls='--', lw=1, alpha=0.5)
        ax.axhline(dof_max, color='red', ls='--', lw=1, alpha=0.5)

        # --- IMPORTANTE: FORZA LO ZOOM DELL'ASSE Y ---
        # Evita che le barre partano forzatamente da 0 rovinando la scala
        padding = (dof_max - dof_min) * 0.1
        if padding == 0: padding = 0.1
        ax.set_ylim(dof_min - padding, dof_max + padding)

        ax.set_title(DOF_NAMES_ALL[i] + title_suffix, fontsize=12)
        ax.set_xlabel("Episodio", fontsize=10)
        ax.set_ylabel("Valore", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(left=-1, right=n_episodes)

        if i in ACTIVE_DOF_INDICES:
            ax.legend(fontsize=8, loc='best')

    # Nascondi l'ottavo grafico vuoto
    axes_flat[7].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    return save_path


def plot_metrics_actor(callback, learning_rate: float = None, n_steps: int = None,
                       save_path: str = None):
    """
    Genera grafici delle metriche della policy (Actor).

    Metriche:
      - Episode reward mean
      - Policy gradient loss
      - Entropy
      - Std (esplorazione)
      - Approx KL
      - Clip fraction
    """

    if not callback.metrics_episodes:
        logger.warning("No metrics to plot (metrics_episodes is empty)")
        return None

    if save_path is None:
        save_path = str(OUTPUT_DIR / "plot_metrics_actor.png")

    ts = np.array(callback.metrics_episodes)

    config = [
        ("ep_rew_mean", "Episode Reward Mean", "navy", None, None),
        ("policy_gradient_loss", "Policy Gradient Loss", "darkslategray", 0.0, "zero"),
        ("entropy_loss", "Entropy H[π] (esplorazione)", "darkorange", None, None),
        ("std", "Std Policy (esplorazione)", "mediumpurple", None, None),
        ("approx_kl", "Approx KL Divergence", "crimson", 0.02, "soglia 0.02"),
        ("clip_fraction", "Clip Fraction (% clippate)", "teal", 0.1, "soglia 0.1"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    fig.suptitle(
        f"Metriche Actor — DOF attivi\n"
        f"Total steps={TOTAL_TIMESTEPS:,}" +
        (f"  LR={learning_rate}" if learning_rate else "") +
        (f"  n_steps={n_steps}" if n_steps else ""),
        fontsize=14, fontweight="bold"
    )

    axes_flat = axes.flatten()

    for idx, (chiave, titolo, colore, ref_val, ref_label) in enumerate(config):
        ax = axes_flat[idx]
        valori = callback.metrics.get(chiave, [])

        if not valori:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=12)
            ax.set_title(titolo, fontsize=11)
            continue

        # Invert entropy (è negativa nei log)
        if chiave == "entropy_loss":
            vals = -np.array(valori)
        else:
            vals = np.array(valori)

        # Plot
        ax.plot(ts[:len(vals)], vals, color=colore, lw=1.5, alpha=0.8, label='value')

        # Media mobile
        if len(vals) >= 5:
            w = max(3, len(vals) // 10)
            ma = np.convolve(vals, np.ones(w) / w, mode='valid')
            ax.plot(ts[w - 1:len(vals)], ma, color=colore, lw=2.5, alpha=0.5,
                    linestyle='--', label=f'MA({w})')

        # Linea riferimento
        if ref_val is not None:
            ax.axhline(ref_val, color='green', ls=':', lw=1.2, alpha=0.7, label=ref_label)

        # Annotazione valore finale
        ax.annotate(
            f"final: {vals[-1]:.4f}",
            xy=(ts[len(vals) - 1], vals[-1]),
            xytext=(-50, 10), textcoords='offset points',
            fontsize=9, color=colore,
            arrowprops=dict(arrowstyle='->', color=colore, lw=0.8)
        )

        ax.set_xlabel('Episode', fontsize=10)
        ax.set_title(titolo, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Actor metrics plot saved: {save_path}")
    return save_path


def plot_metrics_critic(callback, learning_rate: float = None, n_steps: int = None,
                        save_path: str = None):
    """
    Genera grafici delle metriche della value function (Critic).

    Metriche:
      - Explained variance (quanto spiega i returns)
      - Value loss (errore della rete)
    """

    if not callback.metrics_episodes:
        logger.warning("No metrics to plot (metrics_episodes is empty)")
        return None

    if save_path is None:
        save_path = str(OUTPUT_DIR / "plot_metrics_critic.png")

    ts = np.array(callback.metrics_episodes)

    config = [
        ("explained_variance", "Explained Variance (Critic)", "steelblue", 1.0, "ottimo=1.0"),
        ("value_loss", "Value Loss (errore Critic)", "sienna", None, None),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Metriche Critic — DOF attivi\n"
        f"Total steps={TOTAL_TIMESTEPS:,}" +
        (f"  LR={learning_rate}" if learning_rate else "") +
        (f"  n_steps={n_steps}" if n_steps else ""),
        fontsize=14, fontweight="bold"
    )

    axes_flat = axes.flatten()

    for idx, (chiave, titolo, colore, ref_val, ref_label) in enumerate(config):
        ax = axes_flat[idx]
        valori = callback.metrics.get(chiave, [])

        if not valori:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=12)
            ax.set_title(titolo, fontsize=11)
            continue

        vals = np.array(valori)

        # Plot
        ax.plot(ts[:len(vals)], vals, color=colore, lw=1.5, alpha=0.8, label='value')

        # Media mobile
        if len(vals) >= 5:
            w = max(3, len(vals) // 10)
            ma = np.convolve(vals, np.ones(w) / w, mode='valid')
            ax.plot(ts[w - 1:len(vals)], ma, color=colore, lw=2.5, alpha=0.5,
                    linestyle='--', label=f'MA({w})')

        # Linea riferimento
        if ref_val is not None:
            ax.axhline(ref_val, color='green', ls=':', lw=1.2, alpha=0.7, label=ref_label)

        # Annotazione valore finale
        ax.annotate(
            f"final: {vals[-1]:.4f}",
            xy=(ts[len(vals) - 1], vals[-1]),
            xytext=(-50, 10), textcoords='offset points',
            fontsize=9, color=colore,
            arrowprops=dict(arrowstyle='->', color=colore, lw=0.8)
        )

        ax.set_xlabel('Episode', fontsize=10)
        ax.set_title(titolo, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Critic metrics plot saved: {save_path}")
    return save_path


if __name__ == "__main__":
    logger.info("Plots module loaded (for testing, create fake callback)")