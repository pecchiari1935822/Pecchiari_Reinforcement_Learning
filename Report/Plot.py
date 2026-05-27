from Ambiente.Ambiente import  DOF_BOUNDS_ALL, ACTIVE_DOF_INDICES
from Config.Set_input_param import n_dof_totali,DOF_NAMES_ALL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Smith_Chart.Reaction_total_to_total.Smith_chart_reaction_total_to_total import smith_reaction_total_to_total
from Smith_Chart.Action_total_to_static.Smith_chart_action_uscita_assiale import smith_action_assiale
from Smith_Chart.Action_total_to_total.Smith_chart_action_total_to_total import smith_action_total_to_total

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Agente.PPO import BladeCallback


def _plot_results(cb: 'BladeCallback', lr, n_step, save_path="plot_results.png", training_time=None):
    from Agente.PPO import TOTAL_TIMESTEPS
    if not cb.episode_csi:
        print("  Nessun dato da plottare.")
        return

    time_str = ""
    if training_time is not None:
        mins = int(training_time // 60)
        secs = int(training_time % 60)
        time_str = f"  |  Durata: {mins}m {secs}s"

    csi_arr = np.array(cb.episode_csi)
    score_arr = np.array(cb.episode_scores)
    n_ep = len(csi_arr)
    ep_axis = np.arange(n_ep)
    W = min(40, max(n_ep // 5, 2))

    def moving_avg(arr, w):
        if len(arr) < w:
            return ep_axis, arr
        ma = np.convolve(arr, np.ones(w) / w, mode='valid')
        return np.arange(w - 1, len(arr)), ma

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"PPO Blade Opt. — DOF attivi: {[DOF_NAMES_ALL[i] for i in ACTIVE_DOF_INDICES]}\n"
        f"Total steps={TOTAL_TIMESTEPS:,}  "
        f"n_steps={n_step}" 
        f" Learning_Rate={lr}"
        f"{time_str}",
        fontsize=16
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1: CSI per episodio ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(ep_axis, csi_arr, s=8, alpha=0.4, color='orange', label='CSI')
    x_ma, ma = moving_avg(csi_arr, W)
    ax1.plot(x_ma, ma, color='steelblue', lw=1.5, label=f'<CSI> ({W}ep)')
    ax1.plot(ep_axis, np.minimum.accumulate(csi_arr),
             color='green', lw=2, label='Min CSI')
    ax1.axhline(cb.best_csi, color='green', ls='--', lw=1)
    ax1.set_xlabel('Episodio', fontsize=12)
    ax1.set_ylabel('CSI', fontsize=12)
    ax1.set_title('CSI per episodio', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.tick_params(labelsize=11)

    # ── 2: Score per episodio ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(ep_axis, score_arr, s=8, alpha=0.4, color='orange', label='Score')
    x_ma2, ma2 = moving_avg(score_arr, W)
    ax2.plot(x_ma2, ma2, color='steelblue', lw=1.5, label=f'<Score> ({W}ep)')
    ax2.plot(ep_axis, np.maximum.accumulate(score_arr),
             color='green', lw=2, label='Max Score')
    ax2.set_xlabel('Episodio', fontsize=12)
    ax2.set_ylabel('Score (reward cumulativa)', fontsize=12)
    ax2.set_title('Score per episodio', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=11)

    # ── 3: CSI minimo progressivo ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ep_axis, np.minimum.accumulate(csi_arr), color='green', lw=2)
    ax3.annotate(f'Min CSI = {cb.best_csi:.5f}',
                 xy=(0.05, 0.05), xycoords='axes fraction', fontsize=10,
                 color='green',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.8))
    ax3.set_xlabel('Episodio', fontsize=12)
    ax3.set_ylabel('Miglior CSI trovato', fontsize=12)
    ax3.set_title('CSI minimo progressivo', fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.tick_params(labelsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    return save_path


def _plot_dof_evolution(cb: 'BladeCallback', lr, n_step, start_dof = None, save_path="plot_dof_evolution.png"):
    """
    Crea un grafico per ogni DOF mostrando l'andamento del miglior valore trovato
    in ogni episodio per tutta la durata dell'addestramento, colorando i punti in
    base al numero dell'episodio.
    """


    if not cb.episode_best_dofs:
        print("  Nessun dato dei DOF da plottare.")
        return

    # Trasforma la lista di array in una matrice (n_episodi, 7_dof)
    dof_data = np.array(cb.episode_best_dofs)
    n_episodes = dof_data.shape[0]
    ep_axis = np.arange(n_episodes)

    best_ep_idx = int(np.argmin(cb.episode_csi))

    np.random.seed(42)  # Fissa il seed così i colori non cambiano a ogni run
    colori_per_episodio = np.random.uniform(0.1,0.75,size=(n_episodes, 3))

    # Creiamo una griglia 4x2 standard (senza sharex)
    fig, axes = plt.subplots(4, 2, figsize=(16, 8.5))

    axes_flat = axes.flatten()

    for i in range(n_dof_totali):
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

        if start_dof is not None:
            start_val = start_dof[i]
            # Mettiamo il cerchio all'episodio 0 (inizio training)
            ax.plot(0, start_val, marker='o', color='cyan', markeredgecolor='black',
                    markersize=10, linestyle='None', zorder=5, label="Partenza")

        if cb.best_dof is not None:
            best_val = cb.best_dof[i]
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
    axes_flat[n_dof_totali].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    return save_path


def _plot_dof_evolution_barre(cb: 'BladeCallback', lr, n_step,start_dof = None, save_path="plot_dof_evolution_barre.png"):
    """
    Crea un grafico per ogni DOF mostrando l'andamento del miglior valore trovato
    in ogni episodio. Usa barre verticali colorate casualmente per ogni episodio.
    """
    if not cb.episode_best_dofs:
        print("  Nessun dato dei DOF da plottare.")
        return

    # Trasforma la lista di array in una matrice (n_episodi, 7_dof)
    dof_data = np.array(cb.episode_best_dofs)
    n_episodes = dof_data.shape[0]
    ep_axis = np.arange(n_episodes)

    best_ep_idx = int(np.argmin(cb.episode_csi))

    # Generiamo un colore RGB casuale per OGNI episodio
    np.random.seed(42)
    colori_per_episodio = np.random.uniform(0.1,0.75,size=(n_episodes, 3))

    # Creiamo una griglia 4x2 standard
    fig, axes = plt.subplots(4, 2, figsize=(16, 8.5))

    axes_flat = axes.flatten()

    for i in range(n_dof_totali):
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

        if start_dof is not None:
            start_val = start_dof[i]
            # Mettiamo il cerchio all'episodio 0 (inizio training)
            ax.plot(0, start_val, marker='o', color='cyan', markeredgecolor='black',
                    markersize=10, linestyle='None', zorder=5, label="Partenza")

        if cb.best_dof is not None:
            best_val = cb.best_dof[i]
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
    axes_flat[n_dof_totali].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    return save_path


def _plot_training_metrics_actor(cb: 'BladeCallback', lr, n_step, save_path="plot_metrics_actor.png"):
    from Agente.PPO import TOTAL_TIMESTEPS
    if not cb.metrics_episodes:
        print("  Nessuna metrica PPO disponibile (training troppo corto).")
        return

    ts = np.array(cb.metrics_episodes)

    # Configurazione: (chiave, titolo, colore, linea riferimento, label ref)
    config = [("ep_rew_mean", "Episode Reward Mean", "navy", None, None),
            ("policy_gradient_loss", "Policy Gradient Loss (L_CLIP Actor)", "darkslategray", 0.0, "zero"),
            ("entropy_loss", "Entropy H[π] (esplorazione Actor)", "darkorange", None, None),
            ("std", "Std Policy (deviazione standard azioni Actor)", "mediumpurple", None, None),
            ("approx_kl", "Approx KL Divergence (cambio policy per update Actor)", "crimson", 0.02, "soglia 0.02"),
            ("clip_fraction", "Clip Fraction ( % azioni clippate)", "teal", 0.1, "soglia 0.1")
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    fig.suptitle(
        f"Metriche interne Actor — DOF: {[DOF_NAMES_ALL[i] for i in ACTIVE_DOF_INDICES]}\n"
        f"Total steps={TOTAL_TIMESTEPS:,}  n_steps={n_step}  Learning_Rate={lr} ",
        fontsize=16
    )
    axes_flat = axes.flatten()

    for idx, (chiave, titolo, colore, ref_val, ref_label) in enumerate(config):
        ax = axes_flat[idx]
        valori = cb.metrics.get(chiave, [])

        if not valori:
            ax.text(0.5, 0.5, "Nessun dato", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(titolo, fontsize=12)
            continue

        if chiave == "entropy_loss":
            vals = -np.array(valori)
        else:
            vals = np.array(valori)
        ax.plot(ts[:len(vals)], vals, color=colore, lw=1.5, alpha=0.85)

        # Media mobile per leggibilità
        if len(vals) >= 5:
            w = max(3, len(vals) // 10)
            ma = np.convolve(vals, np.ones(w) / w, mode="valid")
            ax.plot(ts[w - 1:len(vals)], ma,
                    color=colore, lw=2.5, alpha=0.5,
                    linestyle="--", label=f"media {w} update")

        # Linea di riferimento (se presente)
        if ref_val is not None:
            ax.axhline(ref_val, color="green", ls=":", lw=1.2,
                       alpha=0.7, label=ref_label)

        # Valore finale annotato
        ax.annotate(
            f"finale: {vals[-1]:.4f}",
            xy=(ts[len(vals) - 1], vals[-1]),
            xytext=(-60, 8), textcoords="offset points",
            fontsize=10, color=colore,
            arrowprops=dict(arrowstyle="->", color=colore, lw=0.8)
        )

        ax.set_xlabel("Episodio", fontsize=10)
        ax.set_title(titolo, fontsize=11, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=11)

    # Disabilita gli ultimi 2 assi (griglia 3x3, metriche sono 7)
    for idx in range(len(config), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    return save_path



def _plot_training_metrics_critic(cb: 'BladeCallback',lr , n_step, save_path="plot_metrics_critic.png"):
    from Agente.PPO import TOTAL_TIMESTEPS
    if not cb.metrics_episodes:
        print("  Nessuna metrica PPO disponibile (training troppo corto).")
        return

    ts = np.array(cb.metrics_episodes)

    # Configurazione: (chiave, titolo, colore, linea riferimento, label ref)
    config = [("explained_variance", "Explained Variance (Critic)", "steelblue", 1.0, "ottimo=1.0"),
            ("value_loss", "Value Loss (errore Critic sui Returns)", "sienna", None, None),
              ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Metriche interne Actor — DOF: {[DOF_NAMES_ALL[i] for i in ACTIVE_DOF_INDICES]}\n"
        f"Total steps={TOTAL_TIMESTEPS:,}  n_steps={n_step}  Learning_Rate={lr} ",
        fontsize=16
    )
    axes_flat = axes.flatten()

    for idx, (chiave, titolo, colore, ref_val, ref_label) in enumerate(config):
        ax = axes_flat[idx]
        valori = cb.metrics.get(chiave, [])

        if not valori:
            ax.text(0.5, 0.5, "Nessun dato", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(titolo, fontsize=15)
            continue

        if chiave == "entropy_loss":
            vals = -np.array(valori)
        else:
            vals = np.array(valori)
        ax.plot(ts[:len(vals)], vals, color=colore, lw=1.5, alpha=0.85)

        # Media mobile per leggibilità
        if len(vals) >= 5:
            w = max(3, len(vals) // 10)
            ma = np.convolve(vals, np.ones(w) / w, mode="valid")
            ax.plot(ts[w - 1:len(vals)], ma,
                    color=colore, lw=2.5, alpha=0.5,
                    linestyle="--", label=f"media {w} update")

        # Linea di riferimento (se presente)
        if ref_val is not None:
            ax.axhline(ref_val, color="green", ls=":", lw=1.2,
                       alpha=0.7, label=ref_label)

        # Valore finale annotato
        ax.annotate(
            f"finale: {vals[-1]:.4f}",
            xy=(ts[len(vals) - 1], vals[-1]),
            xytext=(-60, 8), textcoords="offset points",
            fontsize=10, color=colore,
            arrowprops=dict(arrowstyle="->", color=colore, lw=0.8)
        )

        ax.set_xlabel("Episodio", fontsize=10)
        ax.set_title(titolo, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=11)

    # Disabilita gli ultimi 2 assi (griglia 3x3, metriche sono 7)
    for idx in range(len(config), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

    return save_path

def plot_smith(phi_ottimale, psi_ottimale, deflessione_flusso):
    smith_action_assiale.plot(target_point=(phi_ottimale, psi_ottimale), highlight_deflection=deflessione_flusso,
                              save_path="smith_diagram_action_assiale.png")
    smith_action_total_to_total.plot(target_point=(phi_ottimale, psi_ottimale), highlight_deflection=deflessione_flusso,
                                     save_path="smith_diagram_action_total_to_total.png")
    smith_reaction_total_to_total.plot(target_point=(phi_ottimale, psi_ottimale), highlight_deflection=deflessione_flusso,
                                       save_path="smith_diagram_reaction_total_to_total.png")