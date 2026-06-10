from Ambiente.PPO_Ambiente import  DOF_BOUNDS_ALL, ACTIVE_DOF_INDICES
from Config.Set_input_param import n_dof_totali,DOF_NAMES_ALL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Smith_Chart.Reaction_total_to_total.Smith_chart_reaction_total_to_total import SmithDiagram_Reaction_total_to_total
from Smith_Chart.Action_total_to_static.Smith_chart_action_uscita_assiale import SmithDiagram_Action_Assiale
from Smith_Chart.Action_total_to_total.Smith_chart_action_total_to_total import SmithDiagram_Action_total_to_total
from pathlib import Path
import math

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
        f"PPO Blade Opt. — DOF attivi: Tutti\n"
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


def _plot_dof_evolution(cb: 'BladeCallback', lr, n_step, start_dof=None, base_save_path="plot_dof_evolution"):
    if not cb.episode_best_dofs:
        print("  Nessun dato dei DOF da plottare.")
        return []

    dof_data = np.array(cb.episode_best_dofs)
    n_episodes = dof_data.shape[0]
    ep_axis = np.arange(n_episodes)
    best_ep_idx = int(np.argmin(cb.episode_csi))

    np.random.seed(42)
    colori_per_episodio = np.random.uniform(0.1, 0.75, size=(n_episodes, 3))

    # --- LOGICA DI DIVISIONE (8 DOF per immagine) ---
    dofs_per_fig = 8
    num_figs = math.ceil(n_dof_totali / dofs_per_fig)
    saved_paths = []

    for f in range(num_figs):
        start_idx = f * dofs_per_fig
        end_idx = min(start_idx + dofs_per_fig, n_dof_totali)
        current_dofs = range(start_idx, end_idx)

        # Griglia 4x2
        fig, axes = plt.subplots(4, 2, figsize=(16, 10))
        axes_flat = axes.flatten()

        for idx_in_grid, i in enumerate(current_dofs):
            ax = axes_flat[idx_in_grid]
            y_vals = dof_data[:, i]

            # Scatter plot (Puntini) - Rimosse le Barre come richiesto
            if i in ACTIVE_DOF_INDICES:
                ax.scatter(ep_axis, y_vals, c=colori_per_episodio, s=15, alpha=0.8)
                ax.plot([], [], 'o', color='mediumseagreen', markersize=5, label="Miglior DOF/Ep")
            else:
                ax.scatter(ep_axis, y_vals, color='gray', s=15, alpha=0.5, label="Fisso")

            # Marker di Partenza e Migliore Assoluto
            if start_dof is not None:
                ax.plot(0, start_dof[i], marker='o', color='cyan', markersize=10, label="Partenza", zorder=5)
            if cb.best_dof is not None:
                ax.plot(best_ep_idx, cb.best_dof[i], marker='X', color='red', markersize=12, label="Miglior", zorder=5)

            # Bounds
            dof_min, dof_max = DOF_BOUNDS_ALL[i]
            ax.axhline(dof_min, color='red', ls='--', lw=1, alpha=0.5)
            ax.axhline(dof_max, color='red', ls='--', lw=1, alpha=0.5)

            ax.set_title(f"{DOF_NAMES_ALL[i]}", fontsize=12)
            ax.grid(alpha=0.3)
            if i in ACTIVE_DOF_INDICES: ax.legend(fontsize=8, loc='best')

        # Nascondi assi vuoti se l'ultimo gruppo è < 8
        for j in range(len(current_dofs), 8):
            axes_flat[j].set_visible(False)

        path = f"{base_save_path}_{f + 1}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        saved_paths.append(path)

    return saved_paths

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
        f"Metriche interne Actor — DOF: Tutti\n"
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
        f"Metriche interne Actor — DOF: Tutti\n"
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


def plot_smith(phi_ottimale, psi_ottimale, defl_min=40, defl_max=140, step=1, validate=True):
    """
    Plotta i diagrammi di Smith (Reaction T-T, Action Assiale, Action T-T).

    Carica i dati CSV, interpola le curve per BLOCCHI con flat extrapolation,
    stima la deflessione dal punto (phi, psi) e evidenzia la curva.
    """

    BASE_DIR = Path(__file__).resolve().parents[1]

    # ========== REACTION TOTAL-TO-TOTAL ==========
    smith_reaction_total_to_total = SmithDiagram_Reaction_total_to_total(
        BASE_DIR / "Smith_Chart" / "Reaction_total_to_total" / "csv"
    )

    # Interpola per blocchi con flat extrapolation
    smith_reaction_total_to_total.add_interpolated_deflection_curves_by_blocks(
        step=step,
        n_points=600,
        overwrite=False
    )

    # Validazione (opzionale)
    if validate:
        print("🔍 Validazione interpolazione Reaction T-T...")
        report_reaction = smith_reaction_total_to_total.validate_interpolation(
            defl_min=defl_min, defl_max=defl_max, tolerance_psi=0.04
        )
        if report_reaction['failed'] / report_reaction['total_tests'] > 0.2:
            print("⚠️  ATTENZIONE: >20% dei test falliti!")

    # Stima deflessione
    d_hat = smith_reaction_total_to_total.estimate_deflection_nearest_integer(
        phi_ottimale, psi_ottimale, defl_min=defl_min, defl_max=defl_max
    )



    # Plot
    if d_hat is None:
        print("[Smith Reaction T-T] Punto fuori dominio.")
        smith_reaction_total_to_total.plot(
            target_point=(phi_ottimale, psi_ottimale),
            highlight_deflection=None,
            save_path="smith_diagram_reaction_total_to_total.png"
        )
    else:
        smith_reaction_total_to_total.plot(
            target_point=(phi_ottimale, psi_ottimale),
            highlight_deflection=int(d_hat),
            save_path="smith_diagram_reaction_total_to_total.png"
        )

    # ========== ACTION ASSIALE ==========
    smith_action_assiale = SmithDiagram_Action_Assiale(
        BASE_DIR / "Smith_Chart" / "Action_total_to_static" / "csv"
    )

    smith_action_assiale.add_interpolated_deflection_curves_by_blocks(
        step=step,
        n_points=600,
        overwrite=False
    )

    if validate:
        print("🔍 Validazione interpolazione Action Assiale...")
        report_assiale = smith_action_assiale.validate_interpolation(
            defl_min=defl_min, defl_max=defl_max, tolerance_psi=0.04
        )

    d_hat2 = smith_action_assiale.estimate_deflection_nearest_integer(
        phi_ottimale, psi_ottimale, defl_min=defl_min, defl_max=defl_max
    )

    if d_hat2 is None:
        print("[Smith Action Assiale] Punto fuori dominio.")
        smith_action_assiale.plot(
            target_point=(phi_ottimale, psi_ottimale),
            highlight_deflection=None,
            save_path="smith_diagram_action_assiale.png"
        )
    else:
        smith_action_assiale.plot(
            target_point=(phi_ottimale, psi_ottimale),
            highlight_deflection=int(d_hat2),
            save_path="smith_diagram_action_assiale.png"
        )

    # ========== ACTION TOTAL-TO-TOTAL ==========
    smith_action_total_to_total = SmithDiagram_Action_total_to_total(
        BASE_DIR / "Smith_Chart" / "Action_total_to_total" / "csv"
    )

    smith_action_total_to_total.add_interpolated_deflection_curves_by_blocks(
        step=step,
        n_points=600,
        overwrite=False
    )

    if validate:
        print("🔍 Validazione interpolazione Action T-T...")
        report_total = smith_action_total_to_total.validate_interpolation(
            defl_min=defl_min, defl_max=defl_max, tolerance_psi=0.04
        )

    d_hat3 = smith_action_total_to_total.estimate_deflection_nearest_integer(
        phi_ottimale, psi_ottimale, defl_min=defl_min, defl_max=defl_max
    )

    if d_hat3 is None:
        print("[Smith Action T-T] Punto fuori dominio.")
        smith_action_total_to_total.plot(
            target_point=(phi_ottimale, psi_ottimale),
            highlight_deflection=None,
            save_path="smith_diagram_action_total_to_total.png"
        )
    else:
        smith_action_total_to_total.plot(
            target_point=(phi_ottimale, psi_ottimale),
            highlight_deflection=int(d_hat3),
            save_path="smith_diagram_action_total_to_total.png"
        )

def plot_smith_zoom_reaction(phi_ottimale, psi_ottimale, orig_phi, orig_psi, defl_min=40, defl_max=140,
                             step_deflection=1, step_efficiency=0.2, eta_range=(88, 96), zoom_margin=0.08):
    """
    Plotta SOLO il diagramma di Smith (Reaction T-T) ed esegue uno zoom attorno
    ai due punti passati in ingresso. Vengono interpolate sia le curve di deflessione
    che (nuovo!) le curve di isorendimento, in modo da avere una griglia densa su cui
    confrontare visivamente i due punti.
    """
    BASE_DIR = Path(__file__).resolve().parents[1]

    # 1. Caricamento e inizializzazione base
    smith_zoom = SmithDiagram_Reaction_total_to_total(
        BASE_DIR / "Smith_Chart" / "Reaction_total_to_total" / "csv"
    )

    # 2. Interpola le deflessioni (come fai già)
    smith_zoom.add_interpolated_deflection_curves_by_blocks(
        step=step_deflection,
        n_points=600,
        overwrite=False
    )

    # 3. Interpola le isorendimento (efficienza) per costruire un reticolo denso
    # Utilizza il metodo add_interpolated_efficiency_curves_by_blocks che già possiedi
    smith_zoom.add_interpolated_efficiency_curves_by_blocks(
        step=step_efficiency,
        eta_range=eta_range
    )

    # Stime della deflessione (per sapere se evidenziarne qualcuna, opzionale, non vitale)
    d_hat_ott = smith_zoom.estimate_deflection_nearest_integer(phi_ottimale, psi_ottimale, defl_min=defl_min,
                                                               defl_max=defl_max)

    # 4. Chiama la funzione plot
    # Forziamo show_interpolated_efficiency = True
    fig, ax = smith_zoom.plot(
        figsize=(12, 10),
        target_point=(phi_ottimale, psi_ottimale),
        orig_point=(orig_phi, orig_psi),
        highlight_deflection=int(d_hat_ott) if d_hat_ott is not None else None,
        show_interpolated_efficiency=True,  # <-- Questo farà apparire le nuove curve grigie interpolate
        show_interpolated_deflection=True  # <-- Plotta anche le deflessioni "mancanti" per fare reticolo
    )

    # 5. Effettua uno ZOOM (cambia i limiti deglis assi per fare il crop solo sull'area interessata)
    min_phi = min(phi_ottimale, orig_phi)
    max_phi = max(phi_ottimale, orig_phi)

    min_psi = min(psi_ottimale, orig_psi)
    max_psi = max(psi_ottimale, orig_psi)

    # Aggiungi un piccolo margine per non far collidere i punti con i bordi del grafico
    ax.set_xlim(min_phi - zoom_margin, max_phi + zoom_margin)
    ax.set_ylim(min_psi - zoom_margin, max_psi + zoom_margin)

    ax.set_title("Zoom In - Confronto Punti Operativi\n(Interpolazione Deflessioni e Isorendimento)", fontsize=14,
                 fontweight="bold")

    save_path = "smith_diagram_reaction_ZOOM.png"
    # Facoltativo ma utile: forza prima un tight_layout di matplotlib puro
    fig.tight_layout()
    # SALVA SENZA BBOX_INCHES TIGHT per non innescare il calcolo infinito dello spazio bianco
    fig.savefig(save_path, dpi=120)
    plt.close(fig)

    return save_path
