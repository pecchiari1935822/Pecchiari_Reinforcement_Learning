import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

from Smith_Chart.Action_total_to_total.Smith_chart_action_total_to_total import SmithDiagram_Action_total_to_total


class SmithDiagram_Action_Assiale:
    """
    Carica i file CSV da WebPlotDigitizer e costruisce il diagramma di Smith.
    """

    def __init__(self, csv_folder):
        """
        Args:
            csv_folder: percorso della cartella contenente i file CSV
        """
        self.csv_folder = Path(csv_folder)
        self.deflection_curves = {}
        self.efficiency_curves = {}
        self.interpolators = {}

        # AGGIUNTA: dominio phi per ogni deflessione (per evitare extrapolate "inventato")
        self.deflection_phi_domain = {}

        self._load_csv_files()

    # =========================
    # AGGIUNTA: parse robusti
    # =========================
    def _parse_deflection_from_filename(self, csv_file: Path) -> int:
        """
        Supporta: defl_60.csv, defl_60.0.csv, defl_60,0.csv, defl_60°.csv
        """
        deflection_str = csv_file.stem.split('_', 1)[1]
        deflection_str = deflection_str.replace("°", "").replace(",", ".")
        return int(round(float(deflection_str)))

    def _parse_efficiency_from_filename(self, csv_file: Path) -> float:
        """
        Supporta: eta_0.92.csv, eta_92.csv (92 rimane 92)
        """
        efficiency_str = csv_file.stem.split('_', 1)[1].replace(",", ".")
        return float(efficiency_str)

    def _load_csv_files(self):
        """
        Carica tutti i file CSV dalla cartella.
        """

        # Carica curve di deflessione
        for csv_file in self.csv_folder.glob("defl_*.csv"):
            try:
                # MODIFICA: parse robusto
                deflection_angle = self._parse_deflection_from_filename(csv_file)

                df = pd.read_csv(csv_file, sep=';')
                if len(df.columns) < 2:
                    df = pd.read_csv(csv_file, sep=',', decimal='.')

                for col in df.columns[:2]:
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(',', '.').astype(float)

                phi = df.iloc[:, 0].values
                psi = df.iloc[:, 1].values

                self.deflection_curves[deflection_angle] = np.column_stack([phi, psi])

                # AGGIUNTA: salva dominio phi
                self.deflection_phi_domain[deflection_angle] = (np.min(phi), np.max(phi))

                self.interpolators[deflection_angle] = interp1d(
                    phi, psi, kind='cubic', bounds_error=False,
                    fill_value='extrapolate'
                )

                print(f"✓ Caricato: deflessione {deflection_angle}°")

            except Exception as e:
                print(f"✗ Errore caricamento {csv_file}: {e}")

        # Carica curve di efficienza
        for csv_file in self.csv_folder.glob("eta_*.csv"):
            try:
                # MODIFICA: parse robusto
                efficiency_val = self._parse_efficiency_from_filename(csv_file)

                df = pd.read_csv(csv_file, sep=';')
                if len(df.columns) < 2:
                    df = pd.read_csv(csv_file, sep=',', decimal='.')

                for col in df.columns[:2]:
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(',', '.').astype(float)

                phi = df.iloc[:, 0].values
                psi = df.iloc[:, 1].values

                self.efficiency_curves[efficiency_val] = np.column_stack([phi, psi])

                print(f"✓ Caricato: efficienza {efficiency_val}")

            except Exception as e:
                print(f"✗ Errore caricamento {csv_file}: {e}")

    '''def get_eta_from_phi(self, phi, deflection):
        ...
    '''

    # (non elimino nulla: lascio questi import dentro la classe come nel tuo file)
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.interpolate import interp1d

    # ==========================================================
    # AGGIUNTA: interpolazione curve deflessione tra 60 e 80
    # ==========================================================
    def add_interpolated_deflection_curves(self, defl_a=60, defl_b=80, step=1, n_points=250, overwrite=False):
        if defl_a not in self.interpolators or defl_b not in self.interpolators:
            raise ValueError(
                f"Servono le curve {defl_a}° e {defl_b}° (defl_{defl_a}.csv e defl_{defl_b}.csv). "
                f"Curve caricate: {sorted(self.deflection_curves.keys())}"
            )

        a_min, a_max = self.deflection_phi_domain[defl_a]
        b_min, b_max = self.deflection_phi_domain[defl_b]

        phi_min = max(a_min, b_min)
        phi_max = min(a_max, b_max)
        if phi_max <= phi_min:
            raise ValueError(f"Nessun range di φ in comune tra {defl_a}° e {defl_b}°.")

        phi_grid = np.linspace(phi_min, phi_max, n_points)
        psi_a = self.interpolators[defl_a](phi_grid)
        psi_b = self.interpolators[defl_b](phi_grid)

        low = min(defl_a, defl_b)
        high = max(defl_a, defl_b)

        for defl_new in range(low + step, high, step):
            if (defl_new in self.deflection_curves) and not overwrite:
                continue

            t = (defl_new - defl_a) / (defl_b - defl_a)
            psi_new = (1 - t) * psi_a + t * psi_b

            curve_new = np.column_stack([phi_grid, psi_new])
            self.deflection_curves[defl_new] = curve_new
            self.deflection_phi_domain[defl_new] = (phi_min, phi_max)

            self.interpolators[defl_new] = interp1d(
                phi_grid, psi_new, kind='cubic', bounds_error=False, fill_value='extrapolate'
            )

    # ==========================================================
    # AGGIUNTA: stima deflessione (intero più vicino) da (phi,psi)
    # ==========================================================
    def estimate_deflection_nearest_integer(self, phi, psi_target, defl_min=60, defl_max=80):
        candidates = [d for d in sorted(self.interpolators.keys()) if defl_min <= d <= defl_max]
        if not candidates:
            raise ValueError(f"Nessuna curva/interpolatore tra {defl_min} e {defl_max}.")

        best = None  # (err, defl, psi_curve)

        for d in candidates:
            if d in self.deflection_phi_domain:
                dmin, dmax = self.deflection_phi_domain[d]
                if not (dmin <= phi <= dmax):
                    continue

            psi_curve = float(self.interpolators[d](phi))
            err = abs(psi_curve - psi_target)

            if best is None or err < best[0]:
                best = (err, d, psi_curve)

        if best is None:
            return None

        err, d, psi_curve = best
        return int(round(d))

    def _sort_curve_points(self, points, max_gap=0.5):
        """
        Ordina i punti della curva usando l'algoritmo del 'vicino più prossimo'.
        """
        if len(points) <= 1:
            return points

        sorted_points = []
        remaining = list(points)

        start_idx = np.argmin([p[1] for p in remaining])
        current_point = remaining.pop(start_idx)
        sorted_points.append(current_point)

        while remaining:
            distances = [np.linalg.norm(current_point - p) for p in remaining]
            nearest_idx = np.argmin(distances)
            min_dist = distances[nearest_idx]

            if min_dist > max_gap:
                sorted_points.append(np.array([np.nan, np.nan]))
                start_idx = np.argmin([p[1] for p in remaining])
                current_point = remaining.pop(start_idx)
            else:
                current_point = remaining.pop(nearest_idx)

            sorted_points.append(current_point)

        return np.array(sorted_points)

    def plot(self, figsize=(12, 9), save_path=None, target_point=None, highlight_deflection=None):
        """
        Plotta il diagramma di Smith completo.
        """
        fig, ax = plt.subplots(figsize=figsize)

        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

        # ===== Plotta curve di deflessione =====
        for deflection, points in sorted(self.deflection_curves.items()):
            points_sorted = points[np.argsort(points[:, 0])]

            if highlight_deflection is not None and deflection == highlight_deflection:
                color = 'red'
                linestyle = '-'
                linewidth = 2.5
                alpha = 1.0
                font_color = 'red'
            else:
                color = 'black'
                linestyle = '--'
                linewidth = 1.5
                alpha = 0.25 if highlight_deflection is not None else 0.8
                font_color = 'black'

            # MODIFICA: usa davvero stile+label (legenda funziona)
            ax.plot(
                points_sorted[:, 0], points_sorted[:, 1],
                color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha,
                label=f"{deflection}°"
            )

            if len(points_sorted) > 0:
                idx = len(points_sorted) // 4
                ax.text(points_sorted[idx, 0], points_sorted[idx, 1],
                        f'{deflection}°', fontsize=15, fontweight='bold',
                        color=font_color,
                        ha='center', va='center', bbox=bbox_props)

        # ===== Plotta curve di efficienza =====
        for efficiency, points in sorted(self.efficiency_curves.items()):
            points_sorted = self._sort_curve_points(points)

            ax.plot(points_sorted[:, 0], points_sorted[:, 1], 'k--', linewidth=3, alpha=0.6)

            valid_points = points_sorted[~np.isnan(points_sorted[:, 0])]
            if len(valid_points) > 0:
                ax.text(valid_points[-1, 0], valid_points[-1, 1], f'  η={efficiency:.2f}',
                        fontsize=15,
                        verticalalignment='center_baseline',
                        horizontalalignment='center')

        if target_point is not None:
            phi_p, psi_p = target_point
            ax.plot(phi_p, psi_p, marker='*', color='blue', markersize=14, markeredgecolor='black', zorder=5)
            ax.text(phi_p + 0.03, psi_p + 0.03, f'OP\n({phi_p:.2f}, {psi_p:.2f})',
                    color='blue', fontsize=15, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9), zorder=5)

        ax.set_xlabel('Flow coefficient, φ', fontsize=15, fontweight='bold')
        ax.set_ylabel('Stage loading coefficient, ψ', fontsize=15, fontweight='bold')
        ax.set_title('Smith Diagram Action - uscita assiale', fontsize=15, fontweight='bold')

        ax.legend(loc='upper left', title='Deflection angle', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=15)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafico salvato: {save_path}")

        return fig, ax

    def print_summary(self):
        """Stampa riepilogo delle curve caricate."""
        print("\n" + "=" * 60)
        print("SMITH DIAGRAM - RIEPILOGO")
        print("=" * 60)

        print(f"\n📊 Curve di Deflessione Caricate ({len(self.deflection_curves)}):")
        for deflection in sorted(self.deflection_curves.keys()):
            n_points = len(self.deflection_curves[deflection])
            print(f"   • {deflection}°: {n_points} punti")

        print(f"\n📈 Curve di Efficienza Caricate ({len(self.efficiency_curves)}):")
        for efficiency in sorted(self.efficiency_curves.keys()):
            n_points = len(self.efficiency_curves[efficiency])
            print(f"   • η={efficiency:.2f}: {n_points} punti")

        print("\n" + "=" * 60 + "\n")


# ===== ESEMPIO DI USO =====

# 1. Specifica il percorso della cartella con i CSV
csv_folder = Path(__file__).parent / "csv"

# 2. Carica il diagramma
smith_action_assiale = SmithDiagram_Action_Assiale(csv_folder)

# 3. Stampa riepilogo (se vuoi)
smith_action_assiale.print_summary()

# 3b. Interpola deflessioni tra 60 e 80
smith_action_assiale.add_interpolated_deflection_curves(defl_a=60, defl_b=80, step=1)

# 4. Carica dataset
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "Data" / "database.dat"
df = pd.read_csv(DATASET_PATH)

# 5. Plotta il diagramma
fig, ax = smith_action_assiale.plot(
    figsize=(12, 9),
    save_path="smith_diagram_action_assiale_with_dataset.png"
)

# 6. Scatter dataset
phi = df['OF_phi_OP_01'].values
psi = df['OF_psi_OP_01'].values
csi = df['OF_CSI_OP_01'].values

scatter = ax.scatter(
    phi, psi,
    c=csi, cmap='RdYlGn_r',
    s=50, alpha=0.6,
    edgecolors='black', linewidth=0.5
)
cbar = plt.colorbar(scatter, ax=ax, label='CSI')

plt.tight_layout()
plt.show()

# 7. Deflessione Smith + deflessioni reali (alfa e beta) stampate insieme
nome_colonna_alfa = 'OF_alfa_ex_OP_01'
nome_beta1 = 'DOF_BETA1_GEOM_'
nome_beta2 = 'DOF_BETA2_GEOM_'

alfa_ex = df[nome_colonna_alfa].values
beta1 = df[nome_beta1].values
beta2 = df[nome_beta2].values

print("\nDeflessioni per ogni profilo del dataset:")
print(" - Smith: stimata dal diagramma (phi, psi) nel range 60–80")
print(" - Reale alfa: 10 - alfa_ex")
print(" - Reale beta: beta1 - beta2")
print("-" * 110)

for i in range(len(df)):
    phi_i = float(phi[i])
    psi_i = float(psi[i])

    d_smith = smith_action_assiale.estimate_deflection_nearest_integer(
        phi_i, psi_i, defl_min=60, defl_max=80
    )

    a_ex = float(alfa_ex[i])
    b1 = float(beta1[i])
    b2 = float(beta2[i])

    defl_alfa = 10.0 - a_ex
    defl_beta = b1 - b2

    d_smith_str = f"{d_smith:3d}°" if d_smith is not None else "N/D"

    print(
        f"[{i:5d}] "       
        f"Smith~{d_smith_str:>4} | "
        f"defl_alfa={defl_alfa:8.3f} | "
        f"defl_beta={defl_beta:8.3f}"
    )

# 8. Le tue righe di test (non elimino nulla)
print("\n🧪 TEST FUNZIONI:")
print("-" * 60)

# smith_action_assiale.plot(target_point=(0.6243, 1.2198), highlight_deflection=100,
