import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path


class SmithDiagram_Reaction_total_to_total:
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

        self.deflection_phi_domain = {}
        self._load_csv_files()

    def _parse_deflection_from_filename(self, csv_file: Path) -> int:
        """
        Parse robusto dell'angolo di deflessione dal nome file.
        Supporta: defl_60.csv, defl_60.0.csv, defl_60,0.csv, defl_60°.csv
        """
        deflection_str = csv_file.stem.split('_', 1)[1]  # tutto dopo "defl_"
        deflection_str = deflection_str.replace("°", "").replace(",", ".")
        return int(round(float(deflection_str)))

    def _parse_efficiency_from_filename(self, csv_file: Path) -> float:
        """
        Parse robusto efficienza dal nome file.
        Supporta: eta_0.92.csv, eta_92.csv (se vuoi usarlo come 0.92, vedi nota sotto)
        """
        efficiency_str = csv_file.stem.split('_', 1)[1].replace(",", ".")
        return float(efficiency_str)

    def _load_csv_files(self):
        """
        Carica tutti i file CSV dalla cartella.
        Assume che i nomi dei file seguano un pattern:
        - defl_XX.csv (per le curve di deflessione, XX = angolo)
        - eta_XX.csv (per le isolinee di efficienza, XX = valore efficienza)
        """

        # Carica curve di deflessione
        for csv_file in self.csv_folder.glob("defl_*.csv"):
            try:
                # Estrai l'angolo dal nome del file (robusto)
                deflection_angle = self._parse_deflection_from_filename(csv_file)

                # Leggi il CSV tentando separatore ; e virgola come decimale
                df = pd.read_csv(csv_file, sep=';')
                if len(df.columns) < 2:
                    df = pd.read_csv(csv_file, sep=',', decimal='.')

                # Se i dati sono stati letti come stringhe con virgola decimale, convertiamoli
                for col in df.columns[:2]:
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(',', '.').astype(float)

                # Assumendo che le colonne siano phi e psi
                phi = df.iloc[:, 0].values  # prima colonna
                psi = df.iloc[:, 1].values  # seconda colonna

                # Salva dominio di phi
                self.deflection_phi_domain[deflection_angle] = (np.min(phi), np.max(phi))

                self.deflection_curves[deflection_angle] = np.column_stack([phi, psi])

                # Crea interpolatore (cubic ok, ma se fa oscillazioni prova linear)
                self.interpolators[deflection_angle] = interp1d(
                    phi, psi, kind='cubic', bounds_error=False,
                    fill_value='extrapolate'
                )

            except Exception as e:
                print(f"✗ Errore caricamento {csv_file}: {e}")

        # Carica curve di efficienza
        for csv_file in self.csv_folder.glob("eta_*.csv"):
            try:
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

            except Exception as e:
                print(f"✗ Errore caricamento {csv_file}: {e}")

    def _sort_curve_points(self, points, max_gap=0.5):
        """
        Ordina i punti della curva usando l'algoritmo del 'vicino più prossimo'.
        Risolve il problema dello zig-zag per le curve a forma di 'C'.
        Se la curva è frammentata (es. esce dal grafico e rientra), inserisce
        un NaN per staccare la linea e non unire i pezzi distanti.
        """
        if len(points) <= 1:
            return points

        sorted_points = []
        remaining = list(points)

        # Partiamo dal punto più in basso (Y minima)
        start_idx = np.argmin([p[1] for p in remaining])
        current_point = remaining.pop(start_idx)
        sorted_points.append(current_point)

        while remaining:
            distances = [np.linalg.norm(current_point - p) for p in remaining]
            nearest_idx = np.argmin(distances)
            min_dist = distances[nearest_idx]

            if min_dist > max_gap:
                sorted_points.append(np.array([np.nan, np.nan]))  # Spezza la linea
                start_idx = np.argmin([p[1] for p in remaining])
                current_point = remaining.pop(start_idx)
            else:
                current_point = remaining.pop(nearest_idx)

            sorted_points.append(current_point)

        return np.array(sorted_points)

    def add_interpolated_deflection_curves(self, defl_a=60, defl_b=80, step=1, n_points=250, overwrite=False):
        """
        Genera curve di deflessione interpolate tra defl_a e defl_b (esclusi estremi),
        e le inserisce in deflection_curves/interpolators così vengono plottate.
        """
        if defl_a not in self.interpolators or defl_b not in self.interpolators:
            raise ValueError(
                f"Servono le curve {defl_a}° e {defl_b}° (defl_{defl_a}.csv e defl_{defl_b}.csv). "
                f"Curve caricate: {sorted(self.deflection_curves.keys())}"
            )

        a_min, a_max = self.deflection_phi_domain[defl_a]
        b_min, b_max = self.deflection_phi_domain[defl_b]

        # SOLO range comune (evita extrapolate inventato)
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

    def estimate_deflection_nearest_integer(self, phi, psi_target, defl_min=60, defl_max=80):
        """
        Stima la deflessione come INTERO più vicino (tra defl_min..defl_max),
        scegliendo la curva che minimizza |psi_curve(phi) - psi_target|.
        Ritorna None se phi fuori dominio per tutte le curve.
        """
        candidates = [d for d in sorted(self.interpolators.keys()) if defl_min <= d <= defl_max]
        if not candidates:
            raise ValueError(f"Nessuna curva/interpolatore tra {defl_min} e {defl_max}.")

        best = None  # (err, defl, psi_curve)

        for d in candidates:
            # evita extrapolate fuori range "reale" (se il dominio è noto)
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

    def plot(self, figsize=(12, 9), save_path=None, target_point=None, highlight_deflection=None):
        """
        Plotta il diagramma di Smith completo.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # ===== Plotta curve di deflessione =====
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

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

            ax.plot(
                points_sorted[:, 0], points_sorted[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
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
                        fontsize=15, verticalalignment='center')

        if target_point is not None:
            phi_p, psi_p = target_point
            ax.plot(phi_p, psi_p, marker='*', color='blue', markersize=14, markeredgecolor='black', zorder=5)
            ax.text(phi_p + 0.03, psi_p + 0.03, f'OP\n({phi_p:.2f}, {psi_p:.2f})',
                    color='blue', fontsize=15, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9), zorder=5)

        # ===== Formattazione =====
        ax.set_xlabel('Flow coefficient, φ', fontsize=15, fontweight='bold')
        ax.set_ylabel('Stage loading coefficient, ψ', fontsize=15, fontweight='bold')
        ax.set_title('Smith Diagram Reaction - total to total', fontsize=15, fontweight='bold')
        ax.legend(loc='upper left', title='Deflection angle', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=15)

        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 3.0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Grafico salvato: {save_path}")

        return fig, ax

    def find_intersection(self, efficiency_target, deflection_target):
        """
        Trova il punto di intersezione (phi, psi) tra una curva di deflessione
        e una curva di efficienza.
        """
        from shapely.geometry import LineString

        if deflection_target not in self.deflection_curves:
            raise ValueError(f"Curva di deflessione a {deflection_target}° non trovata.")
        if efficiency_target not in self.efficiency_curves:
            raise ValueError(f"Curva di efficienza a {efficiency_target} non trovata.")

        defl_pts = self.deflection_curves[deflection_target]
        defl_pts = defl_pts[np.argsort(defl_pts[:, 0])]

        eff_pts = self.efficiency_curves[efficiency_target]
        eff_pts = self._sort_curve_points(eff_pts)
        eff_pts = eff_pts[~np.isnan(eff_pts).any(axis=1)]

        line_defl = LineString(defl_pts)
        line_eff = LineString(eff_pts)

        intersection = line_defl.intersection(line_eff)

        if intersection.is_empty:
            return None

        if intersection.geom_type == 'MultiPoint':
            pt = intersection.geoms[0]
            return pt.x, pt.y
        elif intersection.geom_type == 'Point':
            return intersection.x, intersection.y
        else:
            return None

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
smith_reaction_total_to_total = SmithDiagram_Reaction_total_to_total(csv_folder)

# 3. Stampa riepilogo
smith_reaction_total_to_total.print_summary()

# 3b. Aggiungi curve di deflessione interpolate 60..80 (61..79)
smith_reaction_total_to_total.add_interpolated_deflection_curves(defl_a=60, defl_b=80, step=1)

# 4. Carica il dataset
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "Data" / "database.dat"
df = pd.read_csv(DATASET_PATH)

# 5. Plotta il diagramma
fig, ax = smith_reaction_total_to_total.plot(
    figsize=(12, 9),
    save_path="smith_diagram_with_dataset.png"
)

# 6. Aggiungi i punti del dataset come scatter plot
phi = df['OF_phi_OP_01'].values
psi = df['OF_psi_OP_01'].values
csi = df['OF_CSI_OP_01'].values

nome_colonna_alfa = 'OF_alfa_ex_OP_01'
nome_beta1 = 'DOF_BETA1_GEOM_'
nome_beta2 = 'DOF_BETA2_GEOM_'

alfa_ex = df[nome_colonna_alfa].values
beta1 = df[nome_beta1].values
beta2 = df[nome_beta2].values

scatter = ax.scatter(
    phi, psi,
    c=csi, cmap='RdYlGn_r',
    s=50, alpha=0.6,
    edgecolors='black', linewidth=0.5
)

cbar = plt.colorbar(scatter, ax=ax, label='CSI')

plt.tight_layout()
plt.show()

# 7. Stima e stampa la deflessione per ogni profilo del dataset (intero più vicino)
print("\nDeflessione stimata (intero più vicino) per ogni profilo del dataset [range 60–80]:")
print("-" * 80)
for i in range(len(df)):
    phi_i = float(phi[i])
    psi_i = float(psi[i])

    # 1) Deflessione secondo Smith (intero più vicino)
    d_smith = smith_reaction_total_to_total.estimate_deflection_nearest_integer(
        phi_i, psi_i, defl_min=60, defl_max=80
    )

    # 2) Deflessioni "reali" del profilo
    a_ex = float(alfa_ex[i])
    b1 = float(beta1[i])
    b2 = float(beta2[i])

    defl_alfa = 10.0 - a_ex
    defl_beta = b1 - b2

    d_smith_str = f"{d_smith:3d}°" if d_smith is not None else "N/D"

    print(
        f"[{i:5d}] "        
        f"Deflessione Smith~{d_smith_str:>4} | "
        f"Deflessione alfa (flusso)={defl_alfa:8.3f} | "
        f"Deflessione beta (pala)={defl_beta:8.3f}"
    )

# 8. (Tuo test intersezione) -- NOTA: probabilmente vuoi 0.92, non 92, se i file sono eta_0.92.csv
intersezione = smith_reaction_total_to_total.find_intersection(efficiency_target=92, deflection_target=60)

if intersezione:
    phi_int, psi_int = intersezione
    print(f"Punto di intersezione trovato tra Efficienza=0.92 e Deflessione=60°:")
    print(f"Phi (φ) = {phi_int:.4f}")
    print(f"Psi (ψ) = {psi_int:.4f}")
else:
    print("Nessun punto di intersezione trovato. Le curve non si incontrano o l'efficienza non è presente tra quelle caricate.")