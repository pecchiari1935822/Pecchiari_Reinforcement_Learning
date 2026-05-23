import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path


class SmithDiagram_Action_total_to_total:
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

        self._load_csv_files()

    def _load_csv_files(self):
        """
        Carica tutti i file CSV dalla cartella.
        Assume che i nomi dei file seguano un pattern:
        - deflection_XX.csv (per le curve di deflessione, XX = angolo)
        - efficiency_XX.csv (per le isolinee di efficienza, XX = valore efficienza)
        """

        # Carica curve di deflessione
        for csv_file in self.csv_folder.glob("defl_*.csv"):
            try:
                # Estrai l'angolo dal nome del file: deflection_80.csv → 80
                deflection_angle = int(csv_file.stem.split('_')[1])

                # Leggi il CSV tentando separatore ; e virgola come decimale
                df = pd.read_csv(csv_file, sep=';')
                if len(df.columns) < 2:
                    df = pd.read_csv(csv_file, sep=',', decimal='.')
                
                # Se i dati sono stati letti come stringhe con virgola decimale, convertiamoli
                for col in df.columns[:2]:
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(',', '.').astype(float)

                # Assumendo che le colonne siano x e y (o phi e eta)
                phi = df.iloc[:, 0].values  # prima colonna
                psi = df.iloc[:, 1].values  # seconda colonna

                self.deflection_curves[deflection_angle] = np.column_stack([phi, psi])

                # Crea interpolatore
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
                # Estrai il valore efficienza: efficiency_0.86.csv → 0.86
                efficiency_str = csv_file.stem.split('_')[1]
                efficiency_val = float(efficiency_str)

                # Leggi il CSV tentando separatore ; e virgola come decimale
                df = pd.read_csv(csv_file, sep=';')
                if len(df.columns) < 2:
                    df = pd.read_csv(csv_file, sep=',', decimal='.')
                
                # Se i dati sono stati letti come stringhe con virgola decimale, convertiamoli
                for col in df.columns[:2]:
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(',', '.').astype(float)

                phi = df.iloc[:, 0].values
                psi = df.iloc[:, 1].values

                self.efficiency_curves[efficiency_val] = np.column_stack([phi, psi])

                print(f"✓ Caricato: efficienza {efficiency_val}")

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
            # Calcola la distanza tra il punto attuale e tutti i rimanenti
            distances = [np.linalg.norm(current_point - p) for p in remaining]
            nearest_idx = np.argmin(distances)
            min_dist = distances[nearest_idx]

            # Se il punto successivo è troppo lontano (> max_gap), significa che
            # stiamo saltando a un segmento di curva scollegato.
            if min_dist > max_gap:
                sorted_points.append(np.array([np.nan, np.nan]))  # Spezza la linea
                # Ripartiamo dal Y minimo tra i punti rimanenti
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

        # ===== Plotta curve di deflessione =====
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

        for deflection, points in sorted(self.deflection_curves.items()):
            # Ordiniamo per X (le deflessioni non tornano indietro, quindi basta sort_values base)
            points_sorted = points[np.argsort(points[:, 0])]

            if highlight_deflection is not None and deflection == highlight_deflection:
                # Se è la curva scelta: rossa, spessa, continua
                color = 'red'
                linestyle = '-'
                linewidth = 2.5
                alpha = 1.0
                font_color = 'red'
            else:
                # Altrimenti normale (se stiamo evidenziando un'altra, sbiadiamo le restanti)
                color = 'black'
                linestyle = '--'
                linewidth = 1.5
                alpha = 0.25 if highlight_deflection is not None else 0.8
                font_color = 'black'

            # Linea nera tratteggiata (come nell'originale)
            ax.plot(points_sorted[:, 0], points_sorted[:, 1], 'k--', linewidth=1.5, alpha=0.8)

            if len(points_sorted) > 0:
                # Posizioniamo l'etichetta circa a 1/4 del percorso della curva
                idx = len(points_sorted) // 4

                ax.text(points_sorted[idx, 0], points_sorted[idx, 1],
                        f'{deflection}°', fontsize=15, fontweight='bold',
                        ha='center', va='center', bbox=bbox_props)

        # ===== Plotta curve di efficienza (tratteggiato) =====
        for efficiency, points in sorted(self.efficiency_curves.items()):

            # AGGIUNTA: Ordiniamo i punti prima di disegnarli!
            points_sorted = self._sort_curve_points(points)

            eff_alpha = 0.3 if highlight_deflection is not None else 0.8

            ax.plot(points_sorted[:, 0], points_sorted[:, 1], 'k--', linewidth=3, alpha=0.6)

            # Aggiungi etichetta efficienza (ignorando i NaN che abbiamo usato per spezzare)
            valid_points = points_sorted[~np.isnan(points_sorted[:, 0])]
            if len(valid_points) > 0:
                ax.text(valid_points[-1, 0], valid_points[-1, 1], f'  η={efficiency:.2f}',
                        fontsize=15, verticalalignment='center')

        if target_point is not None:
            phi_p, psi_p = target_point
            # Disegna una stella blu grande
            ax.plot(phi_p, psi_p, marker='*', color='blue', markersize=14, markeredgecolor='black', zorder=5)
            # Aggiungi il testo di fianco
            ax.text(phi_p + 0.03, psi_p + 0.03, f'OP\n({phi_p:.2f}, {psi_p:.2f})',
                    color='blue', fontsize=15, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9), zorder=5)

        # ===== Formattazione =====
        ax.set_xlabel('Flow coefficient, φ', fontsize=15, fontweight='bold')
        ax.set_ylabel('Stage loading coefficient, ψ', fontsize=15, fontweight='bold')
        ax.set_title('Smith Diagram Action - total to total', fontsize=15, fontweight='bold')
        ax.legend(loc='upper left', title='Deflection angle', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=15)

        # Imposta i limiti in modo simile all'originale per una migliore resa
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 3.0)

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
csv_folder = Path(__file__).parent / "csv"  # ← Percorso aggiornato

# 2. Carica il diagramma
smith_action_total_to_total = SmithDiagram_Action_total_to_total(csv_folder)

# 3. Stampa riepilogo
#smith_action_total_to_total.print_summary()

# 4. Plotta il diagramma
#smith_action_total_to_total.plot(target_point=(0.6243, 1.2198), highlight_deflection=100, save_path="smith_diagram_action_total_to_total.png")

# 5. Testa le funzioni
print("\n🧪 TEST FUNZIONI:")
print("-" * 60)

