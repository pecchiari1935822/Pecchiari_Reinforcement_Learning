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

    def __init__(self, csv_folder, extrapolation_method='extrapolate',
                 spline_kind='cubic', validation_enabled=True):
        """
        Args:
            csv_folder: percorso cartella CSV
            extrapolation_method: 'extrapolate' (default, pericoloso) o 'clip' (sicuro)
            spline_kind: 'cubic' (default, smooth) o 'linear' (robusto)
            validation_enabled: se True, valida interpolazioni al caricamento
        """
        self.csv_folder = Path(csv_folder)
        self.deflection_curves = {}
        self.efficiency_curves = {}
        self.interpolators = {}
        self.deflection_phi_domain = {}
        self.base_deflections = set()

        # NUOVO: configurazione
        self.extrapolation_method = extrapolation_method
        self.spline_kind = spline_kind
        self.validation_enabled = validation_enabled

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
                self.base_deflections.add(deflection_angle)

                # AGGIUNTA: salva dominio phi
                self.deflection_phi_domain[deflection_angle] = (np.min(phi), np.max(phi))

                self.interpolators[deflection_angle] = interp1d(
                    phi, psi, kind=self.spline_kind, bounds_error=False,
                    fill_value=self.extrapolation_method
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
    def add_interpolated_deflection_curves_by_blocks(
            self,
            step=1,
            n_points=600,
            overwrite=False
    ):
        """
        Interpola nuove curve per BLOCCHI consecutivi di curve base.

        Esempio: se hai curve base a 40°, 60°, 80°, 100°, 120°, 140°
        Interpola:
          - tra 40° e 60° (passo step)
          - tra 60° e 80° (passo step)
          - tra 80° e 100° (passo step)
          - etc.

        Questo mantiene l'interpolazione locale e coerente.

        Args:
            step: incremento deflessione per le curve interpolate (es. 1°)
            n_points: numero di punti nella griglia φ
            overwrite: se True, ricalcola curve già esistenti
        """

        # Ordina le curve base
        base_deflections_sorted = sorted(self.base_deflections)

        if len(base_deflections_sorted) < 2:
            print("⚠️  WARN: Servono almeno 2 curve base per interpolare.")
            return

        print(f"\n📊 Interpolazione per BLOCCHI tra curve base: {base_deflections_sorted}")
        print("=" * 70)

        # Interpola tra ogni coppia consecutiva di curve base
        for i in range(len(base_deflections_sorted) - 1):
            defl_a = base_deflections_sorted[i]
            defl_b = base_deflections_sorted[i + 1]

            print(f"\n🔗 BLOCCO: Interpolazione tra {defl_a}° e {defl_b}°")

            self._interpolate_single_block(
                defl_a=defl_a,
                defl_b=defl_b,
                step=step,
                n_points=n_points,
                overwrite=overwrite
            )

        print("\n" + "=" * 70)
        print("✓ Interpolazione per blocchi completata!")
        print("=" * 70 + "\n")

    def _interpolate_single_block(
            self,
            defl_a,
            defl_b,
            step=1,
            n_points=600,
            overwrite=False
    ):
        """
        Interpola curve tra due curve base consecutive (defl_a e defl_b).

        Usa il dominio φ delle due curve e interpola ψ dove entrambe sono definite.
        """

        if defl_a not in self.deflection_curves or defl_b not in self.deflection_curves:
            raise ValueError(f"Servono le curve {defl_a}° e {defl_b}° nei CSV.")

        # --- Estrai i punti base e ordinali per phi ---
        pts_a = self.deflection_curves[defl_a]
        pts_b = self.deflection_curves[defl_b]
        pts_a = pts_a[np.argsort(pts_a[:, 0])]
        pts_b = pts_b[np.argsort(pts_b[:, 0])]

        phi_a, psi_a_pts = pts_a[:, 0], pts_a[:, 1]
        phi_b, psi_b_pts = pts_b[:, 0], pts_b[:, 1]

        a_min, a_max = self.deflection_phi_domain[defl_a]
        b_min, b_max = self.deflection_phi_domain[defl_b]

        # --- Dominio di interpolazione: intersezione dei due domini ---
        phi_min_common = max(a_min, b_min)
        phi_max_common = min(a_max, b_max)

        if phi_max_common <= phi_min_common:
            print(f"   ⚠️  WARN: Nessun overlap tra i domini di {defl_a}° e {defl_b}°. Salta questo blocco.")
            return

        print(f"   φ dominio comune: [{phi_min_common:.4f}, {phi_max_common:.4f}]")

        # --- Griglia φ comune ---
        phi_grid = np.linspace(phi_min_common, phi_max_common, n_points)

        # --- Valuta ψ sulle due curve base ---
        psi_a_grid = np.interp(phi_grid, phi_a, psi_a_pts)
        psi_b_grid = np.interp(phi_grid, phi_b, psi_b_pts)

        # --- Interpola le curve intermedie ---
        low = min(defl_a, defl_b)
        high = max(defl_a, defl_b)

        curves_created = []
        for defl_new in range(low + step, high, step):
            if (defl_new in self.deflection_curves) and not overwrite:
                continue

            t = (defl_new - defl_a) / (defl_b - defl_a)

            # Interpolazione lineare di ψ
            psi_new = (1 - t) * psi_a_grid + t * psi_b_grid

            curve_new = np.column_stack([phi_grid, psi_new])

            self.deflection_curves[defl_new] = curve_new
            self.deflection_phi_domain[defl_new] = (phi_min_common, phi_max_common)

            self.interpolators[defl_new] = interp1d(
                phi_grid,
                psi_new,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan
            )

            curves_created.append(defl_new)

        if curves_created:
            print(f"   ✓ Create {len(curves_created)} curve: {curves_created}")
        else:
            print(f"   (nessuna nuova curva creata)")

    # ==========================================================
    # AGGIUNTA: stima deflessione (intero più vicino) da (phi,psi)
    # ==========================================================
    def estimate_deflection_nearest_integer(self, phi, psi_target, defl_min=60, defl_max=80):
        candidates = [d for d in sorted(self.interpolators.keys()) if defl_min <= d <= defl_max]
        if not candidates:
            raise ValueError(f"Nessuna curva/interpolatore tra {defl_min} e {defl_max}.")

        best = None  # (err_euclidean, d, psi_curve, err_psi)
        point_is_out_of_all_domains = True

        for d in candidates:
            # Controlla dominio
            if d in self.deflection_phi_domain:
                dmin, dmax = self.deflection_phi_domain[d]
                if not (dmin <= phi <= dmax):
                    continue

            point_is_out_of_all_domains = False

            psi_curve = float(self.interpolators[d](phi))

            # NUOVO: usa distanza euclidea (più robusta)
            err_euclidean = np.sqrt((psi_curve - psi_target) ** 2)  # in questo caso coincide con err_psi
            # Ma se vuoi pesare anche phi, puoi fare:
            # err_euclidean = np.sqrt((psi_curve - psi_target)**2)
            # Oppure con peso: np.sqrt(0.8*(psi_curve - psi_target)**2)

            if best is None or err_euclidean < best[0]:
                best = (err_euclidean, d, psi_curve, abs(psi_curve - psi_target))

        if best is None:
            if point_is_out_of_all_domains:
                print(
                    f"⚠️  WARN: Punto ({phi:.4f}, {psi_target:.4f}) fuori da TUTTI i domini φ nel range [{defl_min}, {defl_max}]")
            return None

        err, d, psi_curve, err_psi = best

        # NUOVO: stampa diagnostica se errore > soglia
        if err_psi > 0.05:  # soglia ψ (ajusta secondo i tuoi dati)
            print(
                f"⚠️  WARN: Deflessione stimata {int(round(d))}° con errore ψ={err_psi:.4f} per punto ({phi:.4f}, {psi_target:.4f})")

        return int(round(d))

    def validate_interpolation(self, test_points=None, defl_min=60, defl_max=80, tolerance_psi=0.03):
        """
        Valida che le curve interpolate siano coerenti.
        Se test_points è None, usa punti lungo le curve caricate.

        Args:
            test_points: lista di (phi, psi, expected_deflection_deg) oppure None
            defl_min, defl_max: range di deflessione
            tolerance_psi: tolleranza massima di errore in ψ

        Returns:
            Rapporto di validazione (dict con risultati)
        """
        report = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': [],
            'errors': []
        }

        # Se non dai test_points, generali automaticamente dai CSV originali
        if test_points is None:
            test_points = []
            # Prendi solo le curve originali (quelle che hai nei CSV)
            original_curves = [d for d in self.deflection_curves.keys()
                               if d in self.deflection_phi_domain]

            for defl in sorted(original_curves):
                if not (defl_min <= defl <= defl_max):
                    continue
                points = self.deflection_curves[defl]
                # Prendi ogni 5° punto sulla curva
                for idx in range(0, len(points), max(1, len(points) // 10)):
                    phi, psi = points[idx]
                    test_points.append((phi, psi, float(defl)))

        # Esegui test
        for phi, psi_target, expected_defl in test_points:
            report['total_tests'] += 1

            estimated_defl = self.estimate_deflection_nearest_integer(
                phi, psi_target, defl_min=defl_min, defl_max=defl_max
            )

            if estimated_defl is None:
                report['failed'] += 1
                report['errors'].append(
                    f"  Punto ({phi:.4f}, {psi_target:.4f}) fuori dominio"
                )
                continue

            # Controlla errore in psi
            psi_estimated = float(self.interpolators[estimated_defl](phi))
            err_psi = abs(psi_estimated - psi_target)
            err_defl = abs(estimated_defl - expected_defl)

            if err_psi <= tolerance_psi and err_defl <= 1:  # ±1 grado è ok
                report['passed'] += 1
            else:
                report['failed'] += 1
                report['warnings'].append(
                    f"  Punto ({phi:.4f}, {psi_target:.4f}): "
                    f"Atteso {expected_defl}°, stimato {estimated_defl}° "
                    f"(errore ψ={err_psi:.4f})"
                )

        # Stampa rapporto
        print("\n" + "=" * 70)
        print(f"VALIDAZIONE INTERPOLAZIONE ({self.__class__.__name__})")
        print("=" * 70)
        print(f"Test totali: {report['total_tests']}")
        print(f"✓ Passati: {report['passed']}")
        print(f"✗ Falliti: {report['failed']}")

        if report['warnings']:
            print(f"\n⚠️  Avvisi ({len(report['warnings'])}):")
            for w in report['warnings'][:5]:  # mostra primi 5
                print(w)
            if len(report['warnings']) > 5:
                print(f"  ... e {len(report['warnings']) - 5} altri")

        if report['errors']:
            print(f"\n❌ Errori ({len(report['errors'])}):")
            for e in report['errors'][:5]:
                print(e)
            if len(report['errors']) > 5:
                print(f"  ... e {len(report['errors']) - 5} altri")

        print("=" * 70 + "\n")

        return report

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

    def build_single_deflection_curve(self, deflection_deg, defl_a=60, defl_b=80, n_points=250):
        """
        Costruisce SOLO la curva di deflessione 'deflection_deg' (anche non intera),
        interpolando linearmente tra le curve note defl_a e defl_b.

        Risultato:
          - self.deflection_curves contiene solo {round(deflection_deg): curve}
          - self.interpolators contiene solo l'interpolatore di quella curva
          - self.deflection_phi_domain aggiornato
        """
        if defl_a not in self.interpolators or defl_b not in self.interpolators:
            raise ValueError(f"Servono {defl_a}° e {defl_b}° già caricate per costruire la curva singola.")

        a_min, a_max = self.deflection_phi_domain[defl_a]
        b_min, b_max = self.deflection_phi_domain[defl_b]
        phi_min = max(a_min, b_min)
        phi_max = min(a_max, b_max)
        if phi_max <= phi_min:
            raise ValueError(f"Nessun range di φ in comune tra {defl_a}° e {defl_b}°.")

        phi_grid = np.linspace(phi_min, phi_max, n_points)

        psi_a = self.interpolators[defl_a](phi_grid)
        psi_b = self.interpolators[defl_b](phi_grid)

        t = (deflection_deg - defl_a) / (defl_b - defl_a)
        psi_new = (1 - t) * psi_a + t * psi_b

        curve_new = np.column_stack([phi_grid, psi_new])

        # etichetta: usa l'intero più vicino (serve per highlight_deflection e legenda coerente)
        defl_label = int(round(deflection_deg))

        # IMPORTANTISSIMO: svuota tutto e lascia solo la curva stimata
        self.deflection_curves[defl_label] = curve_new
        self.deflection_phi_domain[defl_label] = (phi_min, phi_max)
        self.interpolators[defl_label] = interp1d(
            phi_grid, psi_new, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )

        return defl_label

    def plot(self, figsize=(12, 9), save_path=None, target_point=None, highlight_deflection=None):
        """
        Plotta il diagramma di Smith.
        Regole:
          - Plotta SEMPRE le curve base (da CSV) 40,60,80,100,120,140 (e qualsiasi altra base presente).
          - Se esiste una curva extra (interpolata singola) la plotta e la evidenzia.
          - Evidenzia in rosso SOLO la curva con deflessione == highlight_deflection.
        """
        fig, ax = plt.subplots(figsize=figsize)

        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

        # ----------------------------
        # Helper per plottare 1 curva
        # ----------------------------
        def _plot_one_curve(deflection, points, is_highlight: bool, is_base: bool):
            points_sorted = points[np.argsort(points[:, 0])]

            if is_highlight:
                color = "red"
                linestyle = "-"
                linewidth = 2.8
                alpha = 1.0
                font_color = "red"
                zorder = 4
            else:
                # curve base sempre visibili, curve extra non evidenziate molto leggere
                if is_base:
                    color = "black"
                    linestyle = "--"
                    linewidth = 1.6
                    alpha = 0.85 if highlight_deflection is None else 0.35
                else:
                    color = "gray"
                    linestyle = "--"
                    linewidth = 1.2
                    alpha = 0.25
                font_color = "black"
                zorder = 2

            # Label in legenda SOLO per la curva evidenziata (pulito)
            label = f"{deflection}°" if is_highlight else None

            ax.plot(
                points_sorted[:, 0], points_sorted[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                zorder=zorder
            )

            # Testo sulla curva: SOLO curve base (altrimenti la curva interpolata aggiunge “rumore”)
            if is_base and len(points_sorted) > 0:
                idx = len(points_sorted) // 4
                ax.text(
                    points_sorted[idx, 0], points_sorted[idx, 1],
                    f"{deflection}°",
                    fontsize=13, fontweight="bold",
                    color=font_color,
                    ha="center", va="center",
                    bbox=bbox_props,
                    zorder=5
                )

        # ----------------------------
        # 1) PLOT CURVE BASE (CSV)
        # ----------------------------
        if not hasattr(self, "base_deflections"):
            # fallback se non hai aggiunto base_deflections
            self.base_deflections = set(self.deflection_curves.keys())

        base_defs = sorted(self.base_deflections)
        for deflection in base_defs:
            if deflection not in self.deflection_curves:
                continue
            points = self.deflection_curves[deflection]
            is_highlight = (highlight_deflection is not None and deflection == highlight_deflection)
            _plot_one_curve(deflection, points, is_highlight=is_highlight, is_base=True)

        # ----------------------------
        # 2) PLOT CURVE EXTRA (interpolate singole)
        # ----------------------------
        extra_defs = sorted(set(self.deflection_curves.keys()) - set(self.base_deflections))
        for deflection in extra_defs:
            points = self.deflection_curves[deflection]
            is_highlight = (highlight_deflection is not None and deflection == highlight_deflection)
            _plot_one_curve(deflection, points, is_highlight=is_highlight, is_base=False)

        # ----------------------------
        # 3) Curve di efficienza
        # ----------------------------
        for efficiency, points in sorted(self.efficiency_curves.items()):
            points_sorted = self._sort_curve_points(points)

            ax.plot(points_sorted[:, 0], points_sorted[:, 1], "k--", linewidth=2.5, alpha=0.6, zorder=1)

            valid_points = points_sorted[~np.isnan(points_sorted[:, 0])]
            if len(valid_points) > 0:
                ax.text(
                    valid_points[-1, 0], valid_points[-1, 1],
                    f"  η={efficiency:.2f}",
                    fontsize=13,
                    verticalalignment="center_baseline",
                    horizontalalignment="center",
                    zorder=6
                )

        # ----------------------------
        # 4) Punto target OP
        # ----------------------------
        if target_point is not None:
            phi_p, psi_p = target_point
            ax.plot(
                phi_p, psi_p, marker="*", color="blue",
                markersize=14, markeredgecolor="black", zorder=10
            )
            ax.text(
                phi_p + 0.03, psi_p + 0.03,
                f"OP\n({phi_p:.2f}, {psi_p:.2f})",
                color="blue", fontsize=13, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9),
                zorder=10
            )

        # ----------------------------
        # 5) Formattazione
        # ----------------------------
        ax.set_xlabel("Flow coefficient, φ", fontsize=16, fontweight="bold")
        ax.set_ylabel("Stage loading coefficient, ψ", fontsize=16, fontweight="bold")
        ax.set_title("Smith Diagram Action - uscita assiale", fontsize=16, fontweight="bold")

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper left", title="Deflection angle", fontsize=16)

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=14)

        # Mantieni i tuoi limiti originali
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
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
if __name__ == "__main__":
    # 1. Specifica il percorso della cartella con i CSV
    csv_folder = Path(__file__).parent / "csv"

    # 2. Carica il diagramma
    # Nel tuo __main__:
    smith_action_assiale = SmithDiagram_Action_Assiale(csv_folder)

    # Interpola per blocchi
    smith_action_assiale.add_interpolated_deflection_curves_by_blocks(
        step=1,
        n_points=600,
        overwrite=False
    )

    # Valida
    validation_report = smith_action_assiale.validate_interpolation(
        defl_min=60, defl_max=80, tolerance_psi=0.03
    )

    # Plot
    fig, ax = smith_action_assiale.plot(figsize=(12, 9))
    plt.show()

    # 3. Stampa riepilogo (se vuoi)
    '''smith_action_assiale.print_summary()
    
    # 3b. Interpola deflessioni tra 60 e 80
    smith_action_assiale.add_interpolated_deflection_curves(defl_a=60, defl_b=80, step=1)'''

    # 4. Carica dataset
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "Data" / "database.dat"
    df = pd.read_csv(DATASET_PATH)

    # 5. Plotta il diagramma
    '''fig, ax = smith_action_assiale.plot(
        figsize=(12, 9),
        save_path="smith_diagram_action_assiale_with_dataset.png"
    )'''

    # 6. Scatter dataset
    phi = df['OF_phi_OP_01'].values
    psi = df['OF_psi_OP_01'].values
    csi = df['OF_CSI_OP_01'].values

    '''scatter = ax.scatter(
        phi, psi,
        c=csi, cmap='RdYlGn_r',
        s=50, alpha=0.6,
        edgecolors='black', linewidth=0.5
    )
    cbar = plt.colorbar(scatter, ax=ax, label='CSI')
    
    plt.tight_layout()
    plt.show()'''

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


