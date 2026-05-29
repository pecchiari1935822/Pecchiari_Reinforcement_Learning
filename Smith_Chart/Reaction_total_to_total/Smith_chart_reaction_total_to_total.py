import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
import matplotlib.colors as mcolors


class SmithDiagram_Reaction_total_to_total:
    """
    Carica i file CSV da WebPlotDigitizer e costruisce il diagramma di Smith.
    """

    def __init__(self, csv_folder, extrapolation_method='clip',
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
                self.base_deflections.add(deflection_angle)

                # Crea interpolatore (cubic ok, ma se fa oscillazioni prova linear)
                self.interpolators[deflection_angle] = interp1d(
                    phi, psi, kind=self.spline_kind, bounds_error=False,
                    fill_value=self.extrapolation_method
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

    def estimate_deflection_nearest_integer(self, phi, psi_target, defl_min=40, defl_max=140):
        """
        Stima la deflessione come INTERO più vicino (tra defl_min..defl_max),
        scegliendo la curva che minimizza |psi_curve(phi) - psi_target|.
        Ritorna None se phi fuori dominio per tutte le curve.
        """
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

    def add_interpolated_efficiency_curves_by_blocks(self, step=0.5, n_points=600, overwrite=False, eta_range=None):
        base_etas = sorted(self.efficiency_curves.keys())
        if len(base_etas) < 2:
            print("⚠️  Servono almeno 2 curve eta per interpolare.")
            return
        if eta_range is not None:
            base_etas = [e for e in base_etas if eta_range[0] <= e <= eta_range[1]]

        if not hasattr(self, 'interpolated_efficiency_curves'):
            self.interpolated_efficiency_curves = {}

        def arc_length_param(pts):
            diffs = np.diff(pts, axis=0)
            seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
            arc = np.concatenate([[0], np.cumsum(seg_lengths)])
            arc /= arc[-1]
            return arc

        def is_closed(pts, threshold=0.3):
            """Curva chiusa se il primo e l'ultimo punto sono vicini."""
            return np.linalg.norm(pts[0] - pts[-1]) < threshold

        def clean(pts):
            pts = self._sort_curve_points(pts)
            return pts[~np.isnan(pts).any(axis=1)]

        print(f"\n📊 Interpolazione isorendimento tra: {base_etas}")
        print("=" * 70)

        for i in range(len(base_etas) - 1):
            eta_a = base_etas[i]
            eta_b = base_etas[i + 1]
            print(f"\n🔗 BLOCCO: {eta_a} → {eta_b}")

            pts_a = clean(self.efficiency_curves[eta_a])
            pts_b = clean(self.efficiency_curves[eta_b])

            closed_a = is_closed(pts_a)
            closed_b = is_closed(pts_b)

            print(f"   eta_{eta_a}: {'chiusa' if closed_a else 'aperta'} ({len(pts_a)} punti)")
            print(f"   eta_{eta_b}: {'chiusa' if closed_b else 'aperta'} ({len(pts_b)} punti)")

            # Non interpolare tra aperta e chiusa: topologia diversa
            if closed_a != closed_b:
                print(f"   ⚠️  Topologie diverse (aperta↔chiusa). Blocco saltato.")
                continue

            if closed_a and closed_b:
                # Allinea i punti di partenza: entrambe partono dal punto con psi minima
                pts_a = np.roll(pts_a, -np.argmin(pts_a[:, 1]), axis=0)
                pts_b = np.roll(pts_b, -np.argmin(pts_b[:, 1]), axis=0)
            # Se entrambe aperte, lasciale così come sono (già ordinate dal _sort)

            t_a = arc_length_param(pts_a)
            t_b = arc_length_param(pts_b)
            t_grid = np.linspace(0, 1, n_points)

            phi_a_grid = np.interp(t_grid, t_a, pts_a[:, 0])
            psi_a_grid = np.interp(t_grid, t_a, pts_a[:, 1])
            phi_b_grid = np.interp(t_grid, t_b, pts_b[:, 0])
            psi_b_grid = np.interp(t_grid, t_b, pts_b[:, 1])

            eta_new = round(eta_a + step, 9)
            curves_created = []
            while eta_new < eta_b - 1e-9:
                if eta_new not in self.interpolated_efficiency_curves or overwrite:
                    t = (eta_new - eta_a) / (eta_b - eta_a)
                    phi_new = (1 - t) * phi_a_grid + t * phi_b_grid
                    psi_new = (1 - t) * psi_a_grid + t * psi_b_grid
                    self.interpolated_efficiency_curves[round(eta_new, 4)] = np.column_stack([phi_new, psi_new])
                    curves_created.append(round(eta_new, 4))
                eta_new = round(eta_new + step, 9)

            if curves_created:
                print(f"   ✓ Create: {curves_created}")
            else:
                print(f"   (nessuna nuova curva)")

        print("\n" + "=" * 70)
        print("✓ Interpolazione isorendimento completata!")
        print("=" * 70 + "\n")

    def plot(self, figsize=(12, 9), save_path=None, target_point=None, highlight_deflection=None, scatter_data=None,
             show_interpolated_efficiency=False, show_interpolated_deflection=True):
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
        # 2) Curve interpolate extra (deflessione)
        if show_interpolated_deflection:
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

        if show_interpolated_efficiency and hasattr(self, 'interpolated_efficiency_curves'):
            for eta, points in sorted(self.interpolated_efficiency_curves.items()):
                points_sorted = self._sort_curve_points(points)
                ax.plot(points_sorted[:, 0], points_sorted[:, 1],
                        color='gray', linestyle=':', linewidth=1.0, alpha=0.5, zorder=1)
                valid_points = points_sorted[~np.isnan(points_sorted[:, 0])]


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

        # 4) Scatter dataset
        if scatter_data is not None:

            phi_s = np.asarray(scatter_data['phi'])
            psi_s = np.asarray(scatter_data['psi'])
            vals = np.asarray(scatter_data['values'])
            lbl = scatter_data.get('label', 'Losses')
            cmap = scatter_data.get('cmap', 'RdYlGn_r')
            pct = scatter_data.get('percentile_highlight', 10)
            sz = scatter_data.get('size', 40)
            alpha = scatter_data.get('alpha', 0.75)

            vmin, vmax = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            scatter = ax.scatter(phi_s, psi_s, c=vals, cmap=cmap, norm=norm,
                                 s=sz, alpha=alpha, edgecolors='black', linewidths=0.4, zorder=8)
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label(lbl, fontsize=13, fontweight='bold')
            cbar.ax.tick_params(labelsize=11)

            if pct is not None and pct > 0:
                thr_low = np.nanpercentile(vals, pct)
                thr_high = np.nanpercentile(vals, 100 - pct)
                mask_best = vals <= thr_low
                mask_worst = vals >= thr_high

                ax.scatter(phi_s[mask_best], psi_s[mask_best],
                           c=vals[mask_best], cmap=cmap, norm=norm,
                           s=sz * 1.8, edgecolors='lime', linewidths=1.8, zorder=9,
                           label=f'Migliori {pct}% ({lbl} basso)')
                ax.scatter(phi_s[mask_worst], psi_s[mask_worst],
                           c=vals[mask_worst], cmap=cmap, norm=norm,
                           s=sz * 1.8, edgecolors='darkred', linewidths=1.8, zorder=9,
                           label=f'Peggiori {pct}% ({lbl} alto)')

        # ----------------------------
        # 5) Formattazione
        # ----------------------------
        ax.set_xlabel("Flow coefficient, φ", fontsize=16, fontweight="bold")
        ax.set_ylabel("Stage loading coefficient, ψ", fontsize=16, fontweight="bold")
        ax.set_title("Smith Diagram Reaction - Total to total", fontsize=16, fontweight="bold")

        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper left", title="Deflection angle", fontsize=16)

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=14)

        # Mantieni i tuoi limiti originali
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 3)


        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Grafico salvato: {save_path}")

        return fig, ax

    def plot_losses_comparison(self, phi, psi, csi, cpt, figsize=(18, 8), save_path=None):
        """
        Crea due diagrammi di Smith affiancati:
          - Sinistra: scatter colorato per CSI (perdite di pressione statica)
          - Destra:   scatter colorato per CPT (perdite di pressione totale)

        Utile per confrontare come le due misure di perdita si distribuiscono
        sullo stesso spazio (phi, psi).
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        datasets = [
            {'values': csi, 'label': 'CSI  (perdite pressione statica)', 'ax': axes[0]},
            {'values': cpt, 'label': 'CPT  (perdite pressione totale)', 'ax': axes[1]},
        ]

        phi = np.asarray(phi)
        psi = np.asarray(psi)

        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)

        for item in datasets:
            ax = item['ax']
            vals = np.asarray(item['values'])
            lbl = item['label']

            # --- sfondo diagramma di Smith ---
            for deflection in sorted(self.base_deflections):
                if deflection not in self.deflection_curves:
                    continue
                pts = self.deflection_curves[deflection]
                pts = pts[np.argsort(pts[:, 0])]
                ax.plot(pts[:, 0], pts[:, 1], color='black', linestyle='--',
                        linewidth=1.4, alpha=0.4, zorder=1)
                if len(pts) > 0:
                    idx = len(pts) // 4
                    ax.text(pts[idx, 0], pts[idx, 1], f"{deflection}°",
                            fontsize=11, fontweight="bold", color='black',
                            ha="center", va="center", bbox=bbox_props, zorder=5)

            for efficiency, points in sorted(self.efficiency_curves.items()):
                points_sorted = self._sort_curve_points(points)
                ax.plot(points_sorted[:, 0], points_sorted[:, 1], 'k--',
                        linewidth=2.0, alpha=0.5, zorder=1)
                valid_points = points_sorted[~np.isnan(points_sorted[:, 0])]
                if len(valid_points) > 0:
                    ax.text(valid_points[-1, 0], valid_points[-1, 1],
                            f"  η={efficiency:.2f}", fontsize=11,
                            verticalalignment="center_baseline",
                            horizontalalignment="center", zorder=6)

            # --- scatter ---
            vmin = np.nanpercentile(vals, 2)
            vmax = np.nanpercentile(vals, 98)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            sc = ax.scatter(phi, psi, c=vals, cmap='RdYlGn_r', norm=norm,
                            s=35, alpha=0.75, edgecolors='black', linewidths=0.35, zorder=8)

            cbar = plt.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(lbl.split('(')[0].strip(), fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)

            # Evidenzia migliori (verde lime) e peggiori (rosso scuro) — top/bottom 10%
            thr_low = np.nanpercentile(vals, 10)
            thr_high = np.nanpercentile(vals, 90)
            mask_best = vals <= thr_low
            mask_worst = vals >= thr_high

            ax.scatter(phi[mask_best], psi[mask_best],
                       c=vals[mask_best], cmap='RdYlGn_r', norm=norm,
                       s=60, edgecolors='lime', linewidths=1.8, zorder=9,
                       label='Migliori 10%')
            ax.scatter(phi[mask_worst], psi[mask_worst],
                       c=vals[mask_worst], cmap='RdYlGn_r', norm=norm,
                       s=60, edgecolors='darkred', linewidths=1.8, zorder=9,
                       label='Peggiori 10%')

            ax.set_xlabel("Flow coefficient, φ", fontsize=14, fontweight="bold")
            ax.set_ylabel("Stage loading coefficient, ψ", fontsize=14, fontweight="bold")
            ax.set_title(lbl, fontsize=13, fontweight="bold")
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", which="major", labelsize=12)
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, 3)

        fig.suptitle("Distribuzione delle perdite sul Diagramma di Smith",
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Grafico salvato: {save_path}")

        return fig, axes

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
if __name__ == "__main__":
    # 1. Specifica il percorso della cartella con i CSV
    csv_folder = Path(__file__).parent / "csv"

    # 2. Carica il diagramma
    smith_reaction_total_to_total = SmithDiagram_Reaction_total_to_total(csv_folder)

    # Interpola per blocchi
    smith_reaction_total_to_total.add_interpolated_deflection_curves_by_blocks(
        step=1,
        n_points=600,
        overwrite=False
    )

    # Valida
    '''validation_report = smith_reaction_total_to_total.validate_interpolation(
        defl_min=60, defl_max=80, tolerance_psi=0.03
    )'''

    smith_reaction_total_to_total.add_interpolated_efficiency_curves_by_blocks(
        step=0.2, eta_range=(88, 92))

    # Plot del diagramma di Smith senza tutto ti dataset
    fig, ax = smith_reaction_total_to_total.plot(figsize=(12, 9), show_interpolated_efficiency=True,
                                                 show_interpolated_deflection=False)


    # 3. Stampa riepilogo
    smith_reaction_total_to_total.print_summary()

    # Stama diagramma di smith




    # 4. Carica il dataset
    BASE_DIR = Path(__file__).resolve().parents[2]
    DATASET_PATH = BASE_DIR / "Data" / "database.dat"
    df = pd.read_csv(DATASET_PATH)

    # 5. Plotta il diagramma


    # 6. Aggiungi i punti del dataset come scatter plot
    phi = df['OF_phi_OP_01'].values
    psi = df['OF_psi_OP_01'].values
    csi = df['OF_CSI_OP_01'].values
    cpt = df['OF_Cpt_OP_01'].values
    loss_tot = csi+cpt

    nome_colonna_alfa = 'OF_alfa_ex_OP_01'
    nome_beta1 = 'DOF_BETA1_GEOM_'
    nome_beta2 = 'DOF_BETA2_GEOM_'

    alfa_ex = df[nome_colonna_alfa].values
    beta1 = df[nome_beta1].values
    beta2 = df[nome_beta2].values


    # Plot del diagramma di Smith con tutto il dataset per vedere se al ridursi di CSi aumenta l'efficienza
    fig, ax = smith_reaction_total_to_total.plot(
        figsize=(13, 9),show_interpolated_efficiency=True,show_interpolated_deflection=False,
        scatter_data={
            'phi'                  : phi,
            'psi'                  : psi,
            'values'               : csi,
            'label'                : 'CSI  (perdite pressione statica)',
            'cmap'                 : 'RdYlGn_r',   # verde=basso, rosso=alto
            'percentile_highlight' : 10,            # evidenzia top/bottom 10%
            'size'                 : 40,
            'alpha'                : 0.75,
        }
    )
    ax.set_title("Smith Diagram Reaction — Total to total", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------------
    # OPZIONE B — Due diagrammi affiancati: CSI e CPT a confronto
    # -----------------------------------------------------------------------
    fig2, axes2 = smith_reaction_total_to_total.plot_losses_comparison(
        phi=phi, psi=psi, csi=csi, cpt=cpt,
        figsize=(20, 9)
    )
    plt.show()

    # --- 10 profili random sul diagramma di Smith ---
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    np.random.seed(42)
    indices = np.random.choice(len(df), size=10, replace=False)

    smith_reaction_total_to_total.add_interpolated_efficiency_curves_by_blocks(
        step=0.05, eta_range=(88, 92))

    fig2, ax2 = smith_reaction_total_to_total.plot(figsize=(13, 9),show_interpolated_efficiency=True, show_interpolated_deflection=True)

    cmap_10 = cm.get_cmap('tab10')  # 10 colori distinti

    for k, i in enumerate(indices):
        phi_i = float(phi[i])
        psi_i = float(psi[i])
        alfa_ex_i = float(df['OF_alfa_ex_OP_01'].values[i])
        defl_reale = 10.0 - alfa_ex_i  # alfa_in fisso = 10

        color = cmap_10(k)

        # Deflessione stimata da Smith
        d_smith = smith_reaction_total_to_total.estimate_deflection_nearest_integer(
            phi_i, psi_i, defl_min=40, defl_max=140
        )

        # Evidenzia la curva di deflessione corrispondente
        if d_smith is not None and d_smith in smith_reaction_total_to_total.deflection_curves:
            pts = smith_reaction_total_to_total.deflection_curves[d_smith]
            pts = pts[np.argsort(pts[:, 0])]
            ax2.plot(pts[:, 0], pts[:, 1],
                     color=color, linestyle='-', linewidth=2.2,
                     alpha=0.85, zorder=3)

        # Punto
        ax2.scatter(phi_i, psi_i,
                    color=color, s=150,
                    edgecolors='black', linewidths=1.0,
                    zorder=10,
                    label=f'#{i}  |  δ_reale={defl_reale:.1f}°  |  δ_Smith={d_smith}°  |  loss={float(loss_tot[i]):.4f}')

    ax2.set_title("Smith Diagram — 10 profili casuali", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.9, title='#riga  |  Deflessione reale  |  Deflessione Smith  |  CSI')
    margin = 0.15
    ax2.set_xlim(phi[indices].min() - margin, phi[indices].max() + margin)
    ax2.set_ylim(psi[indices].min() - margin, psi[indices].max() + margin)

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
    '''intersezione = smith_reaction_total_to_total.find_intersection(efficiency_target=92, deflection_target=60)
    
    if intersezione:
        phi_int, psi_int = intersezione
        print(f"Punto di intersezione trovato tra Efficienza=0.92 e Deflessione=60°:")
        print(f"Phi (φ) = {phi_int:.4f}")
        print(f"Psi (ψ) = {psi_int:.4f}")
    else:
        print("Nessun punto di intersezione trovato. Le curve non si incontrano o l'efficienza non è presente tra quelle caricate.")'''