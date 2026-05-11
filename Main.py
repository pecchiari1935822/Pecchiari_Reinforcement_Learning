import os
import pandas as pd
import numpy as np
from Agente.Set_input_param import ROW_INDEX, combinazioni_da_testare, learning_rate, n_steps
from Agente.PPO import DOF_BOUNDS_ALL, DOF_NAMES_ALL, PPO_PARAMS, SURROGATE_MODEL_PATH, SCALER_PATH, aggiungi_slide_iterazione, train, pulisci_file_temporanei, Presentation, Path

if __name__ == "__main__":
    # ==========================================
    # TASK 1: Partenza casuale
    # ==========================================
    '''model, best_dof, best_of, best_csi, model_path = train(
        surrogate_path=SURROGATE_MODEL_PATH,
        scaler_path=SCALER_PATH,
        start_dof=None,
        learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size
    )

    # Valutazione finale
    best_dof_end, best_of_eval, best_csi_eval, best_traj, best_csi_list, start_dof_vincente = evaluate(
        model_path=model_path,
        surrogate_path=SURROGATE_MODEL_PATH,
        scaler_path=SCALER_PATH,
        start_dof=None,
        n_episodes=10,
    )

    # 2. Visualizzazioni (usa esattamente la traiettoria estratta sopra!)
    visualize_evaluate_results(
        best_trajectory=best_traj,
        best_csi_steps=best_csi_list,
        best_dof_start=start_dof_vincente,
        best_dof_end=best_dof_end
    )'''

    # ==========================================
    # TASK 2: Ottimizzazione da riga del Dataset
    # ==========================================

    prs = Presentation()

    # 1. Imposta il percorso del tuo dataset e la riga che vuoi analizzare
    DATABASE_DIR = Path(__file__).parent.parent.resolve()
    DATASET_PATH = str(DATABASE_DIR / "Data" / "database.dat")
    TEMPLATE_PATH = "Template.pptx"

    if os.path.exists(TEMPLATE_PATH):
        print(f"Caricamento template da: {TEMPLATE_PATH}")
        prs = Presentation(TEMPLATE_PATH)


    for row_idx in ROW_INDEX:
        print(f"\nLettura dataset: {DATASET_PATH}")
        print(f"Estrazione riga numero: {row_idx}")

        # Leggi il dataset ed estrai la riga
        df = pd.read_csv(DATASET_PATH)
        row = df.iloc[row_idx]
        riga = df.iloc[row_idx].values

        # 3. Estrai il CSI originale (12° valore -> indice 11)
        csi_originale = float(riga[11])

        for active_dof in combinazioni_da_testare:
            # ---> FIX CRITICO: Aggiorniamo dinamicamente le variabili dell'ambiente!
            # Altrimenti Gym continuerebbe ad usare i parametri fissi del file .py
            import Ambiente.Ambiente_claude_senza_keras as env_module

            env_module.ACTIVE_DOF_INDICES = active_dof
            env_module.DOF_BOUNDS = [DOF_BOUNDS_ALL[i] for i in active_dof]

            # Aggiorniamo anche la variabile globale in questo script per i grafici

            ACTIVE_DOF_INDICES = active_dof

            slide_layout_intro = prs.slide_layouts[0]
            slide_intro = prs.slides.add_slide(slide_layout_intro)

            if slide_intro.shapes.title:
                # Prendi direttamente la forma (shape) del titolo
                title_shape = slide_intro.shapes.title
                title_shape.text = f"Ottimizzazione Riga Dataset: {row_idx}"

            # --- 2. GESTIONE DEL SOTTOTITOLO ---
            # Verifica che esista un secondo placeholder (generalmente usato per il sottotitolo)
            if len(slide_intro.placeholders) > 1:
                subtitle_shape = slide_intro.placeholders[1]

                nomi_attivi = [DOF_NAMES_ALL[i].replace("DOF_", "").replace("_GEOM", "").replace("_", "") for i in active_dof]
                subtitle_shape.text = f"Analisi {' e '.join(nomi_attivi).lower()}"

                # 0. Slide per il proiflo alla riga n del dataset
                slide_layout_testo_ppo = prs.slide_layouts[2]
                slide_testo_ppo = prs.slides.add_slide(slide_layout_testo_ppo)
                slide_testo_ppo.shapes.title.text = f"Parametri PPO"
                if len(slide_testo_ppo.placeholders) > 1:
                    corpo_ppo = slide_testo_ppo.placeholders[1]
                    tf_ppo = corpo_ppo.text_frame
                    tf_ppo.clear()  # Pulisce il testo di default
                    for key, value in PPO_PARAMS.items():
                        p = tf_ppo.add_paragraph()
                        run = p.add_run()

                        if key.strip() == "":
                            run.text = ""  # riga vuota
                        else:
                            run.text = f"{key} = {value}"

            # 4. Poni a 0.0 i DOF attivi (quelli che il PPO deve modificare)
            print("\n--- PERTURBAZIONE DEI DOF ATTIVI ---")
            print(f"Estrazione riga numero: {row_idx}")

            # 2. Estrai i 7 DOF (dal 3° al 9° valore -> indici da 2 a 8)
            start_profile = riga[2:9].astype(np.float32).copy()

            # Perturbazione del dof attivo (che si vuole ottimizzare)
            '''for idx in active_dof:
                val_originale = start_profile[idx]
                min_bound, max_bound = DOF_BOUNDS_ALL[idx]

                # Calcoliamo la distanza tra il valore originale e i due limiti
                distanza_dal_min = abs(val_originale - min_bound)
                distanza_dal_max = abs(max_bound - val_originale)

                # Scegliamo il limite più lontano per rendere la sfida più difficile per il PPO!
                if distanza_dal_max > distanza_dal_min:
                    valore_perturbato = max_bound
                else:
                    valore_perturbato = min_bound

                # Applichiamo la perturbazione
                start_profile[idx] = valore_perturbato

                surrogate = load_surrogate(SURROGATE_MODEL_PATH, SCALER_PATH)

                input_dof = start_profile.reshape(1, -1)
                predizione = surrogate(input_dof)
                csi_modificato = float(predizione[11])

                nome_dof = DOF_NAMES_ALL[idx]
                print(f"  {nome_dof:<28}: {val_originale:>10.6f}  -->  Spostato a: {valore_perturbato:>10.6f}")
                print(f"  (CSI originale: {csi_originale:.6f}  -->  Perturbato a: {csi_modificato:.6f})")

            print("-------------------------------------------------")
            print("\nI DOF attivi sono stati spostati ai loro limiti per testare il PPO.")'''

            print(f"\nProfilo di partenza per il PPO: {start_profile}.3f")

            # Testo l'addestramento di ogni profilo (riga) con diversi learning_rate per vedere quale è meglio
            for lr in learning_rate:

                # Per ogni learning_rate, provo anche diversi n_steps per vedere l'effetto sulla convergenza e sul risultato finale
                for n_step in n_steps:
                    if n_step <= 10:
                        batch_size = 10
                    elif n_step <= 50:
                        batch_size = 32
                    else:
                        batch_size = 64

                    # 5. Avvia il Training (Task 2)
                    print(f"\n  Learning_rate attuale : {lr}\n")
                    model, best_dof, best_of, best_csi, model_path = train(
                        surrogate_path=SURROGATE_MODEL_PATH,
                        scaler_path=SCALER_PATH,
                        start_dof=start_profile,
                        learning_rate=lr, n_steps=n_step, batch_size=batch_size
                    )


                    # 6. Confronto Finale PPO vs Dataset
                    miglioramento = csi_originale - best_csi
                    segno = "✓ MIGLIORATO" if miglioramento > 0 else "✗ PEGGIORATO"

                    print("\n" + "=" * 60)
                    print("  RISULTATO TASK 2: CONFRONTO PPO vs DATASET")
                    print("=" * 60)
                    print(f"  CSI Originale (dataset) : {csi_originale:.6f}")
                    print(f"  CSI Migliore PPO        : {best_csi:.6f}")
                    print(f"  Differenza (Delta CSI)  : {miglioramento:+.6f}  ({segno})")
                    print("=" * 60)

                    print(f"\n  Riga {row_idx} (index={row.name}):")
                    print(row.to_frame().T.to_string(index=False))

                    path_img1 = "plot_results.png"
                    path_img2 = "plot_metrics_actor.png"
                    path_img3 = "plot_metrics_critic.png"

                    parametri_iterazione = {
                        "N steps": n_step,
                        "Batch size": batch_size,
                        "CSI Originale": round(csi_originale, 6),
                        "DOF attivi modificati": [f"{DOF_NAMES_ALL[i]} da {riga[i+2]} a {start_profile[i]:.3f}" for i in active_dof],
                        "CSI profilo modificato": None
                    }


                    # 3. Aggiungi i risultati alla presentazione
                    img_paths = [path_img1, path_img2, path_img3]

                    start_of_originali = riga[9:24].astype(np.float32)

                    aggiungi_slide_iterazione(prs, parametri_iterazione, img_paths, row_idx, lr, best_dof, best_of, start_profile, start_of_originali)

    output_pptx = "Report_Simulazioni_PPO.pptx"
    prs.save(output_pptx)

    pulisci_file_temporanei()