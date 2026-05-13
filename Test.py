import os
import numpy as np
import pandas as pd


from stable_baselines3 import PPO
from Ambiente.Ambiente_claude_senza_keras import BladeOptimEnv, load_surrogate, DOF_BOUNDS_ALL
from Agente.PPO import SURROGATE_MODEL_PATH, SCALER_PATH, Path


def ottimizza_con_modello_esistente(percorso_modello, nuovo_profilo, ep_length=20):
    """
    Carica un modello PPO addestrato e lo usa per ottimizzare un nuovo profilo.
    Non esegue training, ma solo inferenza deterministica.
    """
    print("\n" + "=" * 60)
    print(f"  TEST MODELLO ESISTENTE SU NUOVO PROFILO")
    print(f"  Modello: {percorso_modello}")
    print("=" * 60)

    # 1. Carica il modello pre-addestrato
    if not os.path.exists(percorso_modello) and not os.path.exists(percorso_modello + ".zip"):
        raise FileNotFoundError(f"Impossibile trovare il modello {percorso_modello}")

    model = PPO.load(percorso_modello)

    # 2. Prepara l'ambiente con il nuovo profilo
    surrogate = load_surrogate(SURROGATE_MODEL_PATH, SCALER_PATH)
    env = BladeOptimEnv(surrogate, start_dof=nuovo_profilo, episode_length=ep_length)

    # 3. Inizializza l'ambiente
    obs, info = env.reset()

    best_csi = np.inf
    best_dof = None
    best_of = None

    print("\n  Inizio episodi di inferenza...")
    # Fai compiere all'agente i passi consentiti in un episodio
    for step in range(ep_length):
        # deterministic=True è FONDAMENTALE in inferenza!
        # Sceglie l'azione con la massima probabilità (niente esplorazione casuale)
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        current_csi = info["csi"]
        print(f"    Step {step + 1:02d} | CSI: {current_csi:.6f}")

        if current_csi < best_csi:
            best_csi = current_csi
            best_dof = info["dof_full"]
            best_of = info["of"]

        if terminated or truncated:
            break

    print(f"\n  Ottimizzazione completata. Miglior CSI trovato: {best_csi:.6f}")
    return best_dof, best_of, best_csi


if __name__ == "__main__":
    # Assicurati di importare PPO se non lo è già
    from stable_baselines3 import PPO

    # ==========================================
    # TASK 3: Inferenza con Modello Pre-addestrato
    # ==========================================

    # 1. Definisci gli STESSI DOF attivi usati quando hai addestrato il modello della riga 3!
    # Se hai usato "tutti" i DOF, assicurati che la lista corrisponda esattamente.
    import Ambiente.Ambiente_claude_senza_keras as env_module


    active_dof_per_test = [0, 1, 2, 3, 4, 5, 6]  # Sostituisci con quelli che hai effettivamente usato!
    env_module.ACTIVE_DOF_INDICES = active_dof_per_test
    env_module.DOF_BOUNDS = [DOF_BOUNDS_ALL[i] for i in active_dof_per_test]

    DATABASE_DIR = Path(__file__).parent.resolve()
    DATASET_PATH = str(DATABASE_DIR / "Data" / "database.dat")

    # 2. Definisci il nuovo profilo da testare (es. riga 5 del dataset)
    df = pd.read_csv(DATASET_PATH)
    riga_nuova_idx = 3
    riga_nuova = df.iloc[riga_nuova_idx].values
    nuovo_profilo = riga_nuova[2:9].astype(np.float32).copy()
    csi_nuovo_originale = float(riga_nuova[11])

    # 3. Inserisci il nome esatto del file .zip salvato precedentemente (senza .zip)
    # Ad esempio: "ppo_blade_start_profile_DOFPITCH_DOFBETA1_lr3e-05_nsteps200"
    modello_salvato = "ppo_blade_start_profile_DOFPITCH_DOFBETA1_DOFBETA2_DOFW1_DOFW2_DOFTMOVXU_DOFTMOVXL_lr3e-05_nsteps200_riga[3]"

    print(f"\nProfilo da ottimizzare (riga {riga_nuova_idx}):")
    print(nuovo_profilo)
    print(f"CSI di partenza: {csi_nuovo_originale:.6f}")

    best_dof_inf, best_of_inf, best_csi_inf = ottimizza_con_modello_esistente(
        percorso_modello=modello_salvato,
        nuovo_profilo=nuovo_profilo,
        ep_length=20
    )

    print(f"Best DOF: {best_dof_inf}")

    miglioramento = csi_nuovo_originale - best_csi_inf
    print("\n" + "=" * 60)
    print("  RISULTATO INFERENZA PPO vs DATASET")
    print("=" * 60)
    print(f"  CSI Originale (riga {riga_nuova_idx}) : {csi_nuovo_originale:.6f}")
    print(f"  CSI Migliore PPO           : {best_csi_inf:.6f}")
    print(f"  Differenza (Delta CSI)     : {miglioramento:+.6f}")
    print("=" * 60)