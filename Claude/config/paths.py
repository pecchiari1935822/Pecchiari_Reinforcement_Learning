"""
config/paths.py
===============
Gestisce TUTTI i percorsi del progetto in modo centralizzato.
Evita hardcoding di path nel codice.
"""

from pathlib import Path
import os

# ============================================================
# ROOT DIRECTORIES
# ============================================================

# Cartella principale del progetto
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Cartelle principali
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# ============================================================
# DATA FILES
# ============================================================

DATASET_PATH = DATA_DIR / "database.dat"

# ============================================================
# MODEL FILES
# ============================================================

SURROGATE_MODEL_PATH = MODELS_DIR / "best_model.keras"
SCALERS_PATH = MODELS_DIR / "scalers.joblib"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"

# ============================================================
# OUTPUT FILES
# ============================================================

PRESENTATION_TEMPLATE_PATH = PROJECT_ROOT / "Template.pptx"
PRESENTATION_OUTPUT_PATH = OUTPUT_DIR / "Report_Simulazioni_PPO.pptx"

# Plot outputs
PLOT_RESULTS_PATH = OUTPUT_DIR / "plot_results.png"
PLOT_METRICS_ACTOR_PATH = OUTPUT_DIR / "plot_metrics_actor.png"
PLOT_METRICS_CRITIC_PATH = OUTPUT_DIR / "plot_metrics_critic.png"


# ============================================================
# CHECKPOINT DIRECTORIES
# ============================================================

def get_checkpoint_dir(dof_names: list, suffix: str = "") -> Path:
    """
    Crea e restituisce il path per checkpoint specifici.

    Args:
        dof_names: Lista di nomi DOF (es. ["PITCH", "BETA1"])
        suffix: Suffisso aggiuntivo (es. "generale" o "start_profile")

    Returns:
        Path alla cartella di checkpoint
    """
    import re
    safe_names = [re.sub(r'[^0-9A-Za-z]+', '', n) for n in dof_names]
    active_tag = "_".join(safe_names) if safe_names else "ALL"

    dir_name = f"checkpoints_{active_tag}"
    if suffix:
        dir_name = f"checkpoints_{suffix}_{active_tag}"

    checkpoint_path = CHECKPOINTS_DIR / dir_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


def get_log_dir(dof_names: list, suffix: str = "") -> Path:
    """
    Crea e restituisce il path per tensorboard logs specifici.
    """
    import re
    safe_names = [re.sub(r'[^0-9A-Za-z]+', '', n) for n in dof_names]
    active_tag = "_".join(safe_names) if safe_names else "ALL"

    dir_name = f"logs_{active_tag}"
    if suffix:
        dir_name = f"logs_{suffix}_{active_tag}"

    log_path = LOGS_DIR / dir_name
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def get_model_save_path(dof_names: list, learning_rate: float,
                        n_steps: int, suffix: str = "") -> str:
    """
    Crea il nome file per salvare il modello PPO.
    Restituisce il path SENZA .zip (SB3 lo aggiunge automaticamente).
    """
    import re
    safe_names = [re.sub(r'[^0-9A-Za-z]+', '', n) for n in dof_names]
    active_tag = "_".join(safe_names) if safe_names else "ALL"

    base_name = f"ppo_blade_{active_tag}_lr{learning_rate}_nsteps{n_steps}"
    if suffix:
        base_name = f"ppo_blade_{suffix}_{active_tag}_lr{learning_rate}_nsteps{n_steps}"

    return str(CHECKPOINTS_DIR / base_name)


# ============================================================
# ENSURE DIRECTORIES EXIST
# ============================================================

def ensure_directories():
    """Crea tutte le directory necessarie al primo run."""
    directories = [
        DATA_DIR,
        MODELS_DIR,
        OUTPUT_DIR,
        LOGS_DIR,
        CHECKPOINTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Crea directory al primo import
ensure_directories()


# ============================================================
# VALIDATION
# ============================================================

def validate_required_files():
    """Verifica che i file necessari esistano."""
    required_files = [
        (DATASET_PATH, "Database dataset"),
        (SURROGATE_MODEL_PATH, "Surrogate model (neural network)"),
        (SCALERS_PATH, "Scalers for normalization"),
    ]

    missing = []
    for file_path, description in required_files:
        if not file_path.exists():
            missing.append(f"  ❌ {description}: {file_path}")

    if missing:
        print("\n⚠️  FILE MANCANTI:\n")
        print("\n".join(missing))
        print("\nVerifica che i file siano nella cartella 'data/models/'")
        return False

    print("\n✅ Tutti i file richiesti trovati!")
    return True


# Test al primo import
if __name__ == "__main__":
    validate_required_files()