from pathlib import Path
import joblib


data_dir = Path(__file__).parent.parent.resolve()
nuovi_modelli_dir = data_dir


scaler_rete = {
        "X_GLOBAL": str(nuovi_modelli_dir / "scalers" / "scaler_DOF.joblib"),
        "ALFA_EX" : str(nuovi_modelli_dir / "scalers" / "scaler_OF_alfa_ex_FL.joblib"),
        "CPT": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Cpt_FL.joblib"),
        "CSI": str(nuovi_modelli_dir / "scalers" / "scaler_OF_CSI_FL.joblib"),
        "PSI": str(nuovi_modelli_dir / "scalers" / "scaler_OF_PSI_FL.joblib"),
        "PHI": str(nuovi_modelli_dir / "scalers" / "scaler_OF_PHI_FL.joblib"),
        "ZWC": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Zwc_FL.joblib"),
        "DFSS_MIS": str(nuovi_modelli_dir / "scalers" / "scaler_OF_DFss_Mis_LO.joblib"),
        "DS_CP": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Ds_cp_LO.joblib"),
        "TMAX": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Tmax_GM.joblib"),
        "X_TMAX": str(nuovi_modelli_dir / "scalers" / "scaler_OF_X_Tmax_GM.joblib"),
        "WEDGE_TE": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Wedge_TE_GM.joblib"),
        "UGT": str(nuovi_modelli_dir / "scalers" / "scaler_OF_UGT_GM.joblib"),
        "Area": str(nuovi_modelli_dir / "scalers" / "scaler_OF_Area_GM.joblib")
    }

for nome, path in scaler_rete.items():
    obj = joblib.load(path)

    print(f"\n===== {nome} =====")
    print("Tipo:", obj.__class__.__name__)

    # Pipeline
    if hasattr(obj, "steps"):
        for step_name, step in obj.steps:
            print(f"  {step_name}: {step.__class__.__name__}")

    # ColumnTransformer
    elif hasattr(obj, "transformers_"):
        for name, transformer, columns in obj.transformers_:
            print(f"\n  {name} -> colonne {columns}")

            if hasattr(transformer, "steps"):
                for step_name, step in transformer.steps:
                    print(f"    {step_name}: {step.__class__.__name__}")
            else:
                print(f"    {transformer.__class__.__name__}")