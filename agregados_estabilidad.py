import os
import pandas as pd

BASE_DIR = "salidas_tesis"  # ajusta si tu carpeta se llama distinto

files = {
    "Baseline": os.path.join(BASE_DIR, "results_baseline_long.csv"),
    "Tuned": os.path.join(BASE_DIR, "results_tuned_long.csv"),
}

out_files = {
    "Baseline": os.path.join(BASE_DIR, "robustez_ap_group_summary_baseline.csv"),
    "Tuned": os.path.join(BASE_DIR, "robustez_ap_group_summary_tuned.csv"),
}

def normalize_group(x: str) -> str:
    s = str(x).strip().lower()
    if "no" in s:
        return "No Ensemble"
    if "ens" in s:
        return "Ensemble"
    return str(x).strip()

def summarize_robustez_ap(df: pd.DataFrame, config_name: str) -> pd.DataFrame:
    # Validación mínima
    required = {"seed", "group", "average_precision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{config_name}] Faltan columnas en el long: {missing}")

    df = df.copy()
    df["group"] = df["group"].map(normalize_group)

    # 1) AP por seed y grupo (promedio si hubiera más de una fila por seed-grupo)
    ap_seed = (
        df.groupby(["group", "seed"], as_index=False)["average_precision"]
          .mean()
          .rename(columns={"average_precision": "ap_test"})
    )

    # 2) Robustez entre seeds: std + cuartiles
    summary = (
        ap_seed.groupby("group")["ap_test"]
        .agg(
            ap_mean="mean",
            ap_median="median",
            ap_std="std",
            ap_q1=lambda x: x.quantile(0.25),
            ap_q3=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    summary["ap_iqr"] = summary["ap_q3"] - summary["ap_q1"]
    summary.insert(0, "config", config_name)
    summary.insert(2, "n_seeds", ap_seed["seed"].nunique())

    # Orden columnas para tabla
    summary = summary[[
        "config", "group", "n_seeds",
        "ap_mean", "ap_median",
        "ap_std", "ap_q1", "ap_q3", "ap_iqr"
    ]]

    return summary.sort_values("group")

for cfg, path in files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontré el archivo: {path}")

    df = pd.read_csv(path)
    tab = summarize_robustez_ap(df, cfg)

    tab.to_csv(out_files[cfg], index=False)
    print(f"[OK] Tabla robustez AP por grupos ({cfg}) guardada en: {out_files[cfg]}")
    print(tab.round(6))
