import os
import numpy as np
import pandas as pd

from scipy.stats import shapiro, ttest_1samp
from statsmodels.stats.multitest import multipletests

# =========================
# CONFIG
# =========================
BASE_DIR = "salidas_tesis"

FILES = {
    "Baseline": os.path.join(BASE_DIR, "results_baseline_long.csv"),
    "Tuned": os.path.join(BASE_DIR, "results_tuned_long.csv"),
}

MAIN_METRICS = ["average_precision", "roc_auc"]

OUT_DELTAS_GROUP = os.path.join(BASE_DIR, "deltas_principales_grupos_por_seed.csv")
OUT_SHAPIRO = os.path.join(BASE_DIR, "tabla_shapiro_deltas_principal.csv")
OUT_TTEST = os.path.join(BASE_DIR, "tabla_inferencial_t_student_grupos_holm_1cola.csv")

ALPHA = 0.05
ALTERNATIVE = "greater"  # H1: μΔ > 0

# =========================
# HELPERS
# =========================

def normalize_group(x: str) -> str:
    s = str(x).strip().lower()
    if "ens" in s and "no" not in s:
        return "Ensemble"
    if "no" in s:
        return "No Ensemble"
    return str(x).strip()

def build_group_deltas(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    required = {"seed", "group"} | set(MAIN_METRICS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{scenario}] Faltan columnas: {missing}")

    df = df.copy()
    df["__group__"] = df["group"].map(normalize_group)

    agg = df.groupby(["seed", "__group__"], as_index=False)[MAIN_METRICS].mean()

    rows = []
    for met in MAIN_METRICS:
        piv = agg.pivot(index="seed", columns="__group__", values=met).reset_index()

        if "Ensemble" not in piv.columns or "No Ensemble" not in piv.columns:
            raise ValueError(f"[{scenario}] No pude formar grupos correctamente.")

        piv["delta"] = piv["Ensemble"] - piv["No Ensemble"]
        piv["config"] = scenario
        piv["metric"] = met

        rows.append(piv[["seed", "config", "metric", "delta"]])

    return pd.concat(rows, ignore_index=True)

def compute_t_test_summary(deltas: np.ndarray):
    d = np.array(deltas, dtype=float)
    n = len(d)
    dfree = n - 1

    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1))

    t_res = ttest_1samp(d, popmean=0.0, alternative=ALTERNATIVE)

    t_stat = float(t_res.statistic)
    p_val = float(t_res.pvalue)

    # Tamaño de efecto Cohen dz
    dz = mean_d / std_d if std_d != 0 else np.nan

    return {
        "n_seeds": n,
        "delta_mean": mean_d,
        "delta_std": std_d,
        "t_stat": t_stat,
        "df": dfree,
        "p_value": p_val,
        "cohens_dz": dz
    }

# =========================
# MAIN
# =========================

all_deltas = []
shapiro_rows = []

for scenario, path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontré: {path}")

    df = pd.read_csv(path)

    dgroup = build_group_deltas(df, scenario)
    all_deltas.append(dgroup)

group_deltas = pd.concat(all_deltas, ignore_index=True)
group_deltas = group_deltas.sort_values(["config", "metric", "seed"])

group_deltas.to_csv(OUT_DELTAS_GROUP, index=False)
print(f"[OK] deltas grupos: {OUT_DELTAS_GROUP}")

# -------------------------
# Shapiro (diagnóstico)
# -------------------------
for (cfg, met), g in group_deltas.groupby(["config", "metric"]):
    d = g["delta"].to_numpy()
    p_sh = float(shapiro(d).pvalue) if len(d) >= 3 else np.nan

    shapiro_rows.append({
        "config": cfg,
        "metric": met,
        "n_seeds": len(d),
        "shapiro_p": p_sh
    })

shapiro_df = pd.DataFrame(shapiro_rows)
shapiro_df.to_csv(OUT_SHAPIRO, index=False)
print(f"[OK] Shapiro: {OUT_SHAPIRO}")

# -------------------------
# t-Student (una cola)
# -------------------------
rows = []

for (cfg, met), g in group_deltas.groupby(["config", "metric"]):
    d = g["delta"].to_numpy()
    summary = compute_t_test_summary(d)

    rows.append({
        "config": cfg,
        "metric": met,
        **summary
    })

ttest_table = pd.DataFrame(rows)

# Ajuste Holm
ttest_table["p_value_adj_holm"] = multipletests(
    ttest_table["p_value"].to_numpy(),
    alpha=ALPHA,
    method="holm"
)[1]

ttest_table = ttest_table.sort_values(["config", "metric"])
ttest_table.to_csv(OUT_TTEST, index=False)

print(f"[OK] Tabla inferencial t-Student: {OUT_TTEST}")
print(ttest_table)