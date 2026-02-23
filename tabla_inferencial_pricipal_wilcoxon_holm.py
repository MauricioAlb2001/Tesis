import os
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, shapiro, norm
from statsmodels.stats.multitest import multipletests

# =========================
# CONFIG
# =========================
BASE_DIR = "salidas_tesis"  # ajusta si aplica
FILES = {
    "Baseline": os.path.join(BASE_DIR, "results_baseline_long.csv"),
    "Tuned": os.path.join(BASE_DIR, "results_tuned_long.csv"),
}

# Métricas inferenciales:
# Principal: AP_test y ROC_AUC_test
MAIN_METRICS = ["average_precision", "roc_auc"]

# Complementaria (best vs best):
# - se elige el mejor por average_precision
# - se prueba en average_precision y roc_auc
SELECT_METRIC_FOR_BEST = "average_precision"
COMP_METRICS = ["average_precision", "roc_auc"]

# Wilcoxon direccional: H1: delta > 0
WILCOXON_ALTERNATIVE = "greater"  # una cola (Ensemble > No Ensemble)

# Salidas
OUT_DELTAS_GROUP = os.path.join(BASE_DIR, "deltas_principales_grupos_por_seed.csv")
OUT_DELTAS_BEST  = os.path.join(BASE_DIR, "deltas_principales_best_vs_best_por_seed.csv")
OUT_MAIN_GROUP   = os.path.join(BASE_DIR, "tabla_inferencial_grupos_wilcoxon_holm_1cola.csv")
OUT_MAIN_BEST    = os.path.join(BASE_DIR, "tabla_inferencial_best_vs_best_wilcoxon_holm_1cola.csv")
OUT_SHAPIRO      = os.path.join(BASE_DIR, "tabla_shapiro_deltas_principal.csv")

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

def compute_r_from_p_one_sided(p: float, n: int) -> float:
    """
    r aproximado para Wilcoxon usando z ~ N(0,1) desde p-value (one-sided).
    """
    if p <= 0 or p >= 1 or n <= 0:
        return np.nan
    z = norm.isf(p)  # one-sided
    return float(z / np.sqrt(n))

def wilcoxon_summary(deltas: np.ndarray, alternative: str) -> dict:
    d = np.array(deltas, dtype=float)
    n = int(np.sum(d != 0))  # wilcoxon ignora ceros
    # Si todos son cero, wilcoxon no aplica
    if n == 0:
        return {
            "n_seeds": len(d),
            "n_nonzero": 0,
            "delta_mean": float(np.mean(d)),
            "delta_median": float(np.median(d)),
            "wilcoxon_stat": np.nan,
            "wilcoxon_p": np.nan,
            "wilcoxon_r": np.nan,
        }

    w = wilcoxon(d, alternative=alternative)
    p = float(w.pvalue)
    r = compute_r_from_p_one_sided(p, n) if alternative in ["greater", "less"] else np.nan

    return {
        "n_seeds": len(d),
        "n_nonzero": n,
        "delta_mean": float(np.mean(d)),
        "delta_median": float(np.median(d)),
        "wilcoxon_stat": float(w.statistic),
        "wilcoxon_p": p,
        "wilcoxon_r": float(r),
    }

def build_group_deltas(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Por seed: promedio de cada métrica dentro de cada grupo.
    Delta = mean(metric|Ensemble) - mean(metric|No Ensemble)
    """
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
            raise ValueError(f"[{scenario}] No pude formar grupos para métrica {met}. Revisa columna group.")
        piv["ens_mean"] = piv["Ensemble"]
        piv["no_mean"]  = piv["No Ensemble"]
        piv["delta"] = piv["ens_mean"] - piv["no_mean"]
        piv["config"] = scenario
        piv["metric"] = met
        rows.append(piv[["seed","config","metric","ens_mean","no_mean","delta"]])

    return pd.concat(rows, ignore_index=True)

def pick_best_models_by_ap(df: pd.DataFrame) -> dict:
    """
    Elige el mejor modelo dentro de Ensemble y No Ensemble usando promedio de average_precision
    a través de seeds. Devuelve dict: {'Ensemble': model_name, 'No Ensemble': model_name}
    """
    df = df.copy()
    df["__group__"] = df["group"].map(normalize_group)

    # promedio de AP por modelo (a través de seeds)
    ap_by_model = (
        df.groupby(["__group__", "model"], as_index=False)[SELECT_METRIC_FOR_BEST]
          .mean()
          .rename(columns={SELECT_METRIC_FOR_BEST: "ap_mean"})
    )

    winners = {}
    for grp in ["Ensemble", "No Ensemble"]:
        sub = ap_by_model[ap_by_model["__group__"] == grp].copy()
        if sub.empty:
            raise ValueError(f"No encontré modelos para grupo {grp}.")
        winners[grp] = sub.sort_values("ap_mean", ascending=False).iloc[0]["model"]

    return winners

def build_best_vs_best_deltas(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Construye deltas por seed entre:
    (mejor Ensemble por AP) vs (mejor No Ensemble por AP).
    Delta = metric(best_ens) - metric(best_no)
    """
    required = {"seed", "group", "model"} | set(COMP_METRICS) | {SELECT_METRIC_FOR_BEST}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{scenario}] Faltan columnas: {missing}")

    winners = pick_best_models_by_ap(df)
    best_ens = winners["Ensemble"]
    best_no  = winners["No Ensemble"]

    rows = []
    for met in COMP_METRICS:
        ens = df[df["model"] == best_ens][["seed", met]].rename(columns={met:"ens_value"})
        no  = df[df["model"] == best_no][["seed", met]].rename(columns={met:"no_value"})

        merged = pd.merge(ens, no, on="seed", how="inner")
        merged["delta"] = merged["ens_value"] - merged["no_value"]
        merged["config"] = scenario
        merged["metric"] = met
        merged["best_ensemble_model"] = best_ens
        merged["best_no_ensemble_model"] = best_no

        rows.append(merged[["seed","config","metric","best_ensemble_model","best_no_ensemble_model","ens_value","no_value","delta"]])

    return pd.concat(rows, ignore_index=True)

# =========================
# MAIN
# =========================
all_group_deltas = []
all_best_deltas = []
shapiro_rows = []

for scenario, path in FILES.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontré: {path}")

    df = pd.read_csv(path)

    # 1) Deltas por grupos
    dgroup = build_group_deltas(df, scenario)
    all_group_deltas.append(dgroup)

    # 2) Deltas best vs best
    dbest = build_best_vs_best_deltas(df, scenario)
    all_best_deltas.append(dbest)

# Concatenar
group_deltas = pd.concat(all_group_deltas, ignore_index=True).sort_values(["config","metric","seed"])
best_deltas  = pd.concat(all_best_deltas,  ignore_index=True).sort_values(["config","metric","seed"])

group_deltas.to_csv(OUT_DELTAS_GROUP, index=False)
best_deltas.to_csv(OUT_DELTAS_BEST, index=False)

print(f"[OK] deltas grupos:   {OUT_DELTAS_GROUP}")
print(f"[OK] deltas best:     {OUT_DELTAS_BEST}")

# -------------------------
# Shapiro (informativo) sobre deltas de la parte principal (grupos)
# -------------------------
for (cfg, met), g in group_deltas.groupby(["config","metric"]):
    d = g["delta"].to_numpy()
    p_sh = float(shapiro(d).pvalue) if len(d) >= 3 else np.nan
    shapiro_rows.append({"config": cfg, "metric": met, "n_seeds": len(d), "shapiro_p": p_sh})

shapiro_df = pd.DataFrame(shapiro_rows)
shapiro_df.to_csv(OUT_SHAPIRO, index=False)
print(f"[OK] Shapiro (informativo): {OUT_SHAPIRO}")

# -------------------------
# Inferencial principal: grupos
# -------------------------
rows = []
pvals = []

for (cfg, met), g in group_deltas.groupby(["config","metric"]):
    d = g["delta"].to_numpy()
    s = wilcoxon_summary(d, alternative=WILCOXON_ALTERNATIVE)

    rows.append({
        "config": cfg,
        "metric": met,
        **s
    })
    pvals.append(s["wilcoxon_p"])

group_table = pd.DataFrame(rows)

# Holm sobre las 4 pruebas principales
group_table["wilcoxon_p_adj_holm"] = multipletests(
    group_table["wilcoxon_p"].to_numpy(), alpha=0.05, method="holm"
)[1]
group_table["alternative"] = WILCOXON_ALTERNATIVE
group_table = group_table.sort_values(["config","metric"])
group_table.to_csv(OUT_MAIN_GROUP, index=False)

print(f"[OK] Tabla inferencial (grupos): {OUT_MAIN_GROUP}")
print(group_table)

# -------------------------
# Inferencial complementaria: best vs best (mismo bloque de 4 pruebas)
# -------------------------
rows2 = []
pvals2 = []

# Nota: en best_deltas hay columnas extras, pero agrupamos por config/metric
for (cfg, met), g in best_deltas.groupby(["config","metric"]):
    d = g["delta"].to_numpy()
    s = wilcoxon_summary(d, alternative=WILCOXON_ALTERNATIVE)

    # modelos ganadores (son constantes por escenario)
    best_ens = g["best_ensemble_model"].iloc[0]
    best_no  = g["best_no_ensemble_model"].iloc[0]

    rows2.append({
        "config": cfg,
        "metric": met,
        "best_ensemble_model": best_ens,
        "best_no_ensemble_model": best_no,
        **s
    })
    pvals2.append(s["wilcoxon_p"])


best_table = pd.DataFrame(rows2)
best_table["wilcoxon_p_adj_holm"] = multipletests(
    best_table["wilcoxon_p"].to_numpy(), alpha=0.05, method="holm"
)[1]
best_table["alternative"] = WILCOXON_ALTERNATIVE
best_table = best_table.sort_values(["config","metric"])
best_table.to_csv(OUT_MAIN_BEST, index=False)

print(f"[OK] Tabla inferencial (best vs best): {OUT_MAIN_BEST}")
print(best_table)
