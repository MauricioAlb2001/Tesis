import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) Cargar datos
# =========================
df = pd.read_csv("resultados_iteraciones_baseline_tuned.csv")

# =========================
# 1) Normalizar columnas clave
# =========================
model_col = "Modelo"
label_col = "Etiqueta"
seed_col  = "Seed"

metric_cols = [
    "Exactitud","Precision","Sensibilidad","F1",
    "AUC_test","AUC_CV_mean","AUC_CV_std"
]
conf_cols = ["TP","TN","FP","FN"]

# Asegurar numéricos
for c in metric_cols + conf_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Limpiar filas inválidas
df = df.dropna(subset=[model_col, label_col, seed_col] + ["AUC_test"])

# Normalizar etiqueta (por si viene Baseline/Tuned con mayúsculas)
df[label_col] = df[label_col].astype(str).str.strip().str.lower()

# =========================
# 2) Verificación rápida (consistencia del experimento)
# =========================
seeds_per_group = df.groupby([model_col, label_col])[seed_col].nunique().reset_index(name="n_seeds")
print("Seeds por (Modelo, Etiqueta):")
print(seeds_per_group.sort_values(["n_seeds", model_col, label_col]))

# =========================
# 3) Tabla resumen mean±std (Modelo x Etiqueta)
# =========================
agg = df.groupby([model_col, label_col])[metric_cols].agg(["mean","std"])
agg.columns = [f"{m}_{s}" for m, s in agg.columns]
summary = agg.reset_index()

# Formato mean±std (para pegar en tesis)
pretty = summary[[model_col, label_col]].copy()
for m in metric_cols:
    pretty[m] = summary[f"{m}_mean"].map(lambda x: f"{x:.4f}") + " ± " + summary[f"{m}_std"].map(lambda x: f"{x:.4f}")

pretty.to_csv("F5_tabla_resumen_mean_std.csv", index=False)
summary.to_csv("F5_tabla_resumen_numeric.csv", index=False)

# =========================
# 4) Tabla delta Tuned - Baseline (por modelo)
# =========================
base = summary[summary[label_col]=="baseline"].set_index(model_col)
tune = summary[summary[label_col]=="tuned"].set_index(model_col)

common_models = base.index.intersection(tune.index)

delta = pd.DataFrame({model_col: common_models}).set_index(model_col)
for m in metric_cols:
    delta[f"Delta_{m}"] = tune.loc[common_models, f"{m}_mean"] - base.loc[common_models, f"{m}_mean"]

delta = delta.reset_index()
delta.to_csv("F5_tabla_delta_tuned_minus_baseline.csv", index=False)

# =========================
# 5) Ranking (Top) por AUC_test -> F1 -> Sensibilidad
# =========================
rank = summary.copy()
rank = rank.sort_values(
    by=["AUC_test_mean","F1_mean","Sensibilidad_mean","AUC_test_std"],
    ascending=[False, False, False, True]
).reset_index(drop=True)

rank.to_csv("F5_ranking_configuraciones.csv", index=False)

# =========================
# 6) Gráficos (baseline vs tuned) usando AUC_test
# =========================
# 6.1 Barras mean±std por modelo
plot_df = summary.pivot(index=model_col, columns=label_col, values=["AUC_test_mean","AUC_test_std"])
plot_df.columns = [f"{a}_{b}" for a,b in plot_df.columns]
plot_df = plot_df.dropna()

models = plot_df.index.tolist()
x = np.arange(len(models))
w = 0.38

plt.figure()
plt.bar(x - w/2, plot_df["AUC_test_mean_baseline"], yerr=plot_df["AUC_test_std_baseline"], capsize=3, label="baseline")
plt.bar(x + w/2, plot_df["AUC_test_mean_tuned"],    yerr=plot_df["AUC_test_std_tuned"],    capsize=3, label="tuned")
plt.xticks(x, models, rotation=45, ha="right")
plt.ylabel("AUC_test")
plt.title("AUC_test (mean ± std) por modelo: Baseline vs Tuned")
plt.legend()
plt.tight_layout()
plt.savefig("F5_bar_AUC_test_mean_std.png", dpi=200)
plt.close()

# 6.2 Boxplot por modelo (baseline vs tuned)
plt.figure(figsize=(max(8, len(models)*0.8), 5))
data = []
labels = []
for m in models:
    b = df[(df[model_col]==m) & (df[label_col]=="baseline")]["AUC_test"].dropna().values
    t = df[(df[model_col]==m) & (df[label_col]=="tuned")]["AUC_test"].dropna().values
    data.extend([b, t])
    labels.extend([f"{m}\nbase", f"{m}\ntuned"])

plt.boxplot(data, labels=labels, showfliers=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel("AUC_test")
plt.title("Distribución AUC_test (10 seeds) Baseline vs Tuned")
plt.tight_layout()
plt.savefig("F5_boxplot_AUC_test.png", dpi=200)
plt.close()

# 6.3 Delta por modelo
d = delta.set_index(model_col)["Delta_AUC_test"].reindex(models)

plt.figure()
plt.bar(np.arange(len(models)), d.values)
plt.xticks(np.arange(len(models)), models, rotation=45, ha="right")
plt.ylabel("Δ AUC_test (tuned - baseline)")
plt.title("Mejora por tuning: ΔAUC_test por modelo")
plt.tight_layout()
plt.savefig("F5_delta_AUC_test.png", dpi=200)
plt.close()

print("Listo: tablas y gráficos exportados (F5_*.csv / F5_*.png)")
