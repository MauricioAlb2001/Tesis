import os
import numpy as np
import pandas as pd

from scipy.stats import shapiro, wilcoxon, ttest_rel
from sklearn.metrics import precision_recall_curve, roc_curve, auc

import matplotlib.pyplot as plt


OUT_DIR = "salidas_tesis"
FIG_DIR = os.path.join(OUT_DIR, "figuras")
TAB_DIR = os.path.join(OUT_DIR, "tablas_fase5")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

BASELINE_LONG = os.path.join(OUT_DIR, "results_baseline_long.csv")
TUNED_LONG    = os.path.join(OUT_DIR, "results_tuned_long.csv")

DELTAS_BASE = os.path.join(OUT_DIR, "deltas_baseline_ens_minus_no.csv")
DELTAS_TUN  = os.path.join(OUT_DIR, "deltas_tuned_ens_minus_no.csv")

CURVES_BASE = os.path.join(OUT_DIR, "curves_seed42_baseline.csv")
CURVES_TUN  = os.path.join(OUT_DIR, "curves_seed42_tuned.csv")

# Métricas (las que definimos como principales/secundarias para inferencial)
INFER_METRICS = ["average_precision", "roc_auc", "f1", "recall"]

# Nombres “bonitos” para reportes
METRIC_LABELS = {
    "average_precision": "AP (PR-AUC)",
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "recall": "Recall (Sensibilidad)"
}


# -------------------------
# Corrección Holm (p-values)
# -------------------------
def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)

    prev = 0.0
    for i, idx in enumerate(order):
        rank = i + 1
        adj = (m - i) * pvals[idx]
        adj = max(adj, prev)   # monotonicidad
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted


# -------------------------
# Tamaños de efecto
# -------------------------
def cohens_d_paired(x, y):
    # d = mean(diff) / std(diff)
    diff = np.asarray(x) - np.asarray(y)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))

def wilcoxon_r_from_z(z, n):
    # r = |z| / sqrt(n)
    return float(abs(z) / np.sqrt(n))


# -------------------------
# Tablas descriptivas
# -------------------------
def summary_by_model(df_long):
    metric_cols = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    rows = []
    for (cfg, model, grp), sub in df_long.groupby(["config", "model", "group"]):
        row = {"config": cfg, "model": model, "group": grp, "n_seeds": sub["seed"].nunique()}
        for m in metric_cols:
            vals = sub[m].values
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"]  = float(np.std(vals, ddof=1))
            row[f"{m}_median"] = float(np.median(vals))
            row[f"{m}_q1"] = float(np.quantile(vals, 0.25))
            row[f"{m}_q3"] = float(np.quantile(vals, 0.75))
        rows.append(row)
    return pd.DataFrame(rows)

def summary_by_group(df_long):
    metric_cols = ["balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    rows = []
    for (cfg, grp), sub in df_long.groupby(["config", "group"]):
        for m in metric_cols:
            rows.append({
                "config": cfg, "group": grp, "metric": m,
                "mean": float(sub[m].mean()),
                "std":  float(sub[m].std(ddof=1)),
                "median": float(sub[m].median())
            })
    return pd.DataFrame(rows)


# -------------------------
# Inferencial desde deltas
# -------------------------
def inferential_from_deltas(deltas_path, config_name):
    df = pd.read_csv(deltas_path)

    rows = []
    pvals = []
    # Primero calculamos todas las pruebas, luego Holm
    for metric in INFER_METRICS:
        d = df[df["metric"] == metric]["delta_ens_minus_no"].values

        n = len(d)
        sh_p = float(shapiro(d).pvalue) if n >= 3 else np.nan

        # Wilcoxon (pareado sobre delta vs 0)
        # SciPy devuelve statistic y pvalue. Para efecto r necesitamos z:
        # En scipy>=1.10: wilcoxon(..., method="approx") devuelve zstat en res.zstatistic
        w_res = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="approx")
        w_p = float(w_res.pvalue)
        z = float(getattr(w_res, "zstatistic", np.nan))
        w_r = wilcoxon_r_from_z(z, n) if np.isfinite(z) else np.nan

        # t-test pareado (complementario): delta vs 0 equivale a ttest_1samp,
        # pero usamos ttest_rel comparando d con 0 vector para mantener idea pareada.
        t_res = ttest_rel(d, np.zeros_like(d))
        t_p = float(t_res.pvalue)
        d_cohen = cohens_d_paired(d, np.zeros_like(d))

        rows.append({
            "config": config_name,
            "metric": metric,
            "n_seeds": n,
            "delta_mean": float(np.mean(d)),
            "delta_median": float(np.median(d)),
            "shapiro_p": sh_p,
            "wilcoxon_p": w_p,
            "wilcoxon_r": w_r,
            "ttest_p": t_p,
            "cohens_d_paired": d_cohen
        })
        pvals.append(w_p)

    # Holm sobre Wilcoxon
    p_adj = holm_adjust(pvals)
    for i in range(len(rows)):
        rows[i]["wilcoxon_p_adj_holm"] = float(p_adj[i])

    out = pd.DataFrame(rows)
    return out


# -------------------------
# Gráficos
# -------------------------
def bar_mean_std(df_long, metric, config_name, outpath):
    sub = df_long[df_long["config"] == config_name].copy()
    agg = sub.groupby(["model", "group"])[metric].agg(["mean", "std"]).reset_index()
    agg = agg.sort_values("mean", ascending=False)

    plt.figure()
    x = np.arange(len(agg))
    plt.bar(x, agg["mean"].values)
    plt.errorbar(x, agg["mean"].values, yerr=agg["std"].values, fmt="none", capsize=4)
    plt.xticks(x, agg["model"].values, rotation=45, ha="right")
    plt.title(f"{METRIC_LABELS.get(metric, metric)} - {config_name} (mean ± std)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def boxplot_by_model(df_long, metric, config_name, outpath):
    sub = df_long[df_long["config"] == config_name].copy()

    # Orden por media
    order = sub.groupby("model")[metric].mean().sort_values(ascending=False).index.tolist()

    plt.figure()
    data = [sub[sub["model"] == m][metric].values for m in order]
    plt.boxplot(data, labels=order, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{METRIC_LABELS.get(metric, metric)} - {config_name} (Boxplot por seed)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def delta_plot(deltas_path, metric, config_name, outpath):
    df = pd.read_csv(deltas_path)
    d = df[df["metric"] == metric].sort_values("seed")

    plt.figure()
    plt.axhline(0, linestyle="--")
    plt.plot(d["seed"].values, d["delta_ens_minus_no"].values, marker="o")
    plt.title(f"Δ(Ensemble − No) por seed | {METRIC_LABELS.get(metric, metric)} - {config_name}")
    plt.xlabel("Seed")
    plt.ylabel("Delta")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def curves_plot(curves_csv, config_name, out_roc, out_pr):
    df = pd.read_csv(curves_csv)

    # ROC
    plt.figure()
    for model, sub in df.groupby("model"):
        y_true = sub["y_true"].values
        y_score = sub["y_score"].values
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model} (AUC={roc_auc_val:.3f})")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.title(f"ROC - Seed 42 ({config_name})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_roc, dpi=200)
    plt.close()

    # PR
    plt.figure()
    for model, sub in df.groupby("model"):
        y_true = sub["y_true"].values
        y_score = sub["y_score"].values
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc_val = auc(rec, prec)
        plt.plot(rec, prec, label=f"{model} (AUC={pr_auc_val:.3f})")
    plt.title(f"Precision-Recall - Seed 42 ({config_name})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pr, dpi=200)
    plt.close()


def main():
    # --- Leer resultados ---
    df_base = pd.read_csv(BASELINE_LONG)
    df_tun  = pd.read_csv(TUNED_LONG)

    # --- Tablas descriptivas ---
    desc_models = pd.concat([summary_by_model(df_base), summary_by_model(df_tun)], ignore_index=True)
    desc_models.to_csv(os.path.join(TAB_DIR, "tabla_descriptiva_por_modelo.csv"), index=False)

    desc_groups = pd.concat([summary_by_group(df_base), summary_by_group(df_tun)], ignore_index=True)
    desc_groups.to_csv(os.path.join(TAB_DIR, "tabla_descriptiva_por_grupo.csv"), index=False)

    # --- Inferencial (Wilcoxon + Holm + r) ---
    inf_base = inferential_from_deltas(DELTAS_BASE, "Baseline")
    inf_tun  = inferential_from_deltas(DELTAS_TUN,  "Tuned")

    infer_all = pd.concat([inf_base, inf_tun], ignore_index=True)
    infer_all.to_csv(os.path.join(TAB_DIR, "tabla_inferencial_wilcoxon_holm.csv"), index=False)

    # --- Gráficos completos (baseline + tuned) ---
    for cfg, df_long in [("Baseline", df_base), ("Tuned", df_tun)]:
        # Barras mean±std (AP, F1)
        bar_mean_std(df_long, "average_precision", cfg, os.path.join(FIG_DIR, f"bar_{cfg}_AP_mean_std.png"))
        bar_mean_std(df_long, "f1", cfg, os.path.join(FIG_DIR, f"bar_{cfg}_F1_mean_std.png"))

        # Boxplots (AP, F1)
        boxplot_by_model(df_long, "average_precision", cfg, os.path.join(FIG_DIR, f"box_{cfg}_AP.png"))
        boxplot_by_model(df_long, "f1", cfg, os.path.join(FIG_DIR, f"box_{cfg}_F1.png"))

        # Delta plots (AP, F1) - inferencial visual
        deltas_path = DELTAS_BASE if cfg == "Baseline" else DELTAS_TUN
        delta_plot(deltas_path, "average_precision", cfg, os.path.join(FIG_DIR, f"delta_{cfg}_AP.png"))
        delta_plot(deltas_path, "f1", cfg, os.path.join(FIG_DIR, f"delta_{cfg}_F1.png"))

        # (Opcional) Delta también para ROC-AUC y Recall
        delta_plot(deltas_path, "roc_auc", cfg, os.path.join(FIG_DIR, f"delta_{cfg}_ROC_AUC.png"))
        delta_plot(deltas_path, "recall", cfg, os.path.join(FIG_DIR, f"delta_{cfg}_Recall.png"))

    # --- Curvas ROC/PR seed 42 (2 modelos) ---
    curves_plot(CURVES_BASE, "Baseline",
                os.path.join(FIG_DIR, "roc_seed42_baseline.png"),
                os.path.join(FIG_DIR, "pr_seed42_baseline.png"))

    curves_plot(CURVES_TUN, "Tuned",
                os.path.join(FIG_DIR, "roc_seed42_tuned.png"),
                os.path.join(FIG_DIR, "pr_seed42_tuned.png"))

    print("\nLISTO ✅")
    print("Tablas en:", TAB_DIR)
    print("Figuras en:", FIG_DIR)


if __name__ == "__main__":
    main()
