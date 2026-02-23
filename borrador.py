import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE_DIR = "salidas_tesis"
FILES = {
    "Baseline": os.path.join(BASE_DIR, "deltas_baseline_ens_minus_no.csv"),
    "Tuned": os.path.join(BASE_DIR, "deltas_tuned_ens_minus_no.csv"),
}

OUT_DIR = os.path.join(BASE_DIR, "figuras")
os.makedirs(OUT_DIR, exist_ok=True)

# Ajustes globales de legibilidad (puedes subir/bajar)
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# =========================
# HELPERS
# =========================
def detect_delta_column(df: pd.DataFrame) -> str:
    """Encuentra la columna numérica que representa el delta."""
    # candidatos típicos
    candidates = [c for c in df.columns if "delta" in c.lower()]
    # si hay candidatos con 'delta', elige el primero numérico
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c

    # fallback: busca cualquier columna numérica con varianza > 0
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No encontré columnas numéricas (posible problema de lectura del CSV).")
    # elige la numérica que parezca 'valor'
    # (evita seed si existe)
    for c in numeric_cols:
        if c.lower() not in ["seed", "seeds", "random_state", "iter", "iteration"]:
            return c
    return numeric_cols[0]

def build_comparison_label(df: pd.DataFrame) -> pd.Series:
    """Construye una etiqueta por comparación para agrupar boxplots."""
    # 1) si ya existe una columna de comparación
    for col in ["comparison", "pair", "model_pair", "comparacion", "par"]:
        if col in df.columns:
            return df[col].astype(str)

    # 2) si existen columnas de modelos (ens vs no)
    ens_candidates = ["ensemble_model", "ens_model", "model_ensemble", "ens"]
    no_candidates  = ["no_ensemble_model", "no_model", "model_no_ensemble", "no"]

    ens_col = next((c for c in ens_candidates if c in df.columns), None)
    no_col  = next((c for c in no_candidates  if c in df.columns), None)

    if ens_col and no_col:
        return df[ens_col].astype(str) + " − " + df[no_col].astype(str)

    # 3) fallback: si existe 'model'
    if "model" in df.columns:
        return df["model"].astype(str)

    # 4) sin info: etiqueta única
    return pd.Series(["ΔAP"] * len(df))

def plot_boxplot_delta(df: pd.DataFrame, scenario: str, out_path: str):
    delta_col = detect_delta_column(df)
    df = df.copy()
    df["comparison_label"] = build_comparison_label(df)

    # Ordenar comparaciones por mediana (opcional, ayuda a leer)
    order = (
        df.groupby("comparison_label")[delta_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    data = [df.loc[df["comparison_label"] == lbl, delta_col].dropna().values for lbl in order]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(order) + 2)))

    ax.boxplot(
        data,
        vert=False,
        showmeans=True,
        meanline=False,
        whis=1.5,
    )

    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order)

    # Línea de referencia en 0
    ax.axvline(0, linewidth=1)

    ax.set_title(f"ΔAP_test por seed (Ensemble − No Ensemble) — {scenario}")
    ax.set_xlabel("ΔAP_test (Average Precision)")

    # Mostrar a 2 decimales en el eje X (más legible)
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Guardado: {out_path}")
    print(f"     Usé delta_col = '{delta_col}' | #filas = {len(df)} | #comparaciones = {len(order)}")

# =========================
# MAIN
# =========================
for scenario, fpath in FILES.items():
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No encontré el archivo: {fpath}")

    df = pd.read_csv(fpath)

    # Filtrar métrica average_precision si existe columna 'metric'
    if "metric" in df.columns:
        df = df[df["metric"].astype(str).str.lower() == "average_precision"].copy()

    out_path = os.path.join(OUT_DIR, f"boxplot_deltaAP_{scenario.lower()}.png")
    plot_boxplot_delta(df, scenario, out_path)

print("\nListo. Revisa la carpeta:", OUT_DIR)
