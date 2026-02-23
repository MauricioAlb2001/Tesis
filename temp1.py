import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier


# =========================
# CONFIG
# =========================
SEED_REF = 42
DATA_PATH = "GiveMeSomeCredit/cs-training.csv"     # ajusta si tu ruta cambia
SEP = ";"
INDEX_COL = 0

BEST_PARAMS_PATH = "salidas_tesis/best_params.json"
OUT_DIR = "salidas_tesis/figuras"


# =========================
# PIPELINE (igual a tu fase 3)
# =========================
def preparar_datos_local(seed: int):
    df = pd.read_csv(DATA_PATH, sep=SEP, index_col=INDEX_COL)

    y = df["SeriousDlqin2yrs"]
    X = df.drop(columns=["SeriousDlqin2yrs"])

    # 70/30 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    # age=0 -> NA
    X_train.loc[X_train["age"] == 0, "age"] = pd.NA
    X_test.loc[X_test["age"] == 0, "age"] = pd.NA

    # imputación mediana (fit en train)
    med_income = X_train["MonthlyIncome"].median()
    med_dep = X_train["NumberOfDependents"].median()
    med_age = X_train["age"].median()

    X_train["MonthlyIncome"] = X_train["MonthlyIncome"].fillna(med_income)
    X_test["MonthlyIncome"] = X_test["MonthlyIncome"].fillna(med_income)

    X_train["NumberOfDependents"] = X_train["NumberOfDependents"].fillna(med_dep)
    X_test["NumberOfDependents"] = X_test["NumberOfDependents"].fillna(med_dep)

    X_train["age"] = X_train["age"].fillna(med_age)
    X_test["age"] = X_test["age"].fillna(med_age)

    # winsorización 1%–99% (límites en train)
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
        lo = X_train[col].quantile(0.01)
        hi = X_train[col].quantile(0.99)
        X_train[col] = X_train[col].clip(lo, hi)
        X_test[col] = X_test[col].clip(lo, hi)

    # RobustScaler (fit en train)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # SMOTE solo en train
    smote = SMOTE(random_state=seed)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    return X_train_res, y_train_res, X_test_scaled, y_test


# =========================
# CANDIDATOS GLOBALES
# =========================
def build_candidates(seed: int):
    # Baseline: XGBoost baseline (usa tus parámetros baseline “fijos”)
    xgb_baseline = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss"
    )

    # Tuned: Bagging con best_params.json
    if not os.path.exists(BEST_PARAMS_PATH):
        raise FileNotFoundError(f"No encuentro {BEST_PARAMS_PATH}. Revisa la ruta.")

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        best = json.load(f)

    # OJO: el nombre debe coincidir con tu json
    key = "Bagging (Decision Tree)"
    if key not in best:
        raise KeyError(
            f"No encuentro la clave '{key}' en {BEST_PARAMS_PATH}. "
            f"Claves disponibles: {list(best.keys())}"
        )

    bag_params = best[key]
    base_tree_params = bag_params.get("base_estimator_params", {})
    base_tree = DecisionTreeClassifier(random_state=seed, **base_tree_params)

    bag_tuned = BaggingClassifier(
        estimator=base_tree,
        n_estimators=bag_params.get("n_estimators", 800),
        max_samples=bag_params.get("max_samples", 0.8),
        max_features=bag_params.get("max_features", 0.6),
        random_state=seed,
        n_jobs=-1
    )

    return xgb_baseline, bag_tuned


# =========================
# PLOTS
# =========================
def plot_and_save(model, model_name, X_train, y_train, X_test, y_test, tag):
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        raise ValueError(f"{model_name} no tiene predict_proba ni decision_function.")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {model_name} ({tag}) | AUC={roc_auc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"roc_{tag}.png"), dpi=220)
    plt.close()

    # PR
    prec, rec, _ = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {model_name} ({tag}) | AP={ap:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"pr_{tag}.png"), dpi=220)
    plt.close()

    return roc_auc, ap


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[seed {SEED_REF}] preparando datos...")
    X_train_res, y_train_res, X_test_scaled, y_test = preparar_datos_local(SEED_REF)

    xgb_baseline, bag_tuned = build_candidates(SEED_REF)

    print("[curvas] Baseline candidato global: XGBoost")
    roc_auc_b, ap_b = plot_and_save(
        xgb_baseline, "XGBoost",
        X_train_res, y_train_res, X_test_scaled, y_test,
        tag=f"baseline_seed{SEED_REF}_xgboost"
    )
    print(f"  -> AUC={roc_auc_b:.4f} | AP={ap_b:.4f}")

    print("[curvas] Tuned candidato global: Bagging (Decision Tree)")
    roc_auc_t, ap_t = plot_and_save(
        bag_tuned, "Bagging (Decision Tree)",
        X_train_res, y_train_res, X_test_scaled, y_test,
        tag=f"tuned_seed{SEED_REF}_bagging"
    )
    print(f"  -> AUC={roc_auc_t:.4f} | AP={ap_t:.4f}")

    print("\nListo. Figuras guardadas en:", OUT_DIR)


if __name__ == "__main__":
    main()
