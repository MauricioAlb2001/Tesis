# ============================================================
# TESIS - MODELADO (Baseline + Tuning + Tuned)
# Dataset: Give Me Some Credit (cs-training.csv)
# Evaluación: métricas en TEST, 10 seeds (robustez)
# Tuning: complementario, 1 vez (seed_ref), CV interna con SMOTE en pipeline
# Outputs: resultados largos, resumen (12 filas), grupo, deltas, curvas seed_ref
# ============================================================

import os
import json
import time
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    balanced_accuracy_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from scipy.special import expit  # sigmoid, para convertir decision_function a pseudo-proba

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier


# ============================================================
# 0) CONFIGURACIÓN GENERAL
# ============================================================

# --- Usa todos los núcleos cuando sea posible ---
# (Algunas libs respetan OMP_NUM_THREADS; otras usan n_jobs=-1)
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 8)

DATA_PATH = "GiveMeSomeCredit/cs-training.csv"
SEP = ";"
INDEX_COL = 0

SEEDS = list(range(42, 52))      # 10 iteraciones: 42..51
SEED_REF = 42                    # seed de referencia para tuning + curvas

THRESHOLD = 0.5                  # umbral fijo (tesis: comparabilidad)
POS_LABEL = 1                    # clase positiva = incumplimiento (1)

# Banderas: corre por partes sin rehacer todo
RUN_BASELINE = False
RUN_TUNING   = False
RUN_TUNED    = False

# Carpeta de salidas
OUT_DIR = "salidas_tesis"
os.makedirs(OUT_DIR, exist_ok=True)

# Archivos de salida
BASELINE_LONG_CSV = os.path.join(OUT_DIR, "results_baseline_long.csv")
TUNED_LONG_CSV    = os.path.join(OUT_DIR, "results_tuned_long.csv")
SUMMARY_BASE_CSV  = os.path.join(OUT_DIR, "results_baseline_summary_12rows.csv")
SUMMARY_TUNED_CSV = os.path.join(OUT_DIR, "results_tuned_summary_12rows.csv")
GROUP_BASE_CSV    = os.path.join(OUT_DIR, "results_baseline_group_summary.csv")
GROUP_TUNED_CSV   = os.path.join(OUT_DIR, "results_tuned_group_summary.csv")
DELTAS_BASE_CSV   = os.path.join(OUT_DIR, "deltas_baseline_ens_minus_no.csv")
DELTAS_TUNED_CSV  = os.path.join(OUT_DIR, "deltas_tuned_ens_minus_no.csv")

TUNING_REPORT_CSV = os.path.join(OUT_DIR, "tuning_report.csv")
BEST_PARAMS_JSON  = os.path.join(OUT_DIR, "best_params.json")

# Curvas para gráficos (solo seed_ref y 2 modelos: mejor ensamble + mejor no-ensamble)
CURVES_BASELINE_CSV = os.path.join(OUT_DIR, "curves_seed42_baseline.csv")
CURVES_TUNED_CSV    = os.path.join(OUT_DIR, "curves_seed42_tuned.csv")

# ============================================================
# 1) PIPELINE DE DATOS (Fase 3) - por seed
# ============================================================

def load_and_split(seed: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Lee dataset, separa X/y y hace split 70/30 estratificado.
    """
    df = pd.read_csv(DATA_PATH, sep=SEP, index_col=INDEX_COL)
    y = df["SeriousDlqin2yrs"]
    X = df.drop(columns=["SeriousDlqin2yrs"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    return X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy()


def preprocess_fit_transform(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesa:
    - age=0 -> NA
    - imputación por mediana (solo train)
    - winsorización p1–p99 (límites solo train)
    - RobustScaler (fit train, transform test)
    - SMOTE solo en train

    Devuelve:
    - X_train_res, y_train_res, X_test_scaled
    """
    # 1) age=0 -> NA
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train.loc[X_train["age"] == 0, "age"] = pd.NA
    X_test.loc[X_test["age"] == 0, "age"] = pd.NA

    # 2) imputación por mediana (aprendida en train)
    med_income = X_train["MonthlyIncome"].median()
    med_deps   = X_train["NumberOfDependents"].median()
    med_age    = X_train["age"].median()

    X_train["MonthlyIncome"] = X_train["MonthlyIncome"].fillna(med_income)
    X_test["MonthlyIncome"]  = X_test["MonthlyIncome"].fillna(med_income)

    X_train["NumberOfDependents"] = X_train["NumberOfDependents"].fillna(med_deps)
    X_test["NumberOfDependents"]  = X_test["NumberOfDependents"].fillna(med_deps)

    X_train["age"] = X_train["age"].fillna(med_age)
    X_test["age"]  = X_test["age"].fillna(med_age)

    # 3) winsorización p1–p99 (límites del train)
    vars_out = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
    for col in vars_out:
        lo = X_train[col].quantile(0.01)
        hi = X_train[col].quantile(0.99)
        X_train[col] = X_train[col].clip(lo, hi)
        X_test[col]  = X_test[col].clip(lo, hi)

    # 4) escalado robusto (fit train, transform test)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 5) SMOTE solo en train
    smote = SMOTE(random_state=seed)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    return X_train_res, np.asarray(y_train_res), X_test_scaled


# ============================================================
# 2) MÉTRICAS EN TEST (Fase 5)
# ============================================================

def get_test_scores(model, X_test: np.ndarray) -> np.ndarray:
    """
    Devuelve score continuo para ROC-AUC y AP:
    - si predict_proba: probas clase 1
    - si decision_function: raw scores
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    raise ValueError("Modelo sin predict_proba ni decision_function.")


def scores_to_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convierte scores a labels con umbral fijo.
    - Si scores ya son probas: ok
    - Si scores son raw (SVM): aplicamos sigmoid como aproximación
    """
    # Heurística: si hay valores fuera de [0,1], asumimos raw scores
    if np.nanmin(scores) < 0 or np.nanmax(scores) > 1:
        probs = expit(scores)  # convierte a (0,1)
        return (probs >= threshold).astype(int)
    return (scores >= threshold).astype(int)


def evaluate_on_test(
    model,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    threshold: float
) -> Dict[str, Any]:
    """
    Entrena en train y evalúa en test con métricas definidas.
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    scores = get_test_scores(model, X_test)
    y_pred = scores_to_labels(scores, threshold=threshold)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Métricas (todas en TEST)
    roc_auc = float(roc_auc_score(y_test, scores))
    ap      = float(average_precision_score(y_test, scores))

    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    prec    = float(precision_score(y_test, y_pred, zero_division=0))
    rec     = float(recall_score(y_test, y_pred, zero_division=0))
    f1      = float(f1_score(y_test, y_pred, zero_division=0))

    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "fit_seconds": float(fit_seconds),
    }


# ============================================================
# 3) DEFINICIÓN DE MODELOS
# ============================================================

def baseline_models(seed: int) -> List[Tuple[str, str, Any]]:
    """
    Baseline = configuración razonable (no “default puro”), estable y defendible.
    """
    return [
        ("Logistic Regression", "No Ensemble",
         LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1)),

        ("Decision Tree", "No Ensemble",
         DecisionTreeClassifier(random_state=seed)),

        ("SVM (Lineal)", "No Ensemble",
         LinearSVC(random_state=seed)),

        ("Random Forest", "Ensemble",
         RandomForestClassifier(
             n_estimators=300,
             random_state=seed,
             n_jobs=-1
         )),

        ("Bagging (Decision Tree)", "Ensemble",
         BaggingClassifier(
             estimator=DecisionTreeClassifier(random_state=seed),
             n_estimators=300,
             random_state=seed,
             n_jobs=-1
         )),

        ("XGBoost", "Ensemble",
         XGBClassifier(
             n_estimators=400,
             learning_rate=0.05,
             max_depth=4,
             subsample=0.8,
             colsample_bytree=0.8,
             reg_lambda=1.0,
             random_state=seed,
             n_jobs=-1,
             eval_metric="logloss"
         )),
    ]


def build_model_from_params(model_key: str, seed: int, params: Dict[str, Any]):
    """
    Crea modelo a partir de best_params guardados.
    """
    if model_key == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=seed, n_jobs=-1, **params)

    if model_key == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed, **params)

    if model_key == "SVM (Lineal)":
        return LinearSVC(random_state=seed, **params)

    if model_key == "Random Forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)

    if model_key == "Bagging (Decision Tree)":
        base_params = params.pop("base_estimator_params", {})
        estimator = DecisionTreeClassifier(random_state=seed, **base_params)
        return BaggingClassifier(estimator=estimator, random_state=seed, n_jobs=-1, **params)

    if model_key == "XGBoost":
        return XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="logloss", **params)

    raise ValueError(f"Modelo desconocido: {model_key}")


# ============================================================
# 4) TUNING COMPLEMENTARIO (1 vez, seed_ref)
#    - CV interna
#    - SMOTE dentro del pipeline para evitar leakage
#    - scoring = average_precision
# ============================================================

def tuning_search_spaces() -> Dict[str, Dict[str, Any]]:
    """
    Espacios de búsqueda moderados (controlados) para no demorar eternidad.
    Nota: Bagging necesita params del estimador base aparte.
    """
    return {
        "Logistic Regression": {
            "model__C": np.logspace(-4, 2, 20),
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"],
        },
        "Decision Tree": {
            "model__max_depth": [None, 3, 4, 5, 6, 8, 10],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
        },
        "SVM (Lineal)": {
            "model__C": np.logspace(-4, 2, 20),
        },
        "Random Forest": {
            "model__n_estimators": [300, 500, 800],
            "model__max_features": ["sqrt", "log2", None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_depth": [None, 5, 8, 12],
        },
        "Bagging (Decision Tree)": {
            "model__n_estimators": [300, 500, 800],
            "model__max_samples": [0.6, 0.8, 1.0],
            "model__max_features": [0.6, 0.8, 1.0],
            "model__estimator__max_depth": [None, 3, 5, 8],
            "model__estimator__min_samples_leaf": [1, 2, 4, 8],
        },
        "XGBoost": {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_lambda": [1.0, 2.0, 5.0],
        },
    }


def tune_models_once(seed_ref: int) -> Dict[str, Any]:
    """
    Tuning complementario usando el split del seed_ref.
    Importante: tuning se hace SOLO con train (CV interna), nunca con test.
    Retorna best_params por modelo.
    """
    print("\n=== TUNING COMPLEMENTARIO (solo 1 vez) ===")
    X_tr_raw, y_tr, X_te_raw, y_te = load_and_split(seed_ref)

    # Preprocesamiento “sin SMOTE” aquí, porque SMOTE irá dentro del pipeline de CV.
    # Pero sí hacemos imputación/winsor/escalado con fit en train, transform test.
    # Para tuning solo usaremos el TRAIN transformado.
    # Implementamos un mini-preprocess reutilizando la misma lógica (sin SMOTE final).
    # --- age=0 -> NA ---
    X_tr_raw.loc[X_tr_raw["age"] == 0, "age"] = pd.NA
    X_te_raw.loc[X_te_raw["age"] == 0, "age"] = pd.NA

    # --- imputación por mediana (train) ---
    med_income = X_tr_raw["MonthlyIncome"].median()
    med_deps   = X_tr_raw["NumberOfDependents"].median()
    med_age    = X_tr_raw["age"].median()

    for X_ in (X_tr_raw, X_te_raw):
        X_["MonthlyIncome"] = X_["MonthlyIncome"].fillna(med_income)
        X_["NumberOfDependents"] = X_["NumberOfDependents"].fillna(med_deps)
        X_["age"] = X_["age"].fillna(med_age)

    # --- winsor p1-p99 (train) ---
    vars_out = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
    for col in vars_out:
        lo = X_tr_raw[col].quantile(0.01)
        hi = X_tr_raw[col].quantile(0.99)
        X_tr_raw[col] = X_tr_raw[col].clip(lo, hi)
        X_te_raw[col] = X_te_raw[col].clip(lo, hi)

    # --- escalado robusto ---
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    # X_te no se usa en tuning; se deja para coherencia
    _ = scaler.transform(X_te_raw)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_ref)
    spaces = tuning_search_spaces()

    base_defs = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed_ref, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=seed_ref),
        "SVM (Lineal)": LinearSVC(random_state=seed_ref),
        "Random Forest": RandomForestClassifier(random_state=seed_ref, n_jobs=-1),
        "Bagging (Decision Tree)": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=seed_ref),
            random_state=seed_ref,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(random_state=seed_ref, n_jobs=-1, eval_metric="logloss"),
    }

    best_params = {}
    report_rows = []

    for model_name, base_model in base_defs.items():
        print(f"\n[TUNING] {model_name}")

        # Pipeline con SMOTE dentro de CV (evita leakage)
        pipe = ImbPipeline(steps=[
            ("smote", SMOTE(random_state=seed_ref)),
            ("model", base_model)
        ])

        param_dist = spaces[model_name]

        # n_iter moderado; puedes subir si quieres más búsqueda
        n_iter = 25 if model_name in ["XGBoost", "Random Forest", "Bagging (Decision Tree)"] else 20

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            random_state=seed_ref,
            verbose=1
        )

        t0 = time.time()
        search.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        best_score = float(search.best_score_)
        best = search.best_params_

        # Limpieza de keys: model__X o model__estimator__X
        cleaned = {}
        base_estimator_params = {}

        for k, v in best.items():
            if k.startswith("model__estimator__"):
                base_estimator_params[k.replace("model__estimator__", "")] = v
            elif k.startswith("model__"):
                cleaned[k.replace("model__", "")] = v
            else:
                # smote params u otros (normalmente no buscamos)
                pass

        if model_name == "Bagging (Decision Tree)":
            cleaned["base_estimator_params"] = base_estimator_params

        best_params[model_name] = cleaned

        report_rows.append({
            "model": model_name,
            "best_cv_average_precision": best_score,
            "n_iter": n_iter,
            "cv_folds": 5,
            "elapsed_seconds": float(elapsed),
            "best_params": json.dumps(cleaned)
        })

        print("  best CV AP:", round(best_score, 6))
        print("  best params:", cleaned)

    # Guardar outputs de tuning
    pd.DataFrame(report_rows).to_csv(TUNING_REPORT_CSV, index=False)
    with open(BEST_PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    print("\nGuardado:", TUNING_REPORT_CSV)
    print("Guardado:", BEST_PARAMS_JSON)
    return best_params


# ============================================================
# 5) EJECUCIÓN: BASELINE / TUNED (10 seeds)
# ============================================================

def run_experiment(config_name: str, best_params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Corre 10 seeds y 6 modelos.
    config_name: "Baseline" o "Tuned"
    best_params: dict de parámetros tuned (si config_name == "Tuned")
    """
    results = []
    print(f"\n=== RUN: {config_name} ===")

    for seed in SEEDS:
        print(f"\n[SEED {seed}] preparando datos...")
        X_train_raw, y_train, X_test_raw, y_test = load_and_split(seed)

        # Preprocesamiento completo (incluye SMOTE en train)
        # Nota: pos_rate_test se calcula con y_test real acá
        X_train_res, y_train_res, X_test_scaled = preprocess_fit_transform(
            X_train_raw, y_train, X_test_raw, seed
        )

        n_test = int(len(y_test))
        pos_rate_test = float(np.mean(y_test))

        # Modelos
        if config_name == "Baseline":
            models = baseline_models(seed)
        else:
            assert best_params is not None, "Faltan best_params para Tuned."
            models = []
            for model_key in best_params.keys():
                etiqueta = "Ensemble" if model_key in ["Random Forest", "Bagging (Decision Tree)", "XGBoost"] else "No Ensemble"
                m = build_model_from_params(model_key, seed, dict(best_params[model_key]))  # copia
                models.append((model_key, etiqueta, m))

        # Evaluar
        for model_name, group, model in models:
            met = evaluate_on_test(
                model=model,
                X_train=X_train_res, y_train=y_train_res,
                X_test=X_test_scaled, y_test=np.asarray(y_test),
                threshold=THRESHOLD
            )
            row = {
                "seed": seed,
                "config": config_name,
                "model": model_name,
                "group": group,
                "threshold": THRESHOLD,
                "pos_label": POS_LABEL,
                "n_test": n_test,
                "pos_rate_test": pos_rate_test,
                **met
            }
            results.append(row)

    df = pd.DataFrame(results)
    return df


# ============================================================
# 6) RESÚMENES PARA "FICHA DETALLE" Y "FICHA RESUMEN"
# ============================================================

METRIC_COLS = [
    "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"
]

def make_summary_12rows(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    12 filas = 6 modelos × 2 configs si se mezcla, pero aquí se espera un df de una config.
    Entonces: 6 filas por config. (Si quieres 12, concatena baseline+tuned)
    Reporta mean±std y median[IQR] por métrica.
    """
    rows = []
    for model in sorted(df_long["model"].unique()):
        sub = df_long[df_long["model"] == model]
        row = {
            "config": sub["config"].iloc[0],
            "model": model,
            "group": sub["group"].iloc[0],
            "n_seeds": int(sub["seed"].nunique())
        }
        for m in METRIC_COLS:
            vals = sub[m].values
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"]  = float(np.std(vals, ddof=1))
            row[f"{m}_median"] = float(np.median(vals))
            q1 = float(np.quantile(vals, 0.25))
            q3 = float(np.quantile(vals, 0.75))
            row[f"{m}_q1"] = q1
            row[f"{m}_q3"] = q3
        rows.append(row)
    return pd.DataFrame(rows)


def make_group_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen por grupo (Ensemble vs No Ensemble), por métrica.
    """
    rows = []
    for grp in ["Ensemble", "No Ensemble"]:
        sub = df_long[df_long["group"] == grp]
        for m in METRIC_COLS:
            rows.append({
                "config": sub["config"].iloc[0],
                "group": grp,
                "metric": m,
                "mean": float(sub[m].mean()),
                "std": float(sub[m].std(ddof=1)),
                "median": float(sub[m].median()),
            })
    return pd.DataFrame(rows)


def make_deltas_ens_minus_no(df_long: pd.DataFrame, metrics_for_inferential: List[str]) -> pd.DataFrame:
    """
    Genera tabla de deltas por seed:
    delta(seed, metric) = mean(ensamble metric) - mean(no ensamble metric)
    """
    rows = []
    cfg = df_long["config"].iloc[0]
    for seed in sorted(df_long["seed"].unique()):
        dseed = df_long[df_long["seed"] == seed]
        ens = dseed[dseed["group"] == "Ensemble"]
        no  = dseed[dseed["group"] == "No Ensemble"]
        for m in metrics_for_inferential:
            delta = float(ens[m].mean() - no[m].mean())
            rows.append({"config": cfg, "seed": seed, "metric": m, "delta_ens_minus_no": delta})
    return pd.DataFrame(rows)


# ============================================================
# 7) CURVAS PR/ROC (seed_ref + 2 modelos)
# ============================================================

def get_seed_ref_curves(df_long: pd.DataFrame, config_name: str, best_params: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Para seed_ref:
    - identifica mejor ensamble y mejor no-ensamble por AP (average_precision)
    - recalcula scores en test para esos 2 modelos
    - guarda y_test y score en formato largo: (y_true, score, model, group)
    """
    df_ref = df_long[df_long["seed"] == SEED_REF].copy()

    # Mejor por AP dentro de cada grupo
    best_ens = df_ref[df_ref["group"] == "Ensemble"].sort_values("average_precision", ascending=False).iloc[0]["model"]
    best_no  = df_ref[df_ref["group"] == "No Ensemble"].sort_values("average_precision", ascending=False).iloc[0]["model"]

    chosen = [(best_no, "No Ensemble"), (best_ens, "Ensemble")]
    print(f"\n[CURVAS {config_name}] seed={SEED_REF} | best_no={best_no} | best_ens={best_ens}")

    # Prepara datos del seed_ref
    X_train_raw, y_train, X_test_raw, y_test = load_and_split(SEED_REF)
    X_train_res, y_train_res, X_test_scaled = preprocess_fit_transform(
    X_train_raw, y_train, X_test_raw, SEED_REF
    )
    y_test = np.asarray(y_test)

    curves_rows = []
    for model_name, group in chosen:
        if config_name == "Baseline":
            # construye baseline específico
            model_map = {name: m for (name, _, m) in baseline_models(SEED_REF)}
            model = model_map[model_name]
        else:
            assert best_params is not None
            model = build_model_from_params(model_name, SEED_REF, dict(best_params[model_name]))

        model.fit(X_train_res, y_train_res)
        scores = get_test_scores(model, X_test_scaled)

        # guardamos todo el vector para curvas
        for yt, sc in zip(y_test, scores):
            curves_rows.append({
                "config": config_name,
                "seed": SEED_REF,
                "model": model_name,
                "group": group,
                "y_true": int(yt),
                "y_score": float(sc)
            })

    return pd.DataFrame(curves_rows)


# ============================================================
# 8) MAIN
# ============================================================

def main():
    best_params = None

    # --- BASELINE ---
    if RUN_BASELINE:
        df_base = run_experiment(config_name="Baseline")
        df_base.to_csv(BASELINE_LONG_CSV, index=False)
        print("\nGuardado:", BASELINE_LONG_CSV)

        # Resumen 6 filas (baseline) + grupo + deltas
        sum_base = make_summary_12rows(df_base)
        sum_base.to_csv(SUMMARY_BASE_CSV, index=False)

        grp_base = make_group_summary(df_base)
        grp_base.to_csv(GROUP_BASE_CSV, index=False)

        deltas_base = make_deltas_ens_minus_no(
            df_base,
            metrics_for_inferential=["average_precision", "roc_auc", "f1", "recall"]
        )
        deltas_base.to_csv(DELTAS_BASE_CSV, index=False)

        print("Guardado:", SUMMARY_BASE_CSV)
        print("Guardado:", GROUP_BASE_CSV)
        print("Guardado:", DELTAS_BASE_CSV)
    else:
        df_base = pd.read_csv(BASELINE_LONG_CSV)

    # --- TUNING COMPLEMENTARIO ---
    if RUN_TUNING:
        best_params = tune_models_once(seed_ref=SEED_REF)
    else:
        with open(BEST_PARAMS_JSON, "r", encoding="utf-8") as f:
            best_params = json.load(f)

    # --- TUNED ---
    if RUN_TUNED:
        df_tuned = run_experiment(config_name="Tuned", best_params=best_params)
        df_tuned.to_csv(TUNED_LONG_CSV, index=False)
        print("\nGuardado:", TUNED_LONG_CSV)

        sum_tuned = make_summary_12rows(df_tuned)
        sum_tuned.to_csv(SUMMARY_TUNED_CSV, index=False)

        grp_tuned = make_group_summary(df_tuned)
        grp_tuned.to_csv(GROUP_TUNED_CSV, index=False)

        deltas_tuned = make_deltas_ens_minus_no(
            df_tuned,
            metrics_for_inferential=["average_precision", "roc_auc", "f1", "recall"]
        )
        deltas_tuned.to_csv(DELTAS_TUNED_CSV, index=False)

        print("Guardado:", SUMMARY_TUNED_CSV)
        print("Guardado:", GROUP_TUNED_CSV)
        print("Guardado:", DELTAS_TUNED_CSV)
    else:
        df_tuned = pd.read_csv(TUNED_LONG_CSV)

    # --- CURVAS (seed_ref) para gráficos PR/ROC ---
    # Se hacen al final porque elegimos "mejor modelo" según resultados.
    curves_base = get_seed_ref_curves(df_base, config_name="Baseline", best_params=None)
    curves_base.to_csv(CURVES_BASELINE_CSV, index=False)
    print("Guardado:", CURVES_BASELINE_CSV)

    curves_tuned = get_seed_ref_curves(df_tuned, config_name="Tuned", best_params=best_params)
    curves_tuned.to_csv(CURVES_TUNED_CSV, index=False)
    print("Guardado:", CURVES_TUNED_CSV)

    # --- FICHA DETALLE y FICHA RESUMEN ---
    # Ficha detalle = df_long (baseline y tuned por separado o concatenado)
    df_all = pd.concat([df_base, df_tuned], ignore_index=True)
    ficha_detalle = os.path.join(OUT_DIR, "ficha_detalle_120filas.csv")
    df_all.to_csv(ficha_detalle, index=False)
    print("Guardado:", ficha_detalle)

    # Ficha resumen = 12 filas (6 modelos × 2 configs)
    # (Aquí concatenamos summaries baseline y tuned)
    resumen_12 = pd.concat([pd.read_csv(SUMMARY_BASE_CSV), pd.read_csv(SUMMARY_TUNED_CSV)], ignore_index=True)
    ficha_resumen = os.path.join(OUT_DIR, "ficha_resumen_12filas.csv")
    resumen_12.to_csv(ficha_resumen, index=False)
    print("Guardado:", ficha_resumen)

    print("\n=== FIN. Ya tienes todo para Fase 5 (tablas, deltas y curvas). ===")


if __name__ == "__main__":
    main()
