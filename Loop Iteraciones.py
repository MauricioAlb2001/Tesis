# =========================
# 0) IMPORTS
# =========================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from scipy.special import expit  # sigmoid

from xgboost import XGBClassifier


# =========================
# 1) PIPELINE DE DATOS (fase 3)
# =========================
def preparar_datos(seed:int,
                   path="GiveMeSomeCredit/cs-training.csv",
                   sep=";",
                   index_col=0):
    # 1) Cargar dataset limpio
    df = pd.read_csv(path, index_col=index_col, sep=sep)

    # 2) Separar X e y
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])

    # 3) Partición temprana
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    X_train = X_train.copy()
    X_test  = X_test.copy()

    # 4) Corrección de valores inválidos: age = 0 -> NA
    X_train.loc[X_train['age'] == 0, 'age'] = pd.NA
    X_test.loc[X_test['age'] == 0, 'age'] = pd.NA

    # 5) Imputación (mediana aprendida en train)
    median_income = X_train['MonthlyIncome'].median()
    median_dependents = X_train['NumberOfDependents'].median()
    median_age = X_train['age'].median()

    X_train['MonthlyIncome'] = X_train['MonthlyIncome'].fillna(median_income)
    X_test['MonthlyIncome']  = X_test['MonthlyIncome'].fillna(median_income)

    X_train['NumberOfDependents'] = X_train['NumberOfDependents'].fillna(median_dependents)
    X_test['NumberOfDependents']  = X_test['NumberOfDependents'].fillna(median_dependents)

    X_train['age'] = X_train['age'].fillna(median_age)
    X_test['age']  = X_test['age'].fillna(median_age)

    # 6) Outliers (límites aprendidos en train, aplicados a test)
    vars_outliers = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
    for col in vars_outliers:
        lower = X_train[col].quantile(0.01)
        upper = X_train[col].quantile(0.99)
        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col]  = X_test[col].clip(lower, upper)

    # 7) Escalado (fit en train, transform en test)
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

    # 8-9) SMOTE SOLO en train
    smote = SMOTE(random_state=seed)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # CV estratificado
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    return X_train_res, y_train_res, X_test_scaled, y_test, cv


# =========================
# 2) EVALUADOR (AUC CV + AUC test + métricas)
# =========================
def evaluar_modelo(model, model_name, X_train, y_train, X_test, y_test, cv, threshold=0.5):
    # AUC en CV (robustez)
    auc_cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    auc_cv_mean = float(np.mean(auc_cv_scores))
    auc_cv_std  = float(np.std(auc_cv_scores))

    # Entrenar en todo train
    model.fit(X_train, y_train)

    # Obtener scores
    if hasattr(model, "predict_proba"):
        test_scores = model.predict_proba(X_test)[:, 1]
        auc_test = float(roc_auc_score(y_test, test_scores))
        y_pred = (test_scores >= threshold).astype(int)

    elif hasattr(model, "decision_function"):
        # Caso SVM lineal (como lo hiciste): AUC con decision_function, umbral con sigmoid
        raw_scores = model.decision_function(X_test)
        auc_test = float(roc_auc_score(y_test, raw_scores))
        probs = expit(raw_scores)
        y_pred = (probs >= threshold).astype(int)

    else:
        raise ValueError(f"Modelo {model_name} no tiene predict_proba ni decision_function.")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Exactitud": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Sensibilidad": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "AUC_test": auc_test,
        "AUC_CV_mean": auc_cv_mean,
        "AUC_CV_std": auc_cv_std,
        "AUC_CV_scores": auc_cv_scores.round(6).tolist()
    }


# =========================
# 3) MODELOS BASELINE (exactos)
# =========================
def modelos_baseline(seed:int):
    return [
        ("Logistic Regression", "No Ensemble",
         LogisticRegression(max_iter=2000, random_state=seed)),

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


# =========================
# 4) MODELOS TUNED (best params ya encontrados)
# =========================
def modelos_tuned(seed:int):
    return [
        ("Logistic Regression", "No Ensemble",
         LogisticRegression(max_iter=2000, solver="lbfgs", penalty="l2", C=0.001, random_state=seed)),

        ("Decision Tree", "No Ensemble",
         DecisionTreeClassifier(min_samples_split=10, min_samples_leaf=4, max_depth=None, random_state=seed)),

        ("SVM (Lineal)", "No Ensemble",
         LinearSVC(C=0.0001, random_state=seed)),

        ("Random Forest", "Ensemble",
         RandomForestClassifier(
             n_estimators=800,
             min_samples_split=5,
             min_samples_leaf=1,
             max_features="sqrt",
             max_depth=None,
             random_state=seed,
             n_jobs=-1
         )),

        ("Bagging (Decision Tree)", "Ensemble",
         BaggingClassifier(
             estimator=DecisionTreeClassifier(min_samples_leaf=4, max_depth=None, random_state=seed),
             n_estimators=800,
             max_samples=0.8,
             max_features=0.6,
             random_state=seed,
             n_jobs=-1
         )),

        ("XGBoost", "Ensemble",
         XGBClassifier(
             subsample=0.7,
             reg_lambda=1.0,
             n_estimators=500,
             min_child_weight=5,
             max_depth=5,
             learning_rate=0.1,
             colsample_bytree=0.9,
             random_state=seed,
             eval_metric="logloss",
             n_jobs=-1
         )),
    ]


# =========================
# 5) LOOP 10 ITERACIONES (baseline + tuned)
# =========================
seeds = list(range(42, 52))  # 10 iteraciones: 42..51
resultados = []

for seed in seeds:
    X_train_res, y_train_res, X_test_scaled, y_test, cv = preparar_datos(seed=seed)

    # Baseline
    for nombre, etiqueta, model in modelos_baseline(seed):
        met = evaluar_modelo(model, nombre, X_train_res, y_train_res, X_test_scaled, y_test, cv, threshold=0.5)
        met.update({"Seed": seed, "Modelo": nombre, "Etiqueta": etiqueta, "Iteracion": "Baseline"})
        resultados.append(met)

    # Tuned
    for nombre, etiqueta, model in modelos_tuned(seed):
        met = evaluar_modelo(model, nombre, X_train_res, y_train_res, X_test_scaled, y_test, cv, threshold=0.5)
        met.update({"Seed": seed, "Modelo": nombre, "Etiqueta": etiqueta, "Iteracion": "Tuned"})
        resultados.append(met)

df_iter = pd.DataFrame(resultados)

print("Resultados shape:", df_iter.shape)
print(df_iter.head())

df_iter.to_csv("resultados_iteraciones_baseline_tuned.csv", index=False)
print("Guardado: resultados_iteraciones_baseline_tuned.csv")
