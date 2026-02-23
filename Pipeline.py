import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# 1) Cargar dataset limpio
df = pd.read_csv("GiveMeSomeCredit/cs-training.csv", index_col=0, sep= ";")

# 2) Separar X e y
y = df['SeriousDlqin2yrs']
X = df.drop(columns=['SeriousDlqin2yrs'])

# 3) Partición temprana
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train = X_train.copy()
X_test  = X_test.copy()

# 4) Corrección de valores inválidos: age = 0 -> NA (antes de imputar)
age0_train = (X_train['age'] == 0).sum()
age0_test  = (X_test['age'] == 0).sum()

print("Registros con age=0 (train):", age0_train)
print("Registros con age=0 (test):", age0_test)

X_train.loc[X_train['age'] == 0, 'age'] = pd.NA
X_test.loc[X_test['age'] == 0, 'age'] = pd.NA

age0_train_before = (X_train['age'] == 0).sum()
age0_test_before = (X_test['age'] == 0).sum()

print("Registros con age=0 (train):", age0_train_before)
print("Registros con age=0 (test):", age0_test_before)

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
# Estadísticos y percentiles en TRAIN antes del tratamiento
X_train.describe(percentiles=[0.01, 0.05, 0.95, 0.99])

vars_outliers = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
for col in vars_outliers:
    lower = X_train[col].quantile(0.01)
    upper = X_train[col].quantile(0.99)
    X_train[col] = X_train[col].clip(lower, upper)
    X_test[col]  = X_test[col].clip(lower, upper)

# Estadísticos y percentiles en TRAIN luego del tratamiento
X_train.describe(percentiles=[0.01, 0.05, 0.95, 0.99])    

# 7) Escalado (fit en train, transform en test)
scaler = RobustScaler ()
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

#Validacion de columnas, escala y nombres de variables
X_train_scaled.shape, X_test_scaled.shape,  X_train_scaled.columns.equals(X_train.columns)


# 8) Diagnóstico antes de SMOTE
print("Antes de SMOTE (train):")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True).round(4))

# 9) SMOTE SOLO en train
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("\nDespués de SMOTE (train):")
print(y_train_res.value_counts())
print(y_train_res.value_counts(normalize=True).round(4))


#10 Tamaño de los conjuntos de datos finales
print("\nShapes Train:")
print(X_train_res.shape, y_train_res.shape)

print("\nShapes Test:")
print(X_test_scaled.shape, y_test   .shape)

