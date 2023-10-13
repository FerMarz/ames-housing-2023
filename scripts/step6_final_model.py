import joblib
import pandas as pd
from time import time

from default_xgb import default_xgb

# Cargar datos de entrenamiento y test
train_data = pd.read_csv("AmesHousing_train_set.csv")
test_data = pd.read_csv("AmesHousing_test_set.csv")

# Dividir los datos en características (X) y variable objetivo (y)
x_train = train_data.drop(columns=["sale_price"])
y_train = train_data["sale_price"]
x_test = test_data.drop(columns=["sale_price"])
y_test = test_data["sale_price"]

# Define el número de hilos para XGBoost y GridSearchCV
xgb_threads = 12

# Crear el modelo XGBoost
xgb_model = default_xgb(
    n_jobs=xgb_threads,
    tree_method="exact",
    n_estimators=15000,
    learning_rate=0.02,
    max_depth=4, 
    min_child_weight=3,
    subsample=0.7,
    colsample_bytree=0.3,
    reg_alpha=1000,
    reg_lambda=1,
)

# Entrenar el modelo
ti = time()
evals = [(x_train, y_train), (x_test, y_test)]
xgb_model.fit(
    x_train, y_train, 
    eval_set=evals,
    early_stopping_rounds=0, 
    verbose=50
)
tf = time()
print(f"Tiempo en segundos: {tf - ti}")

# Guardar los resultados de grid_search en un archivo .pkl
joblib.dump(xgb_model, "Step6_final_model.pkl")
print("Model finished!")