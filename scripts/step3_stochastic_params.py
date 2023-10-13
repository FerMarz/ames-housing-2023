import joblib
import pandas as pd
from time import time
from sklearn.model_selection import GridSearchCV

from default_xgb import default_xgb

# Cargar datos de entrenamiento y test
train_data = pd.read_csv("AmesHousing_train_set.csv")
test_data = pd.read_csv("AmesHousing_test_set.csv")

# Dividir los datos en características (X) y variable objetivo (y)
x_train = train_data.drop(columns=["sale_price"])
y_train = train_data["sale_price"]
x_test = test_data.drop(columns=["sale_price"])
y_test = test_data["sale_price"]

# Define los parámetros para la búsqueda de cuadrícula
param_grid = {
    "subsample": [
        0.1,
        0.15,
        0.20,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1,
    ],
    "colsample_bytree": [
        0.1,
        0.15,
        0.20,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1,
    ],
}

# Define el número de hilos para XGBoost y GridSearchCV
xgb_threads = 3
grid_search_threads = 8

# Crear el modelo XGBoost
xgb_model = default_xgb(
    n_jobs=xgb_threads,
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
)

# Configurar GridSearchCV con 5-Fold
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    verbose=2,
    n_jobs=grid_search_threads,
)

# Realizar la búsqueda de cuadrícula
ti = time()
grid_search.fit(x_train, y_train)
tf = time()
print(f"Tiempo en segundos: {tf - ti}")
print(grid_search.best_params_)

# Guardar los resultados de grid_search en un archivo .pkl
joblib.dump(grid_search, "Step3_GridSearchCV_AmesHousing_stochastic_params.pkl")
print("Search finished!")
