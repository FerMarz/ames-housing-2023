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
    "max_depth": [3, 4, 5, 6, 7, 8, 9],
    "min_child_weight": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}

# Define el número de hilos para XGBoost y GridSearchCV
xgb_threads = 4
grid_search_threads = 6

# Crear el modelo XGBoost
xgb_model = default_xgb(
    n_jobs=xgb_threads,
    n_estimators=3000,
    learning_rate=0.05,
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
joblib.dump(grid_search, "Step2_GridSearchCV_AmesHousing_tree_parameters.pkl")
print("Search finished!")
