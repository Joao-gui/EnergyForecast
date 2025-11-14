from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
import numpy as np
import joblib
import os
import json

# Condfigurando o KFold para o cross validation
def kfold_model (n_splits, random_state):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cv

# Utilizando LInera Regression
def linearRegression_model(input, target, cv, scoring: str):
    # Transformação logarítmica e depois volta com exponencial com o TransformedTargerRegressor
    model = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log1p, inverse_func=np.expm1)
    param_grid = {}  # LinearRegression não tem hiperparâmetros relevantes aqui
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(input, target)

    preds = grid_search.predict(input)
    
    metrics = {
        "R2": r2_score(target, preds),
        "MAE": mean_absolute_error(target, preds),
        "MSE": mean_squared_error(target, preds),
        "RMSE": np.sqrt(mean_squared_error(target, preds))
    }
    
    return metrics, preds, target

# Utilizando LInera Regression
def randomForest_model(input, target, cv, scoring: str, random_state, verbose):
    # Transformação logarítmica e depois volta com exponencial com o TransformedTargerRegressor
    model = TransformedTargetRegressor(regressor=RandomForestRegressor(), func=np.log1p, inverse_func=np.expm1)
    param_grid = dict(
        n_estimators=[100], #50, 200
        criterion=['squared_error'],
        min_samples_split=[2], #5,10
        min_samples_leaf = [1], #2,3
        random_state = [random_state],
        verbose = [verbose])
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(input, target)

    preds = grid_search.predict(input)
    
    metrics = {
        "R2": r2_score(target, preds),
        "MAE": mean_absolute_error(target, preds),
        "MSE": mean_squared_error(target, preds),
        "RMSE": np.sqrt(mean_squared_error(target, preds))
    }
    
    return metrics, preds, target, grid_search.best_estimator_

# Salvando modelo no arquivo reports/model
def save_model(model, filename, folder='../reports/model'):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{filename}.joblib")
    joblib.dump(model, filepath)
    print(f"Modelo salvo em {filepath}")

    return

# Salvando métrica do modelo no arquivo reports/metrics
def save_metrics(metrics: dict, filename: str, folder='../reports/metrics'):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{filename}.json")

    # Abre o arquivo e escre o dict no .json
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'Metrics salvas em {filepath}')

    return