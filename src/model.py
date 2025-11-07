from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Condfigurando o KFold para o cross validation
def kfold_model (n_splits, random_state):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cv

# Utilizando LInera Regression
def linearRegression_model(input, target, cv, scoring: str):
    model = LinearRegression()
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
    model = RandomForestRegressor()
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
    
    return metrics, preds, target