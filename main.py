import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import matplotlib.pyplot as plt
from src.features import prepare_feature
from src.utils import plot_regressor_model_ultils

# Import para visualizar as pastas
import os
import sys

# Adiciona a pasta raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from pathlib import Path

# Carregando modelo .joblib salvo em reports/model
MODEL_PATH = "reports/model/randomForest_model.joblib"
model = load(MODEL_PATH)
print("Modelo carregado com sucesso!")

# Carregar novos dados para testar modelo de data/processed
df = pd.read_csv('data/processed/AEP_2010.csv')

# Pré-Processamento do dataframe
df = prepare_feature(df)

# Separando Target de Prediction
X = df[['year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend']]
y = df['AEP_MW']

# Fazendo as previsões
y_pred = model.predict(X)
df['prediction'] = y_pred
print('Previsões geradas com sucesso!')

# Calcular métricas e salvando em reports/final/final_metrics.json
if "AEP_MW" in df.columns:
    metrics = {
        "R2": r2_score(y, y_pred),
        "MAE": mean_absolute_error(y, y_pred),
        "MSE": mean_squared_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred))
        }
    print('\n Métricas:')
    for name, value in metrics.items():
        print(f'{name}: {value}')

    # Salvando as métricas e verificando se a pasta reports/final existe, se não ele cria
    folder = 'reports/final'
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "final_metrics.json")

    # Abre o arquivo e escreve o dict no .json
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'Metrics salvas em {filepath}')

# Salvando previsões em reports/final/predictions.csv
# A pasta já foi verificada se existe ou não no processo anterior, com isso só será salvo o arquivo
df.to_csv("reports/final/predictions.csv", index=False)
print('Previsões salvas em reports/final')

# Gerando plot salvando em reports/final/finalGraphic.png
fig = plot_regressor_model_ultils(pred=y_pred, target=y, title="Gráfico Regressão modelo Final", xlabel='Amostras', ylabel="AEP_MW")
#Salvando o plot
fig.savefig('reports/final/finalGraphic.png')
print('Plot salvo em reports/final')