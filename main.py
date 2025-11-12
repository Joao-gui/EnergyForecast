import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import matplotlib.pyplot as plt

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

# Carregar novos dados em data/processed
df = pd.read_csv('data/processed/AEP_2010.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Pré-Processamento
df['year'] = df['Datetime'].dt.year
df['month'] = df['Datetime'].dt.month
df['day'] = df['Datetime'].dt.day
df['hour'] = df['Datetime'].dt.hour
df['dayofweek'] = df['Datetime'].dt.dayofweek
df['is_weekend'] = (df['Datetime'].dt.dayofweek >= 5).astype(int)

X = df[['year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend']]
y = df['AEP_MW']

# Fazendo as previsões
y_pred = model.predict(X)
df['prediction'] = y_pred
print('Previsões geradas com sucesso!')

# Calcular métricas e salvando em reports/metrics
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

    # Salvando as métricas
    folder = 'reports/final'
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, "final_metrics.json")

    # Abre o arquivo e escre o dict no .json
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'Metrics salvas em {filepath}')

# Salvando previsões
df.to_csv("reports/final/predictions.csv", index=False)
print('Previsões salvas em reports/final')

# Gerando plot salvando
df_temp = pd.DataFrame({'Desejado': y, 'Estimado': y_pred}) # Criação de um dataframe com os dados desejados e os estimados na predição
df_temp = df_temp.head(40) # Armazena a quantidade de elementos a serem apresentados no gráfico, pois pode ser visualmente difícil de abstrair caso tenham muitas informações
ax = df_temp.plot(kind='bar',figsize=(10,6)) # Configuração do tipo de gráfico 'bar' e tamanho da figura
fig = ax.get_figure()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray') # Configuração do grid do gráfico
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='blue') # Configuração do grid do gráfico
plt.xlabel('Amostras')
plt.ylabel('AEP_MW')
plt.title('Gráfico Final')
plt.show()
#Salvando o plot
fig.savefig('reports/final/finalGraphic.png')
print('Plot salvo em reports/final')