# Projeto: Aprendizado de máquinas e séries temporais para prever o consumo de energia elétrica nas próximas horas ou dias.

---

## Descrição Projeto

---

Este projeto tem como objetivo **prever o consumo de energia elétrica** nas horas e dias seguintes, utilizando **modelos de apredizado de máquina** aplicados a **séries temporais**.

O conjunto de dados utilizados foi o [Hourly Energy Consumption (AEP)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data?select=AEP_hourly.csv) disponibilizado no Kaggle, contendo leituras horárias de consumo de energia elétrica de uma região dos Estados Unidos por megawatts (MW). Utilizando o arquivo AEP_hourly.csv.

*O [AEP (American Eletric Power)](https://en.wikipedia.org/wiki/American_Electric_Power) é uma companhia elétrica dos **Estados Unidos**, sendo a maior companhia do país atuando em 11 estados.*

O foco principal é explorar e comparar diferentes abordagens de regressão, analisando a performace dos modelos e avaliando a qualidade das previsões para uso em cenários reais, como **planejamento de demanda energética** e **otimização de recursos**.

---

## Metodologia

---

O projeto foi dividido em etapas:

- **Coleta e Entendimento dos Dados**
  - Utilizando do dataset `AEP_hourly.csv` (consumo energético horários).
  - Análise emploratória dos padrões temporais (tendências diárias, sazionais, etc.).
- **Pre Provcessamento dos Atributos**
  - A partir da coluna `Datetime`, foram criadas novas variáveis para capturar padrẽos temporais:

    - `yeas`, `month`, `day`, `hour`
    - `dayofweek` (dia da semana)
    - `is_weekend` (indicados de fim de semana)

    Essas estruturas permitem ao modelo aprender como o consumo varia ao longo do tempo.
- **Treinamento dos Modelo**
  - Foram testados dois algoritmos de regressão:
    - **Regressão Linear**
    - **Random Forest Regressor**
  - *O Processo de validação foi feito utilizando **K-fold Cross Validation***
- **Avaliação das Métricas**
  - As ericas utilizadas foram:
    - **R² (Coeficiente de Determinação)** - mede a qualidade do ajuste do modelo.
    - **MAE (Erro Absoluto Médio)** - média das diferenças absolutas entre valores reais e previstos.
    - **MSE (Erro Quadrático Médio)** - penaliza grandes erros.
    - **RMSE (Raiz do Erro Quadrático Médio)** - erro médio em mesma escala da variável alvo.

---

## Resultados Obtidos

---

| **Modelo**  | **R²** | **MAE** | **MSE** | **RMSE** |
| ----------------- | ------------- | ------------- | ------------- | -------------- |
| Regressão LInear | 0.2984        | 1754          | 4.71e6        | 2170           |
| Random Forest     | 0.9950        | 111.1         | 33110         | 181.96         |

O modelo **Random Forest** apresentou desempenho significativamente superior, capturando de forma eficiente os padrões não lineares do consumo energético.

---

## Estrutura do Projeto

---

Os modelos gŕaficos gerados durante o experimento são automaticamente salvos em diretórios:

```
EnergyForecast/
|
|-- data/
|   |-- raw/                       # Dados brutos originais (sem alteração)
|   |   |-- AEP_hourly.csv
|   |
|   |-- processed/                 # Dados intermediários, entre pré-processamento e features, e dados tratados e prontos para modelagem
|   |   |-- AEP_20XX.csv
|   |   |-- AEP_ready.csv
|   |   |-- AEP_hourly_update.csv
|
|-- notebooks/			   
|   |-- 01_exploracao.ipynb
|   |-- 02_preprocessamento.ipynb
|   |-- 03_modelagem.ipynb
|
|-- reports/
|   |-- figures/                   # Gráficos e visualizações geradas
|   |-- metrics/                   # Métricas e resultados de avaliação
|   |-- final/                     # Resultados finais (gráficos + previsões + relatórios)
|   |-- models/			   # Modelos salbos (.joblib, .pkl)
|   |   |-- best_model.joblib
|
|-- src/                           # Código-fonte principal do projeto
|   |-- data.py
|   |-- features.py
|   |-- model.py
|   |-- utils.py
|
|-- main.py                        # Script principal para rodar o pipeline completo
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- .gitignore
```
