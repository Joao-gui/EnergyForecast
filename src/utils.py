import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_eda(df, year, figures_path='../reports/figures', data_path='../data/processed'):

        '''Gera e salva gŕaficos EDA e o subset de dados correspondente.'''     
        
        # Garantir que as pastas existam
        Path(figures_path).mkdir(parents=True, exist_ok=True)
        Path(data_path).mkdir(parents=True, exist_ok=True)

        # Criar gráfico
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

        # Série temporal
        sns.lineplot(data=df, x='Datetime', y='AEP_MW', ax=ax1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_title(f'Série Temporal AEP_MW - {year}')

        # Histograma com KDE
        sns.histplot(data=df, x='AEP_MW', kde=True, ax=ax2)
        ax2.set_title(f'Distribuição de AEP_MW - {year}')

        plt.tight_layout()

        plt.show()

        # Salvando gráfico
        fig.savefig(f"{figures_path}/eda_{year}.png", dpi=300)
        plt.close(fig)

        # Salvar subset processado
        df.to_csv(f"{data_path}/AEP_{year}.csv", index=False)

        print(f"Arquivo EDA gerado e salvo para {year} em:")
        print(f' -> {figures_path}/eda_{year}.png')
        print(f' -> {data_path}/AEP_{year}.csv')

# Plot Gráfico de avaliação de modelo de regressão
def plot_regressor_model_ultils(pred, target):
        df_temp = pd.DataFrame({'Desejado': target, 'Estimado': pred}) # Criação de um dataframe com os dados desejados e os estimados na predição
        df_temp = df_temp.head(60) # Armazena a quantidade de elementos a serem apresentados no gráfico, pois pode ser visualmente difícil de abstrair caso tenham muitas informações
        df_temp.plot(kind='bar',figsize=(10,6)) # Configuração do tipo de gráfico 'bar' e tamanho da figura
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray') # Configuração do grid do gráfico
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='blue') # Configuração do grid do gráfico
        plt.show() # Apresenta o gráfico comparando o desejado e o estimado pelo modelo neural

        return