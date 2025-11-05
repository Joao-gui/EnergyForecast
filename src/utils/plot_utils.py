import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_eda(df, year, figures_path='../reports/figures', 
            data_path='../data/processed'):

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