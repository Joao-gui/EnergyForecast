import kagglehub
import pandas as pd
import os
import shutil
#from kagglehub import KaggleDatasetAdapter

def load_raw_csv(filename="AEP_hourly.csv"):
    # Caminho para o arquvio .csv
    custom_path = os.path.abspath("../data/raw") # Caminho para salvar os dados
    file_path_destiny = os.path.join(custom_path, filename)

    # Caminho padrão de donwload do kaggle    
    dataset_path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
    file_path_origin = os.path.join(dataset_path, filename)

    # Garante que o diretório destino exista
    custom_path = os.path.dirname(custom_path)
    if not os.path.exists(custom_path):
        os.makedirs(custom_path)

    # Copia arquivo
    shutil.copy(file_path_origin, file_path_destiny)

    # Criando o dataframe do arquivo .csv
    df = pd.read_csv(file_path_destiny)
    return df

def load_custom_csv(filename: str):
    # Caminho para o arquvio .csv
    file_path = os.path.abspath(os.path.join("../data/processed", filename)) # Caminho dos arquivos

    # Verifica se o arquivo existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    # le arquivo .csv
    df = pd.read_csv(file_path)
    
    return df