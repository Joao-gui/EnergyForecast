import kagglehub
import pandas as pd
import os
#from kagglehub import KaggleDatasetAdapter

def load_csv(filename="AEP_hourly.csv"):
    dataset_path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
    file_path = os.path.join(dataset_path, filename)
    df = pd.read_csv(file_path)
    return df