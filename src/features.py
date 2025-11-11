from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd

def std_scaler_feature(df, columns):
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled, scaler

def minmax_scaler_feature(df, columns):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled, scaler

def robust_scaler_feature(df, columns):
    scaler = RobustScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled, scaler

def prepare_feature(df, datetime_col= 'Datetime'):
    df_features = df.copy()
    df_features[datetime_col] = pd.to_datetime(df_features[datetime_col])
    df_features['year'] = df_features[datetime_col].dt.year
    df_features['month'] = df_features[datetime_col].dt.month
    df_features['day'] = df_features[datetime_col].dt.day
    df_features['hour'] = df_features[datetime_col].dt.hour
    df_features['dayofweek'] = df_features[datetime_col].dt.dayofweek
    df_features['is_weekend'] = (df_features[datetime_col].dt.dayofweek >= 5).astype(int)

    return df_features