from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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