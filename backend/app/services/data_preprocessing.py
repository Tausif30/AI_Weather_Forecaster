import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import os

SCALER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'parameters')
os.makedirs(SCALER_DIR, exist_ok=True)


def add_temporal_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12.0)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['week_of_year_sin'] = np.sin(2 * np.pi * df.index.isocalendar().week / 52.0)
    df['week_of_year_cos'] = np.cos(2 * np.pi * df.index.isocalendar().week / 52.0)
    return df


def add_lag_features(df):
    if 'temperature' in df.columns:
        df['temp_lag_1h'] = df['temperature'].shift(1)
        df['temp_lag_3h'] = df['temperature'].shift(3)
        df['temp_lag_6h'] = df['temperature'].shift(6)
        df['temp_lag_12h'] = df['temperature'].shift(12)
        df['temp_lag_24h'] = df['temperature'].shift(24)
        df['temp_change_1h'] = df['temperature'].diff(1)
        df['temp_change_6h'] = df['temperature'].diff(6)
        df['temp_rolling_mean_24h'] = df['temperature'].rolling(24, min_periods=1).mean()
        df['temp_rolling_std_24h'] = df['temperature'].rolling(24, min_periods=1).std()

    if 'humidity' in df.columns:
        df['humidity_lag_1h'] = df['humidity'].shift(1)
        df['humidity_lag_3h'] = df['humidity'].shift(3)
        df['humidity_lag_6h'] = df['humidity'].shift(6)
        df['humidity_lag_12h'] = df['humidity'].shift(12)
        df['humidity_change_1h'] = df['humidity'].diff(1)
        df['humidity_change_3h'] = df['humidity'].diff(3)
        df['humidity_change_6h'] = df['humidity'].diff(6)
        df['humidity_rolling_mean_12h'] = df['humidity'].rolling(12, min_periods=1).mean()
        df['humidity_rolling_std_12h'] = df['humidity'].rolling(12, min_periods=1).std()

    if 'wind_speed' in df.columns:
        df['wind_lag_1h'] = df['wind_speed'].shift(1)
        df['wind_lag_3h'] = df['wind_speed'].shift(3)
        df['wind_lag_6h'] = df['wind_speed'].shift(6)
        df['wind_change_1h'] = df['wind_speed'].diff(1)
        df['wind_change_3h'] = df['wind_speed'].diff(3)
        df['wind_rolling_mean_6h'] = df['wind_speed'].rolling(6, min_periods=1).mean()
        df['wind_rolling_std_6h'] = df['wind_speed'].rolling(6, min_periods=1).std()

    if 'pressure' in df.columns:
        df['pressure_lag_1h'] = df['pressure'].shift(1)
        df['pressure_lag_2h'] = df['pressure'].shift(2)
        df['pressure_lag_3h'] = df['pressure'].shift(3)
        df['pressure_lag_6h'] = df['pressure'].shift(6)
        df['pressure_lag_12h'] = df['pressure'].shift(12)
        df['pressure_lag_24h'] = df['pressure'].shift(24)
        df['pressure_change_1h'] = df['pressure'].diff(1)
        df['pressure_change_3h'] = df['pressure'].diff(3)
        df['pressure_change_6h'] = df['pressure'].diff(6)
        df['pressure_change_12h'] = df['pressure'].diff(12)
        df['pressure_rolling_mean_6h'] = df['pressure'].rolling(6, min_periods=1).mean()
        df['pressure_rolling_mean_12h'] = df['pressure'].rolling(12, min_periods=1).mean()
        df['pressure_rolling_mean_24h'] = df['pressure'].rolling(24, min_periods=1).mean()
        df['pressure_rolling_std_24h'] = df['pressure'].rolling(24, min_periods=1).std()

    if 'temperature' in df.columns and 'humidity' in df.columns:
        a, b = 17.27, 237.7
        alpha = ((a * df['temperature']) / (b + df['temperature'])) + np.log(df['humidity'] / 100.0)
        df['dewpoint'] = (b * alpha) / (a - alpha)
        df['temp_dewpoint_spread'] = df['temperature'] - df['dewpoint']

    if 'wind_speed' in df.columns and 'wind_angle' in df.columns:
        df['wind_angle'] = pd.to_numeric(df['wind_angle'], errors='coerce').fillna(0)
        df['wind_u'] = df['wind_speed'] * np.cos(np.radians(df['wind_angle']))
        df['wind_v'] = df['wind_speed'] * np.sin(np.radians(df['wind_angle']))

    return df


def create_sequences(data_df, target_columns, look_back_steps, look_ahead_steps=1):
    for col in data_df.columns:
        if data_df[col].dtype == 'object':
            data_df = data_df.drop(columns=[col])

    data_df = data_df.ffill().bfill()

    feature_cols = [col for col in data_df.columns if col not in target_columns]
    features = data_df[feature_cols].values
    targets = data_df[target_columns].values

    X, y = [], []

    for i in range(len(data_df) - look_back_steps - look_ahead_steps + 1):
        X.append(features[i:(i + look_back_steps)])

        if look_ahead_steps == 1:
            y.append(targets[i + look_back_steps])
        else:
            y.append(targets[i + look_back_steps:i + look_back_steps + look_ahead_steps])

    return np.array(X), np.array(y)


def preprocess_data_for_training(raw_df, target_columns, look_back_steps=24*7,
                                 look_ahead_steps=1, scaler_type='robust',
                                 feature_scaler=None, target_scaler=None):
    df = raw_df.copy()

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    df.columns = [col.replace('.', '_') for col in df.columns]

    core_features = [
        'temperature', 'feels_like', 'wind_chill', 'wind_speed', 'wind_angle',
        'pressure', 'humidity', 'cloud_cover_total', 'ozone'
    ]
    
    available_features = [f for f in core_features if f in df.columns]
    
    for col in target_columns:
        if col not in available_features and col in df.columns:
            available_features.append(col)
    
    print(f"Using features: {available_features}")
    df = df[available_features].copy()
    
    df = df.interpolate(method='time', limit_direction='both')
    df = df.ffill().bfill()
    
    df = add_lag_features(df)
    df = add_temporal_features(df)
    
    df = df.ffill().bfill().fillna(0)
    
    feature_columns = [col for col in df.columns if col not in target_columns]
    
    if feature_scaler is None:
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        
        df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])
        df[target_columns] = target_scaler.fit_transform(df[target_columns])
        
        feature_scaler.feature_names_in_ = feature_columns
        target_scaler.feature_names_in_ = target_columns
    else:
        expected_features = list(feature_scaler.feature_names_in_)
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            for f in missing_features:
                df[f] = 0
        
        df_features = df[expected_features].copy()
        df_targets = df[target_columns].copy()
        
        df[expected_features] = feature_scaler.transform(df_features)
        df[target_columns] = target_scaler.transform(df_targets)
    
    X, y = create_sequences(df, target_columns, look_back_steps, look_ahead_steps)
    
    print(f"Preprocessing complete: X shape={X.shape}, y shape={y.shape}")
    
    return X, y, feature_scaler, target_scaler, df


def preprocess_data_for_inference(df_recent, feature_scaler, look_back_steps):
    df = df_recent.copy()
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
    
    df = df.sort_index()
    df.columns = [col.replace('.', '_') for col in df.columns]
    
    df = add_lag_features(df)
    df = add_temporal_features(df)
    
    expected_features = list(feature_scaler.feature_names_in_)
    
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for f in missing_features:
            df[f] = 0
    
    df = df[expected_features].copy()
    df = df.ffill().bfill().fillna(0)
    
    if len(df) < look_back_steps:
        raise ValueError(f"Need {look_back_steps} steps, got {len(df)}")
    
    df = df.tail(look_back_steps)
    scaled_features = feature_scaler.transform(df.values)
    
    return scaled_features.reshape(1, look_back_steps, scaled_features.shape[1])


def save_scalers(feature_scaler, target_scaler, region_name):
    joblib.dump(feature_scaler, os.path.join(SCALER_DIR, f'feature_scaler_{region_name}.pkl'))
    joblib.dump(target_scaler, os.path.join(SCALER_DIR, f'target_scaler_{region_name}.pkl'))
    print(f"Scalers for {region_name} saved.")


def load_scalers(region_name):
    try:
        feature_scaler = joblib.load(os.path.join(SCALER_DIR, f'feature_scaler_{region_name}.pkl'))
        target_scaler = joblib.load(os.path.join(SCALER_DIR, f'target_scaler_{region_name}.pkl'))
        return feature_scaler, target_scaler
    except:
        return None, None