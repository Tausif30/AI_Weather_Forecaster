import os
from datetime import datetime, timedelta
import pandas as pd

from backend.app.crud import data_access
from backend.app.services import data_preprocessing
from backend.app.models import weather_model
from backend.app.config import MODEL_CONFIG, REGIONS, DATA_START_DATE

# Use core regions for daily updates
DAILY_UPDATE_REGIONS = data_access.CORE_REGIONS_PLACE_IDS


def run_daily_update_and_retrain():
    print(f"--- Starting Daily Weather Data Update & Model Retraining ({datetime.now()}) ---")

    yesterday_date = datetime.now() - timedelta(days=1)
    yesterday_date_str = yesterday_date.strftime('%Y-%m-%d')

    for region_name, place_id in DAILY_UPDATE_REGIONS.items():
        print(f"\n--- Processing: {region_name} ---")

        try:
            data_access.append_daily_api_data_to_csv(region_name, place_id, yesterday_date)
            print(f"Data updated for {yesterday_date_str}")
        except Exception as e:
            print(f"Failed to update data: {e}")
            continue 

        print(f"Retraining model...")

        try:
            csv_path = os.path.join(data_access.RAW_DATA_DIR, f"{region_name}.csv")
            raw_df = data_access.load_raw_data_from_csv(csv_path)

            raw_df['time'] = pd.to_datetime(raw_df['time'])
            raw_df = raw_df[
                (raw_df['time'] >= pd.to_datetime(DATA_START_DATE)) & 
                (raw_df['time'] <= yesterday_date)
            ].sort_values('time').reset_index(drop=True)

            if raw_df.empty:
                print(f"No data available for {region_name}")
                continue

            temp_df = raw_df.set_index('time')
            temp_stats = weather_model.calculate_temperature_range_stats(temp_df, region_name)

            X, y, feature_scaler, target_scaler, _ = \
                data_preprocessing.preprocess_data_for_training(
                    raw_df.copy(),
                    target_columns=MODEL_CONFIG['target_variables'], 
                    look_back_steps=MODEL_CONFIG['look_back_hours'],
                    look_ahead_steps=MODEL_CONFIG['look_ahead_hours'],
                    scaler_type=MODEL_CONFIG['scaler_type']
                )

            if X is None or y is None:
                print(f"Preprocessing failed")
                continue
            if len(y.shape) == 3 and MODEL_CONFIG['look_ahead_hours'] == 24:
                y = y.reshape(y.shape[0], MODEL_CONFIG['look_ahead_hours'] * len(MODEL_CONFIG['target_variables']))

            print(f"Data points: {len(X)}")

            train_size = int(0.8 * len(X))
            val_size = int(0.1 * len(X))

            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]

            if X_train.shape[0] == 0:
                print(f"Insufficient training data")
                continue

            trained_model, history = weather_model.train_lstm_model(
                X_train, y_train, X_val, y_val,
                region_name=region_name,
                **MODEL_CONFIG
            )

            if trained_model:
                data_preprocessing.save_scalers(feature_scaler, target_scaler, region_name)
                
                if temp_stats:
                    import joblib
                    stats_path = os.path.join(weather_model.PARAMETERS_DIR, 
                                            f'temp_stats_{region_name}.pkl')
                    joblib.dump(temp_stats, stats_path)
                
                print(f"Model retrained successfully")
            else:
                print(f"Model retraining failed")

        except Exception as e:
            print(f"Error during retraining: {e}")

    print(f"\n--- Daily Update & Retraining Complete ({datetime.now()}) ---")


if __name__ == "__main__":
    os.makedirs(data_access.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(weather_model.PARAMETERS_DIR, exist_ok=True)
    
    run_daily_update_and_retrain()