import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from backend.app.crud import data_access
from backend.app.services import data_preprocessing
from backend.app.models import weather_model
from backend.app.config import MODEL_CONFIG, REGIONS, DATA_START_DATE, DATA_END_DATE


def split_data(raw_df, test_months=6):
    raw_df = raw_df.sort_values('time').reset_index(drop=True)
    
    total_len = len(raw_df)
    test_size = test_months * 30 * 24
    val_size = int((total_len - test_size) * MODEL_CONFIG['validation_split'])
    
    train_df = raw_df[:-test_size-val_size]
    val_df = raw_df[-test_size-val_size:-test_size]
    test_df = raw_df[-test_size:]
    
    print(f"Data splits:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/(30*24):.1f} months)")
    print(f"  Val: {len(val_df)} samples ({len(val_df)/(30*24):.1f} months)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/(30*24):.1f} months)")
    
    return train_df, val_df, test_df


def train_and_evaluate_models():
    print(f"STARTING MODEL TRAINING at {datetime.now()}")
    print("=" * 80)
    for region_name in REGIONS:
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL FOR: {region_name}")
        print(f"{'='*60}")
        try:
            csv_path = os.path.join(data_access.RAW_DATA_DIR, f"{region_name}.csv")
            if not os.path.exists(csv_path):
                print(f"ERROR: Data file not found for {region_name}")
                continue

            raw_df = data_access.load_raw_data_from_csv(csv_path)
            raw_df['time'] = pd.to_datetime(raw_df['time'])
            start_date = pd.to_datetime(DATA_START_DATE)
            end_date = pd.to_datetime(DATA_END_DATE)
            raw_df = raw_df[(raw_df['time'] >= start_date) & (raw_df['time'] <= end_date)]
            raw_df = raw_df.sort_values('time').reset_index(drop=True)
            print(f"Total data points: {len(raw_df)}")

            train_df, val_df, test_df = split_data(raw_df, MODEL_CONFIG['test_months'])
            temp_df = raw_df.set_index('time')
            temp_stats = weather_model.calculate_temperature_range_stats(temp_df, region_name)
            print(f"\nPreprocessing data...")
            X_train_val, y_train_val, feature_scaler, target_scaler, _ = \
                data_preprocessing.preprocess_data_for_training(
                    pd.concat([train_df, val_df]).reset_index(drop=True),
                    target_columns=MODEL_CONFIG['target_variables'].copy(),
                    look_back_steps=MODEL_CONFIG['look_back_hours'],
                    look_ahead_steps=MODEL_CONFIG['look_ahead_hours'],
                    scaler_type=MODEL_CONFIG['scaler_type']
                )

            if X_train_val is None or y_train_val is None:
                print(f"ERROR: Preprocessing failed for {region_name}")
                continue

            print(f"Preprocessed data shapes - X: {X_train_val.shape}, y: {y_train_val.shape}")
            if len(y_train_val.shape) == 3 and MODEL_CONFIG['look_ahead_hours'] == 24:
                n_samples = y_train_val.shape[0]
                y_train_val = y_train_val.reshape(n_samples, MODEL_CONFIG['look_ahead_hours'] * len(MODEL_CONFIG['target_variables']))
                print(f"Reshaped y to: {y_train_val.shape} for 24-hour prediction")

            split_idx = int(len(X_train_val) * (1 - MODEL_CONFIG['validation_split']))
            X_train = X_train_val[:split_idx]
            y_train = y_train_val[:split_idx]
            X_val = X_train_val[split_idx:]
            y_val = y_train_val[split_idx:]
            print(f"\nFinal training data splits:")
            print(f"  Train: {X_train.shape[0]} samples")
            print(f"  Val: {X_val.shape[0]} samples")
            print(f"\nTraining model...")
            trained_model, history = weather_model.train_lstm_model(
                X_train, y_train, X_val, y_val,
                region_name=region_name,
                **MODEL_CONFIG
            )

            if trained_model is None:
                print(f"ERROR: Model training failed for {region_name}")
                continue

            print(f"\nTraining Results:")
            print(f"  Epochs completed: {len(history['loss'])}")
            print(f"  Final training loss: {history['loss'][-1]:.6f}")
            print(f"  Final validation loss: {history['val_loss'][-1]:.6f}")

            print(f"\nSaving model artifacts...")
            data_preprocessing.save_scalers(feature_scaler, target_scaler, region_name)

            config_path = os.path.join(weather_model.PARAMETERS_DIR, f'config_{region_name}.pkl')
            joblib.dump(MODEL_CONFIG, config_path)

            if temp_stats:
                stats_path = os.path.join(weather_model.PARAMETERS_DIR, f'temp_stats_{region_name}.pkl')
                joblib.dump(temp_stats, stats_path)

            weather_model.plot_training_history(history, region_name)

            print(f"\n{'='*40}")
            print(f"EVALUATING MODEL FOR {region_name}")
            print(f"{'='*40}")

            X_test, y_test, _, _, _ = data_preprocessing.preprocess_data_for_training(
                test_df.copy(),
                target_columns=MODEL_CONFIG['target_variables'].copy(),
                look_back_steps=MODEL_CONFIG['look_back_hours'],
                look_ahead_steps=MODEL_CONFIG['look_ahead_hours'],
                scaler_type=MODEL_CONFIG['scaler_type'],
                feature_scaler=feature_scaler,
                target_scaler=target_scaler
            )

            if X_test is not None and y_test is not None:
                # Adjust reshaping for 4 target variables
                if len(y_test.shape) == 3 and MODEL_CONFIG['look_ahead_hours'] == 24:
                    y_test = y_test.reshape(y_test.shape[0], MODEL_CONFIG['look_ahead_hours'] * len(MODEL_CONFIG['target_variables']))

                print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
                y_pred_scaled = trained_model.predict(X_test)

                if MODEL_CONFIG['look_ahead_hours'] == 24 and y_pred_scaled.shape[1] == (24 * len(MODEL_CONFIG['target_variables'])):
                    n_samples = y_pred_scaled.shape[0]
                    # Changed from 5 to 4 target variables
                    y_pred_scaled_reshaped = y_pred_scaled.reshape(n_samples * 24, len(MODEL_CONFIG['target_variables']))
                    y_pred_temp = target_scaler.inverse_transform(y_pred_scaled_reshaped)
                    y_pred = y_pred_temp.reshape(n_samples, 24, len(MODEL_CONFIG['target_variables']))
                    y_test_reshaped = y_test.reshape(n_samples * 24, len(MODEL_CONFIG['target_variables']))
                    y_test_temp = target_scaler.inverse_transform(y_test_reshaped)
                    y_test_original = y_test_temp.reshape(n_samples, 24, len(MODEL_CONFIG['target_variables']))
                    y_pred = y_pred[:, 0, :]
                    y_test_original = y_test_original[:, 0, :]
                else:
                    y_pred = target_scaler.inverse_transform(y_pred_scaled)
                    y_test_original = target_scaler.inverse_transform(y_test)

                y_pred_corrected = weather_model.apply_comprehensive_bias_correction(
                    y_pred, region_name, timestamps=test_df['time'].tail(len(y_test_original)).tolist()
                )

                metrics = weather_model.evaluate_model(
                    trained_model, X_test, y_test, target_scaler, region_name
                )

            print(f"\nSuccessfully trained and evaluated model for {region_name}")

        except Exception as e:
            print(f"ERROR training {region_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"TRAINING AND EVALUATION COMPLETED at {datetime.now()}")


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    os.makedirs(data_access.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(weather_model.PARAMETERS_DIR, exist_ok=True)
    os.makedirs(weather_model.PLOTS_DIR, exist_ok=True)

    print("WEATHER MODEL TRAINING AND EVALUATION")
    print("=" * 80)

    train_and_evaluate_models()

    print(f"\nALL PROCESSES COMPLETE!")
    print(f"Check the plots directory for training history.")