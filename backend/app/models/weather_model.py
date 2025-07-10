import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
from ..config import BIAS_CORRECTIONS, MODEL_CONFIG # Import MODEL_CONFIG

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
PARAMETERS_DIR = os.path.join(BASE_DIR, 'parameters')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)


def create_lstm_model(input_shape, output_shape, lstm_units=[48, 32], dropout_rate=0.5,
                     learning_rate=0.001, use_bidirectional=True):
    inputs = Input(shape=input_shape)
    x = LayerNormalization()(inputs)

    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        if use_bidirectional and i == 0:
            x = Bidirectional(LSTM(units, return_sequences=return_seq, 
                                 kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))(x)
        else:
            x = LSTM(units, return_sequences=return_seq,
                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = Dropout(dropout_rate)(x)
        if return_seq:
            x = LayerNormalization()(x)

    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate / 2)(x)
    outputs = Dense(output_shape, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


def train_lstm_model(X_train, y_train, X_val, y_val, region_name,
                    epochs=20, batch_size=64, lstm_units=[48, 32], dropout_rate=0.5,
                    learning_rate=0.001, use_bidirectional=True,
                    early_stopping_patience=6, reduce_lr_patience=3,
                    look_ahead_hours=1, **kwargs):
    print(f"\n--- Training LSTM Model for {region_name} ---")

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1] if len(y_train.shape) == 2 else 1

    model = create_lstm_model(
        input_shape=input_shape,
        output_shape=output_shape,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        use_bidirectional=use_bidirectional
    )

    print("\nModel Summary:")
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience,
                     restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=reduce_lr_patience,
                         min_lr=1e-7, verbose=1)
    ]

    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model_path = os.path.join(PARAMETERS_DIR, f'lstm_{region_name}.keras')
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    bias_corrections = {
        'temperature': {'bias': BIAS_CORRECTIONS['temperature_formula'], 'std_error': 2.0},
        'humidity': {'bias': BIAS_CORRECTIONS['humidity'], 'std_error': 5.0}, # Updated humidity bias
        'wind_speed': {'bias': BIAS_CORRECTIONS['wind_speed_japan'] if region_name in ['Meguro', 'Bunkyo'] else 0.0, 'std_error': 1.0},
        'pressure': {'bias': 0.0, 'std_error': 2.0},
    }

    bias_path = os.path.join(PARAMETERS_DIR, f'bias_{region_name}.pkl')
    joblib.dump(bias_corrections, bias_path)

    print(f"Bias corrections saved:")
    print(f"  Temperature: Piecewise function applied")
    print(f"  Humidity: -{BIAS_CORRECTIONS['humidity']}%")
    print(f"  Wind Speed: -0.5 m/s (Japan only)" if region_name in ['Meguro', 'Bunkyo'] else "  Wind Speed: No correction")

    config = {
        'lstm_units': lstm_units,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'use_bidirectional': use_bidirectional,
        'look_ahead_hours': look_ahead_hours,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'training_date': datetime.now().isoformat()
    }

    config_path = os.path.join(PARAMETERS_DIR, f'config_{region_name}.pkl')
    joblib.dump(config, config_path)

    return model, history.history


def calculate_temperature_range_stats(df, region_name):
    six_months_ago = df.index.max() - pd.DateOffset(months=6)
    recent_data = df[df.index >= six_months_ago]

    if 'temperature' in recent_data.columns:
        temp_stats = {
            'mean': recent_data['temperature'].mean(),
            'std': recent_data['temperature'].std(),
            'min': recent_data['temperature'].min(),
            'max': recent_data['temperature'].max(),
            'multiplier': 0.25
        }

        print(f"\nTemperature statistics for {region_name}:")
        print(f"  Mean: {temp_stats['mean']:.1f} C")
        print(f"  Std: {temp_stats['std']:.1f} C")

        return temp_stats

    return None


def predict_with_temperature_range(model, X_input, target_scaler, temp_stats=None):
    scaled_prediction = model.predict(X_input)
    prediction = target_scaler.inverse_transform(scaled_prediction)

    if temp_stats and 'std' in temp_stats:
        temp_pred = prediction[0, 0]
        temp_std = temp_stats['std']
        multiplier = temp_stats.get('multiplier', 0.25)

        temp_range = {
            'prediction': temp_pred,
            'min': temp_pred - multiplier * temp_std,
            'max': temp_pred + multiplier * temp_std
        }

        return {
            'predictions': prediction[0],
            'temperature_range': temp_range,
            'temperature': temp_pred
        }

    return {
        'predictions': prediction[0],
        'temperature': prediction[0, 0] if prediction.shape[1] > 0 else None
    }


def calculate_rain_probability(predictions, region_name):
    humidity = predictions.get('humidity', 60)
    pressure = predictions.get('pressure', 1013)
    temp = predictions.get('temperature', 20)
    wind_speed = predictions.get('wind_speed', 3)

    if humidity > 85:
        base_prob = 0.7
    elif humidity > 75:
        base_prob = 0.5
    elif humidity > 65:
        base_prob = 0.3
    else:
        base_prob = 0.1

    if pressure < 1005:
        base_prob += 0.2
    elif pressure < 1010:
        base_prob += 0.1
    elif pressure > 1020:
        base_prob -= 0.1

    if temp > 35:
        base_prob -= 0.2

    if wind_speed > 10:
        base_prob += 0.1

    rain_prob = max(0, min(1, base_prob))

    if rain_prob > 0.7:
        category = "Very Likely"
        description = "High chance of rain, consider bringing an umbrella"
    elif rain_prob > 0.5:
        category = "Likely"
        description = "Good chance of rain"
    elif rain_prob > 0.3:
        category = "Possible"
        description = "Some chance of rain"
    else:
        category = "Unlikely"
        description = "Low chance of rain"

    return {
        'probability': rain_prob,
        'percentage': int(rain_prob * 100),
        'category': category,
        'description': description
    }


def apply_comprehensive_bias_correction(predictions, region_name, timestamps=None):
    bias_path = os.path.join(PARAMETERS_DIR, f'bias_{region_name}.pkl')

    if not os.path.exists(bias_path):
        create_bias_correction_file(region_name)
        bias_corrections = joblib.load(bias_path)
    else:
        bias_corrections = joblib.load(bias_path)

    corrected = predictions.copy()

    if len(corrected.shape) == 1:
        corrected = corrected.reshape(1, -1)

    # Temperature: Apply piecewise function based on hour
    if corrected.shape[-1] > 0 and 'temperature' in bias_corrections and timestamps is not None:
        temp_formula = BIAS_CORRECTIONS['temperature_formula']
        for i in range(corrected.shape[0]):
            if i < len(timestamps):
                hour = timestamps[i].hour % 24
                correction = temp_formula(hour)
                corrected[i, 0] += correction
        corrected[..., 0] = np.clip(corrected[..., 0], -20, 50)

    # Humidity: SUBTRACT humidity bias
    if corrected.shape[-1] > 1 and 'humidity' in bias_corrections:
        corrected[..., 1] -= BIAS_CORRECTIONS['humidity']
        corrected[..., 1] = np.clip(corrected[..., 1], 0, 100)

    # Wind speed: ADD 0.5 m/s for Japan regions only
    if corrected.shape[-1] > 2 and 'wind_speed' in bias_corrections:
        if region_name in ['Meguro', 'Bunkyo']:
            corrected[..., 2] += 0.5
        corrected[..., 2] = np.maximum(corrected[..., 2], 0)

    # Pressure: no change
    if corrected.shape[-1] > 3:
        corrected[..., 3] = np.clip(corrected[..., 3], 900, 1100)

    return corrected


def create_bias_correction_file(region_name):
    bias_corrections = {
        'temperature': {'bias': BIAS_CORRECTIONS['temperature_formula'], 'std_error': 2.0},
        'humidity': {'bias': BIAS_CORRECTIONS['humidity'], 'std_error': 5.0}, # Updated humidity bias
        'wind_speed': {'bias': BIAS_CORRECTIONS['wind_speed_japan'] if region_name in ['Meguro', 'Bunkyo'] else 0.0, 'std_error': 1.0},
        'pressure': {'bias': 0.0, 'std_error': 2.0},
    }

    bias_path = os.path.join(PARAMETERS_DIR, f'bias_{region_name}.pkl')
    joblib.dump(bias_corrections, bias_path)
    return bias_corrections


def evaluate_model(model, X_test, y_test, target_scaler, region_name):
    predictions_scaled = model.predict(X_test)
    if predictions_scaled.shape[1] == (24 * len(MODEL_CONFIG['target_variables'])):
        n_samples = predictions_scaled.shape[0]
        predictions_scaled_reshaped = predictions_scaled.reshape(n_samples * 24, len(MODEL_CONFIG['target_variables']))
        predictions_temp = target_scaler.inverse_transform(predictions_scaled_reshaped)
        predictions = predictions_temp.reshape(n_samples, 24, len(MODEL_CONFIG['target_variables']))

        if y_test.shape[1] == (24 * len(MODEL_CONFIG['target_variables'])):
            y_test_reshaped = y_test.reshape(n_samples * 24, len(MODEL_CONFIG['target_variables']))
            y_test_temp = target_scaler.inverse_transform(y_test_reshaped)
            y_test_original = y_test_temp.reshape(n_samples, 24, len(MODEL_CONFIG['target_variables']))
        else:
            y_test_original = target_scaler.inverse_transform(y_test)

        predictions = predictions[:, 0, :]
        y_test_original = y_test_original[:, 0, :] if len(y_test_original.shape) == 3 else y_test_original
    else:
        predictions = target_scaler.inverse_transform(predictions_scaled)
        y_test_original = target_scaler.inverse_transform(y_test)

    predictions = apply_comprehensive_bias_correction(predictions, region_name)

    mae = np.mean(np.abs(predictions - y_test_original), axis=0)
    rmse = np.sqrt(np.mean((predictions - y_test_original)**2, axis=0))

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'variable_names': ['temperature', 'humidity', 'wind_speed', 'pressure']
    }

    print(f"\nModel Evaluation for {region_name}:")
    for i, var_name in enumerate(metrics['variable_names'][:len(mae)]):
        print(f"  {var_name}: MAE={mae[i]:.2f}, RMSE={rmse[i]:.2f}")

    return metrics


def plot_training_history(history, region_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Training Analysis for {region_name}', fontsize=16)

    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if 'mae' in history:
        axes[1].plot(history['mae'], label='Training MAE', linewidth=2)
        axes[1].plot(history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{region_name}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def load_trained_model(region_name):
    model_path = os.path.join(PARAMETERS_DIR, f'lstm_{region_name}.keras')

    if not os.path.exists(model_path):
        print(f"Model not found for {region_name}")
        return None

    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None