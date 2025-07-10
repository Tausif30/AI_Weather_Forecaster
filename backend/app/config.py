#Model Configuration
MODEL_CONFIG = {
    'model_type': 'advanced',
    'look_back_hours': 24 * 7,
    'look_ahead_hours': 24,
    'target_variables': ['temperature', 'humidity', 'wind_speed', 'pressure'],
    'epochs': 20,
    'batch_size': 64,
    'early_stopping_patience': 6,
    'reduce_lr_patience': 3,
    'validation_split': 0.25,
    'test_months': 6,
    'lstm_units': [48, 32],
    'dropout_rate': 0.5,
    'learning_rate': 0.001,
    'use_attention': False,
    'use_bidirectional': True,
    'use_augmentation': False,
    'scaler_type': 'robust'
}

# Data Frame
REGIONS = ["Meguro", "Bunkyo", "Dhaka"]
DATA_START_DATE = "2022-06-01"
DATA_END_DATE = "2025-07-09"

# API Configuration
METEOSOURCE_SUBSCRIPTION_TIER = "flexi"

def get_temperature_bias(h):
    if 0 <= h < 4:
        return 3 - (12 * (h - 0) / 32)
    elif 4 <= h < 14:
        return 5 + (72 * (h - 4) / 80)
    elif 14 <= h < 24:
        return 10 - (56 * (h - 14) / 80)
    else:
        return 0

BIAS_CORRECTIONS = {
    'temperature_formula': get_temperature_bias,
    'humidity': 25.0,
    'wind_speed_japan': -0.5,
}