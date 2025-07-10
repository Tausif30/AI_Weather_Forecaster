from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field 
from datetime import datetime, timedelta
import numpy as np
import os
import pandas as pd
import joblib

from .crud import data_access
from .models import weather_model
from .services import data_preprocessing
from .config import MODEL_CONFIG, BIAS_CORRECTIONS

app = FastAPI(
    title="AI Weather Forecast API",
    description="Weather predictions using LSTM",
    version="0.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOOK_BACK_HOURS = MODEL_CONFIG['look_back_hours']


class ForecastRequest(BaseModel):
    region_name: str
    forecast_hours: int = Field(24, gt=0, le=72)


class ForecastResponse(BaseModel):
    predicted_temperature: float
    predicted_max_temperature: float
    predicted_min_temperature: float
    predicted_humidity: float
    predicted_wind_speed: float
    predicted_pressure: float
    rain_probability: float
    rain_percentage: int
    rain_category: str
    rain_description: str
    temperature_unit: str = "°C"
    wind_speed_unit: str = "m/s"
    pressure_unit: str = "hPa"
    forecast_for_time: datetime
    forecast_date: str
    region: str
    temperature_std: float
    last_known_wind_direction: str = 'N/A'


class LiveWeatherRequest(BaseModel):
    city_name: str


class LiveWeatherResponse(BaseModel):
    temperature: float
    feels_like: float
    humidity: float
    wind_speed: float
    wind_direction: str
    pressure: float
    weather_description: str
    cloud_cover: float
    temperature_unit: str = "°C"
    wind_speed_unit: str = "m/s"
    pressure_unit: str = "hPa"


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "AI Weather Forecast API is running"}


@app.post("/forecast", response_model=ForecastResponse)
async def get_weather_forecast(request: ForecastRequest):
    region_name = request.region_name
    forecast_hours_requested = request.forecast_hours # Store original requested hours
    
    model = weather_model.load_trained_model(region_name)
    feature_scaler, target_scaler = data_preprocessing.load_scalers(region_name)
    
    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(404, f"Model or scalers not found for '{region_name}'")
    
    config_path = os.path.join(weather_model.PARAMETERS_DIR, f'config_{region_name}.pkl')
    model_look_ahead = 1
    
    if os.path.exists(config_path):
        try:
            config = joblib.load(config_path)
            model_look_ahead = config.get('look_ahead_hours', 1)
        except:
            pass
    
    temp_stats = None
    stats_path = os.path.join(weather_model.PARAMETERS_DIR, f'temp_stats_{region_name}.pkl')
    if os.path.exists(stats_path):
        try:
            temp_stats = joblib.load(stats_path)
        except:
            pass
    
    try:
        csv_path = os.path.join(data_access.RAW_DATA_DIR, f"{region_name}.csv")
        df = data_access.load_raw_data_from_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').set_index('time')
        
        recent_data = df.tail(LOOK_BACK_HOURS).copy()
        
        if len(recent_data) < LOOK_BACK_HOURS:
            raise HTTPException(25, f"Model doesn't work well for forecasting more than a day.")
        
        if temp_stats is None:
            temp_stats = weather_model.calculate_temperature_range_stats(df, region_name)
        
        X_input = data_preprocessing.preprocess_data_for_inference(
            recent_data.reset_index(), 
            feature_scaler,
            LOOK_BACK_HOURS
        )
        
        if X_input is None:
            raise HTTPException(25, "Failed to preprocess data")
        
        num_target_variables = len(MODEL_CONFIG['target_variables'])
        last_timestamp = recent_data.index[-1]
        
        if forecast_hours_requested == 0:
            target_forecast_time = last_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            if target_forecast_time <= last_timestamp:
                target_forecast_time += timedelta(days=1)
            
        elif forecast_hours_requested == 24:
            next_midnight = last_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            if next_midnight <= last_timestamp:
                next_midnight += timedelta(days=1)
            target_forecast_time = next_midnight + timedelta(hours=24)
            
        else:
            target_forecast_time = last_timestamp + timedelta(hours=forecast_hours_requested)
        
        actual_offset_hours = int((target_forecast_time - last_timestamp).total_seconds() / 3600)

        if model_look_ahead == 24:
            prediction = model.predict(X_input, verbose=0)
            prediction_reshaped = prediction.reshape(24, num_target_variables)
            predictions_original = target_scaler.inverse_transform(prediction_reshaped)
            hour_index_for_prediction_array = min(max(actual_offset_hours - 1, 0), 23)
            selected_prediction = predictions_original[hour_index_for_prediction_array].copy()
            
            # Apply bias correction
            selected_prediction_corrected = weather_model.apply_comprehensive_bias_correction(
                selected_prediction.reshape(1, -1), region_name, timestamps=[target_forecast_time]
            )[0]
            
            pred_dict = {
                'temperature': selected_prediction_corrected[0],
                'humidity': selected_prediction_corrected[1],
                'wind_speed': max(selected_prediction_corrected[2], 0),
                'pressure': selected_prediction_corrected[3],
            }
        else:
            result = weather_model.predict_with_temperature_range(
                model, X_input, target_scaler, temp_stats
            )
            predictions = result['predictions']
            
            current_forecast_time = last_timestamp + timedelta(hours=forecast_hours_requested)
            predictions_corrected = weather_model.apply_comprehensive_bias_correction(
                predictions.reshape(1, -1), region_name, timestamps=[current_forecast_time]
            )[0]
            
            pred_dict = {
                'temperature': predictions_corrected[0],
                'humidity': predictions_corrected[1] if len(predictions_corrected) > 1 else 60.0,
                'wind_speed': max(predictions_corrected[2] if len(predictions_corrected) > 2 else 3.0, 0),
                'pressure': predictions_corrected[3] if len(predictions_corrected) > 3 else 1013.0,
            }
        
        if temp_stats and 'std' in temp_stats:
            temp_std = temp_stats['std']
            multiplier = temp_stats.get('multiplier', 0.5)
            max_temp = pred_dict['temperature'] + multiplier * temp_std
            min_temp = pred_dict['temperature'] - multiplier * temp_std
        else:
            max_temp = pred_dict['temperature'] + 3
            min_temp = pred_dict['temperature'] - 3
            temp_std = 3.0
        
        rain_info = weather_model.calculate_rain_probability(pred_dict, region_name)
        
        forecast_timestamp = target_forecast_time
        
        last_data = recent_data.iloc[-1].to_dict()
        wind_direction = last_data.get('wind_dir', 'N/A')
        
        return ForecastResponse(
            predicted_temperature=round(pred_dict['temperature'], 1),
            predicted_max_temperature=round(max_temp, 1),
            predicted_min_temperature=round(min_temp, 1),
            predicted_humidity=round(pred_dict['humidity'], 1),
            predicted_wind_speed=round(pred_dict['wind_speed'], 1),
            predicted_pressure=round(pred_dict['pressure'], 1),
            rain_probability=rain_info['probability'],
            rain_percentage=rain_info['percentage'],
            rain_category=rain_info['category'],
            rain_description=rain_info['description'],
            forecast_for_time=forecast_timestamp,
            forecast_date=forecast_timestamp.strftime('%Y-%m-%d'),
            region=region_name,
            temperature_std=round(temp_std, 2),
            last_known_wind_direction=wind_direction
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(25, f"Error generating forecast: {str(e)}")


@app.post("/live_weather", response_model=LiveWeatherResponse)
async def get_live_weather(request: LiveWeatherRequest):
    place_id_lookup = {
        "meguro": "meguro-11790374",
        "bunkyo": "bunkyo-11791351",
        "dhaka": "dhaka"
    }
    
    place_id = place_id_lookup.get(request.city_name.lower(), request.city_name.lower())
    
    try:
        data = data_access.fetch_current_weather_api(place_id)
        
        if not data:
            raise HTTPException(404, f"No weather data found for '{request.city_name}'")
        
        return LiveWeatherResponse(
            temperature=round(data.get('temperature', 20), 1),
            feels_like=round(data.get('feels_like', 20), 1),
            humidity=round(data.get('humidity', 60), 1),
            wind_speed=round(data.get('wind', {}).get('speed', 3), 1),
            wind_direction=data.get('wind', {}).get('dir', 'N/A'),
            pressure=round(data.get('pressure', 1013), 1),
            weather_description=data.get('weather', 'N/A'),
            cloud_cover=round(data.get('cloud_cover', 0), 1),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(25, f"Error fetching weather: {str(e)}")


@app.on_event("startup")
async def startup_event():
    parameters_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'parameters')
    
    print("Starting AI Weather Forecast API...")
    
    for region in ["Meguro", "Bunkyo", "Dhaka"]:
        model_path = os.path.join(parameters_dir, f'lstm_{region}.keras')
        if os.path.exists(model_path):
            print(f"  - {region}: Model found")
        else:
            print(f"  - {region}: Model not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)