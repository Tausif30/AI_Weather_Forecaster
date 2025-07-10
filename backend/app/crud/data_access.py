import pandas as pd
import os
from datetime import datetime
import requests
from dotenv import load_dotenv
import time
from ..config import REGIONS, METEOSOURCE_SUBSCRIPTION_TIER

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../.env'))

# API Key
METEOSOURCE_API_KEY = os.getenv("METEOSOURCE_API_KEY")

# Data directory
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
os.makedirs(RAW_DATA_DIR, exist_ok=True) 

# Place IDs
CORE_REGIONS_PLACE_IDS = {
    "Meguro": "meguro-11790374", 
    "Bunkyo": "bunkyo-11791351", 
    "Dhaka": "dhaka",
}


def load_raw_data_from_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)

    if 'date' in df.columns:
        df = df.rename(columns={'date': 'time'})

    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)

    return df


def get_all_regions_data(start_date_str, end_date_str):
    all_regions_data = {}

    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')

    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith('.csv'):
            region_name = os.path.splitext(filename)[0]
            file_path = os.path.join(RAW_DATA_DIR, filename)

            try:
                df = load_raw_data_from_csv(file_path)
                df = df[(df['time'] >= start_dt) & (df['time'] <= end_dt)]
                df = df.sort_values('time').reset_index(drop=True)

                if not df.empty:
                    all_regions_data[region_name] = df

            except Exception as e:
                print(f"Error loading {region_name}: {e}")

    return all_regions_data


def _make_meteosource_api_request(base_url, params):
    try:
        response = requests.get(base_url, params=params)
        time.sleep(0.5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API error: {e}")
        raise


def fetch_daily_historical_data_from_api(place_id, date_to_fetch):
    if not METEOSOURCE_API_KEY:
        raise ValueError("METEOSOURCE_API_KEY not set")

    date_str = date_to_fetch.strftime('%Y-%m-%d')
    base_url = f"https://www.meteosource.com/api/v1/{METEOSOURCE_SUBSCRIPTION_TIER}/time_machine"

    params = {
        "place_id": place_id,
        "date": date_str,
        "key": METEOSOURCE_API_KEY,
        "units": "auto"
    }

    data = _make_meteosource_api_request(base_url, params)

    if 'hourly' in data and data['hourly'] and 'data' in data['hourly']:
        hourly_records = data['hourly']['data']
        df = pd.DataFrame(hourly_records)
        df['time'] = df['hr'].apply(lambda x: f"{date_str}T{str(x).zfill(2)}:00:00")
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)
        df.columns = [col.replace('.', '_') for col in df.columns]
        return df

    return pd.DataFrame()


def append_daily_api_data_to_csv(region_name, place_id, date_to_fetch_dt):
    csv_file_path = os.path.join(RAW_DATA_DIR, f"{region_name}.csv")

    new_data_df = fetch_daily_historical_data_from_api(place_id, date_to_fetch_dt)

    if new_data_df.empty:
        return

    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')

        missing_cols = set(header) - set(new_data_df.columns)
        for col in missing_cols:
            new_data_df[col] = pd.NA

        new_data_df = new_data_df[header]
        new_data_df.to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        new_data_df.to_csv(csv_file_path, mode='w', header=True, index=False)


def fetch_current_weather_api(place_id):
    if not METEOSOURCE_API_KEY:
        raise ValueError("METEOSOURCE_API_KEY not set")

    base_url = f"https://www.meteosource.com/api/v1/{METEOSOURCE_SUBSCRIPTION_TIER}/point"

    params = {
        "place_id": place_id,
        "key": METEOSOURCE_API_KEY,
        "sections": "current",
        "units": "auto"
    }

    data = _make_meteosource_api_request(base_url, params)
    return data.get('current', {})