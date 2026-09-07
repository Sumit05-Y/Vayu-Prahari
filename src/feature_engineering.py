import pandas as pd
import numpy as np
import os

def engineer_features(year):
    """Loads a cleaned dataset for a given year, applies feature engineering, and saves it."""
    input_file = f"../data/cleaned/{year}cleaned.csv"
    output_file = f"../data/cleaned/day3_features_{year}.csv"
    
    if not os.path.exists(input_file):
        print(f"Skipping {year}: File not found at {input_file}")
        return

    print(f"Processing {year} data...")
    df = pd.read_csv(input_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Core variables
    weather = df[["timestamp", "temperature", "humidity", "pressure"]].copy()

    # Previous-Value Features
    weather["temp_prev"] = weather["temperature"].shift(1)
    weather["rhum_prev"] = weather["humidity"].shift(1)
    weather["pres_prev"] = weather["pressure"].shift(1)

    # Difference Features
    weather["temp_diff"] = weather["temperature"].diff()
    weather["rhum_diff"] = weather["humidity"].diff()
    weather["pres_diff"] = weather["pressure"].diff()

    # Rate of Change
    weather["time_diff_hours"] = weather["timestamp"].diff().dt.total_seconds() / 3600
    weather["temp_rate"] = weather["temp_diff"] / weather["time_diff_hours"]
    weather["rhum_rate"] = weather["rhum_diff"] / weather["time_diff_hours"]
    weather["pres_rate"] = weather["pres_diff"] / weather["time_diff_hours"]

    # Rolling Statistics
    weather["temp_roll_mean_3"] = weather["temperature"].rolling(3).mean()
    weather["rhum_roll_mean_3"] = weather["humidity"].rolling(3).mean()
    weather["pres_roll_mean_3"] = weather["pressure"].rolling(3).mean()

    weather["temp_roll_std_3"] = weather["temperature"].rolling(3).std()
    weather["rhum_roll_std_3"] = weather["humidity"].rolling(3).std()
    weather["pres_roll_std_3"] = weather["pressure"].rolling(3).std()

    # Cyclical Time Encoding
    weather["hour"] = weather["timestamp"].dt.hour
    weather["month"] = weather["timestamp"].dt.month
    weather["hour_sin"] = np.sin(2 * np.pi * weather["hour"] / 24)
    weather["hour_cos"] = np.cos(2 * np.pi * weather["hour"] / 24)
    weather["month_sin"] = np.sin(2 * np.pi * weather["month"] / 12)
    weather["month_cos"] = np.cos(2 * np.pi * weather["month"] / 12)

    # Clean up NaNs and save
    weather_features = weather.dropna().reset_index(drop=True)
    weather_features.to_csv(output_file, index=False)
    print(f"Success! Saved {output_file} ({weather_features.shape[0]} rows, {weather_features.shape[1]} columns)")

# Execute the pipeline for all 5 years
for y in range(2020, 2025):
    engineer_features(y)

print("\nAll years processed successfully!")