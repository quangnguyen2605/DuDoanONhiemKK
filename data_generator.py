import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_aqi_data(n_samples=10000):
    """
    Generate synthetic Hanoi AQI data for 2024-2026
    Includes realistic pollution patterns and seasonal variations
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate date range (2024-2026)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2026, 1, 7)
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Use all dates in the range for complete coverage
    dates = date_range
    n_samples = len(date_range)
    
    # Initialize data dictionary
    data = {'Date': dates}
    
    # Generate realistic pollution data with seasonal patterns
    for i, date in enumerate(dates):
        # Seasonal factors (higher pollution in winter, lower in summer)
        month = date.month
        hour = date.hour
        
        # Winter (Nov-Feb): Higher pollution
        if month in [11, 12, 1, 2]:
            seasonal_factor = 1.5
        # Spring (Mar-May): Moderate pollution
        elif month in [3, 4, 5]:
            seasonal_factor = 1.2
        # Summer (Jun-Aug): Lower pollution
        elif month in [6, 7, 8]:
            seasonal_factor = 0.8
        # Fall (Sep-Oct): Moderate pollution
        else:
            seasonal_factor = 1.1
        
        # Hourly patterns (rush hours have higher pollution)
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            hourly_factor = 1.3
        elif 10 <= hour <= 16:
            hourly_factor = 1.0
        else:
            hourly_factor = 0.9
        
        # Random variation
        random_factor = np.random.normal(1.0, 0.2)
        random_factor = max(0.5, min(1.5, random_factor))  # Clamp between 0.5 and 1.5
        
        # Combined factor
        combined_factor = seasonal_factor * hourly_factor * random_factor
        
        # Generate pollutant concentrations
        # PM2.5 (μg/m³) - Primary pollutant in Hanoi
        pm25_base = np.random.normal(35, 15)
        pm25_base = max(5, min(200, pm25_base))  # Clamp between 5 and 200
        pm25 = pm25_base * combined_factor
        
        # PM10 (μg/m³) - Usually higher than PM2.5
        pm10_base = np.random.normal(50, 20)
        pm10_base = max(10, min(300, pm10_base))
        pm10 = pm10_base * combined_factor
        
        # NO2 (μg/m³) - Traffic-related
        no2_base = np.random.normal(30, 15)
        no2_base = max(5, min(150, no2_base))
        no2 = no2_base * combined_factor
        
        # SO2 (μg/m³) - Industrial/traffic
        so2_base = np.random.normal(15, 8)
        so2_base = max(2, min(100, so2_base))
        so2 = so2_base * combined_factor
        
        # CO (mg/m³) - Incomplete combustion
        co_base = np.random.normal(2, 1)
        co_base = max(0.5, min(10, co_base))
        co = co_base * combined_factor
        
        # O3 (μg/m³) - Photochemical, higher in afternoon
        if 12 <= hour <= 16:
            o3_factor = 1.5
        else:
            o3_factor = 0.8
        o3_base = np.random.normal(60, 25)
        o3_base = max(10, min(200, o3_base))
        o3 = o3_base * o3_factor * seasonal_factor
        
        # Meteorological factors
        # Temperature (°C) - Seasonal variation
        temp_base = 25 + 10 * np.sin(2 * np.pi * (month - 1) / 12)
        temperature = temp_base + np.random.normal(0, 3)
        
        # Humidity (%) - Higher in summer
        humidity_base = 70 + 20 * np.sin(2 * np.pi * (month - 4) / 12)
        humidity = humidity_base + np.random.normal(0, 10)
        humidity = max(30, min(95, humidity))
        
        # Wind Speed (m/s) - Lower in winter
        wind_base = 3 - 1.5 * np.sin(2 * np.pi * (month - 1) / 12)
        wind_speed = wind_base + np.random.normal(0, 1)
        wind_speed = max(0.5, min(10, wind_speed))
        
        # Pressure (hPa) - Random variation around standard
        pressure = 1013 + np.random.normal(0, 10)
        
        # Rainfall (mm) - Seasonal (more in summer)
        if month in [6, 7, 8]:
            rain_prob = 0.3
        elif month in [9, 10, 4, 5]:
            rain_prob = 0.15
        else:
            rain_prob = 0.05
        
        if np.random.random() < rain_prob:
            rainfall = np.random.exponential(2)
            rainfall = min(50, rainfall)  # Cap at 50mm
        else:
            rainfall = 0
        
        # Store values
        if i == 0:
            data.update({
                'PM2.5': [pm25],
                'PM10': [pm10],
                'NO2': [no2],
                'SO2': [so2],
                'CO': [co],
                'O3': [o3],
                'Temperature': [temperature],
                'Humidity': [humidity],
                'Wind_Speed': [wind_speed],
                'Pressure': [pressure],
                'Rainfall': [rainfall]
            })
        else:
            data['PM2.5'].append(pm25)
            data['PM10'].append(pm10)
            data['NO2'].append(no2)
            data['SO2'].append(so2)
            data['CO'].append(co)
            data['O3'].append(o3)
            data['Temperature'].append(temperature)
            data['Humidity'].append(humidity)
            data['Wind_Speed'].append(wind_speed)
            data['Pressure'].append(pressure)
            data['Rainfall'].append(rainfall)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate AQI based on pollutant concentrations
    df['AQI'] = calculate_aqi(df)
    
    # Classify pollution levels
    df['Pollution_Level'] = classify_pollution_level(df['AQI'])
    
    # Add some missing values to make it more realistic
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
        df.loc[missing_indices[:len(missing_indices)//6], col] = np.nan
    
    # Add outliers
    outlier_indices = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
    for idx in outlier_indices:
        pollutant = np.random.choice(['PM2.5', 'PM10', 'NO2'])
        df.loc[idx, pollutant] *= np.random.uniform(2, 4)
    
    return df

def calculate_aqi(df):
    """
    Calculate AQI based on pollutant concentrations
    Uses simplified AQI calculation based on PM2.5 as primary pollutant
    """
    aqi_values = []
    
    for _, row in df.iterrows():
        # PM2.5 breakpoints (μg/m³)
        pm25 = row['PM2.5']
        
        if pm25 <= 12:
            aqi_pm25 = (50 / 12) * pm25
        elif pm25 <= 35.4:
            aqi_pm25 = 50 + ((100 - 50) / (35.4 - 12)) * (pm25 - 12)
        elif pm25 <= 55.4:
            aqi_pm25 = 100 + ((150 - 100) / (55.4 - 35.4)) * (pm25 - 35.4)
        elif pm25 <= 150.4:
            aqi_pm25 = 150 + ((200 - 150) / (150.4 - 55.4)) * (pm25 - 55.4)
        elif pm25 <= 250.4:
            aqi_pm25 = 200 + ((300 - 200) / (250.4 - 150.4)) * (pm25 - 150.4)
        elif pm25 <= 350.4:
            aqi_pm25 = 300 + ((400 - 300) / (350.4 - 250.4)) * (pm25 - 250.4)
        else:
            aqi_pm25 = 400 + ((500 - 400) / (500.4 - 350.4)) * (pm25 - 350.4)
        
        # PM10 breakpoints (μg/m³)
        pm10 = row['PM10']
        
        if pm10 <= 54:
            aqi_pm10 = (50 / 54) * pm10
        elif pm10 <= 154:
            aqi_pm10 = 50 + ((100 - 50) / (154 - 54)) * (pm10 - 54)
        elif pm10 <= 254:
            aqi_pm10 = 100 + ((150 - 100) / (254 - 154)) * (pm10 - 154)
        elif pm10 <= 354:
            aqi_pm10 = 150 + ((200 - 150) / (354 - 254)) * (pm10 - 254)
        elif pm10 <= 424:
            aqi_pm10 = 200 + ((300 - 200) / (424 - 354)) * (pm10 - 354)
        elif pm10 <= 504:
            aqi_pm10 = 300 + ((400 - 300) / (504 - 424)) * (pm10 - 424)
        else:
            aqi_pm10 = 400 + ((500 - 400) / (604 - 504)) * (pm10 - 504)
        
        # Take the maximum AQI from all pollutants
        aqi = max(aqi_pm25, aqi_pm10)
        aqi = min(500, aqi)  # Cap at 500
        aqi_values.append(aqi)
    
    return aqi_values

def classify_pollution_level(aqi_values):
    """
    Classify AQI values into pollution levels
    """
    levels = []
    for aqi in aqi_values:
        if aqi <= 50:
            levels.append("Tốt")
        elif aqi <= 100:
            levels.append("Trung Bình")
        elif aqi <= 150:
            levels.append("Kém")
        elif aqi <= 200:
            levels.append("Xấu")
        elif aqi <= 300:
            levels.append("Rất Xấu")
        else:
            levels.append("Nguy Hiểm")
    return levels

def get_data_info(df):
    """
    Get basic information about the dataset
    """
    info = {
        'total_samples': len(df),
        'date_range': (df['Date'].min(), df['Date'].max()),
        'features': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'pollution_levels': df['Pollution_Level'].value_counts().to_dict(),
        'aqi_stats': {
            'mean': df['AQI'].mean(),
            'std': df['AQI'].std(),
            'min': df['AQI'].min(),
            'max': df['AQI'].max(),
            'median': df['AQI'].median()
        }
    }
    return info

if __name__ == "__main__":
    # Generate sample data
    print("Generating Hanoi AQI dataset...")
    df = generate_aqi_data(5000)
    
    # Display information
    info = get_data_info(df)
    print(f"Dataset generated with {info['total_samples']} samples")
    print(f"Date range: {info['date_range'][0]} to {info['date_range'][1]}")
    print(f"Features: {info['features']}")
    print(f"Pollution levels distribution: {info['pollution_levels']}")
    print(f"AQI statistics: {info['aqi_stats']}")
    
    # Save to CSV
    df.to_csv('hanoi_aqi_data.csv', index=False)
    print("Dataset saved to 'hanoi_aqi_data.csv'")
