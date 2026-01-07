import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_real_aqi_data(csv_path='hanoi_air_quality_recent.csv'):
    """
    Load real AQI data from CSV file and process it for ML models
    """
    try:
        # Load the real data
        df = pd.read_csv(csv_path)
        print(f"✅ Đã tải dữ liệu thật: {len(df)} bản ghi")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Rename columns to match our expected format
        column_mapping = {
            'date': 'Date',
            'PM25': 'PM2.5',
            'PM10': 'PM10', 
            'NO2': 'NO2',
            'SO2': 'SO2',
            'CO': 'CO',
            'O3': 'O3',
            'temperature': 'Temperature',
            'humidity': 'Humidity',
            'wind_speed': 'Wind_Speed',
            'pressure': 'Pressure',
            'Rainfall': 'Rainfall'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Calculate AQI based on pollutant concentrations
        df['AQI'] = calculate_aqi_from_pollutants(df)
        
        # Classify pollution levels
        df['Pollution_Level'] = classify_pollution_level(df['AQI'])
        
        # Handle Rainfall column - if it's empty or all zeros, generate realistic rainfall data
        if 'Rainfall' not in df.columns or df['Rainfall'].isna().all() or (df['Rainfall'] == 0).all():
            print("🌧️ Cột Rainfall trống, đang tạo dữ liệu mưa giả lập thực tế...")
            df['Rainfall'] = generate_realistic_rainfall(df)
        
        # Handle any missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Remove any rows with all zeros in pollutants
        pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        df = df[~(df[pollutant_cols] == 0).all(axis=1)]
        
        print(f"📊 Kết quả xử lý dữ liệu:")
        print(f"- Tổng bản ghi: {len(df)}")
        print(f"- Khoảng thời gian: {df['Date'].min()} đến {df['Date'].max()}")
        print(f"- AQI trung bình: {df['AQI'].mean():.2f}")
        print(f"- Phân phối mức độ ô nhiễm: {df['Pollution_Level'].value_counts().to_dict()}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {csv_path}")
        print("🔄 Sử dụng dữ liệu giả lập thay thế...")
        return None
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        return None

def calculate_aqi_from_pollutants(df):
    """
    Calculate AQI based on real pollutant concentrations
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

def generate_realistic_rainfall(df):
    """
    Generate realistic rainfall data based on weather patterns and seasons
    """
    np.random.seed(42)  # For reproducibility
    
    rainfall_data = []
    
    for _, row in df.iterrows():
        date = row['Date']
        humidity = row['Humidity']
        pressure = row['Pressure']
        
        # Seasonal rainfall patterns for Hanoi
        month = date.month
        
        # Define seasonal rainfall probability and intensity
        if 3 <= month <= 5:  # Spring (March-May)
            rain_prob = 0.3
            rain_intensity = 2.5
        elif 6 <= month <= 8:  # Summer (June-August) - rainy season
            rain_prob = 0.7
            rain_intensity = 8.0
        elif 9 <= month <= 11:  # Fall (September-November)
            rain_prob = 0.4
            rain_intensity = 3.5
        else:  # Winter (December-February)
            rain_prob = 0.2
            rain_intensity = 1.5
        
        # Adjust probability based on humidity and pressure
        if humidity > 80:
            rain_prob += 0.2
        elif humidity < 60:
            rain_prob -= 0.1
            
        if pressure < 1010:
            rain_prob += 0.1
        elif pressure > 1020:
            rain_prob -= 0.1
        
        # Ensure probability stays in valid range
        rain_prob = max(0.05, min(0.95, rain_prob))
        
        # Generate rainfall
        if np.random.random() < rain_prob:
            # Generate rainfall amount with some variation
            rainfall = np.random.exponential(rain_intensity)
            # Cap at reasonable maximum (50mm for single day)
            rainfall = min(rainfall, 50.0)
        else:
            rainfall = 0.0
        
        rainfall_data.append(rainfall)
    
    return rainfall_data

def classify_pollution_level(aqi_values):
    """
    Classify AQI values into pollution levels (Vietnamese)
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
    # Test loading real data
    print("🧪 Kiểm tra tải dữ liệu thật...")
    df = load_real_aqi_data()
    
    if df is not None:
        info = get_data_info(df)
        print(f"✅ Tải dữ liệu thành công!")
        print(f"Dữ liệu: {info['total_samples']} mẫu")
        print(f"Thời gian: {info['date_range'][0]} đến {info['date_range'][1]}")
        print(f"Đặc trưng: {info['features']}")
        print(f"Phân phối ô nhiễm: {info['pollution_levels']}")
        print(f"Thống kê AQI: {info['aqi_stats']}")
    else:
        print("❌ Không thể tải dữ liệu thật")
