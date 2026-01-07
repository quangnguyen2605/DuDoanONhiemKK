import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for air pollution prediction
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.pollution_level_mapping = None
        
    def fit_transform(self, data):
        """
        Complete preprocessing pipeline
        Returns: X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf
        """
        print("🔧 Starting data preprocessing...")
        
        # Step 1: Handle missing values
        print("📋 Step 1: Handling missing values...")
        data_clean = self._handle_missing_values(data)
        
        # Step 2: Remove outliers
        print("📋 Step 2: Removing outliers...")
        data_clean = self._remove_outliers(data_clean)
        
        # Step 3: Feature engineering
        print("📋 Step 3: Feature engineering...")
        data_clean = self._feature_engineering(data_clean)
        
        # Step 4: Encode categorical variables
        print("📋 Step 4: Encoding categorical variables...")
        data_clean = self._encode_categorical(data_clean)
        
        # Step 5: Split features and targets
        print("📋 Step 5: Splitting features and targets...")
        X, y_reg, y_clf = self._split_features_targets(data_clean)
        
        # Step 6: Split train-test
        print("📋 Step 6: Splitting train-test sets...")
        X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = self._train_test_split(X, y_reg, y_clf)
        
        # Step 7: Feature scaling
        print("📋 Step 7: Feature scaling...")
        X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
        
        print("✅ Preprocessing completed successfully!")
        return X_train_scaled, X_test_scaled, y_train_reg, y_test_reg, y_train_clf, y_test_clf
    
    def _handle_missing_values(self, data):
        """Handle missing values using appropriate strategies"""
        data_copy = data.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        categorical_cols = data_copy.select_dtypes(include=['object']).columns
        
        # Handle numeric missing values with median imputation
        if len(numeric_cols) > 0:
            imputer_numeric = SimpleImputer(strategy='median')
            data_copy[numeric_cols] = imputer_numeric.fit_transform(data_copy[numeric_cols])
        
        # Handle categorical missing values with mode imputation
        if len(categorical_cols) > 0:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            data_copy[categorical_cols] = imputer_categorical.fit_transform(data_copy[categorical_cols])
        
        return data_copy
    
    def _remove_outliers(self, data):
        """Remove outliers using IQR method"""
        data_copy = data.copy()
        numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
        
        # Remove outliers for each numeric column
        for col in numeric_cols:
            if col in ['AQI']:  # Don't remove outliers from target variable
                continue
                
            Q1 = data_copy[col].quantile(0.25)
            Q3 = data_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers
            data_copy = data_copy[(data_copy[col] >= lower_bound) & (data_copy[col] <= upper_bound)]
        
        return data_copy
    
    def _feature_engineering(self, data):
        """Create additional features"""
        data_copy = data.copy()
        
        # Extract temporal features
        if 'Date' in data_copy.columns:
            data_copy['Date'] = pd.to_datetime(data_copy['Date'])
            data_copy['Hour'] = data_copy['Date'].dt.hour
            data_copy['DayOfWeek'] = data_copy['Date'].dt.dayofweek
            data_copy['Month'] = data_copy['Date'].dt.month
            data_copy['Season'] = data_copy['Month'].apply(self._get_season)
            
            # Create cyclical features for hour and month
            data_copy['Hour_sin'] = np.sin(2 * np.pi * data_copy['Hour'] / 24)
            data_copy['Hour_cos'] = np.cos(2 * np.pi * data_copy['Hour'] / 24)
            data_copy['Month_sin'] = np.sin(2 * np.pi * data_copy['Month'] / 12)
            data_copy['Month_cos'] = np.cos(2 * np.pi * data_copy['Month'] / 12)
        
        # Create pollution ratios
        if all(col in data_copy.columns for col in ['PM2.5', 'PM10']):
            data_copy['PM25_PM10_Ratio'] = data_copy['PM2.5'] / (data_copy['PM10'] + 1e-6)
        
        # Create air quality indicators
        if all(col in data_copy.columns for col in ['NO2', 'SO2', 'CO']):
            data_copy['Traffic_Pollution_Index'] = data_copy['NO2'] + data_copy['CO']
            data_copy['Industrial_Pollution_Index'] = data_copy['SO2']
        
        # Create weather interaction features
        if all(col in data_copy.columns for col in ['Temperature', 'Humidity']):
            data_copy['Temp_Humidity_Interaction'] = data_copy['Temperature'] * data_copy['Humidity']
        
        if all(col in data_copy.columns for col in ['Wind_Speed', 'PM2.5']):
            data_copy['Wind_Pollution_Interaction'] = data_copy['Wind_Speed'] / (data_copy['PM2.5'] + 1e-6)
        
        # Create pollution severity indicators
        pollution_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        available_pollution_cols = [col for col in pollution_cols if col in data_copy.columns]
        
        if available_pollution_cols:
            data_copy['Total_Pollution'] = data_copy[available_pollution_cols].sum(axis=1)
            data_copy['Max_Pollutant'] = data_copy[available_pollution_cols].max(axis=1)
            data_copy['Pollution_Std'] = data_copy[available_pollution_cols].std(axis=1)
        
        return data_copy
    
    def _get_season(self, month):
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _encode_categorical(self, data):
        """Encode categorical variables"""
        data_copy = data.copy()
        categorical_cols = data_copy.select_dtypes(include=['object']).columns
        
        # Encode pollution levels
        if 'Pollution_Level' in categorical_cols:
            self.pollution_level_mapping = {
                'Tốt': 0,
                'Trung Bình': 1,
                'Kém': 2,
                'Xấu': 3,
                'Rất Xấu': 4,
                'Nguy Hiểm': 5
            }
            data_copy['Pollution_Level_Encoded'] = data_copy['Pollution_Level'].map(self.pollution_level_mapping)
        
        # Encode other categorical variables
        for col in categorical_cols:
            if col != 'Pollution_Level':
                # Use label encoding for simplicity
                data_copy[f'{col}_Encoded'] = self.label_encoder.fit_transform(data_copy[col].astype(str))
                # Drop original categorical column
                data_copy = data_copy.drop(col, axis=1)
        
        return data_copy
    
    def _split_features_targets(self, data):
        """Split data into features and targets"""
        # Define feature columns (exclude targets and date)
        exclude_cols = ['AQI', 'Pollution_Level', 'Pollution_Level_Encoded', 'Date']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = data[feature_cols]
        y_reg = data['AQI']
        y_clf = data['Pollution_Level_Encoded']
        
        return X, y_reg, y_clf
    
    def _train_test_split(self, X, y_reg, y_clf, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
            X, y_reg, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
        )
        
        return X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf
    
    def _scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        # Separate numeric and categorical columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        # Scale only numeric features
        if len(numeric_cols) > 0:
            X_train_scaled = self.scaler.fit_transform(X_train[numeric_cols])
            X_test_scaled = self.scaler.transform(X_test[numeric_cols])
            
            # Convert back to DataFrame
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=X_test.index)
            
            # Add categorical columns back if any
            if len(categorical_cols) > 0:
                X_train_scaled = pd.concat([X_train_scaled, X_train[categorical_cols]], axis=1)
                X_test_scaled = pd.concat([X_test_scaled, X_test[categorical_cols]], axis=1)
        else:
            # No numeric columns to scale
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
        
        return X_train_scaled, X_test_scaled
    
    def transform_new_data(self, new_data):
        """Transform new data using fitted preprocessing pipeline"""
        data_copy = new_data.copy()
        
        # Apply same preprocessing steps
        data_copy = self._handle_missing_values(data_copy)
        data_copy = self._feature_engineering(data_copy)
        data_copy = self._encode_categorical(data_copy)
        
        # Select only the feature columns used in training
        X = data_copy[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        return X_scaled
    
    def get_preprocessing_info(self):
        """Get information about the preprocessing pipeline"""
        info = {
            'feature_columns': self.feature_columns,
            'pollution_level_mapping': self.pollution_level_mapping,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }
        return info
    
    def analyze_data_quality(self, data):
        """Analyze data quality before preprocessing"""
        quality_report = {
            'total_samples': len(data),
            'total_features': len(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object']).columns),
            'duplicate_rows': data.duplicated().sum(),
            'memory_usage': data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Statistical summary for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            quality_report['numeric_summary'] = data[numeric_cols].describe().to_dict()
        
        return quality_report

if __name__ == "__main__":
    # Test the preprocessing pipeline
    from data_generator import generate_aqi_data
    
    print("🧪 Testing data preprocessing pipeline...")
    
    # Generate sample data
    data = generate_aqi_data(1000)
    print(f"Generated dataset with {len(data)} samples")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Analyze data quality
    quality_report = preprocessor.analyze_data_quality(data)
    print(f"Data quality analysis completed")
    
    # Apply preprocessing
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = preprocessor.fit_transform(data)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Regression target shape: {y_train_reg.shape}")
    print(f"Classification target shape: {y_train_clf.shape}")
    
    print("✅ Preprocessing pipeline test completed successfully!")
