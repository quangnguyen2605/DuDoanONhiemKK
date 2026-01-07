# 🌫️ Hanoi Air Pollution Prediction System

## 📋 Project Overview

This comprehensive demo application implements machine learning algorithms to predict air quality index (AQI) and classify pollution levels in Hanoi, Vietnam. The system compares four different algorithms to determine the optimal approach for air pollution forecasting.

### 🎯 Objectives

- **Regression Task**: Predict continuous AQI values using environmental parameters
- **Classification Task**: Classify pollution levels into categories (Good, Moderate, Poor, etc.)
- **Algorithm Comparison**: Evaluate Linear Regression, Decision Tree (CART), SVM, and Logistic Regression
- **Interactive Demo**: Provide user-friendly interface for real-time predictions

### 🏗️ Project Structure

```
BLTHocMay/
├── main.py                    # Main Streamlit application
├── data_generator.py          # Synthetic Hanoi AQI data generation
├── data_preprocessing.py      # Data preprocessing pipeline
├── models.py                  # ML model implementations
├── evaluation.py              # Model evaluation and comparison
├── visualization.py          # Data visualization tools
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone or download the project files**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run main.py
   ```

The application will open in your web browser at `http://localhost:8501`

## 📊 Dataset Information

### Synthetic Hanoi AQI Dataset (2024-2025)

The system generates realistic air pollution data for Hanoi with the following characteristics:

#### **Pollutants Measured**:
- **PM2.5** (μg/m³) - Fine particulate matter
- **PM10** (μg/m³) - Coarse particulate matter  
- **NO₂** (μg/m³) - Nitrogen dioxide
- **SO₂** (μg/m³) - Sulfur dioxide
- **CO** (mg/m³) - Carbon monoxide
- **O₃** (μg/m³) - Ozone

#### **Meteorological Factors**:
- Temperature (°C)
- Humidity (%)
- Wind Speed (m/s)
- Atmospheric Pressure (hPa)
- Rainfall (mm)

#### **Target Variables**:
- **AQI** (Air Quality Index) - Continuous value (0-500)
- **Pollution_Level** - Categorical classification:
  - Good (0-50)
  - Moderate (51-100)
  - Kém/Poor (101-150)
  - Xấu/Unhealthy (151-200)
  - Rất xấu/Very Unhealthy (201-300)
  - Nguy hại/Hazardous (301+)

### Data Features

- **Temporal Patterns**: Seasonal variations, hourly cycles
- **Realistic Correlations**: Pollutants interact with weather conditions
- **Missing Values**: 2% missing data for realism
- **Outliers**: 1% extreme values for robustness testing

## 🤖 Machine Learning Algorithms

### 1. **Linear Regression** (Mạnh)
- **Purpose**: AQI value prediction
- **Strengths**: Simple, interpretable, fast training
- **Mathematical Foundation**: $y = \beta_0 + \beta_1x_1 + ... + \beta_nx_n + \epsilon$

### 2. **Decision Tree (CART)** (Quang)
- **Purpose**: AQI prediction and feature importance analysis
- **Strengths**: Non-linear relationships, easy visualization
- **Algorithm**: Classification and Regression Trees

### 3. **Support Vector Machine (SVM)** (Tiến)
- **Purpose**: Pollution level classification
- **Strengths**: High accuracy, effective in high-dimensional spaces
- **Kernels**: Linear, RBF, Polynomial

### 4. **Logistic Regression** (Thương)
- **Purpose**: Pollution level classification
- **Strengths**: Probabilistic output, fast prediction
- **Mathematical Foundation**: $P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1x_1 + ... + \beta_nx_n)}}$

## 📈 Evaluation Metrics

### Regression Metrics
- **MSE** (Mean Squared Error): $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **RMSE** (Root Mean Squared Error): $\sqrt{MSE}$
- **MAE** (Mean Absolute Error): $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$
- **R²** (Coefficient of Determination): $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$

### Classification Metrics
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{Precision \times Recall}{Precision + Recall}$

## 🎨 Application Features

### 1. **Data Generation & Exploration**
- Automatic synthetic dataset generation
- Interactive data visualization
- Statistical analysis and correlation matrices
- Time series analysis

### 2. **Data Preprocessing**
- Missing value imputation
- Outlier detection and removal
- Feature engineering
- Data scaling and encoding

### 3. **Model Training**
- Multi-algorithm training
- Hyperparameter tuning (Grid Search)
- Cross-validation
- Performance comparison

### 4. **Model Evaluation**
- Comprehensive metric analysis
- Visual performance comparison
- Feature importance analysis
- Best model recommendations

### 5. **Real-time Prediction**
- Interactive parameter input
- Instant AQI prediction
- Pollution level classification
- Health recommendations

### 6. **Conclusions & Recommendations**
- Algorithm performance summary
- Use case recommendations
- Future improvement suggestions

## 🔧 Usage Guide

### Step-by-Step Workflow

1. **Launch the Application**
   ```bash
   streamlit run main.py
   ```

2. **Data Generation**
   - Navigate to "Data Generation & Exploration"
   - Review dataset statistics and visualizations

3. **Data Preprocessing**
   - Go to "Data Preprocessing" section
   - Click "Apply Preprocessing" to clean the data

4. **Model Training**
   - Select desired algorithms in "Model Training"
   - Configure training parameters
   - Click "🚀 Train Selected Models"

5. **Model Evaluation**
   - View performance comparisons in "Model Evaluation & Comparison"
   - Analyze detailed metrics and visualizations

6. **Real-time Prediction**
   - Use "Real-time Prediction" for instant forecasts
   - Input environmental parameters
   - Get AQI predictions and health advice

### Parameter Input Guidelines

#### **Pollutant Concentrations**:
- PM2.5: 5-200 μg/m³ (typical range)
- PM10: 10-300 μg/m³
- NO₂: 5-150 μg/m³
- SO₂: 2-100 μg/m³
- CO: 0.5-10 mg/m³
- O₃: 10-200 μg/m³

#### **Weather Parameters**:
- Temperature: -10°C to 50°C
- Humidity: 30% to 95%
- Wind Speed: 0.5 to 10 m/s
- Pressure: 900 to 1100 hPa
- Rainfall: 0 to 100 mm

## 📊 Expected Results

### Algorithm Performance (Typical Results)

| Algorithm | Task | RMSE | R² | Accuracy | F1-Score |
|-----------|------|------|----|----------|----------|
| Linear Regression | AQI Prediction | 18-25 | 0.82-0.88 | - | - |
| Decision Tree | AQI Prediction | 15-22 | 0.85-0.90 | - | - |
| SVM | Classification | - | - | 0.84-0.89 | 0.82-0.87 |
| Logistic Regression | Classification | - | - | 0.80-0.86 | 0.78-0.84 |

### Best Use Cases

- **Highest Accuracy**: SVM for classification
- **Fastest Training**: Linear Regression for regression
- **Best Interpretability**: Decision Tree for analysis
- **Most Reliable**: Logistic Regression for production

## 🔬 Technical Implementation

### Data Preprocessing Pipeline

1. **Missing Value Handling**
   - Numeric variables: Median imputation
   - Categorical variables: Mode imputation

2. **Outlier Removal**
   - IQR method with 1.5×IQR threshold
   - Applied to all numeric features except targets

3. **Feature Engineering**
   - Temporal features: Hour, day of week, month, season
   - Cyclical encoding: sin/cos transformations
   - Interaction terms: pollutant ratios, weather interactions
   - Composite indices: traffic/industrial pollution

4. **Data Scaling**
   - StandardScaler for numeric features
   - Label encoding for categorical variables

### Model Training Process

1. **Data Splitting**
   - 80% training, 20% testing
   - Stratified sampling for classification

2. **Cross-Validation**
   - 5-fold CV for robust evaluation
   - Prevents overfitting

3. **Hyperparameter Tuning**
   - Grid Search CV (optional)
   - Optimizes for RMSE (regression) or accuracy (classification)

## 🚀 Future Improvements

### **Technical Enhancements**
- **Deep Learning**: LSTM/GRU for time series prediction
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Feature Selection**: Recursive Feature Elimination
- **Hyperparameter Optimization**: Bayesian Optimization

### **Data Improvements**
- **Real Data Integration**: Connect to actual monitoring stations
- **Geographic Expansion**: Include other Vietnam cities
- **Additional Features**: Traffic data, industrial emissions
- **Temporal Resolution**: Real-time data streaming

### **Application Enhancements**
- **Mobile App**: iOS/Android applications
- **API Development**: RESTful API for integration
- **Alert System**: Automated pollution warnings
- **Historical Analysis**: Long-term trend analysis

## 🏆 Project Achievements

### **Academic Contributions**
- ✅ Comprehensive ML algorithm comparison
- ✅ Realistic synthetic dataset generation
- ✅ Complete preprocessing pipeline
- ✅ Robust evaluation framework

### **Practical Applications**
- ✅ Interactive web interface
- ✅ Real-time prediction capability
- ✅ Health recommendation system
- ✅ Performance optimization

### **Technical Excellence**
- ✅ Modular, maintainable code
- ✅ Comprehensive documentation
- ✅ Error handling and validation
- ✅ Scalable architecture

## 👥 Team Contributions

| Member | Algorithm | Responsibilities |
|--------|-----------|------------------|
| Mạnh | Linear Regression | Algorithm implementation, mathematical foundations |
| Quang | Decision Tree (CART) | Tree-based methods, feature importance analysis |
| Tiến | SVM | Support vector machines, kernel optimization |
| Thương | Logistic Regression | Classification algorithms, probabilistic modeling |

## 📞 Support & Contact

For questions, issues, or contributions:

1. **Documentation**: Refer to this README and inline code comments
2. **Issues**: Check console output for error messages
3. **Debugging**: Use the built-in data exploration tools
4. **Performance**: Monitor training time and memory usage

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning algorithms
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

---

**🎉 Thank you for using the Hanoi Air Pollution Prediction System!**

This project demonstrates the practical application of machine learning in environmental monitoring and public health protection. The comprehensive comparison of algorithms provides valuable insights for both academic research and practical implementation.
