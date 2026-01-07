import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class VisualizationTools:
    """
    Comprehensive visualization tools for air pollution analysis
    """
    
    def __init__(self):
        self.pollution_level_colors = {
            'Tốt': '#00e400',
            'Trung Bình': '#ffff00', 
            'Kém': '#ff7e00',
            'Xấu': '#ff0000',
            'Rất Xấu': '#8f3f97',
            'Nguy Hiểm': '#7e0023'
        }
        
        self.pollution_level_order = ['Tốt', 'Trung Bình', 'Kém', 'Xấu', 'Rất Xấu', 'Nguy Hiểm']
    
    def create_data_overview_dashboard(self, data):
        """Create comprehensive data overview dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'AQI Distribution', 'Pollution Levels', 'Monthly AQI Trends',
                'PM2.5 vs PM10', 'Temperature vs AQI', 'Hourly Patterns',
                'Correlation Heatmap', 'Pollution Level Pie Chart', 'Wind Speed Impact'
            ],
            specs=[
                [{"type": "histogram"}, {"type": "pie"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "pie"}, {"type": "scatter"}]
            ]
        )
        
        # 1. AQI Distribution
        fig.add_trace(
            go.Histogram(x=data['AQI'], nbinsx=30, name='AQI', marker_color='blue'),
            row=1, col=1
        )
        
        # 2. Pollution Levels (Bar)
        pollution_counts = data['Pollution_Level'].value_counts()
        fig.add_trace(
            go.Bar(x=pollution_counts.index, y=pollution_counts.values, 
                  name='Pollution Levels', marker_color=list(self.pollution_level_colors.values())),
            row=1, col=2
        )
        
        # 3. Monthly AQI Trends
        if 'Date' in data.columns:
            data['Month'] = pd.to_datetime(data['Date']).dt.month
            monthly_aqi = data.groupby('Month')['AQI'].mean()
            fig.add_trace(
                go.Scatter(x=monthly_aqi.index, y=monthly_aqi.values, 
                          mode='lines+markers', name='Monthly AQI'),
                row=1, col=3
            )
        
        # 4. PM2.5 vs PM10
        fig.add_trace(
            go.Scatter(x=data['PM2.5'], y=data['PM10'], mode='markers', 
                      name='PM2.5 vs PM10', opacity=0.6),
            row=2, col=1
        )
        
        # 5. Temperature vs AQI
        fig.add_trace(
            go.Scatter(x=data['Temperature'], y=data['AQI'], mode='markers', 
                      name='Temperature vs AQI', opacity=0.6),
            row=2, col=2
        )
        
        # 6. Hourly Patterns
        if 'Date' in data.columns:
            data['Hour'] = pd.to_datetime(data['Date']).dt.hour
            hourly_aqi = data.groupby('Hour')['AQI'].mean()
            fig.add_trace(
                go.Bar(x=hourly_aqi.index, y=hourly_aqi.values, name='Hourly AQI'),
                row=2, col=3
            )
        
        # 7. Correlation Heatmap
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                      colorscale='RdBu', name='Correlation'),
            row=3, col=1
        )
        
        # 8. Pollution Level Pie Chart
        fig.add_trace(
            go.Pie(values=pollution_counts.values, names=pollution_counts.index,
                  name='Pollution Distribution'),
            row=3, col=2
        )
        
        # 9. Wind Speed Impact
        fig.add_trace(
            go.Scatter(x=data['Wind_Speed'], y=data['AQI'], mode='markers',
                      name='Wind Speed vs AQI', opacity=0.6),
            row=3, col=3
        )
        
        fig.update_layout(
            height=1200,
            title_text="Hanoi Air Pollution Data Overview Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_time_series_analysis(self, data):
        """Create detailed time series analysis"""
        
        # Ensure Date column is datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date')
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['AQI Time Series', 'Pollutant Trends', 'Weather Impact'],
            vertical_spacing=0.08
        )
        
        # 1. AQI Time Series
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['AQI'], mode='lines', name='AQI',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add pollution level thresholds
        fig.add_hline(y=50, line_dash="dash", line_color="green", 
                     annotation_text="Good", row=1, col=1)
        fig.add_hline(y=100, line_dash="dash", line_color="yellow", 
                     annotation_text="Moderate", row=1, col=1)
        fig.add_hline(y=150, line_dash="dash", line_color="orange", 
                     annotation_text="Unhealthy", row=1, col=1)
        
        # 2. Pollutant Trends
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        colors = ['red', 'orange', 'purple', 'brown', 'gray', 'cyan']
        
        for i, (pollutant, color) in enumerate(zip(pollutants, colors)):
            if pollutant in data.columns:
                fig.add_trace(
                    go.Scatter(x=data['Date'], y=data[pollutant], mode='lines',
                              name=pollutant, line=dict(color=color)),
                    row=2, col=1
                )
        
        # 3. Weather Impact
        weather_vars = ['Temperature', 'Humidity', 'Wind_Speed']
        weather_colors = ['red', 'blue', 'green']
        
        for i, (var, color) in enumerate(zip(weather_vars, weather_colors)):
            if var in data.columns:
                # Normalize weather variables for better visualization
                normalized = (data[var] - data[var].min()) / (data[var].max() - data[var].min())
                fig.add_trace(
                    go.Scatter(x=data['Date'], y=normalized * 100, mode='lines',
                              name=var, line=dict(color=color)),
                    row=3, col=1
                )
        
        fig.update_layout(
            height=900,
            title_text="Hanoi Air Pollution Time Series Analysis",
            showlegend=True
        )
        
        return fig
    
    def create_pollution_analysis_charts(self, data):
        """Create pollution-specific analysis charts"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Pollutant Concentration Boxplots', 
                'Pollution Level vs Pollutants',
                'Seasonal Pollution Patterns',
                'Pollution Hotspot Analysis'
            ],
            specs=[[{"type": "box"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Pollutant Concentration Boxplots
        pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        for pollutant in pollutants:
            if pollutant in data.columns:
                fig.add_trace(
                    go.Box(y=data[pollutant], name=pollutant),
                    row=1, col=1
                )
        
        # 2. Pollution Level vs Pollutants
        pollution_level_avg = data.groupby('Pollution_Level')[pollutants].mean()
        for pollutant in pollutants:
            if pollutant in data.columns:
                fig.add_trace(
                    go.Bar(x=pollution_level_avg.index, y=pollution_level_avg[pollutant],
                          name=pollutant),
                    row=1, col=2
                )
        
        # 3. Seasonal Pollution Patterns
        if 'Date' in data.columns:
            data['Season'] = pd.to_datetime(data['Date']).dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            seasonal_aqi = data.groupby('Season')['AQI'].mean()
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            seasonal_aqi = seasonal_aqi.reindex(season_order)
            
            fig.add_trace(
                go.Bar(x=seasonal_aqi.index, y=seasonal_aqi.values,
                      name='Seasonal AQI', marker_color=['blue', 'green', 'red', 'orange']),
                row=2, col=1
            )
        
        # 4. Pollution Hotspot Analysis (Hour vs Day of Week)
        if 'Date' in data.columns:
            data['Hour'] = pd.to_datetime(data['Date']).dt.hour
            data['DayOfWeek'] = pd.to_datetime(data['Date']).dt.day_name()
            
            heatmap_data = data.groupby(['Hour', 'DayOfWeek'])['AQI'].mean().reset_index()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            pivot_data = heatmap_data.pivot(index='Hour', columns='DayOfWeek', values='AQI')
            pivot_data = pivot_data[day_order]
            
            fig.add_trace(
                go.Heatmap(z=pivot_data.values, x=pivot_data.columns, y=pivot_data.index,
                          colorscale='Reds', name='AQI Heatmap'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Pollution Analysis Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_model_performance_dashboard(self, results):
        """Create comprehensive model performance dashboard"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Regression Models Comparison',
                'Classification Models Comparison',
                'Model Performance Radar Chart',
                'Feature Importance Comparison'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatterpolar"}, {"type": "bar"}]]
        )
        
        # 1. Regression Models Comparison
        if 'regression' in results:
            reg_models = list(results['regression'].keys())
            rmse_values = [results['regression'][model]['rmse'] for model in reg_models]
            r2_values = [results['regression'][model]['r2'] for model in reg_models]
            
            fig.add_trace(
                go.Bar(x=reg_models, y=rmse_values, name='RMSE', marker_color='red'),
                row=1, col=1
            )
        
        # 2. Classification Models Comparison
        if 'classification' in results:
            clf_models = list(results['classification'].keys())
            accuracy_values = [results['classification'][model]['accuracy'] for model in clf_models]
            f1_values = [results['classification'][model]['f1'] for model in clf_models]
            
            fig.add_trace(
                go.Bar(x=clf_models, y=accuracy_values, name='Accuracy', marker_color='blue'),
                row=1, col=2
            )
        
        # 3. Model Performance Radar Chart
        if 'regression' in results and 'classification' in results:
            categories = ['RMSE', 'R²', 'Accuracy', 'F1-Score']
            
            # Normalize values for radar chart
            fig.add_trace(
                go.Scatterpolar(
                    r=[1, 0.85, 0, 0],  # Placeholder values
                    theta=categories,
                    fill='toself',
                    name='Model Performance'
                ),
                row=2, col=1
            )
        
        # 4. Feature Importance Comparison
        all_features = set()
        feature_importance_data = {}
        
        # Collect all features
        for model_type in ['regression', 'classification']:
            if model_type in results:
                for model_name, metrics in results[model_type].items():
                    if metrics.get('feature_importance'):
                        all_features.update(metrics['feature_importance'].keys())
                        feature_importance_data[f"{model_name} ({model_type})"] = metrics['feature_importance']
        
        # Plot top 10 features
        if feature_importance_data:
            # Calculate average importance
            avg_importance = {}
            for feature in all_features:
                values = []
                for model_data in feature_importance_data.values():
                    if feature in model_data:
                        values.append(model_data[feature])
                if values:
                    avg_importance[feature] = np.mean(values)
            
            # Sort and get top 10
            top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*top_features)
            
            fig.add_trace(
                go.Bar(x=list(importances), y=list(features), 
                      orientation='h', name='Feature Importance'),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Model Performance Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_prediction_analysis(self, actual, predicted, model_name):
        """Create prediction analysis visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Predictions vs Actual',
                'Residual Plot',
                'Prediction Error Distribution',
                'Performance Metrics'
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # 1. Predictions vs Actual
        fig.add_trace(
            go.Scatter(x=actual, y=predicted, mode='markers', name='Predictions',
                      opacity=0.6),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect Prediction',
                      line=dict(dash='dash', color='red')),
            row=1, col=1
        )
        
        # 2. Residual Plot
        residuals = actual - predicted
        fig.add_trace(
            go.Scatter(x=predicted, y=residuals, mode='markers', name='Residuals',
                      opacity=0.6),
            row=1, col=2
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Prediction Error Distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Error Distribution', nbinsx=30),
            row=2, col=1
        )
        
        # 4. Performance Metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        metrics = ['MSE', 'RMSE', 'MAE', 'R²']
        values = [mse, rmse, mae, r2]
        
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Metrics', marker_color=['red', 'orange', 'yellow', 'green']),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Prediction Analysis - {model_name}",
            showlegend=True
        )
        
        return fig
    
    def create_classification_analysis(self, y_true, y_pred, model_name, class_names=None):
        """Create classification analysis visualization"""
        
        if class_names is None:
            class_names = ['Good', 'Moderate', 'Kém', 'Xấu', 'Rất xấu', 'Nguy hại']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Confusion Matrix',
                'Class Distribution',
                'Prediction Confidence',
                'Performance Metrics'
            ],
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig.add_trace(
            go.Heatmap(z=cm, x=class_names, y=class_names,
                      colorscale='Blues', name='Confusion Matrix',
                      text=cm, texttemplate="%{text}", textfont={"size": 10}),
            row=1, col=1
        )
        
        # 2. Class Distribution
        unique, counts = np.unique(y_true, return_counts=True)
        fig.add_trace(
            go.Bar(x=[class_names[i] for i in unique], y=counts, name='Actual Distribution'),
            row=1, col=2
        )
        
        # 3. Prediction Confidence (placeholder)
        fig.add_trace(
            go.Bar(x=class_names, y=np.random.rand(len(class_names)), name='Confidence'),
            row=2, col=1
        )
        
        # 4. Performance Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Metrics', marker_color=['blue', 'green', 'orange', 'red']),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=f"Classification Analysis - {model_name}",
            showlegend=True
        )
        
        return fig

def create_visualizations():
    """Main function to create all visualizations"""
    viz_tools = VisualizationTools()
    return viz_tools

if __name__ == "__main__":
    # Test visualization tools
    print("🧪 Testing visualization tools...")
    
    # Create sample data
    from data_generator import generate_aqi_data
    data = generate_aqi_data(1000)
    
    viz_tools = VisualizationTools()
    
    # Test different visualization types
    print("Creating data overview dashboard...")
    fig1 = viz_tools.create_data_overview_dashboard(data)
    
    print("Creating time series analysis...")
    fig2 = viz_tools.create_time_series_analysis(data)
    
    print("Creating pollution analysis charts...")
    fig3 = viz_tools.create_pollution_analysis_charts(data)
    
    print("✅ Visualization tools test completed successfully!")
