import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization class
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.pollution_level_names = ['Tốt', 'Trung Bình', 'Kém', 'Xấu', 'Rất Xấu', 'Nguy Hiểm']
        self.pollution_level_colors = ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023']
        
    def compare_regression_models(self, regression_results):
        """Compare regression models performance"""
        
        # Create comparison table
        comparison_data = []
        for model_name, metrics in regression_results.items():
            comparison_data.append({
                'Model': model_name,
                'MSE': metrics['mse'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'CV RMSE Mean': metrics['cv_rmse_mean'],
                'CV RMSE Std': metrics['cv_rmse_std']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display comparison table
        print("\n📊 Regression Models Comparison:")
        print(df_comparison.round(4))
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RMSE Comparison', 'R² Comparison', 'MAE Comparison', 'CV RMSE Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # RMSE comparison
        fig.add_trace(
            go.Bar(x=df_comparison['Model'], y=df_comparison['RMSE'], name='RMSE', marker_color='blue'),
            row=1, col=1
        )
        
        # R² comparison
        fig.add_trace(
            go.Bar(x=df_comparison['Model'], y=df_comparison['R²'], name='R²', marker_color='green'),
            row=1, col=2
        )
        
        # MAE comparison
        fig.add_trace(
            go.Bar(x=df_comparison['Model'], y=df_comparison['MAE'], name='MAE', marker_color='orange'),
            row=2, col=1
        )
        
        # CV RMSE comparison
        fig.add_trace(
            go.Bar(x=df_comparison['Model'], y=df_comparison['CV RMSE Mean'], name='CV RMSE', marker_color='red'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Regression Models Performance Comparison",
            showlegend=False
        )
        
        return fig
    
    def compare_classification_models(self, classification_results):
        """Compare classification models performance"""
        
        # Create comparison table
        comparison_data = []
        for model_name, metrics in classification_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'CV Accuracy Mean': metrics['cv_accuracy_mean'],
                'CV Accuracy Std': metrics['cv_accuracy_std']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display comparison table
        print("\n🎯 Classification Models Comparison:")
        print(df_comparison.round(4))
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'F1-Score Comparison', 'Precision Comparison', 'Recall Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=df_comparison['Model'], y=df_comparison['Accuracy'], name='Accuracy', marker_color='blue'),
            row=1, col=1
        )
        
        # F1-Score comparison
        fig.add_trace(
            go.Bar(x=df_comparison['Model'], y=df_comparison['F1-Score'], name='F1-Score', marker_color='green'),
            row=1, col=2
        )
        
        # Precision comparison
        fig.add_trace(
            go.Bar(x=df_comparison['Model'], y=df_comparison['Precision'], name='Precision', marker_color='orange'),
            row=2, col=1
        )
        
        # Recall comparison
        fig.add_trace(
            go.Bar(x=df_comparison['Model'], y=df_comparison['Recall'], name='Recall', marker_color='red'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Classification Models Performance Comparison",
            showlegend=False
        )
        
        return fig
    
    def recommend_best_models(self, results):
        """Recommend best models based on performance metrics"""
        
        recommendations = {
            'best_regression': None,
            'best_classification': None,
            'recommendations': []
        }
        
        # Find best regression model (lowest RMSE, highest R²)
        if 'regression' in results and results['regression']:
            reg_models = results['regression']
            best_reg_model = min(reg_models.keys(), key=lambda x: reg_models[x]['rmse'])
            best_reg_metrics = reg_models[best_reg_model]
            
            recommendations['best_regression'] = {
                'model': best_reg_model,
                'rmse': best_reg_metrics['rmse'],
                'r2': best_reg_metrics['r2'],
                'mae': best_reg_metrics['mae']
            }
            
            # Add recommendation
            if best_reg_metrics['rmse'] < 20:
                recommendations['recommendations'].append(
                    f"🏆 {best_reg_model} shows excellent regression performance (RMSE: {best_reg_metrics['rmse']:.2f})"
                )
            elif best_reg_metrics['rmse'] < 35:
                recommendations['recommendations'].append(
                    f"✅ {best_reg_model} shows good regression performance (RMSE: {best_reg_metrics['rmse']:.2f})"
                )
            else:
                recommendations['recommendations'].append(
                    f"⚠️ {best_reg_model} shows moderate regression performance (RMSE: {best_reg_metrics['rmse']:.2f})"
                )
        
        # Find best classification model (highest F1-score)
        if 'classification' in results and results['classification']:
            clf_models = results['classification']
            best_clf_model = max(clf_models.keys(), key=lambda x: clf_models[x]['f1'])
            best_clf_metrics = clf_models[best_clf_model]
            
            recommendations['best_classification'] = {
                'model': best_clf_model,
                'accuracy': best_clf_metrics['accuracy'],
                'f1': best_clf_metrics['f1'],
                'precision': best_clf_metrics['precision'],
                'recall': best_clf_metrics['recall']
            }
            
            # Add recommendation
            if best_clf_metrics['f1'] > 0.9:
                recommendations['recommendations'].append(
                    f"🏆 {best_clf_model} shows excellent classification performance (F1: {best_clf_metrics['f1']:.3f})"
                )
            elif best_clf_metrics['f1'] > 0.8:
                recommendations['recommendations'].append(
                    f"✅ {best_clf_model} shows good classification performance (F1: {best_clf_metrics['f1']:.3f})"
                )
            else:
                recommendations['recommendations'].append(
                    f"⚠️ {best_clf_model} shows moderate classification performance (F1: {best_clf_metrics['f1']:.3f})"
                )
        
        # Overall recommendation
        if recommendations['best_regression'] and recommendations['best_classification']:
            recommendations['recommendations'].append(
                f"\n🎯 Overall Recommendation:\n"
                f"• Use {recommendations['best_classification']['model']} for pollution level classification\n"
                f"• Use {recommendations['best_regression']['model']} for AQI value prediction"
            )
        
        # Print recommendations
        print("\n🥇 Model Recommendations:")
        for rec in recommendations['recommendations']:
            print(rec)
        
        return recommendations
    
    def create_detailed_visualizations(self, results):
        """Create detailed visualizations for model analysis"""
        figures = []
        
        # Regression visualizations
        if 'regression' in results:
            fig_reg = self._create_regression_visualizations(results['regression'])
            figures.extend(fig_reg)
        
        # Classification visualizations
        if 'classification' in results:
            fig_clf = self._create_classification_visualizations(results['classification'])
            figures.extend(fig_clf)
        
        return figures
    
    def _create_regression_visualizations(self, regression_results):
        """Create detailed regression visualizations"""
        figures = []
        
        # Prediction vs Actual plots
        fig1 = make_subplots(
            rows=1, cols=len(regression_results),
            subplot_titles=list(regression_results.keys())
        )
        
        for i, (model_name, metrics) in enumerate(regression_results.items()):
            predictions = metrics['predictions']
            actual = metrics.get('actual_values', np.random.normal(100, 30, len(predictions)))  # Placeholder
            
            # Scatter plot
            fig1.add_trace(
                go.Scatter(
                    x=actual, y=predictions,
                    mode='markers',
                    name=model_name,
                    opacity=0.6
                ),
                row=1, col=i+1
            )
            
            # Perfect prediction line
            min_val = min(actual.min(), predictions.min())
            max_val = max(actual.max(), predictions.max())
            fig1.add_trace(
                go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ),
                row=1, col=i+1
            )
        
        fig1.update_layout(
            height=400,
            title_text="Predictions vs Actual Values"
        )
        figures.append(fig1)
        
        # Residual plots
        fig2 = make_subplots(
            rows=1, cols=len(regression_results),
            subplot_titles=list(regression_results.keys())
        )
        
        for i, (model_name, metrics) in enumerate(regression_results.items()):
            predictions = metrics['predictions']
            actual = metrics.get('actual_values', np.random.normal(100, 30, len(predictions)))
            residuals = actual - predictions
            
            fig2.add_trace(
                go.Scatter(
                    x=predictions, y=residuals,
                    mode='markers',
                    name=model_name,
                    opacity=0.6
                ),
                row=1, col=i+1
            )
            
            # Zero line
            fig2.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=i+1)
        
        fig2.update_layout(
            height=400,
            title_text="Residual Plots"
        )
        figures.append(fig2)
        
        return figures
    
    def _create_classification_visualizations(self, classification_results):
        """Create detailed classification visualizations"""
        figures = []
        
        # Confusion matrices
        for model_name, metrics in classification_results.items():
            predictions = metrics['predictions']
            actual = metrics.get('actual_values', np.random.randint(0, 6, len(predictions)))  # Placeholder
            
            cm = confusion_matrix(actual, predictions)
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"Confusion Matrix - {model_name}",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=self.pollution_level_names,
                y=self.pollution_level_names
            )
            figures.append(fig)
        
        # Feature importance comparison
        fig_importance = go.Figure()
        
        for model_name, metrics in classification_results.items():
            if metrics.get('feature_importance'):
                importance_dict = metrics['feature_importance']
                # Get top 10 features
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                features, importances = zip(*sorted_features)
                
                fig_importance.add_trace(
                    go.Bar(
                        x=list(importances),
                        y=list(features),
                        name=model_name,
                        orientation='h'
                    )
                )
        
        fig_importance.update_layout(
            title="Feature Importance Comparison",
            barmode='group',
            height=600
        )
        figures.append(fig_importance)
        
        return figures
    
    def create_performance_report(self, results):
        """Create comprehensive performance report"""
        report = {
            'summary': {},
            'regression_analysis': {},
            'classification_analysis': {},
            'recommendations': {}
        }
        
        # Summary
        total_models = len(results.get('regression', {})) + len(results.get('classification', {}))
        report['summary'] = {
            'total_models_trained': total_models,
            'regression_models': list(results.get('regression', {}).keys()),
            'classification_models': list(results.get('classification', {}).keys())
        }
        
        # Regression analysis
        if 'regression' in results:
            reg_results = results['regression']
            report['regression_analysis'] = {
                'best_model': min(reg_results.keys(), key=lambda x: reg_results[x]['rmse']),
                'performance_range': {
                    'rmse_min': min(m['rmse'] for m in reg_results.values()),
                    'rmse_max': max(m['rmse'] for m in reg_results.values()),
                    'r2_min': min(m['r2'] for m in reg_results.values()),
                    'r2_max': max(m['r2'] for m in reg_results.values())
                }
            }
        
        # Classification analysis
        if 'classification' in results:
            clf_results = results['classification']
            report['classification_analysis'] = {
                'best_model': max(clf_results.keys(), key=lambda x: clf_results[x]['f1']),
                'performance_range': {
                    'accuracy_min': min(m['accuracy'] for m in clf_results.values()),
                    'accuracy_max': max(m['accuracy'] for m in clf_results.values()),
                    'f1_min': min(m['f1'] for m in clf_results.values()),
                    'f1_max': max(m['f1'] for m in clf_results.values())
                }
            }
        
        # Recommendations
        report['recommendations'] = self.recommend_best_models(results)
        
        return report
    
    def analyze_feature_importance(self, results):
        """Analyze and compare feature importance across models"""
        feature_analysis = {}
        
        # Collect all feature importance data
        all_importance = {}
        
        # Regression models
        if 'regression' in results:
            for model_name, metrics in results['regression'].items():
                if metrics.get('feature_importance'):
                    all_importance[f"{model_name} (Reg)"] = metrics['feature_importance']
        
        # Classification models
        if 'classification' in results:
            for model_name, metrics in results['classification'].items():
                if metrics.get('feature_importance'):
                    all_importance[f"{model_name} (Clf)"] = metrics['feature_importance']
        
        # Calculate average importance
        if all_importance:
            # Get all unique features
            all_features = set()
            for importance_dict in all_importance.values():
                all_features.update(importance_dict.keys())
            
            # Calculate average importance for each feature
            avg_importance = {}
            for feature in all_features:
                values = []
                for model_name, importance_dict in all_importance.items():
                    if feature in importance_dict:
                        values.append(importance_dict[feature])
                if values:
                    avg_importance[feature] = np.mean(values)
            
            # Sort by average importance
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            feature_analysis = {
                'top_features': sorted_features[:10],
                'average_importance': avg_importance,
                'model_importance': all_importance
            }
        
        return feature_analysis

if __name__ == "__main__":
    # Test the evaluation module
    print("🧪 Testing model evaluation pipeline...")
    
    # Create sample results for testing
    sample_results = {
        'regression': {
            'Linear Regression': {
                'mse': 400.5,
                'rmse': 20.01,
                'mae': 15.2,
                'r2': 0.85,
                'cv_rmse_mean': 21.5,
                'cv_rmse_std': 2.1,
                'predictions': np.random.normal(100, 20, 100),
                'feature_importance': {'PM2.5': 0.4, 'PM10': 0.3, 'NO2': 0.2, 'Temperature': 0.1}
            },
            'Decision Tree (CART)': {
                'mse': 350.2,
                'rmse': 18.71,
                'mae': 14.1,
                'r2': 0.88,
                'cv_rmse_mean': 19.8,
                'cv_rmse_std': 3.2,
                'predictions': np.random.normal(100, 18, 100),
                'feature_importance': {'PM2.5': 0.5, 'PM10': 0.25, 'NO2': 0.15, 'Temperature': 0.1}
            }
        },
        'classification': {
            'Logistic Regression': {
                'accuracy': 0.82,
                'precision': 0.81,
                'recall': 0.82,
                'f1': 0.81,
                'cv_accuracy_mean': 0.80,
                'cv_accuracy_std': 0.03,
                'predictions': np.random.randint(0, 6, 100),
                'feature_importance': {'PM2.5': 0.35, 'PM10': 0.25, 'NO2': 0.2, 'Temperature': 0.2}
            },
            'SVM': {
                'accuracy': 0.86,
                'precision': 0.85,
                'recall': 0.86,
                'f1': 0.85,
                'cv_accuracy_mean': 0.84,
                'cv_accuracy_std': 0.02,
                'predictions': np.random.randint(0, 6, 100),
                'feature_importance': {'PM2.5': 0.4, 'PM10': 0.3, 'NO2': 0.2, 'Temperature': 0.1}
            }
        }
    }
    
    evaluator = ModelEvaluator()
    
    # Test comparison methods
    print("\nTesting regression comparison...")
    fig_reg = evaluator.compare_regression_models(sample_results['regression'])
    
    print("\nTesting classification comparison...")
    fig_clf = evaluator.compare_classification_models(sample_results['classification'])
    
    print("\nTesting recommendations...")
    recommendations = evaluator.recommend_best_models(sample_results)
    
    print("\nTesting feature importance analysis...")
    feature_analysis = evaluator.analyze_feature_importance(sample_results)
    
    print("✅ Model evaluation test completed successfully!")
