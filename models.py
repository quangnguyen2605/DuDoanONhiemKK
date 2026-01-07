import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive model training class for air pollution prediction
    Implements Linear Regression, Decision Tree, SVM, and Logistic Regression
    """
    
    def __init__(self):
        self.regression_models = {}
        self.classification_models = {}
        self.training_history = {}
        
    def train_models(self, X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf,
                    regression_models, classification_models, cv_folds=5, enable_grid_search=False, custom_params=None):
        """
        Train selected regression and classification models with custom hyperparameters
        """
        results = {
            'regression': {},
            'classification': {}
        }
        
        print("🚀 Starting model training...")
        
        # Train regression models
        if regression_models:
            print("📈 Training regression models...")
            for model_name in regression_models:
                print(f"  Training {model_name}...")
                model_results = self._train_regression_model(
                    model_name, X_train, X_test, y_train_reg, y_test_reg, cv_folds, enable_grid_search, custom_params
                )
                results['regression'][model_name] = model_results
        
        # Train classification models
        if classification_models:
            print("🎯 Training classification models...")
            for model_name in classification_models:
                print(f"  Training {model_name}...")
                model_results = self._train_classification_model(
                    model_name, X_train, X_test, y_train_clf, y_test_clf, cv_folds, enable_grid_search, custom_params
                )
                results['classification'][model_name] = model_results
        
        print("✅ Model training completed!")
        return results
    
    def _train_regression_model(self, model_name, X_train, X_test, y_train, y_test, cv_folds, enable_grid_search, custom_params=None):
        """Train a single regression model with custom hyperparameters"""
        
        # Support both Vietnamese and English model names
        if model_name in ["Hồi Quy Tuyến Tính", "Linear Regression"]:
            model = LinearRegression()
            param_grid = {}  # No hyperparameters to tune
            
        elif model_name in ["Cây Quyết Định (CART)", "Decision Tree (CART)"]:
            model = DecisionTreeRegressor(random_state=42)
            
            # Use custom parameters if available
            if enable_grid_search and custom_params and 'cart_params' in custom_params:
                cart_params = custom_params['cart_params']
                param_grid = {
                    'max_depth': cart_params.get('max_depth', [5]),
                    'min_samples_split': cart_params.get('min_samples_split', [2]),
                    'min_samples_leaf': cart_params.get('min_samples_leaf', [1]),
                    'max_features': cart_params.get('max_features', [None])
                }
                # Remove None from max_features if present
                if None in param_grid['max_features']:
                    param_grid['max_features'] = [x for x in param_grid['max_features'] if x is not None] + ['None']
            elif enable_grid_search:
                param_grid = {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            else:
                param_grid = {'max_depth': [5], 'min_samples_split': [2]}
        
        else:
            raise ValueError(f"Unknown regression model: {model_name}")
        
        # Grid search if enabled
        if enable_grid_search and param_grid:
            print(f"    Performing grid search for {model_name}...")
            grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = model.get_params()
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Store model
        self.regression_models[model_name] = best_model
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
        elif hasattr(best_model, 'coef_'):
            feature_importance = dict(zip(X_train.columns, np.abs(best_model.coef_)))
        
        results = {
            'model': best_model,
            'predictions': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'best_params': best_params,
            'feature_importance': feature_importance
        }
        
        print(f"    {model_name} - RMSE: {rmse:.3f}, R²: {r2:.3f}")
        return results
    
    def _train_classification_model(self, model_name, X_train, X_test, y_train, y_test, cv_folds, enable_grid_search, custom_params=None):
        """Train a single classification model with custom hyperparameters"""
        
        # Support both Vietnamese and English model names
        if model_name in ["Hồi Quy Logistic", "Logistic Regression"]:
            model = LogisticRegression(random_state=42, max_iter=1000)
            
            # Use custom parameters if available
            if enable_grid_search and custom_params and 'lr_params' in custom_params:
                lr_params = custom_params['lr_params']
                param_grid = {
                    'C': lr_params.get('C', [1]),
                    'penalty': lr_params.get('penalty', ['l2']),
                    'solver': lr_params.get('solver', ['liblinear']),
                    'max_iter': lr_params.get('max_iter', [1000])
                }
                # Handle solver-penalty compatibility
                valid_combinations = []
                for solver in param_grid['solver']:
                    for penalty in param_grid['penalty']:
                        if (solver == 'liblinear' and penalty in ['l1', 'l2']) or \
                           (solver == 'saga' and penalty in ['l1', 'l2']):
                            valid_combinations.append((solver, penalty))
                
                if valid_combinations:
                    param_grid['solver'] = [combo[0] for combo in valid_combinations]
                    param_grid['penalty'] = [combo[1] for combo in valid_combinations]
                else:
                    param_grid = {'C': [1]}
            elif enable_grid_search:
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            else:
                param_grid = {'C': [1]}
        
        elif model_name in ["SVM", "Support Vector Machine"]:
            model = SVC(random_state=42, probability=True)
            
            # Use custom parameters if available
            if enable_grid_search and custom_params and 'svm_params' in custom_params:
                svm_params = custom_params['svm_params']
                param_grid = {
                    'C': svm_params.get('C', [1]),
                    'kernel': svm_params.get('kernel', ['rbf']),
                    'gamma': svm_params.get('gamma', ['scale']),
                    'degree': svm_params.get('degree', [3])
                }
                # Handle kernel-specific parameters
                if 'linear' in param_grid['kernel']:
                    param_grid['gamma'] = ['scale']  # gamma not used for linear kernel
                if 'poly' not in param_grid['kernel']:
                    param_grid['degree'] = [3]  # degree only used for poly kernel
            elif enable_grid_search:
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto'],
                    'degree': [2, 3, 4, 5]
                }
            else:
                param_grid = {'C': [1], 'kernel': ['rbf']}
        
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
        
        # Grid search if enabled
        if enable_grid_search and param_grid:
            print(f"    Performing grid search for {model_name}...")
            grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model.fit(X_train, y_train)
            best_model = model
            best_params = model.get_params()
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        
        # Store model
        self.classification_models[model_name] = best_model
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
        elif hasattr(best_model, 'coef_'):
            # For multi-class, take average of absolute coefficients
            coef = best_model.coef_
            if len(coef.shape) > 1:
                coef = np.mean(np.abs(coef), axis=0)
            else:
                coef = np.abs(coef)
            feature_importance = dict(zip(X_train.columns, coef))
        elif model_name in ["SVM", "Support Vector Machine"]:
            # For SVM with non-linear kernels, use permutation importance
            try:
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(best_model, X_test, y_test, 
                                                        n_repeats=10, random_state=42, n_jobs=-1)
                feature_importance = dict(zip(X_train.columns, perm_importance.importances_mean))
            except ImportError:
                # If sklearn.inspection is not available, use a simple approach
                # For linear kernel SVM, we can still use coefficients
                if best_model.kernel == 'linear' and hasattr(best_model, 'coef_'):
                    coef = best_model.coef_
                    if len(coef.shape) > 1:
                        coef = np.mean(np.abs(coef), axis=0)
                    else:
                        coef = np.abs(coef)
                    feature_importance = dict(zip(X_train.columns, coef))
                else:
                    # For non-linear kernels, create a simple importance based on feature variance
                    feature_importance = dict(zip(X_train.columns, np.var(X_train, axis=0)))
        else:
            # Fallback: use feature variance as a simple importance measure
            feature_importance = dict(zip(X_train.columns, np.var(X_train, axis=0)))
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        results = {
            'model': best_model,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'best_params': best_params,
            'feature_importance': feature_importance,
            'classification_report': class_report
        }
        
        print(f"    {model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        return results
    
    def predict_regression(self, X, model_name):
        """Make regression predictions - supports both Vietnamese and English model names"""
        # Try to find model with Vietnamese name first, then English
        if model_name in self.regression_models:
            model = self.regression_models[model_name]
        else:
            # Map Vietnamese names to English for lookup
            name_mapping = {
                "Hồi Quy Tuyến Tính": "Linear Regression",
                "Cây Quyết Định (CART)": "Decision Tree (CART)",
                "Linear Regression": "Hồi Quy Tuyến Tính",
                "Decision Tree (CART)": "Cây Quyết Định (CART)"
            }
            mapped_name = name_mapping.get(model_name, model_name)
            if mapped_name in self.regression_models:
                model = self.regression_models[mapped_name]
            else:
                raise ValueError(f"Model {model_name} not found in trained regression models")
        
        predictions = model.predict(X)
        return predictions
    
    def predict_classification(self, X, model_name):
        """Make classification predictions - supports both Vietnamese and English model names"""
        # Try to find model with Vietnamese name first, then English
        if model_name in self.classification_models:
            model = self.classification_models[model_name]
        else:
            # Map Vietnamese names to English for lookup
            name_mapping = {
                "Hồi Quy Logistic": "Logistic Regression",
                "SVM": "Support Vector Machine",
                "Logistic Regression": "Hồi Quy Logistic",
                "Support Vector Machine": "SVM"
            }
            mapped_name = name_mapping.get(model_name, model_name)
            if mapped_name in self.classification_models:
                model = self.classification_models[mapped_name]
            else:
                raise ValueError(f"Model {model_name} not found in trained classification models")
        
        predictions = model.predict(X)
        return predictions
    
    def get_feature_importance(self, model_name, model_type):
        """Get feature importance for a trained model"""
        if model_type == 'regression':
            if model_name not in self.regression_models:
                raise ValueError(f"Regression model {model_name} not found")
            model = self.regression_models[model_name]
        elif model_type == 'classification':
            if model_name not in self.classification_models:
                raise ValueError(f"Classification model {model_name} not found")
            model = self.classification_models[model_name]
        else:
            raise ValueError("model_type must be 'regression' or 'classification'")
        
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_)
            if len(feature_importance.shape) > 1:
                feature_importance = np.mean(feature_importance, axis=0)
        
        return feature_importance
    
    def save_models(self, filepath_prefix):
        """Save trained models to disk"""
        # Save regression models
        for name, model in self.regression_models.items():
            filename = f"{filepath_prefix}_regression_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
            joblib.dump(model, filename)
        
        # Save classification models
        for name, model in self.classification_models.items():
            filename = f"{filepath_prefix}_classification_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
            joblib.dump(model, filename)
        
        print(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix, model_names):
        """Load trained models from disk"""
        # Load regression models
        for name in model_names.get('regression', []):
            filename = f"{filepath_prefix}_regression_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
            try:
                self.regression_models[name] = joblib.load(filename)
                print(f"Loaded regression model: {name}")
            except FileNotFoundError:
                print(f"Could not find regression model file: {filename}")
        
        # Load classification models
        for name in model_names.get('classification', []):
            filename = f"{filepath_prefix}_classification_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
            try:
                self.classification_models[name] = joblib.load(filename)
                print(f"Loaded classification model: {name}")
            except FileNotFoundError:
                print(f"Could not find classification model file: {filename}")
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        summary = {
            'regression_models': list(self.regression_models.keys()),
            'classification_models': list(self.classification_models.keys()),
            'total_models': len(self.regression_models) + len(self.classification_models)
        }
        return summary
    
    def compare_models(self, results):
        """Compare performance of all trained models"""
        comparison = {
            'regression': {},
            'classification': {}
        }
        
        # Compare regression models
        if 'regression' in results:
            reg_results = results['regression']
            best_reg_model = min(reg_results.keys(), key=lambda x: reg_results[x]['rmse'])
            comparison['regression'] = {
                'best_model': best_reg_model,
                'best_rmse': reg_results[best_reg_model]['rmse'],
                'best_r2': reg_results[best_reg_model]['r2'],
                'all_models': {name: {'rmse': metrics['rmse'], 'r2': metrics['r2']} 
                              for name, metrics in reg_results.items()}
            }
        
        # Compare classification models
        if 'classification' in results:
            clf_results = results['classification']
            best_clf_model = max(clf_results.keys(), key=lambda x: clf_results[x]['f1'])
            comparison['classification'] = {
                'best_model': best_clf_model,
                'best_accuracy': clf_results[best_clf_model]['accuracy'],
                'best_f1': clf_results[best_clf_model]['f1'],
                'all_models': {name: {'accuracy': metrics['accuracy'], 'f1': metrics['f1']} 
                              for name, metrics in clf_results.items()}
            }
        
        return comparison

if __name__ == "__main__":
    # Test the model training
    from data_generator import generate_aqi_data
    from data_preprocessing import DataPreprocessor
    
    print("🧪 Testing model training pipeline...")
    
    # Generate and preprocess data
    data = generate_aqi_data(1000)
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = preprocessor.fit_transform(data)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train models
    regression_models = ["Linear Regression", "Decision Tree (CART)"]
    classification_models = ["Logistic Regression", "SVM"]
    
    results = trainer.train_models(
        X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf,
        regression_models, classification_models, cv_folds=3, enable_grid_search=False
    )
    
    # Display results
    print("\n📊 Training Results:")
    for model_type, models in results.items():
        print(f"\n{model_type.upper()} MODELS:")
        for model_name, metrics in models.items():
            if model_type == 'regression':
                print(f"  {model_name}: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")
            else:
                print(f"  {model_name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
    
    print("✅ Model training test completed successfully!")
