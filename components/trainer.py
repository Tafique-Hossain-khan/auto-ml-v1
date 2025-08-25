"""
Training and Evaluation Component for AutoML System
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.pipeline import Pipeline
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import time
import traceback
from components.pipeline_builder import create_pipeline
from components.utils import save_model_artifacts, safe_json_serialize
from config.settings import DEFAULT_CV_FOLDS, DEFAULT_RANDOM_STATE

warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

class AutoMLTrainer:
    """Handles training and evaluation of multiple model pipelines"""

    def __init__(self, cv_folds: int = DEFAULT_CV_FOLDS, random_state: int = DEFAULT_RANDOM_STATE,
                 time_limit_minutes: int = 5):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.time_limit_seconds = time_limit_minutes * 60
        self.results = []
        self.best_pipeline = None
        self.best_score = -np.inf
        self.task_type = None

    def train_multiple_models(self, df: pd.DataFrame, target_column: str, 
                            model_configs: List[Dict[str, Any]], task_type: str,
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train multiple models and return best performing pipeline

        Args:
            df: Training DataFrame
            target_column: Target column name
            model_configs: List of model configurations from ModelAdvisor
            task_type: 'classification' or 'regression'
            metadata: Dataset metadata

        Returns:
            Training results dictionary
        """
        self.task_type = task_type
        start_time = time.time()

        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Setup cross-validation
        cv = self._setup_cross_validation(y, task_type)

        # Train each model
        for i, model_config in enumerate(model_configs):
            if time.time() - start_time > self.time_limit_seconds:
                print(f"Time limit reached. Stopping after {i} models.")
                break

            try:
                result = self._train_single_model(
                    X, y, model_config, task_type, cv, metadata
                )
                self.results.append(result)

                # Update best model
                if result['cv_score'] > self.best_score:
                    self.best_score = result['cv_score']
                    self.best_pipeline = result['pipeline']

                print(f"✓ {model_config['name']}: {result['cv_score']:.4f}")

            except Exception as e:
                error_result = {
                    'model_name': model_config['name'],
                    'model_class': model_config['model'],
                    'status': 'failed',
                    'error': str(e),
                    'cv_score': -np.inf
                }
                self.results.append(error_result)
                print(f"✗ {model_config['name']}: Failed ({str(e)})")

        # Train best model on full dataset
        if self.best_pipeline:
            print(f"\nRetraining best model on full dataset...")
            self.best_pipeline.fit(X, y)

        # Calculate training summary
        training_summary = self._create_training_summary(df, target_column, start_time)

        return training_summary

    def _train_single_model(self, X: pd.DataFrame, y: pd.Series, 
                           model_config: Dict[str, Any], task_type: str,
                           cv, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train a single model and evaluate with cross-validation"""

        # Create pipeline
        pipeline = create_pipeline(
            pd.concat([X, y], axis=1), y.name, model_config, task_type, metadata
        )

        # Perform cross-validation
        scoring = self._get_scoring_metric(task_type)
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        # Calculate additional metrics on a single fold for detailed analysis
        detailed_metrics = self._calculate_detailed_metrics(pipeline, X, y, task_type, cv)

        return {
            'model_name': model_config['name'],
            'model_class': model_config['model'],
            'model_params': model_config['params'],
            'pipeline': pipeline,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'detailed_metrics': detailed_metrics,
            'status': 'success'
        }

    def _setup_cross_validation(self, y: pd.Series, task_type: str):
        """Setup appropriate cross-validation strategy"""
        if task_type == 'classification':
            # Check if stratification is possible
            min_class_count = y.value_counts().min()
            if min_class_count >= self.cv_folds:
                return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            else:
                # Fall back to regular KFold if stratification not possible
                print(f"Warning: Using KFold instead of StratifiedKFold due to small class sizes")
                return KFold(n_splits=min(self.cv_folds, min_class_count), shuffle=True, random_state=self.random_state)
        else:
            return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    def _get_scoring_metric(self, task_type: str) -> str:
        """Get primary scoring metric for cross-validation"""
        if task_type == 'classification':
            return 'accuracy'  # Can be changed to 'f1_weighted' for imbalanced datasets
        else:
            return 'r2'

    def _calculate_detailed_metrics(self, pipeline: Pipeline, X: pd.DataFrame, 
                                  y: pd.Series, task_type: str, cv) -> Dict[str, float]:
        """Calculate detailed metrics using train-test split from CV"""
        try:
            # Use first fold for detailed metrics
            train_idx, test_idx = next(cv.split(X, y))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            if task_type == 'classification':
                return self._calculate_classification_metrics(y_test, y_pred, pipeline, X_test)
            else:
                return self._calculate_regression_metrics(y_test, y_pred)

        except Exception as e:
            print(f"Warning: Could not calculate detailed metrics: {e}")
            return {}

    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                        pipeline: Pipeline, X_test: pd.DataFrame) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        # Add ROC AUC for binary/multiclass classification
        try:
            if hasattr(pipeline, 'predict_proba'):
                y_proba = pipeline.predict_proba(X_test)
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multiclass classification
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception:
            pass  # Skip ROC AUC if it fails

        return {k: float(v) for k, v in metrics.items()}

    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics"""
        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }

        # Add MAPE if no zero values in y_true
        if not (y_true == 0).any():
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = mape

        return {k: float(v) for k, v in metrics.items()}

    def _create_training_summary(self, df: pd.DataFrame, target_column: str, 
                               start_time: float) -> Dict[str, Any]:
        """Create comprehensive training summary"""

        # Sort results by score
        successful_results = [r for r in self.results if r.get('status') == 'success']
        successful_results.sort(key=lambda x: x['cv_score'], reverse=True)

        # Best model info
        best_model_info = None
        if successful_results:
            best_result = successful_results[0]
            best_model_info = {
                'name': best_result['model_name'],
                'class': best_result['model_class'],
                'cv_score': best_result['cv_score'],
                'cv_std': best_result['cv_std'],
                'params': best_result['model_params'],
                'detailed_metrics': best_result.get('detailed_metrics', {})
            }

        # Model comparison
        model_comparison = []
        for result in self.results:
            if result.get('status') == 'success':
                model_comparison.append({
                    'model_name': result['model_name'],
                    'cv_score': result['cv_score'],
                    'cv_std': result['cv_std'],
                    'detailed_metrics': result.get('detailed_metrics', {})
                })
            else:
                model_comparison.append({
                    'model_name': result['model_name'],
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error')
                })

        training_summary = {
            'training_completed': True,
            'task_type': self.task_type,
            'n_models_trained': len([r for r in self.results if r.get('status') == 'success']),
            'n_models_failed': len([r for r in self.results if r.get('status') == 'failed']),
            'training_time_seconds': time.time() - start_time,
            'cv_folds': self.cv_folds,
            'best_model': best_model_info,
            'model_comparison': model_comparison,
            'dataset_info': {
                'n_samples': len(df),
                'n_features': len(df.columns) - 1,
                'target_column': target_column
            }
        }

        return safe_json_serialize(training_summary)

    def save_best_model(self, model_name: str = "automl_best_model") -> Optional[Dict[str, str]]:
        """Save the best model artifacts"""
        if not self.best_pipeline:
            print("No best model available to save")
            return None

        # Create metadata
        best_result = max(self.results, key=lambda x: x.get('cv_score', -np.inf))
        metadata = {
            'model_name': best_result['model_name'],
            'model_class': best_result['model_class'],
            'cv_score': best_result['cv_score'],
            'cv_std': best_result['cv_std'],
            'task_type': self.task_type,
            'detailed_metrics': best_result.get('detailed_metrics', {}),
            'training_timestamp': pd.Timestamp.now().isoformat()
        }

        # Extract preprocessor and model
        preprocessor = self.best_pipeline.named_steps['preprocessor']
        model = self.best_pipeline.named_steps['model']

        # Save artifacts
        try:
            file_paths = save_model_artifacts(model, preprocessor, metadata, model_name)
            print(f"Model artifacts saved successfully!")
            return file_paths
        except Exception as e:
            print(f"Error saving model artifacts: {e}")
            return None

    def get_best_pipeline(self) -> Optional[Pipeline]:
        """Get the best trained pipeline"""
        return self.best_pipeline

    def get_training_results(self) -> List[Dict[str, Any]]:
        """Get all training results"""
        return self.results

def train_automl_models(df: pd.DataFrame, target_column: str, 
                       model_configs: List[Dict[str, Any]], task_type: str,
                       metadata: Dict[str, Any] = None, 
                       cv_folds: int = DEFAULT_CV_FOLDS,
                       time_limit_minutes: int = 5) -> Dict[str, Any]:
    """
    Convenience function to train multiple AutoML models

    Args:
        df: Training DataFrame
        target_column: Target column name
        model_configs: Model configurations from ModelAdvisor
        task_type: 'classification' or 'regression'
        metadata: Dataset metadata
        cv_folds: Number of cross-validation folds
        time_limit_minutes: Time limit for training

    Returns:
        Training results and best model
    """
    trainer = AutoMLTrainer(cv_folds=cv_folds, time_limit_minutes=time_limit_minutes)
    results = trainer.train_multiple_models(df, target_column, model_configs, task_type, metadata)

    # Save best model
    file_paths = trainer.save_best_model()
    if file_paths:
        results['saved_artifacts'] = file_paths

    results['best_pipeline'] = trainer.get_best_pipeline()

    return results
