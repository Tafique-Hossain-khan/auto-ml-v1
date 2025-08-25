"""
Pipeline Builder for AutoML System
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import warnings

# Safely import XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. XGB models will be skipped.")

warnings.filterwarnings('ignore')

class PipelineBuilder:
    """Builds scikit-learn pipelines for AutoML"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = None
        self.feature_columns = []
        self.target_column = None

    def build_preprocessing_pipeline(self, df: pd.DataFrame, target_column: str, 
                                   metadata: Dict[str, Any] = None) -> ColumnTransformer:
        """
        Build preprocessing pipeline based on data characteristics

        Args:
            df: Input DataFrame
            target_column: Name of target column
            metadata: Dataset metadata from DataAnalyzer

        Returns:
            Fitted ColumnTransformer for preprocessing
        """
        self.target_column = target_column
        self.feature_columns = [col for col in df.columns if col != target_column]

        # Analyze feature types
        numeric_features = df[self.feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df[self.feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove high cardinality categorical features (>100 unique values)
        high_cardinality_features = []
        for col in categorical_features.copy():
            if df[col].nunique() > 100:
                high_cardinality_features.append(col)
                categorical_features.remove(col)

        if high_cardinality_features:
            print(f"Dropping high cardinality features: {high_cardinality_features}")

        # Build transformers
        transformers = []

        # Numeric pipeline
        if numeric_features:
            numeric_pipeline = self._build_numeric_pipeline(df, numeric_features, metadata)
            transformers.append(('numeric', numeric_pipeline, numeric_features))

        # Categorical pipeline
        if categorical_features:
            categorical_pipeline = self._build_categorical_pipeline(df, categorical_features, metadata)
            transformers.append(('categorical', categorical_pipeline, categorical_features))

        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop high cardinality and other columns
            sparse_threshold=0  # Return dense arrays
        )

        return self.preprocessor

    def _build_numeric_pipeline(self, df: pd.DataFrame, numeric_features: List[str], 
                               metadata: Dict[str, Any] = None) -> Pipeline:
        """Build preprocessing pipeline for numeric features"""
        steps = []

        # Imputation
        # Choose strategy based on missing value percentage
        missing_percentages = df[numeric_features].isnull().mean()
        if missing_percentages.max() > 0:
            if missing_percentages.max() > 0.3:
                # High missing values - use median
                imputer = SimpleImputer(strategy='median')
            else:
                # Low missing values - use mean
                imputer = SimpleImputer(strategy='mean')
            steps.append(('imputer', imputer))

        # Scaling
        # Choose scaler based on outlier presence
        outlier_info = metadata.get('outliers_summary', {}) if metadata else {}
        has_many_outliers = any(
            info.get('outlier_percentage', 0) > 5 
            for feature in numeric_features 
            for info in [outlier_info.get(feature, {})]
        )

        if has_many_outliers:
            scaler = RobustScaler()  # Less sensitive to outliers
        else:
            scaler = StandardScaler()  # Standard scaling

        steps.append(('scaler', scaler))

        return Pipeline(steps)

    def _build_categorical_pipeline(self, df: pd.DataFrame, categorical_features: List[str],
                                   metadata: Dict[str, Any] = None) -> Pipeline:
        """Build preprocessing pipeline for categorical features"""
        steps = []

        # Imputation
        imputer = SimpleImputer(strategy='most_frequent')
        steps.append(('imputer', imputer))

        # Encoding
        # Choose encoding strategy based on cardinality
        max_cardinality = max([df[col].nunique() for col in categorical_features])

        if max_cardinality <= 10:
            # Low cardinality - use OneHotEncoder
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        else:
            # Higher cardinality - use OrdinalEncoder
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        steps.append(('encoder', encoder))

        return Pipeline(steps)

    def build_model_pipeline(self, model_config: Dict[str, Any], preprocessor: ColumnTransformer,
                           task_type: str) -> Pipeline:
        """
        Build complete model pipeline with preprocessing and model

        Args:
            model_config: Model configuration from ModelAdvisor
            preprocessor: Fitted preprocessing pipeline
            task_type: 'classification' or 'regression'

        Returns:
            Complete sklearn Pipeline
        """
        # Get model instance
        model = self._get_model_instance(model_config, task_type)

        # Create pipeline steps
        steps = [
            ('preprocessor', preprocessor),
            ('model', model)
        ]

        # Add feature selection if dataset is high-dimensional
        if hasattr(preprocessor, 'transformers_'):
            # Estimate feature count after preprocessing
            n_features_estimate = self._estimate_features_after_preprocessing(preprocessor)
            if n_features_estimate > 100:  # High dimensional
                if task_type == 'classification':
                    selector = SelectKBest(score_func=f_classif, k=min(50, n_features_estimate // 2))
                else:
                    selector = SelectKBest(score_func=f_regression, k=min(50, n_features_estimate // 2))

                # Insert feature selection before model
                steps.insert(-1, ('feature_selection', selector))

        return Pipeline(steps)

    def _get_model_instance(self, model_config: Dict[str, Any], task_type: str):
        """Get sklearn model instance from configuration"""
        model_name = model_config['model']
        params = model_config.get('params', {})

        # Add random state to relevant parameters
        if 'random_state' not in params and self._model_uses_random_state(model_name):
            params['random_state'] = self.random_state

        # Model mapping
        model_map = {
            # Classification models
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'SVC': SVC,
            'GaussianNB': GaussianNB,
            'KNeighborsClassifier': KNeighborsClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,

            # Regression models
            'LinearRegression': LinearRegression,
            'RandomForestRegressor': RandomForestRegressor,
            'SVR': SVR,
            'Ridge': Ridge,
            'ElasticNet': ElasticNet,
            'DecisionTreeRegressor': DecisionTreeRegressor,
        }

        # Add XGBoost models if available
        if XGBOOST_AVAILABLE:
            model_map.update({
                'XGBClassifier': XGBClassifier,
                'XGBRegressor': XGBRegressor
            })

        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")

        try:
            model_class = model_map[model_name]
            return model_class(**params)
        except Exception as e:
            print(f"Warning: Failed to create {model_name} with params {params}: {e}")
            # Return default instance
            return model_class()

    def _model_uses_random_state(self, model_name: str) -> bool:
        """Check if model supports random_state parameter"""
        random_state_models = [
            'LogisticRegression', 'RandomForestClassifier', 'RandomForestRegressor',
            'XGBClassifier', 'XGBRegressor', 'DecisionTreeClassifier', 'DecisionTreeRegressor'
        ]
        return model_name in random_state_models

    def _estimate_features_after_preprocessing(self, preprocessor: ColumnTransformer) -> int:
        """Estimate number of features after preprocessing"""
        total_features = 0

        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder':
                continue

            if name == 'numeric':
                # Numeric features keep same count
                total_features += len(columns)
            elif name == 'categorical':
                # For OneHotEncoder, estimate based on unique values
                # For OrdinalEncoder, keeps same count
                if hasattr(transformer, 'steps'):
                    encoder = transformer.steps[-1][1]  # Last step is encoder
                    if isinstance(encoder, OneHotEncoder):
                        # Rough estimate: assume average 3-5 categories per feature
                        total_features += len(columns) * 4
                    else:
                        total_features += len(columns)
                else:
                    total_features += len(columns)

        return total_features

    def create_pipeline_from_config(self, df: pd.DataFrame, target_column: str,
                                   model_config: Dict[str, Any], task_type: str,
                                   metadata: Dict[str, Any] = None) -> Pipeline:
        """
        Convenience method to create complete pipeline from configuration

        Args:
            df: Input DataFrame
            target_column: Target column name
            model_config: Model configuration
            task_type: Task type ('classification' or 'regression')
            metadata: Dataset metadata

        Returns:
            Complete fitted pipeline
        """
        # Build preprocessing pipeline
        preprocessor = self.build_preprocessing_pipeline(df, target_column, metadata)

        # Build complete model pipeline
        pipeline = self.build_model_pipeline(model_config, preprocessor, task_type)

        return pipeline

    def get_feature_names_out(self, pipeline: Pipeline) -> List[str]:
        """Get feature names after preprocessing"""
        try:
            if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
                return pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()
            else:
                # Fallback to generic names
                n_features = pipeline.named_steps['preprocessor'].transform(
                    pd.DataFrame(columns=self.feature_columns).head(1)
                ).shape[1]
                return [f'feature_{i}' for i in range(n_features)]
        except:
            return [f'feature_{i}' for i in range(len(self.feature_columns))]

def create_pipeline(df: pd.DataFrame, target_column: str, model_config: Dict[str, Any],
                   task_type: str, metadata: Dict[str, Any] = None) -> Pipeline:
    """
    Convenience function to create a complete pipeline

    Args:
        df: Input DataFrame
        target_column: Target column name
        model_config: Model configuration from ModelAdvisor
        task_type: 'classification' or 'regression'
        metadata: Dataset metadata from DataAnalyzer

    Returns:
        Complete sklearn Pipeline ready for training
    """
    builder = PipelineBuilder()
    return builder.create_pipeline_from_config(df, target_column, model_config, task_type, metadata)
