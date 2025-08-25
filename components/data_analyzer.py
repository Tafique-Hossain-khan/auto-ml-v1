"""
Data Analyzer Component for AutoML System
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Analyzes datasets and extracts metadata for AutoML"""

    def __init__(self):
        self.metadata = {}
        self.df = None
        self.target_column = None

    def analyze_dataset(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Main method to analyze dataset and extract comprehensive metadata

        Args:
            df: Input DataFrame
            target_column: Name of target column (if None, will try to detect)

        Returns:
            Dictionary containing dataset metadata
        """
        self.df = df.copy()
        self.target_column = target_column

        # If no target specified, try to detect it
        if self.target_column is None:
            self.target_column = self._detect_target_column()

        # Basic dataset info
        basic_info = self._get_basic_info()

        # Feature analysis
        feature_info = self._analyze_features()

        # Target analysis
        target_info = self._analyze_target()

        # Data quality assessment
        quality_info = self._assess_data_quality()

        # Task type detection
        task_type = self._detect_task_type()

        # Combine all metadata
        self.metadata = {
            **basic_info,
            **feature_info,
            **target_info,
            **quality_info,
            "task_type": task_type,
            "target_column": self.target_column,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }

        return self.metadata

    def _get_basic_info(self) -> Dict[str, Any]:
        """Extract basic dataset information"""
        return {
            "n_rows": len(self.df),
            "n_features": len(self.df.columns) - (1 if self.target_column else 0),
            "total_columns": len(self.df.columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "duplicate_rows": self.df.duplicated().sum()
        }

    def _analyze_features(self) -> Dict[str, Any]:
        """Analyze feature types and characteristics"""
        feature_columns = [col for col in self.df.columns if col != self.target_column]

        numeric_cols = self.df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df[feature_columns].select_dtypes(include=['datetime64']).columns.tolist()
        boolean_cols = self.df[feature_columns].select_dtypes(include=['bool']).columns.tolist()

        # Detect text columns (object columns with high unique values)
        text_cols = []
        for col in categorical_cols.copy():
            if self.df[col].nunique() / len(self.df) > 0.9:  # High cardinality
                avg_length = self.df[col].astype(str).str.len().mean()
                if avg_length > 20:  # Average length > 20 characters
                    text_cols.append(col)
                    categorical_cols.remove(col)

        # Analyze cardinality
        cardinality_info = {}
        for col in categorical_cols:
            cardinality_info[col] = {
                "unique_values": self.df[col].nunique(),
                "top_categories": self.df[col].value_counts().head(5).to_dict()
            }

        return {
            "n_numeric": len(numeric_cols),
            "n_categorical": len(categorical_cols),
            "n_text": len(text_cols),
            "n_datetime": len(datetime_cols),
            "n_boolean": len(boolean_cols),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "text_columns": text_cols,
            "datetime_columns": datetime_cols,
            "boolean_columns": boolean_cols,
            "cardinality_info": cardinality_info
        }

    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze target variable characteristics"""
        if not self.target_column or self.target_column not in self.df.columns:
            return {"target_analysis": "No target column specified or found"}

        target_series = self.df[self.target_column]

        # Basic target info
        target_info = {
            "target_dtype": str(target_series.dtype),
            "target_unique_values": target_series.nunique(),
            "target_missing_count": target_series.isnull().sum(),
            "target_missing_percentage": round(target_series.isnull().mean() * 100, 2)
        }

        # Determine if target is numeric or categorical
        if pd.api.types.is_numeric_dtype(target_series):
            # Numeric target - could be regression or classification
            target_info.update({
                "target_min": target_series.min(),
                "target_max": target_series.max(),
                "target_mean": target_series.mean(),
                "target_std": target_series.std(),
                "target_median": target_series.median()
            })

            # Check if it might be classification (few unique values)
            if target_series.nunique() <= 20:
                target_info["possible_classification"] = True
                target_info["class_distribution"] = target_series.value_counts().to_dict()
        else:
            # Categorical target - classification
            target_info.update({
                "class_distribution": target_series.value_counts().to_dict(),
                "class_balance": self._assess_class_balance(target_series)
            })

        return target_info

    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess overall data quality"""
        quality_info = {
            "missing_values_summary": {},
            "outliers_summary": {},
            "data_quality_score": 0
        }

        # Missing values analysis
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100

        for col in self.df.columns:
            if missing_counts[col] > 0:
                quality_info["missing_values_summary"][col] = {
                    "count": int(missing_counts[col]),
                    "percentage": round(missing_percentages[col], 2)
                }

        # Outliers detection for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.target_column:
                outliers = self._detect_outliers(self.df[col])
                if outliers["outlier_count"] > 0:
                    quality_info["outliers_summary"][col] = outliers

        # Calculate overall quality score
        missing_penalty = min(missing_percentages.mean() / 10, 30)  # Max 30 point penalty
        duplicate_penalty = min((self.df.duplicated().sum() / len(self.df)) * 100 / 2, 20)  # Max 20 point penalty

        quality_info["data_quality_score"] = max(0, 100 - missing_penalty - duplicate_penalty)

        return quality_info

    def _detect_task_type(self) -> str:
        """Detect whether this is a classification or regression task"""
        if not self.target_column or self.target_column not in self.df.columns:
            return "unknown"

        target_series = self.df[self.target_column].dropna()

        # If target is clearly non-numeric, it's classification
        if not pd.api.types.is_numeric_dtype(target_series):
            return "classification"

        # If numeric, check uniqueness ratio and value types
        unique_ratio = target_series.nunique() / len(target_series)

        # If very few unique values or all integers with small range, likely classification
        if target_series.nunique() <= 20 or (
            target_series.dtype in ['int64', 'int32'] and 
            target_series.nunique() <= max(50, len(target_series) * 0.05)
        ):
            return "classification"

        # If continuous values with high uniqueness, likely regression
        if unique_ratio > 0.05:
            return "regression"

        # Edge case: check if all values are 0s and 1s (binary classification)
        if set(target_series.unique()).issubset({0, 1}):
            return "classification"

        # Default to regression for numeric targets
        return "regression"

    def _detect_target_column(self) -> Optional[str]:
        """Attempt to automatically detect target column"""
        # Common target column names
        target_candidates = [
            'target', 'y', 'label', 'class', 'outcome', 'result', 
            'prediction', 'score', 'value', 'price', 'amount'
        ]

        # Look for exact matches (case insensitive)
        for col in self.df.columns:
            if col.lower() in target_candidates:
                return col

        # Look for columns containing target-like words
        for col in self.df.columns:
            col_lower = col.lower()
            for candidate in target_candidates:
                if candidate in col_lower:
                    return col

        # If nothing found, return the last column as a heuristic
        return self.df.columns[-1] if len(self.df.columns) > 1 else None

    def _assess_class_balance(self, target_series: pd.Series) -> Dict[str, Any]:
        """Assess class balance for classification targets"""
        value_counts = target_series.value_counts()
        total_samples = len(target_series)

        # Calculate balance metrics
        majority_class_ratio = value_counts.iloc[0] / total_samples
        minority_class_ratio = value_counts.iloc[-1] / total_samples

        # Determine balance status
        if majority_class_ratio > 0.9:
            balance_status = "severely_imbalanced"
        elif majority_class_ratio > 0.7:
            balance_status = "moderately_imbalanced"
        elif majority_class_ratio > 0.6:
            balance_status = "slightly_imbalanced"
        else:
            balance_status = "balanced"

        return {
            "balance_status": balance_status,
            "majority_class_ratio": round(majority_class_ratio, 3),
            "minority_class_ratio": round(minority_class_ratio, 3),
            "imbalance_ratio": round(majority_class_ratio / minority_class_ratio, 2)
        }

    def _detect_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        if series.dtype not in [np.number]:
            return {"outlier_count": 0}

        series_clean = series.dropna()
        if len(series_clean) == 0:
            return {"outlier_count": 0}

        Q1 = series_clean.quantile(0.25)
        Q3 = series_clean.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series_clean[(series_clean < lower_bound) | (series_clean > upper_bound)]

        return {
            "outlier_count": len(outliers),
            "outlier_percentage": round(len(outliers) / len(series_clean) * 100, 2),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }

    def get_preprocessing_recommendations(self) -> Dict[str, List[str]]:
        """Generate preprocessing recommendations based on analysis"""
        recommendations = {
            "required": [],
            "suggested": [],
            "optional": []
        }

        if not self.metadata:
            return recommendations

        # Missing values recommendations
        if self.metadata.get("missing_values_summary"):
            recommendations["required"].append("Handle missing values through imputation")

        # Scaling recommendations
        if self.metadata.get("n_numeric", 0) > 0:
            recommendations["suggested"].append("Apply feature scaling/normalization")

        # Encoding recommendations  
        if self.metadata.get("n_categorical", 0) > 0:
            recommendations["required"].append("Apply categorical encoding (OneHot/Ordinal)")

        # Class imbalance recommendations
        target_info = self.metadata.get("class_balance", {})
        if target_info.get("balance_status") in ["severely_imbalanced", "moderately_imbalanced"]:
            recommendations["suggested"].append("Consider class balancing techniques (SMOTE, class weights)")

        # Outlier handling
        if any(info.get("outlier_percentage", 0) > 5 for info in self.metadata.get("outliers_summary", {}).values()):
            recommendations["optional"].append("Consider outlier removal or robust scaling")

        # High cardinality features
        high_card_features = [
            col for col, info in self.metadata.get("cardinality_info", {}).items()
            if info.get("unique_values", 0) > 50
        ]
        if high_card_features:
            recommendations["optional"].append("Consider feature selection or dimensionality reduction")

        return recommendations

def analyze_uploaded_data(df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
    """
    Convenience function to analyze uploaded data

    Args:
        df: Uploaded DataFrame
        target_column: Target column name (optional)

    Returns:
        Complete analysis metadata
    """
    analyzer = DataAnalyzer()
    metadata = analyzer.analyze_dataset(df, target_column)
    recommendations = analyzer.get_preprocessing_recommendations()

    return {
        "metadata": metadata,
        "preprocessing_recommendations": recommendations
    }
