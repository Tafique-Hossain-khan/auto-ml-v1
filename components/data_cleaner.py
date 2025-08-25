"""
Data Cleaning Agent for AutoML System v2 - Fixed JSON Serialization
Uses GOOGLE_API_KEY2 for AI-powered data cleaning suggestions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import warnings
import json

# LangChain imports for Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    warnings.warn("langchain-google-genai not available")

from langchain.schema import HumanMessage, SystemMessage
from config.settings import DATA_CLEANING_CONFIG

warnings.filterwarnings('ignore')

def safe_json_serialize(obj: Any) -> Any:
    """Safely serialize objects to JSON-compatible format"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    elif hasattr(obj, 'item'):  # NumPy scalars
        return obj.item()
    else:
        return obj

def generate_cache_key(data_dict: Dict[str, Any]) -> str:
    """Generate a unique cache key from data dictionary"""
    import hashlib
    # Convert to JSON-safe format first
    safe_data = safe_json_serialize(data_dict)
    data_str = json.dumps(safe_data, sort_keys=True, default=str)
    return hashlib.md5(data_str.encode()).hexdigest()

def save_to_cache(key: str, data: Any, cache_dir: str = "cache") -> None:
    """Save data to cache with timestamp"""
    from pathlib import Path
    from datetime import datetime

    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    # Convert data to JSON-safe format
    safe_data = safe_json_serialize(data)

    cache_data = {
        "data": safe_data,
        "timestamp": datetime.now().isoformat(),
        "key": key
    }

    cache_file = cache_path / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, default=str)

def load_from_cache(key: str, cache_dir: str = "cache", max_age_hours: int = 24) -> Optional[Any]:
    """Load data from cache if it exists and is not expired"""
    from pathlib import Path
    from datetime import datetime, timedelta

    cache_file = Path(cache_dir) / f"{key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        # Check if cache is expired
        cache_time = datetime.fromisoformat(cache_data["timestamp"])
        if datetime.now() - cache_time > timedelta(hours=max_age_hours):
            return None

        return cache_data["data"]
    except (json.JSONDecodeError, KeyError, ValueError):
        return None

class DataCleaningAgent:
    """AI-powered data cleaning agent using Gemini - Fixed JSON serialization"""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.llm = None
        self.active_provider = "offline"
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini for data cleaning"""
        if GEMINI_AVAILABLE and DATA_CLEANING_CONFIG["api_key"]:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=DATA_CLEANING_CONFIG["model"],
                    google_api_key=DATA_CLEANING_CONFIG["api_key"],
                    temperature=DATA_CLEANING_CONFIG["temperature"],
                    max_output_tokens=DATA_CLEANING_CONFIG["max_output_tokens"]
                )
                self.active_provider = "gemini"
                print("✅ Data Cleaning Agent: Gemini initialized")
            except Exception as e:
                print(f"⚠️ Gemini initialization failed: {e}")
                self.llm = None
                self.active_provider = "offline"

    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality and return comprehensive assessment - Fixed JSON serialization"""

        # Basic statistics
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols

        # Missing values analysis - Convert to native Python types
        missing_counts = df.isnull().sum()
        total_missing = int(missing_counts.sum())  # Convert to int
        missing_percentage = float((total_missing / total_cells) * 100)  # Convert to float

        missing_by_column = {}
        for col in df.columns:
            missing_count = int(missing_counts[col])  # Convert to int
            if missing_count > 0:
                missing_by_column[col] = {
                    "count": missing_count,
                    "percentage": round(float((missing_count / n_rows) * 100), 2)  # Convert to float
                }

        # Duplicate analysis - Convert to native Python types
        duplicate_count = int(df.duplicated().sum())  # Convert to int
        duplicate_percentage = float((duplicate_count / n_rows) * 100)  # Convert to float

        # Calculate overall quality score
        quality_score = 100.0
        quality_score -= min(missing_percentage * 2, 40)
        quality_score -= min(duplicate_percentage * 1.5, 30)
        quality_score = max(0.0, quality_score)

        # Return with all native Python types
        return {
            "quality_score": float(quality_score),
            "basic_stats": {
                "n_rows": int(n_rows),
                "n_columns": int(n_cols),
                "total_cells": int(total_cells)
            },
            "missing_values": {
                "total_missing": total_missing,
                "total_missing_percentage": float(missing_percentage),
                "by_column": missing_by_column
            },
            "duplicates": {
                "count": duplicate_count,
                "percentage": float(duplicate_percentage)
            }
        }

    def suggest_cleaning_operations(self, df: pd.DataFrame, quality_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get AI-powered cleaning suggestions - Fixed caching"""

        # Create cache key with safe data
        cache_data = {
            "shape": [int(df.shape[0]), int(df.shape[1])],  # Convert to native int
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},  # Convert to str
            "quality_score": float(quality_analysis["quality_score"]),  # Ensure float
            "provider": self.active_provider
        }

        cache_key = generate_cache_key(cache_data)

        # Try cache first
        if self.use_cache:
            cached_suggestions = load_from_cache(cache_key, max_age_hours=1)
            if cached_suggestions:
                print("📋 Using cached cleaning suggestions")
                return cached_suggestions

        # Generate suggestions (simplified for clean UI)
        suggestions = self._get_rule_based_suggestions(df, quality_analysis)

        # Cache results - data is already JSON-safe
        if self.use_cache:
            save_to_cache(cache_key, suggestions)

        return suggestions

    def _get_rule_based_suggestions(self, df: pd.DataFrame, quality_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate rule-based cleaning suggestions - All native Python types"""

        suggestions = []

        # Missing values
        missing_pct = float(quality_analysis["missing_values"]["total_missing_percentage"])
        if missing_pct > 5.0:
            suggestions.append({
                "operation": "handle_missing_values",
                "description": "Handle missing values in dataset",
                "rationale": f"Dataset has {missing_pct:.1f}% missing values",
                "parameters": {"strategy": "intelligent", "threshold": 50},
                "priority": "high",
                "apply": True
            })

        # Duplicates
        dup_pct = float(quality_analysis["duplicates"]["percentage"])
        dup_count = int(quality_analysis["duplicates"]["count"])
        if dup_pct > 1.0:
            suggestions.append({
                "operation": "remove_duplicates",
                "description": "Remove duplicate rows",
                "rationale": f"Found {dup_count} duplicate rows ({dup_pct:.1f}%)",
                "parameters": {"keep": "first"},
                "priority": "high",
                "apply": True
            })

        print(f"📋 Generated {len(suggestions)} cleaning suggestions")
        return suggestions

    def apply_cleaning_operations(self, df: pd.DataFrame, operations: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply cleaning operations to dataset - Fixed stats conversion"""

        cleaned_df = df.copy()
        before_stats = self._get_data_stats(df)

        for operation in operations:
            if not operation.get('apply', False):
                continue

            try:
                op_type = operation['operation']
                params = operation['parameters']

                if op_type == "handle_missing_values":
                    cleaned_df = self._handle_missing_values(cleaned_df, params)
                elif op_type == "remove_duplicates":
                    cleaned_df = self._remove_duplicates(cleaned_df, params)

                print(f"✅ Applied: {operation['description']}")

            except Exception as e:
                print(f"❌ Failed to apply {operation['operation']}: {e}")

        after_stats = self._get_data_stats(cleaned_df)

        # Calculate changes - ensure native Python types
        changes_summary = {
            "rows_removed": int(before_stats["n_rows"] - after_stats["n_rows"]),
            "missing_values_handled": int(before_stats["missing_values"] - after_stats["missing_values"]),
            "duplicates_removed": int(before_stats["duplicates"] - after_stats["duplicates"]),
            "memory_reduction_mb": float(before_stats["memory_mb"] - after_stats["memory_mb"])
        }

        cleaning_report = {
            "before_stats": before_stats,
            "after_stats": after_stats,
            "changes_summary": changes_summary
        }

        return cleaned_df, cleaning_report

    def _handle_missing_values(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values"""
        df_cleaned = df.copy()

        # Simple imputation
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns

        # Fill numeric with median
        for col in numeric_cols:
            if df_cleaned[col].isnull().sum() > 0:
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        # Fill categorical with mode
        for col in categorical_cols:
            if df_cleaned[col].isnull().sum() > 0:
                mode_value = df_cleaned[col].mode()
                if len(mode_value) > 0:
                    df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])

        return df_cleaned

    def _remove_duplicates(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Remove duplicate rows"""
        keep = params.get("keep", "first")
        return df.drop_duplicates(keep=keep).reset_index(drop=True)

    def _get_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the dataset - Convert to native Python types"""
        return {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        }

def create_data_cleaning_agent(use_cache: bool = True) -> DataCleaningAgent:
    """Create a new data cleaning agent"""
    return DataCleaningAgent(use_cache=use_cache)
