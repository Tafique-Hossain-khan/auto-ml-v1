"""
Utility functions for AutoML System
"""
import os
import json
import hashlib
import pickle
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

def create_directories():
    """Create necessary directories for the AutoML system"""
    directories = ["models", "cache", "artifacts"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def generate_cache_key(data_dict: Dict[str, Any]) -> str:
    """Generate a unique cache key from data dictionary"""
    data_str = json.dumps(data_dict, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

def save_to_cache(key: str, data: Any, cache_dir: str = "cache") -> None:
    """Save data to cache with timestamp"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)

    cache_data = {
        "data": data,
        "timestamp": datetime.now().isoformat(),
        "key": key
    }

    cache_file = cache_path / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

def load_from_cache(key: str, cache_dir: str = "cache", max_age_hours: int = 24) -> Optional[Any]:
    """Load data from cache if it exists and is not expired"""
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

def save_model_artifacts(model, preprocessor, metadata: Dict[str, Any], model_name: str = "best_model") -> Dict[str, str]:
    """Save model artifacts and return file paths"""
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Generate unique timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save preprocessor
    preprocessor_path = artifacts_dir / f"{model_name}_preprocessor_{timestamp}.pkl"
    joblib.dump(preprocessor, preprocessor_path)

    # Save model
    model_path = artifacts_dir / f"{model_name}_model_{timestamp}.pkl"
    joblib.dump(model, model_path)

    # Save metadata
    metadata_path = artifacts_dir / f"{model_name}_metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "preprocessor_path": str(preprocessor_path),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path)
    }

def load_model_artifacts(artifacts_paths: Dict[str, str]) -> tuple:
    """Load saved model artifacts"""
    # Load preprocessor
    preprocessor = joblib.load(artifacts_paths["preprocessor_path"])

    # Load model
    model = joblib.load(artifacts_paths["model_path"])

    # Load metadata
    with open(artifacts_paths["metadata_path"], "r") as f:
        metadata = json.load(f)

    return model, preprocessor, metadata

def safe_json_serialize(obj: Any) -> Any:
    """Safely serialize objects to JSON-compatible format"""
    if hasattr(obj, 'tolist'):  # NumPy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # NumPy scalars
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    else:
        return obj

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def validate_dataframe(df) -> Dict[str, Any]:
    """Validate uploaded dataframe and return validation results"""
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }

    # Check if dataframe is empty
    if df.empty:
        validation_results["is_valid"] = False
        validation_results["errors"].append("Uploaded file is empty")
        return validation_results

    # Check minimum rows
    if len(df) < 10:
        validation_results["warnings"].append("Dataset has very few rows (<10). Results may not be reliable.")

    # Check for all NaN columns
    all_nan_cols = df.columns[df.isnull().all()].tolist()
    if all_nan_cols:
        validation_results["warnings"].append(f"Columns with all missing values: {all_nan_cols}")

    # Check memory usage
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    validation_results["info"]["memory_usage_mb"] = round(memory_usage_mb, 2)

    if memory_usage_mb > 500:  # 500MB threshold
        validation_results["warnings"].append(f"Large dataset detected ({memory_usage_mb:.1f}MB). Processing may be slow.")

    return validation_results

class ModelPredictor:
    """Class for making predictions with saved models"""

    def __init__(self, model_path: str, preprocessor_path: str, metadata_path: str):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def predict(self, X):
        """Make predictions on new data"""
        # Preprocess the data
        X_processed = self.preprocessor.transform(X)

        # Make predictions
        predictions = self.model.predict(X_processed)

        # Get prediction probabilities for classification
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_processed)
            return predictions, probabilities

        return predictions, None

    def get_feature_names(self):
        """Get feature names from preprocessor if available"""
        if hasattr(self.preprocessor, "get_feature_names_out"):
            return self.preprocessor.get_feature_names_out()
        return None
