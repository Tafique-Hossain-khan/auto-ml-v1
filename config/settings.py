"""
Configuration settings for AutoML System v2 - Enhanced Gemini Integration
Features: Model Training + Data Cleaning + Chat with Data
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration - Separate keys for each feature
GOOGLE_API_KEY1 = os.getenv("GOOGLE_API_KEY1", "")  # Model Training
GOOGLE_API_KEY2 = os.getenv("GOOGLE_API_KEY2", "")  # Data Cleaning  
GOOGLE_API_KEY3 = os.getenv("GOOGLE_API_KEY3", "")  # Chat with Data

# Fallback to single key if separate keys not provided
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY1 and GOOGLE_API_KEY:
    GOOGLE_API_KEY1 = GOOGLE_API_KEY
if not GOOGLE_API_KEY2 and GOOGLE_API_KEY:
    GOOGLE_API_KEY2 = GOOGLE_API_KEY
if not GOOGLE_API_KEY3 and GOOGLE_API_KEY:
    GOOGLE_API_KEY3 = GOOGLE_API_KEY

# Alternative LLM providers (fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Default LLM Settings - All using Gemini Flash for free tier
DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"  # Free tier optimized
ALTERNATIVE_GEMINI_MODEL = "gemini-1.5-flash"  # Backup
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
ANTHROPIC_MODEL = "claude-3-haiku-20240307"

# Feature-specific model configurations
MODEL_TRAINING_CONFIG = {
    "api_key": GOOGLE_API_KEY1,
    "model": "gemini-1.5-flash",
    "temperature": 0.1,
    "top_p": 0.8,
    "top_k": 10,
    "max_output_tokens": 2048
}

DATA_CLEANING_CONFIG = {
    "api_key": GOOGLE_API_KEY2,
    "model": "gemini-1.5-flash",
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 20,
    "max_output_tokens": 3072
}

CHAT_WITH_DATA_CONFIG = {
    "api_key": GOOGLE_API_KEY3,
    "model": "gemini-1.5-flash",
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 30,
    "max_output_tokens": 4096
}

# Free API limits strategy
MAX_MODELS_PER_CALL = 3
USE_CACHING = True
CACHE_DURATION_HOURS = 24

# Legacy Gemini settings (for backward compatibility)
GEMINI_TEMPERATURE = 0.1
GEMINI_TOP_P = 0.8
GEMINI_TOP_K = 10
GEMINI_MAX_OUTPUT_TOKENS = 2048

# Model Training Settings
DEFAULT_CV_FOLDS = 5
DEFAULT_TIME_LIMIT_MINUTES = 5
DEFAULT_RANDOM_STATE = 42

# File paths
MODELS_DIR = "models"
CACHE_DIR = "cache"
ARTIFACTS_DIR = "artifacts"

# Preprocessing settings
NUMERIC_IMPUTATION_STRATEGY = "median"
CATEGORICAL_IMPUTATION_STRATEGY = "most_frequent"
FEATURE_SELECTION_THRESHOLD = 0.01

# Classification metrics priority
CLASSIFICATION_METRICS = ["accuracy", "f1_weighted", "roc_auc_ovr"]
REGRESSION_METRICS = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

# Default model candidates for fallback
DEFAULT_CLASSIFICATION_MODELS = [
    {"name": "logistic_regression", "model": "LogisticRegression", "params": {"max_iter": 1000}},
    {"name": "random_forest", "model": "RandomForestClassifier", "params": {"n_estimators": 100}},
    {"name": "xgboost", "model": "XGBClassifier", "params": {"n_estimators": 100}}
]

DEFAULT_REGRESSION_MODELS = [
    {"name": "linear_regression", "model": "LinearRegression", "params": {}},
    {"name": "random_forest", "model": "RandomForestRegressor", "params": {"n_estimators": 100}},
    {"name": "xgboost", "model": "XGBRegressor", "params": {"n_estimators": 100}}
]

# Streamlit settings
STREAMLIT_PAGE_TITLE = "AutoML System v2 - Enhanced with Gemini AI"
STREAMLIT_PAGE_ICON = "🤖"
STREAMLIT_LAYOUT = "wide"
MAX_FILE_SIZE_MB = 200

# Data Cleaning Settings
DATA_CLEANING_OPERATIONS = [
    "handle_missing_values",
    "remove_duplicates", 
    "detect_outliers",
    "standardize_formats",
    "validate_data_types",
    "feature_engineering"
]

# Chat with Data Settings
CHAT_MAX_HISTORY = 10
CHAT_RESPONSE_TYPES = ["text", "visualization", "statistical_summary", "insights"]

# Gemini API Safety Settings
GEMINI_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH", 
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]
