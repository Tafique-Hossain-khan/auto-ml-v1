"""
AutoML System Components v2 - Enhanced with Data Cleaning and Chat Agents

Core components:
- DataAnalyzer: Dataset analysis and profiling  
- ModelAdvisor: Gemini-powered model selection (GOOGLE_API_KEY1)
- PipelineBuilder: Sklearn pipeline construction
- Trainer: Model training and evaluation
- Utils: Utility functions and helpers

New v2 components:
- DataCleaningAgent: Gemini-powered data cleaning (GOOGLE_API_KEY2)
- ChatWithDataAgent: Conversational data analysis (GOOGLE_API_KEY3)
"""

from .data_analyzer import DataAnalyzer, analyze_uploaded_data
from .model_advisor import ModelAdvisor, get_model_suggestions, get_gemini_advisor_info
from .pipeline_builder import PipelineBuilder, create_pipeline
from .trainer import AutoMLTrainer, train_automl_models
from .utils import (
    save_model_artifacts, load_model_artifacts,
    ModelPredictor, validate_dataframe
)
from .data_cleaner import DataCleaningAgent, create_data_cleaning_agent
from .chat_agent import ChatWithDataAgent, create_chat_agent, quick_data_summary

__all__ = [
    # Core components
    'DataAnalyzer', 'analyze_uploaded_data',
    'ModelAdvisor', 'get_model_suggestions', 'get_gemini_advisor_info', 
    'PipelineBuilder', 'create_pipeline',
    'AutoMLTrainer', 'train_automl_models',
    'save_model_artifacts', 'load_model_artifacts',
    'ModelPredictor', 'validate_dataframe',

    # v2 New features
    'DataCleaningAgent', 'create_data_cleaning_agent', 'quick_clean_data',
    'ChatWithDataAgent', 'create_chat_agent', 'quick_data_summary'
]
