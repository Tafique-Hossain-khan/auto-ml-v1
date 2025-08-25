"""
LangChain-based Model Advisor for AutoML System - Gemini Compatible
Uses GOOGLE_API_KEY1 for model selection
"""
import json
import logging
from typing import Dict, List, Any, Optional
import warnings

# LangChain imports for Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    warnings.warn("langchain-google-genai not available. Install with: pip install langchain-google-genai")

# Fallback imports
try:
    from langchain.chat_models import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from langchain.schema import HumanMessage, SystemMessage
from config.settings import (
    MODEL_TRAINING_CONFIG, OPENAI_API_KEY, DEFAULT_OPENAI_MODEL, 
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MAX_MODELS_PER_CALL, 
    DEFAULT_CLASSIFICATION_MODELS, DEFAULT_REGRESSION_MODELS,
    DEFAULT_LLM_PROVIDER
)
from components.utils import generate_cache_key, save_to_cache, load_from_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAdvisor:
    """Gemini-powered model selection advisor with fallback options"""

    def __init__(self, use_cache: bool = True, preferred_provider: str = DEFAULT_LLM_PROVIDER):
        self.use_cache = use_cache
        self.preferred_provider = preferred_provider
        self.llm = None
        self.active_provider = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LangChain LLM with Gemini as primary and fallbacks"""

        # Try Gemini first (primary) - using GOOGLE_API_KEY1
        if self.preferred_provider == "gemini" and GEMINI_AVAILABLE and MODEL_TRAINING_CONFIG["api_key"]:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=MODEL_TRAINING_CONFIG["model"],
                    google_api_key=MODEL_TRAINING_CONFIG["api_key"],
                    temperature=MODEL_TRAINING_CONFIG["temperature"],
                    top_p=MODEL_TRAINING_CONFIG["top_p"],
                    top_k=MODEL_TRAINING_CONFIG["top_k"],
                    max_output_tokens=MODEL_TRAINING_CONFIG["max_output_tokens"],
                )
                self.active_provider = "gemini"
                logger.info(f"✅ Initialized Gemini LLM: {MODEL_TRAINING_CONFIG['model']}")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Gemini: {e}")

        # Try OpenAI as fallback
        if OPENAI_AVAILABLE and OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model=DEFAULT_OPENAI_MODEL,
                    temperature=MODEL_TRAINING_CONFIG["temperature"],
                    openai_api_key=OPENAI_API_KEY
                )
                self.active_provider = "openai"
                logger.info(f"✅ Initialized OpenAI LLM (fallback): {DEFAULT_OPENAI_MODEL}")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize OpenAI: {e}")

        # Try Anthropic as fallback
        if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
            try:
                self.llm = ChatAnthropic(
                    model=ANTHROPIC_MODEL,
                    temperature=MODEL_TRAINING_CONFIG["temperature"],
                    anthropic_api_key=ANTHROPIC_API_KEY
                )
                self.active_provider = "anthropic"
                logger.info(f"✅ Initialized Anthropic LLM (fallback): {ANTHROPIC_MODEL}")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Anthropic: {e}")

        # No LLM available
        logger.warning("❌ No LLM available. Will use fallback models only.")
        self.llm = None
        self.active_provider = "offline"

    def suggest_models(self, metadata: Dict[str, Any], user_constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Suggest optimal models based on dataset metadata and user constraints

        Args:
            metadata: Dataset metadata from DataAnalyzer
            user_constraints: User-defined constraints (time_limit, interpretability, etc.)

        Returns:
            List of suggested model configurations
        """
        # Create cache key
        cache_key = generate_cache_key({
            "metadata_subset": self._extract_cache_relevant_metadata(metadata),
            "constraints": user_constraints or {},
            "provider": self.active_provider
        })

        # Try to load from cache
        if self.use_cache:
            cached_result = load_from_cache(cache_key)
            if cached_result:
                logger.info(f"📋 Using cached model suggestions (provider: {self.active_provider})")
                return cached_result

        # Get suggestions from LLM or fallback
        if self.llm and self.active_provider != "offline":
            suggestions = self._get_llm_suggestions(metadata, user_constraints)
        else:
            suggestions = self._get_fallback_suggestions(metadata, user_constraints)

        # Cache the results
        if self.use_cache:
            save_to_cache(cache_key, suggestions)

        return suggestions

    def _extract_cache_relevant_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only cache-relevant metadata to avoid cache misses"""
        return {
            "task_type": metadata.get("task_type"),
            "n_rows": metadata.get("n_rows"),
            "n_features": metadata.get("n_features"),
            "n_numeric": metadata.get("n_numeric"),
            "n_categorical": metadata.get("n_categorical"),
            "class_balance": metadata.get("class_balance", {}).get("balance_status"),
            "data_quality_score": metadata.get("data_quality_score")
        }

    def _get_llm_suggestions(self, metadata: Dict[str, Any], user_constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get model suggestions using LangChain LLM (Gemini primary, others fallback)"""
        try:
            # Prepare constraints
            constraints = user_constraints or {}
            time_limit = constraints.get("time_limit_minutes", 5)
            interpretability = constraints.get("interpretability", "medium")

            # Create the prompt
            prompt = self._create_model_selection_prompt(metadata, constraints)

            # Create system message optimized for Gemini
            system_message = self._create_system_message_for_gemini()

            # Get response from LLM
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]

            response = self.llm(messages)

            # Parse the response
            suggestions = self._parse_llm_response(response.content, metadata["task_type"])

            logger.info(f"🤖 {self.active_provider.title()} suggested {len(suggestions)} models")
            return suggestions

        except Exception as e:
            logger.error(f"❌ {self.active_provider.title()} suggestion failed: {e}")
            return self._get_fallback_suggestions(metadata, user_constraints)

    def _create_system_message_for_gemini(self) -> str:
        """Create system message optimized for Gemini's instruction following"""
        return """You are an expert AutoML consultant specializing in machine learning model selection. Your task is to analyze dataset characteristics and recommend the optimal sklearn models with appropriate hyperparameters.

IMPORTANT INSTRUCTIONS:
1. You must respond with ONLY a valid JSON array
2. Suggest exactly 3 models maximum
3. Use only sklearn model class names
4. Include realistic hyperparameter values
5. Consider dataset size, features, and task complexity
6. Prioritize models that work well with the given constraints

Response format must be exactly:
[
  {"name": "descriptive_name", "model": "SklearnClassName", "params": {"param": value}},
  {"name": "descriptive_name", "model": "SklearnClassName", "params": {"param": value}},
  {"name": "descriptive_name", "model": "SklearnClassName", "params": {"param": value}}
]

Do not include any explanations, markdown formatting, or additional text."""

    def _create_model_selection_prompt(self, metadata: Dict[str, Any], constraints: Dict[str, Any]) -> str:
        """Create detailed prompt for model selection optimized for Gemini"""

        # Prepare dataset summary
        dataset_summary = f"""Dataset Analysis:
- Task: {metadata.get('task_type', 'unknown')}
- Samples: {metadata.get('n_rows', 0):,}
- Features: {metadata.get('n_features', 0)} total
- Numeric: {metadata.get('n_numeric', 0)}
- Categorical: {metadata.get('n_categorical', 0)}
- Quality Score: {metadata.get('data_quality_score', 0):.0f}/100"""

        # Add task-specific info
        if metadata.get('task_type') == 'classification':
            balance_info = metadata.get('class_balance', {})
            dataset_summary += f"\n- Class Balance: {balance_info.get('balance_status', 'unknown')}"
            if 'imbalance_ratio' in balance_info:
                dataset_summary += f" (ratio: {balance_info['imbalance_ratio']:.1f})"

        # Prepare constraints
        time_limit = constraints.get('time_limit_minutes', 5)
        interpretability = constraints.get('interpretability', 'medium')

        constraints_summary = f"""Constraints:
- Time Limit: {time_limit} minutes
- Interpretability Need: {interpretability}
- Max Models: {MAX_MODELS_PER_CALL}"""

        # Available models based on task
        if metadata.get('task_type') == 'classification':
            models_list = "LogisticRegression, RandomForestClassifier, XGBClassifier, SVC, GaussianNB, KNeighborsClassifier"
        else:
            models_list = "LinearRegression, RandomForestRegressor, XGBRegressor, SVR, Ridge, ElasticNet"

        # Create the main prompt
        prompt = f"""{dataset_summary}

{constraints_summary}

Available Models: {models_list}

Selection Criteria:
- Choose models suitable for dataset size ({metadata.get('n_rows', 0):,} samples)
- Consider feature count ({metadata.get('n_features', 0)} features)
- Account for time constraints ({time_limit} minutes)
- Match interpretability requirements ({interpretability})
- Handle data quality issues (score: {metadata.get('data_quality_score', 0):.0f}/100)

Return exactly 3 optimal models as JSON array with realistic hyperparameters."""

        return prompt

    def _parse_llm_response(self, response: str, task_type: str) -> List[Dict[str, Any]]:
        """Parse and validate LLM response with improved Gemini compatibility"""
        try:
            # Clean the response
            response = response.strip()

            # Remove common markdown formatting that Gemini might add
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                json_str = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                json_str = response[start:end].strip()
            else:
                json_str = response

            # Find JSON array bounds more robustly
            start_bracket = json_str.find('[')
            end_bracket = json_str.rfind(']') + 1

            if start_bracket != -1 and end_bracket > start_bracket:
                json_str = json_str[start_bracket:end_bracket]

            # Parse JSON
            suggestions = json.loads(json_str)

            # Validate suggestions
            validated_suggestions = []
            for suggestion in suggestions[:MAX_MODELS_PER_CALL]:
                if self._validate_suggestion(suggestion, task_type):
                    validated_suggestions.append(suggestion)

            if not validated_suggestions:
                raise ValueError("No valid suggestions found in LLM response")

            logger.info(f"✅ Successfully parsed {len(validated_suggestions)} model suggestions from {self.active_provider}")
            return validated_suggestions

        except Exception as e:
            logger.error(f"❌ Failed to parse {self.active_provider} response: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            # Return fallback suggestions
            return self._get_default_models(task_type)

    def _validate_suggestion(self, suggestion: Dict[str, Any], task_type: str) -> bool:
        """Validate a single model suggestion"""
        required_keys = ['name', 'model', 'params']

        # Check required keys
        if not all(key in suggestion for key in required_keys):
            logger.warning(f"Missing required keys in suggestion: {suggestion}")
            return False

        # Check model name validity
        valid_models = self._get_valid_models(task_type)
        if suggestion['model'] not in valid_models:
            logger.warning(f"Invalid model name: {suggestion['model']}")
            return False

        # Check if params is a dictionary
        if not isinstance(suggestion['params'], dict):
            logger.warning(f"Invalid params format: {suggestion['params']}")
            return False

        return True

    def _get_valid_models(self, task_type: str) -> List[str]:
        """Get list of valid sklearn model class names"""
        if task_type == 'classification':
            return [
                'LogisticRegression', 'RandomForestClassifier', 'XGBClassifier',
                'SVC', 'GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier'
            ]
        else:
            return [
                'LinearRegression', 'RandomForestRegressor', 'XGBRegressor',
                'SVR', 'Ridge', 'ElasticNet', 'DecisionTreeRegressor'
            ]

    def _get_fallback_suggestions(self, metadata: Dict[str, Any], user_constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get fallback model suggestions when LLM is unavailable"""
        logger.info(f"🔄 Using fallback model suggestions (provider: {self.active_provider})")

        task_type = metadata.get('task_type', 'classification')
        n_rows = metadata.get('n_rows', 0)
        n_features = metadata.get('n_features', 0)

        if task_type == 'classification':
            suggestions = self._get_default_models('classification')
        else:
            suggestions = self._get_default_models('regression')

        # Adjust hyperparameters based on dataset characteristics
        adjusted_suggestions = []
        for suggestion in suggestions:
            adjusted = self._adjust_hyperparameters(suggestion, n_rows, n_features)
            adjusted_suggestions.append(adjusted)

        return adjusted_suggestions[:MAX_MODELS_PER_CALL]

    def _get_default_models(self, task_type: str) -> List[Dict[str, Any]]:
        """Get default model configurations"""
        if task_type == 'classification':
            return DEFAULT_CLASSIFICATION_MODELS.copy()
        else:
            return DEFAULT_REGRESSION_MODELS.copy()

    def _adjust_hyperparameters(self, suggestion: Dict[str, Any], n_rows: int, n_features: int) -> Dict[str, Any]:
        """Adjust hyperparameters based on dataset characteristics"""
        model_name = suggestion['model']
        params = suggestion['params'].copy()

        # Adjust based on dataset size
        if 'RandomForest' in model_name:
            if n_rows > 10000:
                params['n_estimators'] = 200
                params['max_depth'] = 15
            elif n_rows > 1000:
                params['n_estimators'] = 100
                params['max_depth'] = 10
            else:
                params['n_estimators'] = 50
                params['max_depth'] = 5

        elif 'XGB' in model_name:
            if n_rows > 10000:
                params['n_estimators'] = 300
                params['max_depth'] = 8
                params['learning_rate'] = 0.1
            elif n_rows > 1000:
                params['n_estimators'] = 200
                params['max_depth'] = 6
                params['learning_rate'] = 0.1
            else:
                params['n_estimators'] = 100
                params['max_depth'] = 4
                params['learning_rate'] = 0.15

        return {
            'name': suggestion['name'],
            'model': model_name,
            'params': params
        }

    def get_active_provider(self) -> str:
        """Get the currently active LLM provider"""
        return self.active_provider

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the active provider"""
        return {
            "active_provider": self.active_provider,
            "gemini_available": GEMINI_AVAILABLE,
            "openai_available": OPENAI_AVAILABLE,
            "anthropic_available": ANTHROPIC_AVAILABLE,
            "model_name": MODEL_TRAINING_CONFIG["model"] if self.active_provider == "gemini" else "N/A"
        }

def get_model_suggestions(metadata: Dict[str, Any], user_constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to get model suggestions using Gemini

    Args:
        metadata: Dataset metadata from DataAnalyzer
        user_constraints: Optional user constraints

    Returns:
        List of model suggestions
    """
    advisor = ModelAdvisor()
    return advisor.suggest_models(metadata, user_constraints)

def get_gemini_advisor_info() -> Dict[str, Any]:
    """Get information about Gemini advisor capabilities"""
    advisor = ModelAdvisor()
    return advisor.get_provider_info()
