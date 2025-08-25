"""
Chat with Data Agent powered by Google Gemini
Uses GOOGLE_API_KEY3 for natural language data interaction
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import warnings
import plotly.express as px
import plotly.graph_objects as go

# LangChain imports for Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    warnings.warn("langchain-google-genai not available for chat agent")

from langchain.schema import HumanMessage, SystemMessage
from config.settings import (
    CHAT_WITH_DATA_CONFIG, CHAT_MAX_HISTORY, CHAT_RESPONSE_TYPES,
    GEMINI_SAFETY_SETTINGS
)
from components.utils import generate_cache_key, save_to_cache, load_from_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatWithDataAgent:
    """Gemini-powered conversational data analysis agent"""

    def __init__(self, df: pd.DataFrame, use_cache: bool = True):
        self.df = df
        self.use_cache = use_cache
        self.llm = None
        self.chat_history = []
        self.data_context = self._build_data_context()
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize Gemini LLM for chat with data"""
        api_key = CHAT_WITH_DATA_CONFIG["api_key"]

        if GEMINI_AVAILABLE and api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=CHAT_WITH_DATA_CONFIG["model"],
                    google_api_key=api_key,
                    temperature=CHAT_WITH_DATA_CONFIG["temperature"],
                    top_p=CHAT_WITH_DATA_CONFIG["top_p"],
                    top_k=CHAT_WITH_DATA_CONFIG["top_k"],
                    max_output_tokens=CHAT_WITH_DATA_CONFIG["max_output_tokens"],
                    safety_settings=GEMINI_SAFETY_SETTINGS
                )
                logger.info(f"✅ Chat Agent initialized with Gemini")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Gemini for chat: {e}")
                self.llm = None
        else:
            logger.warning("❌ Gemini not available for chat")
            self.llm = None

    def _build_data_context(self) -> Dict[str, Any]:
        """Build comprehensive context about the dataset"""
        context = {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "dtypes": self.df.dtypes.to_dict(),
            "basic_stats": {},
            "sample_data": self.df.head(3).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "unique_counts": self.df.nunique().to_dict()
        }

        # Add basic statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            context["basic_stats"][col] = {
                "mean": float(self.df[col].mean()),
                "std": float(self.df[col].std()),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "median": float(self.df[col].median())
            }

        # Add value counts for categorical columns (top 5)
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        context["categorical_distributions"] = {}
        for col in categorical_cols:
            if self.df[col].nunique() <= 20:  # Only for low cardinality
                context["categorical_distributions"][col] = self.df[col].value_counts().head(5).to_dict()

        return context

    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process user message and return response"""

        # Add to chat history
        self.chat_history.append({"role": "user", "message": user_message})

        # Keep history manageable
        if len(self.chat_history) > CHAT_MAX_HISTORY * 2:
            self.chat_history = self.chat_history[-CHAT_MAX_HISTORY:]

        # Generate response
        if self.llm:
            response = self._generate_ai_response(user_message)
        else:
            response = self._generate_fallback_response(user_message)

        # Add assistant response to history
        self.chat_history.append({"role": "assistant", "message": response["text"]})

        return response

    def _generate_ai_response(self, user_message: str) -> Dict[str, Any]:
        """Generate AI-powered response using Gemini"""

        try:
            # Create system message with data context
            system_message = self._create_chat_system_message()

            # Create prompt with user message and context
            prompt = self._create_chat_prompt(user_message)

            # Get response from Gemini
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]

            response = self.llm(messages)

            # Parse and execute response
            parsed_response = self._parse_chat_response(response.content, user_message)

            logger.info(f"🤖 Gemini chat response generated")
            return parsed_response

        except Exception as e:
            logger.error(f"❌ AI chat failed: {e}")
            return self._generate_fallback_response(user_message)

    def _create_chat_system_message(self) -> str:
        """Create system message for chat agent"""
        return f"""You are an expert data analyst AI assistant. You help users understand and explore their dataset through natural conversation.

DATASET CONTEXT:
- Shape: {self.data_context['shape']}
- Columns: {', '.join(self.data_context['columns'])}
- Data Types: {json.dumps(self.data_context['dtypes'], indent=None)}

INSTRUCTIONS:
1. Answer questions about the data accurately using the provided context
2. Provide insights, patterns, and recommendations
3. Suggest relevant visualizations when appropriate
4. Keep responses helpful, clear, and concise
5. If you need to perform calculations, describe what you would calculate
6. For visualization requests, specify the chart type and variables

RESPONSE FORMAT:
Always respond with a JSON object containing:
{{
  "text": "Your main response text",
  "response_type": "text|insight|visualization|statistical_summary",
  "chart_suggestion": {{"type": "chart_type", "x": "column", "y": "column", "title": "title"}},
  "code_suggestion": "python code if applicable",
  "confidence": "high|medium|low"
}}

Remember: You are analyzing a real dataset with {self.data_context['shape'][0]} rows and {self.data_context['shape'][1]} columns."""

    def _create_chat_prompt(self, user_message: str) -> str:
        """Create prompt for chat response"""

        # Include recent chat history for context
        history_context = ""
        if len(self.chat_history) > 1:
            recent_history = self.chat_history[-4:]  # Last 2 exchanges
            history_context = "\nRecent conversation:\n"
            for entry in recent_history:
                role = "User" if entry["role"] == "user" else "Assistant"
                history_context += f"{role}: {entry['message'][:100]}...\n"

        # Create comprehensive prompt
        prompt = f"""Current User Question: {user_message}

DATASET DETAILS:
- Rows: {self.data_context['shape'][0]:,}
- Columns: {self.data_context['shape'][1]}
- Missing Values: {sum(self.data_context['missing_values'].values())} total

SAMPLE DATA (first 3 rows):
{json.dumps(self.data_context['sample_data'], indent=2)}

BASIC STATISTICS:
{json.dumps(self.data_context['basic_stats'], indent=2)}

{history_context}

Please analyze this question and provide a helpful response about the dataset. Be specific and actionable."""

        return prompt

    def _parse_chat_response(self, response: str, user_message: str) -> Dict[str, Any]:
        """Parse Gemini's chat response"""

        try:
            # Clean response
            response = response.strip()

            # Try to extract JSON
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]

                try:
                    parsed = json.loads(json_str)
                    if 'text' in parsed:
                        return self._enhance_response(parsed, user_message)
                except json.JSONDecodeError:
                    pass

            # Fallback: treat entire response as text
            return {
                "text": response,
                "response_type": "text",
                "chart_suggestion": None,
                "code_suggestion": None,
                "confidence": "medium"
            }

        except Exception as e:
            logger.error(f"❌ Failed to parse chat response: {e}")
            return self._generate_simple_response(user_message)

    def _enhance_response(self, parsed_response: Dict[str, Any], user_message: str) -> Dict[str, Any]:
        """Enhance response with additional analysis"""

        # Add actual data calculations if needed
        if "statistics" in user_message.lower() or "summary" in user_message.lower():
            parsed_response["actual_stats"] = self._generate_summary_stats()

        # Add visualization if suggested
        if parsed_response.get("chart_suggestion") and parsed_response["chart_suggestion"]:
            chart_data = self._create_suggested_visualization(parsed_response["chart_suggestion"])
            if chart_data:
                parsed_response["chart_data"] = chart_data

        return parsed_response

    def _generate_fallback_response(self, user_message: str) -> Dict[str, Any]:
        """Generate fallback response when AI is unavailable"""

        message_lower = user_message.lower()

        # Pattern matching for common queries
        if any(word in message_lower for word in ['shape', 'size', 'rows', 'columns']):
            return {
                "text": f"This dataset has {self.data_context['shape'][0]:,} rows and {self.data_context['shape'][1]} columns. The columns are: {', '.join(self.data_context['columns'])}",
                "response_type": "statistical_summary",
                "chart_suggestion": None,
                "code_suggestion": None,
                "confidence": "high"
            }

        elif any(word in message_lower for word in ['missing', 'null', 'nan']):
            total_missing = sum(self.data_context['missing_values'].values())
            return {
                "text": f"There are {total_missing:,} missing values in total across all columns. Here's the breakdown by column: {self.data_context['missing_values']}",
                "response_type": "statistical_summary",
                "chart_suggestion": None,
                "code_suggestion": None,
                "confidence": "high"
            }

        elif any(word in message_lower for word in ['summary', 'describe', 'overview']):
            return {
                "text": self._generate_dataset_summary(),
                "response_type": "statistical_summary",
                "chart_suggestion": None,
                "code_suggestion": None,
                "confidence": "high"
            }

        else:
            return {
                "text": f"I can help you analyze this dataset with {self.data_context['shape'][0]:,} rows and {self.data_context['shape'][1]} columns. Try asking about data summary, missing values, column distributions, or correlations.",
                "response_type": "text",
                "chart_suggestion": None,
                "code_suggestion": None,
                "confidence": "medium"
            }

    def _generate_simple_response(self, user_message: str) -> Dict[str, Any]:
        """Generate simple response for error cases"""
        return {
            "text": "I'm here to help you explore your data. Try asking about dataset summary, missing values, or specific columns.",
            "response_type": "text",
            "chart_suggestion": None,
            "code_suggestion": None,
            "confidence": "low"
        }

    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        return {
            "shape": self.data_context["shape"],
            "missing_values_total": sum(self.data_context["missing_values"].values()),
            "numeric_columns_count": len(self.data_context["basic_stats"]),
            "categorical_columns_count": len(self.data_context["categorical_distributions"]),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }

    def _generate_dataset_summary(self) -> str:
        """Generate a comprehensive dataset summary"""
        summary = f"""Dataset Overview:

📊 **Basic Info:**
- Rows: {self.data_context['shape'][0]:,}
- Columns: {self.data_context['shape'][1]}
- Memory: {round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2)} MB

📈 **Column Types:**
- Numeric: {len(self.data_context['basic_stats'])}
- Categorical: {len(self.data_context['categorical_distributions'])}

❓ **Data Quality:**
- Missing values: {sum(self.data_context['missing_values'].values()):,} total
- Complete rows: {self.df.dropna().shape[0]:,}

🔍 **Key Columns:** {', '.join(self.data_context['columns'][:5])}{'...' if len(self.data_context['columns']) > 5 else ''}
"""
        return summary

    def _create_suggested_visualization(self, chart_suggestion: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create visualization data based on suggestion"""

        try:
            chart_type = chart_suggestion.get("type", "").lower()
            x_col = chart_suggestion.get("x")
            y_col = chart_suggestion.get("y")

            if not x_col or x_col not in self.df.columns:
                return None

            if chart_type in ["histogram", "hist"]:
                return {
                    "type": "histogram",
                    "data": self.df[x_col].dropna().tolist(),
                    "title": chart_suggestion.get("title", f"Distribution of {x_col}")
                }

            elif chart_type == "scatter" and y_col and y_col in self.df.columns:
                return {
                    "type": "scatter",
                    "x_data": self.df[x_col].dropna().tolist(),
                    "y_data": self.df[y_col].dropna().tolist(),
                    "title": chart_suggestion.get("title", f"{x_col} vs {y_col}")
                }

            elif chart_type == "bar":
                value_counts = self.df[x_col].value_counts().head(10)
                return {
                    "type": "bar",
                    "categories": value_counts.index.tolist(),
                    "values": value_counts.values.tolist(),
                    "title": chart_suggestion.get("title", f"{x_col} Distribution")
                }

        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")

        return None

    def get_data_insights(self) -> Dict[str, Any]:
        """Generate comprehensive data insights"""

        insights = {
            "basic_info": {
                "shape": self.data_context["shape"],
                "memory_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "missing_percentage": (sum(self.data_context["missing_values"].values()) / 
                                     (self.data_context["shape"][0] * self.data_context["shape"][1])) * 100
            },
            "column_insights": {},
            "correlations": {},
            "recommendations": []
        }

        # Column-specific insights
        for col in self.df.columns:
            col_info = {
                "type": str(self.df[col].dtype),
                "unique_count": int(self.df[col].nunique()),
                "missing_count": int(self.df[col].isnull().sum())
            }

            if self.df[col].dtype in [np.number]:
                col_info.update({
                    "mean": float(self.df[col].mean()),
                    "std": float(self.df[col].std()),
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max())
                })

            insights["column_insights"][col] = col_info

        # Correlations for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            # Get top correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:  # Strong correlation
                        corr_pairs.append({
                            "var1": corr_matrix.columns[i],
                            "var2": corr_matrix.columns[j],
                            "correlation": float(corr_value)
                        })
            insights["correlations"] = sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:5]

        # Generate recommendations
        if insights["basic_info"]["missing_percentage"] > 5:
            insights["recommendations"].append("Consider handling missing values before analysis")

        if len(insights["correlations"]) > 0:
            insights["recommendations"].append("Strong correlations detected - useful for feature selection")

        return insights

    def reset_conversation(self):
        """Reset chat history"""
        self.chat_history = []
        logger.info("Chat history reset")

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get current chat history"""
        return self.chat_history.copy()

# Convenience functions
def create_chat_agent(df: pd.DataFrame) -> ChatWithDataAgent:
    """Create a new chat with data agent"""
    return ChatWithDataAgent(df)

def quick_data_summary(df: pd.DataFrame) -> str:
    """Get a quick data summary"""
    agent = ChatWithDataAgent(df, use_cache=False)
    return agent._generate_dataset_summary()
