"""
FIXED: Dynamic AI-Powered Chat Agent for AutoML System Integration
Integrates with existing app.py - replaces chat_agent.py completely
"""

import pandas as pd
import numpy as np
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import builtins

# LangChain imports for Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
    print("✅ langchain_google_genai imported successfully")
except ImportError:
    GEMINI_AVAILABLE = False
    warnings.warn("langchain-google-genai not available for chat agent")
    print("❌ langchain_google_genai import failed")

try:
    from langchain.schema import HumanMessage, SystemMessage
    print("✅ LangChain schema imported successfully")
except ImportError:
    print("❌ LangChain schema import failed")

try:
    from config.settings import (
        CHAT_WITH_DATA_CONFIG, CHAT_MAX_HISTORY,
        GEMINI_SAFETY_SETTINGS
    )
    print("✅ Config settings imported successfully")
except ImportError:
    print("❌ Config import failed - using fallback")
    # Fallback config
    CHAT_WITH_DATA_CONFIG = {"api_key": None}
    CHAT_MAX_HISTORY = 5
    GEMINI_SAFETY_SETTINGS = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrulyDynamicChatAgent:
    """
    FIXED VERSION: AI-DRIVEN Chat Agent for AutoML System Integration
    - Works with existing app.py structure
    - Generates actual visualizations from natural language
    - Optimized for Gemini free tier
    """

    def __init__(self, df: pd.DataFrame, use_cache: bool = True):
        print("🔄 Initializing TrulyDynamicChatAgent...")
        self.df = df
        self.use_cache = use_cache
        self.llm = None
        self.chat_history = []

        print(f"📊 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"📋 Columns: {list(df.columns)}")

        self.data_info = self._create_comprehensive_data_info()
        self._initialize_gemini()

        print(f"✅ Chat agent initialized. AI available: {self.llm is not None}")

    def _initialize_gemini(self):
        api_key = CHAT_WITH_DATA_CONFIG.get("api_key")
        if GEMINI_AVAILABLE and api_key:
            try:
                # Adapt safety settings from list to dict as needed by Gemini
                if isinstance(GEMINI_SAFETY_SETTINGS, list):
                    # Convert list of dicts to a single dict with proper structure, e.g.:
                    adapted_safety_settings = {item["category"]: item["threshold"] for item in GEMINI_SAFETY_SETTINGS}
                else:
                    adapted_safety_settings = GEMINI_SAFETY_SETTINGS

                self.llm = ChatGoogleGenerativeAI(
                    model=CHAT_WITH_DATA_CONFIG["model"],
                    google_api_key=api_key,
                    temperature=CHAT_WITH_DATA_CONFIG["temperature"],
                    top_p=CHAT_WITH_DATA_CONFIG.get("top_p"),
                    top_k=CHAT_WITH_DATA_CONFIG.get("top_k"),
                    max_output_tokens=CHAT_WITH_DATA_CONFIG["max_output_tokens"],
                    safety_settings=adapted_safety_settings 
                )
                print("✅ Gemini Chat Agent initialized successfully!")
            except Exception as e:
                print(f"❌ Gemini initialization failed: {e}")
                self.llm = None
        else:
            print("❌ Gemini not available - check API key and installation")
            self.llm = None


    def _create_comprehensive_data_info(self) -> Dict[str, Any]:
        """Create efficient data information for AI context"""
        print("🔄 Creating data information...")

        info = {
            "shape": {"rows": len(self.df), "columns": len(self.df.columns)},
            "columns": {},
        }

        # Efficient column analysis
        for col in self.df.columns:
            col_info = {
                "name": col,
                "dtype": str(self.df[col].dtype),
                "unique_count": int(self.df[col].nunique()),
                "missing_count": int(self.df[col].isnull().sum()),
                "is_numeric": pd.api.types.is_numeric_dtype(self.df[col]),
                "is_categorical": pd.api.types.is_categorical_dtype(self.df[col]) or self.df[col].dtype == 'object'
            }

            # Add sample values and stats
            if col_info["is_numeric"]:
                col_info.update({
                    "min": float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None,
                    "max": float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None,
                    "mean": float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None
                })
            elif col_info["is_categorical"]:
                unique_vals = self.df[col].dropna().unique()
                col_info["sample_values"] = unique_vals[:3].tolist() if len(unique_vals) > 0 else []

            info["columns"][col] = col_info

        print(f"✅ Data info created for {len(info['columns'])} columns")
        return info

    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Main chat method - integrates with existing AutoML app structure
        """
        print(f"💬 Processing query: '{user_message}'")

        user_message = user_message.strip()
        if not user_message:
            print("❌ Empty user message")
            return self._create_response("Please ask me something about your data!")

        # Add to history
        self.chat_history.append({"role": "user", "message": user_message})
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-8:]

        try:
            if self.llm:
                print("🤖 Using AI-powered response generation...")
                response = self._generate_ai_powered_response(user_message)
            else:
                print("⚠️ AI not available, using smart fallback...")
                response = self._create_smart_fallback_response(user_message)

            # Add to history
            self.chat_history.append({
                "role": "assistant",
                "message": response["text"][:150] + "..." if len(response["text"]) > 150 else response["text"]
            })

            print(f"✅ Response generated successfully")
            return response

        except Exception as e:
            print(f"❌ Chat processing failed: {e}")
            logger.error(f"❌ Chat processing failed: {e}")
            return self._create_response(f"I encountered an error: {str(e)}. Please try rephrasing your question.")

    def _generate_ai_powered_response(self, user_message: str) -> Dict[str, Any]:
        """Generate AI-powered response optimized for free tier"""
        print("🔄 Generating AI response...")

        try:
            # Create efficient prompts
            system_prompt = self._create_ai_system_prompt()
            user_prompt = self._create_ai_user_prompt(user_message)

            print(f"📝 System prompt length: {len(system_prompt)}")
            print(f"📝 User prompt length: {len(user_prompt)}")

            # Get AI response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            print("🔄 Calling Gemini AI...")
            ai_response = self.llm.invoke(messages)
            print(f"✅ AI response received, length: {len(ai_response.content)}")

            # Process AI response
            return self._process_ai_response(ai_response.content, user_message)

        except Exception as e:
            print(f"❌ AI response generation failed: {e}")
            logger.error(f"❌ AI response generation failed: {e}")
            return self._create_response(f"AI analysis failed: {str(e)}. Please try a different question.")

    def _create_ai_system_prompt(self) -> str:
        """Create token-efficient system prompt for Gemini"""
        # Get column info efficiently
        numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        categorical_cols = [col for col in self.df.columns if not pd.api.types.is_numeric_dtype(self.df[col])]

        return f"""You are a data visualization expert. Analyze user queries and provide appropriate responses.

DATASET: {self.df.shape[0]} rows × {self.df.shape[1]} columns
NUMERIC: {numeric_cols[:5]}
CATEGORICAL: {categorical_cols[:5]}

RULES:
1. For charts/plots/graphs: Generate Python plotly code
2. For analysis: Provide insights + calculations  
3. Use 'df' as dataframe variable
4. Create 'fig' variable for visualizations
5. Be concise due to token limits

RESPONSE FORMAT (JSON only):
{{
    "query_type": "visualization|analysis|information",
    "response_text": "Brief explanation",
    "python_code": "plotly code or null",
    "columns_used": ["col1"],
    "chart_type": "histogram|scatter|bar|line|box|heatmap"
}}

No explanations outside JSON."""

    def _create_ai_user_prompt(self, user_message: str) -> str:
        """Create efficient user prompt"""
        return f"""USER QUERY: "{user_message}"

CONTEXT: Dataset with {len(self.df)} rows, columns: {list(self.df.columns)[:5]}

Generate appropriate response. If visualization requested, provide working plotly code that creates 'fig' variable."""

    def _process_ai_response(self, ai_response: str, user_message: str) -> Dict[str, Any]:
        """Process AI response and execute visualization code"""
        print("🔄 Processing AI response...")

        try:
            # Parse JSON response
            parsed_response = self._parse_ai_json_response(ai_response)

            if not parsed_response:
                print("⚠️ Could not parse JSON, using fallback...")
                return self._create_smart_fallback_response(user_message)

            print(f"✅ Parsed AI response: {parsed_response.get('query_type')}")

            response_text = parsed_response.get("response_text", "Analysis completed")
            python_code = parsed_response.get("python_code")
            print(f"AI returned code:\n{python_code}")


            # Execute visualization code if provided
            chart = None
            if python_code and python_code.strip() and python_code != "null":
                print("🔄 Executing visualization code...")
                chart = self._execute_visualization_code(python_code)
                if chart:
                    print("✅ Visualization created successfully")
                else:
                    print("❌ Visualization creation failed")
                    response_text += "\n\n*Note: Chart generation encountered an issue, but analysis is complete.*"

            return self._create_response(
                response_text,
                chart,
                python_code if python_code and python_code.strip() and python_code != "null" else None
            )

        except Exception as e:
            print(f"❌ Response processing failed: {e}")
            return self._create_smart_fallback_response(user_message)

    def _parse_ai_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from AI response"""
        try:
            response = response.strip()

            # Extract JSON from markdown
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.rfind("```")
                response = response[start:end].strip()

            # Find JSON object
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ JSON parsing failed: {e}")

        return None
    def _execute_visualization_code(self, code: str) -> Optional[go.Figure]:
        """Safely execute Python visualization code with improved error handling"""
        try:
            print(f"🔄 Executing code:\n{code}")
            
            # Use default builtins without modification
            safe_globals = {
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'df': self.df,
                'make_subplots': make_subplots
                # Don't modify __builtins__ - let Python handle it
            }
            
            local_vars = {}
            exec(code, safe_globals, local_vars)
            
            # Look for figure in local variables
            for var_name, var_value in local_vars.items():
                if isinstance(var_value, (go.Figure, go.FigureWidget)):
                    print(f"✅ Found figure: {var_name}")
                    var_value.update_layout(height=450, showlegend=True)
                    return var_value
            
            # Check for 'fig' specifically
            if 'fig' in local_vars and isinstance(local_vars['fig'], (go.Figure, go.FigureWidget)):
                local_vars['fig'].update_layout(height=450, showlegend=True)
                return local_vars['fig']
            
            return None
            
        except Exception as e:
            print(f"❌ Code execution failed: {e}")
            logger.error(f"❌ Code execution failed: {e}")
            return None


    # def _execute_visualization_code(self, code: str) -> Optional[go.Figure]:
    #     """Safely execute Python visualization code"""
    #     try:
    #         # Safe execution environment  
    #         # safe_builtins = {'__import__': builtins.__import__}
    #         safe_globals = {
    #             'pd': pd,
    #             'np': np,
    #             'px': px,
    #             'go': go,
    #             'df': self.df,
    #             'make_subplots': make_subplots,
    #             '__builtins__': None
    #         }

    #         # Execute code
    #         local_vars = {}
    #         exec(code, safe_globals, local_vars)

    #         # Find plotly figure
    #         for var_name, var_value in local_vars.items():
    #             if isinstance(var_value, (go.Figure, go.FigureWidget)):
    #                 var_value.update_layout(height=450, showlegend=True)
    #                 return var_value

    #         # Check globals for 'fig'
    #         if 'fig' in safe_globals and isinstance(safe_globals['fig'], (go.Figure, go.FigureWidget)):
    #             safe_globals['fig'].update_layout(height=450, showlegend=True)
    #             return safe_globals['fig']
            
    #     except Exception as e:
    #         print(f"❌ Code execution failed: {e}")
    #         logger.error(f"❌ Code execution failed: {e}")

    #     return None

    # Replace your _create_smart_fallback_response method with this fixed version:

    def _create_smart_fallback_response(self, user_message: str) -> Dict[str, Any]:
        """DIAGNOSTIC VERSION - Let's see exactly what's happening"""
        
        try:
            print(f"🔍 FALLBACK CALLED with: '{user_message}'")
            
            message_lower = user_message.lower()
            print(f"🔍 Message lower: '{message_lower}'")
            print(f"🔍 Available columns: {list(self.df.columns)}")
            
            # Test each step explicitly
            plot_keywords = ['plot', 'chart', 'graph', 'histogram', 'distribution']
            found_keywords = [word for word in plot_keywords if word in message_lower]
            print(f"🔍 Found keywords: {found_keywords}")
            
            has_plot_request = len(found_keywords) > 0
            print(f"🔍 Has plot request: {has_plot_request}")
            
            if has_plot_request:
                print("✅ PLOT REQUEST DETECTED - Looking for columns...")
                
                # Column detection
                column_name = None
                for col in self.df.columns:
                    print(f"🔍 Checking column '{col}' in message...")
                    if col.lower() in message_lower:
                        column_name = col
                        print(f"✅ FOUND COLUMN: {col}")
                        break
                
                if column_name:
                    print(f"✅ CREATING CHART FOR: {column_name}")
                    
                    # Check column type
                    is_numeric = pd.api.types.is_numeric_dtype(self.df[column_name])
                    print(f"🔍 Column {column_name} is numeric: {is_numeric}")
                    
                    try:
                        if is_numeric:
                            print("📊 Creating histogram...")
                            fig = px.histogram(self.df, x=column_name, title=f"Distribution of {column_name}")
                            print("✅ Histogram created successfully!")
                            
                            return self._create_response(
                                f"Here's the distribution of {column_name}.",
                                fig,
                                f"Chart created for {column_name}"
                            )
                        else:
                            print("📊 Creating bar chart...")
                            value_counts = self.df[column_name].value_counts()
                            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                        title=f"Distribution of {column_name}")
                            print("✅ Bar chart created successfully!")
                            
                            return self._create_response(
                                f"Here's the distribution of {column_name}.",
                                fig,
                                f"Chart created for {column_name}"
                            )
                            
                    except Exception as chart_error:
                        print(f"❌ CHART CREATION FAILED: {chart_error}")
                        import traceback
                        print(traceback.format_exc())
                        return self._create_response(f"Chart creation failed: {str(chart_error)}")
                
                else:
                    print("❌ NO COLUMN FOUND")
                    return self._create_response(
                        f"I'd be happy to create a visualization! Available columns: {', '.join(self.df.columns)}"
                    )
            
            # Test other patterns
            elif any(word in message_lower for word in ['summary', 'describe', 'overview', 'stats']):
                print("✅ SUMMARY REQUEST DETECTED")
                summary_stats = self._generate_summary_stats()
                return self._create_response(
                    f"Dataset Summary:\n- Shape: {summary_stats['shape']}\n- Missing values: {summary_stats['missing_total']}\n- Numeric columns: {summary_stats['numeric_count']}\n- Categorical columns: {summary_stats['categorical_count']}"
                )
            
            else:
                print("❌ NO PATTERNS MATCHED - USING DEFAULT")
                return self._create_response(
                    f"I can help you analyze this dataset with {len(self.df):,} rows and {len(self.df.columns)} columns. "
                    f"Available columns: {', '.join(self.df.columns)}"
                )
        
        except Exception as e:
            print(f"❌ FALLBACK METHOD CRASHED: {e}")
            import traceback
            print(traceback.format_exc())
            return self._create_response(f"Fallback method error: {str(e)}")


    # ALSO ADD: Method to force console output in Streamlit
    def debug_fallback_execution(self, user_message: str):
        """Force execution of fallback and show results"""
        import sys
        
        print("="*60)
        print("FORCING FALLBACK EXECUTION FOR DEBUG")
        print("="*60)
        
        # Force flush
        sys.stdout.flush()
        
        try:
            response = self._create_smart_fallback_response(user_message)
            print(f"FALLBACK RESPONSE: {response}")
            sys.stdout.flush()
            return response
        except Exception as e:
            print(f"DEBUG EXECUTION FAILED: {e}")
            sys.stdout.flush()
            raise

    # Also add this debugging method to check AI status:
    def get_ai_status(self) -> Dict[str, Any]:
        """Get AI initialization status for debugging"""
        return {
            "gemini_available": GEMINI_AVAILABLE,
            "api_key_exists": CHAT_WITH_DATA_CONFIG.get("api_key") is not None,
            "api_key_length": len(CHAT_WITH_DATA_CONFIG.get("api_key", "")) if CHAT_WITH_DATA_CONFIG.get("api_key") else 0,
            "llm_initialized": self.llm is not None,
            "active_ai": "Gemini" if self.llm else "None (using fallback)"
        }

    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            "shape": f"{len(self.df)} rows × {len(self.df.columns)} columns",
            "missing_total": self.df.isnull().sum().sum(),
            "numeric_count": len(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_count": len(self.df.select_dtypes(include=['object', 'category']).columns),
        }

    def _create_response(self, text: str, chart: Optional[go.Figure] = None, code: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized response format for AutoML app integration"""
        return {
            "text": text,
            "chart": chart,
            "code": code,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def reset_conversation(self):
        """Reset chat history"""
        self.chat_history = []

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get chat history"""
        return self.chat_history.copy()

# Factory functions for compatibility with existing AutoML app
def create_chat_agent(df: pd.DataFrame) -> TrulyDynamicChatAgent:
    """Create chat agent - compatible with existing app.py"""
    print("🔄 create_chat_agent() called")
    return TrulyDynamicChatAgent(df)

def quick_chat_with_data(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """Quick chat function - compatible with existing app.py"""
    print(f"🔄 quick_chat_with_data() called with: {question}")
    agent = TrulyDynamicChatAgent(df, use_cache=False)
    return agent.chat(question)

# Aliases for compatibility with existing AutoML system
ChatWithDataAgent = TrulyDynamicChatAgent
EnhancedChatAgent = TrulyDynamicChatAgent

print("✅ FIXED CHAT AGENT MODULE LOADED - READY FOR INTEGRATION")
