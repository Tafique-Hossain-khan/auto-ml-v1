"""
Data Analyzer Component for AutoML System
Enhanced with Graph Generation and Chat Capabilities
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import warnings
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# LangChain imports for Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import ChatPromptTemplate
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    warnings.warn("langchain-google-genai not available for enhanced analysis")

from config.settings import CHAT_WITH_DATA_CONFIG, GEMINI_SAFETY_SETTINGS

warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Analyzes datasets and extracts metadata for AutoML with enhanced visualization"""

    def __init__(self):
        self.metadata = {}
        self.df = None
        self.target_column = None
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize Gemini LLM for enhanced analysis"""
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
            except Exception as e:
                warnings.warn(f"Failed to initialize Gemini for analysis: {e}")
                self.llm = None
        else:
            self.llm = None

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

    def generate_visualizations(self, chart_type: str = "auto", columns: List[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive visualizations for the dataset
        
        Args:
            chart_type: Type of chart to generate ("auto", "distribution", "correlation", "missing", "outliers")
            columns: Specific columns to visualize (if None, uses all relevant columns)
            
        Returns:
            Dictionary containing plotly figures and metadata
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Call analyze_dataset() first.")
        
        if columns is None:
            columns = self.df.columns.tolist()
        
        visualizations = {}
        
        if chart_type == "auto" or chart_type == "distribution":
            visualizations["distributions"] = self._generate_distribution_plots(columns)
            
        if chart_type == "auto" or chart_type == "correlation":
            visualizations["correlations"] = self._generate_correlation_plots()
            
        if chart_type == "auto" or chart_type == "missing":
            visualizations["missing_data"] = self._generate_missing_data_plot()
            
        if chart_type == "auto" or chart_type == "outliers":
            visualizations["outliers"] = self._generate_outlier_plots(columns)
            
        if chart_type == "auto":
            visualizations["summary"] = self._generate_summary_dashboard()
            
        return visualizations

    def _generate_distribution_plots(self, columns: List[str]) -> Dict[str, Any]:
        """Generate distribution plots for numerical and categorical columns"""
        plots = {}
        
        # Numerical columns
        numeric_cols = [col for col in columns if col in self.df.columns and 
                       pd.api.types.is_numeric_dtype(self.df[col])]
        
        for col in numeric_cols[:5]:  # Limit to 5 for performance
            try:
                # Histogram
                fig_hist = px.histogram(
                    self.df, x=col, nbins=30,
                    title=f"Distribution of {col}",
                    labels={col: col, 'count': 'Frequency'}
                )
                fig_hist.update_layout(showlegend=False)
                
                # Box plot
                fig_box = px.box(
                    self.df, y=col,
                    title=f"Box Plot of {col}",
                    labels={col: col}
                )
                
                plots[f"{col}_histogram"] = fig_hist
                plots[f"{col}_boxplot"] = fig_box
                
            except Exception as e:
                warnings.warn(f"Failed to create distribution plot for {col}: {e}")
        
        # Categorical columns
        categorical_cols = [col for col in columns if col in self.df.columns and 
                           not pd.api.types.is_numeric_dtype(self.df[col])]
        
        for col in categorical_cols[:5]:  # Limit to 5 for performance
            try:
                # Bar chart
                value_counts = self.df[col].value_counts().head(10)
                fig_bar = px.bar(
                    x=value_counts.index, y=value_counts.values,
                    title=f"Top 10 Values in {col}",
                    labels={'x': col, 'y': 'Count'}
                )
                fig_bar.update_layout(showlegend=False)
                
                # Pie chart (if not too many categories)
                if len(value_counts) <= 8:
                    fig_pie = px.pie(
                        values=value_counts.values, names=value_counts.index,
                        title=f"Distribution of {col}"
                    )
                    plots[f"{col}_pie"] = fig_pie
                
                plots[f"{col}_bar"] = fig_bar
                
            except Exception as e:
                warnings.warn(f"Failed to create categorical plot for {col}: {e}")
        
        return plots

    def _generate_correlation_plots(self) -> Dict[str, Any]:
        """Generate correlation plots for numerical columns"""
        plots = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return plots
        
        try:
            # Correlation heatmap
            corr_matrix = self.df[numeric_cols].corr()
            
            fig_heatmap = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            fig_heatmap.update_layout(
                xaxis_title="Features",
                yaxis_title="Features"
            )
            
            plots["correlation_heatmap"] = fig_heatmap
            
            # Scatter plot matrix (for smaller datasets)
            if len(numeric_cols) <= 6:
                fig_scatter_matrix = px.scatter_matrix(
                    self.df[numeric_cols],
                    title="Scatter Plot Matrix"
                )
                plots["scatter_matrix"] = fig_scatter_matrix
            
        except Exception as e:
            warnings.warn(f"Failed to create correlation plots: {e}")
        
        return plots

    def _generate_missing_data_plot(self) -> Dict[str, Any]:
        """Generate missing data visualization"""
        plots = {}
        
        try:
            # Missing values by column
            missing_counts = self.df.isnull().sum()
            missing_percentages = (missing_counts / len(self.df)) * 100
            
            fig_missing = px.bar(
                x=missing_counts.index,
                y=missing_percentages,
                title="Missing Data Percentage by Column",
                labels={'x': 'Columns', 'y': 'Missing Percentage (%)'}
            )
            fig_missing.update_layout(showlegend=False)
            
            plots["missing_data"] = fig_missing
            
            # Missing data heatmap
            if missing_counts.sum() > 0:
                missing_data = self.df.isnull()
                fig_missing_heatmap = px.imshow(
                    missing_data.T,
                    title="Missing Data Pattern",
                    color_continuous_scale='Blues'
                )
                plots["missing_pattern"] = fig_missing_heatmap
            
        except Exception as e:
            warnings.warn(f"Failed to create missing data plots: {e}")
        
        return plots

    def _generate_outlier_plots(self, columns: List[str]) -> Dict[str, Any]:
        """Generate outlier detection plots"""
        plots = {}
        
        numeric_cols = [col for col in columns if col in self.df.columns and 
                       pd.api.types.is_numeric_dtype(self.df[col])]
        
        for col in numeric_cols[:3]:  # Limit to 3 for performance
            try:
                # Box plot with outliers
                fig_box = px.box(
                    self.df, y=col,
                    title=f"Outlier Detection for {col}",
                    labels={col: col}
                )
                
                plots[f"{col}_outliers"] = fig_box
                
            except Exception as e:
                warnings.warn(f"Failed to create outlier plot for {col}: {e}")
        
        return plots

    def _generate_summary_dashboard(self) -> Dict[str, Any]:
        """Generate a comprehensive summary dashboard"""
        plots = {}
        
        try:
            # Create subplots for summary statistics
            fig_summary = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Dataset Overview', 'Column Types', 'Data Quality', 'Memory Usage'),
                specs=[[{"type": "indicator"}, {"type": "pie"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # Dataset overview
            fig_summary.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=len(self.df),
                    title={"text": "Total Rows"},
                    delta={"reference": 0}
                ),
                row=1, col=1
            )
            
            # Column types pie chart
            numeric_count = len(self.df.select_dtypes(include=[np.number]).columns)
            categorical_count = len(self.df.select_dtypes(include=['object', 'category']).columns)
            
            fig_summary.add_trace(
                go.Pie(
                    labels=['Numeric', 'Categorical'],
                    values=[numeric_count, categorical_count],
                    name="Column Types"
                ),
                row=1, col=2
            )
            
            # Data quality bar chart
            missing_percentages = (self.df.isnull().sum() / len(self.df)) * 100
            fig_summary.add_trace(
                go.Bar(
                    x=list(missing_percentages.index),
                    y=list(missing_percentages.values),
                    name="Missing %"
                ),
                row=2, col=1
            )
            
            # Memory usage
            memory_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
            fig_summary.add_trace(
                go.Indicator(
                    mode="number",
                    value=memory_mb,
                    title={"text": "Memory (MB)"}
                ),
                row=2, col=2
            )
            
            fig_summary.update_layout(height=600, title_text="Dataset Summary Dashboard")
            plots["summary_dashboard"] = fig_summary
            
        except Exception as e:
            warnings.warn(f"Failed to create summary dashboard: {e}")
        
        return plots

    def chat_with_data(self, user_query: str) -> Dict[str, Any]:
        """
        Chat with the dataset using Gemini AI
        
        Args:
            user_query: User's question about the data
            
        Returns:
            Dictionary containing response and suggested visualizations
        """
        if self.llm is None:
            return self._generate_fallback_chat_response(user_query)
        
        try:
            # Create system message
            system_message = self._create_chat_system_message()
            
            # Create user prompt
            user_prompt = self._create_chat_prompt(user_query)
            
            # Get response from Gemini
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            
            # Parse response
            parsed_response = self._parse_chat_response(response.content, user_query)
            
            return parsed_response
            
        except Exception as e:
            warnings.warn(f"Chat failed: {e}")
            return self._generate_fallback_chat_response(user_query)

    def _create_chat_system_message(self) -> str:
        """Create system message for chat agent"""
        return f"""You are an expert data analyst AI assistant. You help users understand and explore their dataset through natural conversation.

DATASET CONTEXT:
- Shape: {self.df.shape}
- Columns: {', '.join(self.df.columns.tolist())}
- Data Types: {dict(self.df.dtypes)}

INSTRUCTIONS:
1. Answer questions about the data accurately using the provided context
2. Provide insights, patterns, and recommendations
3. Suggest relevant visualizations when appropriate
4. Keep responses helpful, clear, and concise
5. Use efficient token usage for free tier API limits

RESPONSE FORMAT:
Always respond with a JSON object containing:
{{
  "text": "Your main response text",
  "response_type": "text|insight|visualization|statistical_summary",
  "chart_suggestion": {{"type": "chart_type", "x": "column", "y": "column", "title": "title"}},
  "confidence": "high|medium|low"
}}

Remember: You are analyzing a real dataset with {len(self.df)} rows and {len(self.df.columns)} columns."""

    def _create_chat_prompt(self, user_query: str) -> str:
        """Create prompt for chat response"""
        
        # Include basic dataset info
        basic_stats = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Limit to 3 for efficiency
            basic_stats[col] = {
                "mean": float(self.df[col].mean()),
                "std": float(self.df[col].std()),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max())
            }
        
        prompt = f"""Current User Question: {user_query}

DATASET DETAILS:
- Rows: {len(self.df):,}
- Columns: {len(self.df.columns)}
- Missing Values: {self.df.isnull().sum().sum()} total

SAMPLE DATA (first 3 rows):
{self.df.head(3).to_dict()}

BASIC STATISTICS:
{json.dumps(basic_stats, indent=2)}

Please analyze this question and provide a helpful response about the dataset. Be specific and actionable."""

        return prompt

    def _parse_chat_response(self, response: str, user_query: str) -> Dict[str, Any]:
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
                        return self._enhance_chat_response(parsed, user_query)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: treat entire response as text
            return {
                "text": response,
                "response_type": "text",
                "chart_suggestion": None,
                "confidence": "medium"
            }
            
        except Exception as e:
            warnings.warn(f"Failed to parse chat response: {e}")
            return self._generate_simple_chat_response(user_query)

    def _enhance_chat_response(self, parsed_response: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Enhance response with additional analysis"""
        
        # Add actual data calculations if needed
        if "statistics" in user_query.lower() or "summary" in user_query.lower():
            parsed_response["actual_stats"] = self._generate_summary_stats()
        
        # Add visualization if suggested
        if parsed_response.get("chart_suggestion") and parsed_response["chart_suggestion"]:
            chart_data = self._create_suggested_visualization(parsed_response["chart_suggestion"])
            if chart_data:
                parsed_response["chart_data"] = chart_data
        
        return parsed_response

    def _generate_fallback_chat_response(self, user_query: str) -> Dict[str, Any]:
        """Generate fallback response when AI is unavailable"""
        
        message_lower = user_query.lower()
        
        # Pattern matching for common queries
        if any(word in message_lower for word in ['shape', 'size', 'rows', 'columns']):
            return {
                "text": f"This dataset has {len(self.df):,} rows and {len(self.df.columns)} columns. The columns are: {', '.join(self.df.columns.tolist())}",
                "response_type": "statistical_summary",
                "chart_suggestion": None,
                "confidence": "high"
            }
        
        elif any(word in message_lower for word in ['missing', 'null', 'nan']):
            total_missing = self.df.isnull().sum().sum()
            return {
                "text": f"There are {total_missing:,} missing values in total across all columns. Here's the breakdown by column: {self.df.isnull().sum().to_dict()}",
                "response_type": "statistical_summary",
                "chart_suggestion": None,
                "confidence": "high"
            }
        
        elif any(word in message_lower for word in ['summary', 'describe', 'overview']):
            return {
                "text": self._generate_dataset_summary(),
                "response_type": "statistical_summary",
                "chart_suggestion": None,
                "confidence": "high"
            }
        
        else:
            return {
                "text": f"I can help you analyze this dataset with {len(self.df):,} rows and {len(self.df.columns)} columns. Try asking about data summary, missing values, column distributions, or correlations.",
                "response_type": "text",
                "chart_suggestion": None,
                "confidence": "medium"
            }

    def _generate_simple_chat_response(self, user_query: str) -> Dict[str, Any]:
        """Generate simple response for error cases"""
        return {
            "text": "I'm here to help you explore your data. Try asking about dataset summary, missing values, or specific columns.",
            "response_type": "text",
            "chart_suggestion": None,
            "confidence": "low"
        }

    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        return {
            "shape": self.df.shape,
            "missing_values_total": self.df.isnull().sum().sum(),
            "numeric_columns_count": len(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns_count": len(self.df.select_dtypes(include=['object', 'category']).columns),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }

    def _generate_dataset_summary(self) -> str:
        """Generate a comprehensive dataset summary"""
        summary = f"""Dataset Overview:

📊 **Basic Info:**
- Rows: {len(self.df):,}
- Columns: {len(self.df.columns)}
- Memory: {round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2)} MB

📈 **Column Types:**
- Numeric: {len(self.df.select_dtypes(include=[np.number]).columns)}
- Categorical: {len(self.df.select_dtypes(include=['object', 'category']).columns)}

❓ **Data Quality:**
- Missing values: {self.df.isnull().sum().sum():,} total
- Complete rows: {self.df.dropna().shape[0]:,}

🔍 **Key Columns:** {', '.join(self.df.columns[:5].tolist())}{'...' if len(self.df.columns) > 5 else ''}
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
            warnings.warn(f"Failed to create visualization: {e}")
        
        return None

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
