"""
Streamlit Frontend for AutoML System v2 - Enhanced with 3 Gemini Features
Features: Model Training + Data Cleaning + Chat with Data
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import zipfile
from pathlib import Path
import time
import uuid

# Import AutoML components
from main import AutoMLSystemV2, run_automl_pipeline_v2, quick_chat_with_data
from components.utils import validate_dataframe, ModelPredictor
from components.model_advisor import get_gemini_advisor_info
from components.data_cleaner_backup import create_data_cleaning_agent
from components.chat_agent_backup import create_chat_agent
from config.settings import (
    STREAMLIT_PAGE_TITLE, STREAMLIT_PAGE_ICON, STREAMLIT_LAYOUT,
    MAX_FILE_SIZE_MB, DEFAULT_TIME_LIMIT_MINUTES,
    MODEL_TRAINING_CONFIG, DATA_CLEANING_CONFIG, CHAT_WITH_DATA_CONFIG
)

# Page configuration
st.set_page_config(
    page_title="AutoML System v2 - Enhanced",
    page_icon="🚀",
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'automl_results' not in st.session_state:
        st.session_state.automl_results = None
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'chart_counter' not in st.session_state:
        st.session_state.chart_counter = 0
    if 'cleaning_suggestions' not in st.session_state:
        st.session_state.cleaning_suggestions = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None

def get_unique_chart_key(base_name):
    """Generate a unique key for charts"""
    st.session_state.chart_counter += 1
    return f"{base_name}_{st.session_state.chart_counter}"

def main():
    """Main Streamlit application"""
    initialize_session_state()

    # App header with v2 branding
    st.title("🚀 AutoML System v2")
    st.markdown("**Enhanced with 3 Gemini Features • Model Training + Data Cleaning + Chat with Data**")

    # Display enhanced Gemini status
    display_enhanced_gemini_status()

    st.markdown("---")

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration v2")
    display_enhanced_sidebar_info()

    # Main application tabs - now with 6 tabs for new features
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Upload", 
        "🧹 Data Cleaning", 
        "🔍 Analysis", 
        "🏋️ Training", 
        "💬 Chat with Data",
        "📈 Results"
    ])

    with tab1:
        data_upload_section()

    with tab2:
        data_cleaning_section()

    with tab3:
        data_analysis_section()

    with tab4:
        training_section()

    with tab5:
        chat_with_data_section()

    with tab6:
        results_section()

def display_enhanced_gemini_status():
    """Display enhanced Gemini integration status for all 3 features"""
    try:
        st.subheader("🧠 Gemini Integration Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎯 Model Training**")
            if MODEL_TRAINING_CONFIG["api_key"]:
                st.success("✅ API Key 1 Configured")
                st.info(f"Model: {MODEL_TRAINING_CONFIG['model']}")
            else:
                st.error("❌ GOOGLE_API_KEY1 Missing")

        with col2:
            st.markdown("**🧹 Data Cleaning**")
            if DATA_CLEANING_CONFIG["api_key"]:
                st.success("✅ API Key 2 Configured")
                st.info(f"Model: {DATA_CLEANING_CONFIG['model']}")
            else:
                st.error("❌ GOOGLE_API_KEY2 Missing")

        with col3:
            st.markdown("**💬 Chat with Data**")
            if CHAT_WITH_DATA_CONFIG["api_key"]:
                st.success("✅ API Key 3 Configured")  
                st.info(f"Model: {CHAT_WITH_DATA_CONFIG['model']}")
            else:
                st.error("❌ GOOGLE_API_KEY3 Missing")

        # Show setup instructions if needed
        missing_keys = []
        if not MODEL_TRAINING_CONFIG["api_key"]:
            missing_keys.append("GOOGLE_API_KEY1")
        if not DATA_CLEANING_CONFIG["api_key"]:
            missing_keys.append("GOOGLE_API_KEY2")
        if not CHAT_WITH_DATA_CONFIG["api_key"]:
            missing_keys.append("GOOGLE_API_KEY3")

        if missing_keys:
            st.warning(f"💡 **Setup Required**: Set {', '.join(missing_keys)} in your environment")
            st.info("Get your API keys from: https://makersuite.google.com/app/apikey")

    except Exception as e:
        st.error(f"Could not get Gemini status: {e}")

def display_enhanced_sidebar_info():
    """Display enhanced sidebar information"""
    st.sidebar.markdown("### 🚀 AutoML v2 Features")
    st.sidebar.markdown("""
    **🎯 Model Training (Key 1)**
    - AI-powered model selection
    - Intelligent hyperparameter tuning
    - Automated pipeline creation

    **🧹 Data Cleaning (Key 2)**  
    - Quality analysis & scoring
    - Intelligent cleaning suggestions
    - Automated preprocessing

    **💬 Chat with Data (Key 3)**
    - Natural language queries
    - Interactive data exploration  
    - AI-powered insights
    """)

    st.sidebar.markdown("### 📚 Quick Guide")
    st.sidebar.markdown("""
    1. **Upload** your CSV dataset
    2. **Clean** data with AI suggestions
    3. **Analyze** with enhanced profiling
    4. **Train** models with Gemini
    5. **Chat** with your data
    6. **Download** trained models
    """)

    # API key status
    st.sidebar.markdown("### 🔑 API Keys Status")
    key1_status = "✅" if MODEL_TRAINING_CONFIG["api_key"] else "❌"
    key2_status = "✅" if DATA_CLEANING_CONFIG["api_key"] else "❌"
    key3_status = "✅" if CHAT_WITH_DATA_CONFIG["api_key"] else "❌"

    st.sidebar.markdown(f"""
    - Key 1 (Training): {key1_status}
    - Key 2 (Cleaning): {key2_status}
    - Key 3 (Chat): {key3_status}
    """)

def data_upload_section():
    """Enhanced data upload and preview section"""
    st.header("📊 Data Upload & Preview")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
    )

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df

            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            with col4:
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("Memory", f"{memory_mb:.1f} MB")

            # Data validation
            validation = validate_dataframe(df)
            if not validation["is_valid"]:
                st.error("❌ Data validation failed:")
                for error in validation["errors"]:
                    st.error(f"• {error}")
                return

            if validation["warnings"]:
                st.warning("⚠️ Data validation warnings:")
                for warning in validation["warnings"]:
                    st.warning(f"• {warning}")

            # Data preview with enhanced info
            st.subheader("Data Preview")
            preview_rows = st.slider("Rows to preview", 5, 100, 10)
            st.dataframe(df.head(preview_rows), use_container_width=True)

            # Enhanced column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Non-Null Count': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Null %': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique Values': [df[col].nunique() for col in df.columns],
                'Unique %': [(df[col].nunique() / len(df) * 100).round(2) for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)

            # Data quality overview
            st.subheader("📊 Data Quality Overview")
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            complete_percentage = ((total_cells - missing_cells) / total_cells) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Completeness", f"{complete_percentage:.1f}%")
            with col2:
                st.metric("Duplicates", f"{df.duplicated().sum():,}")
            with col3:
                st.metric("Duplicate %", f"{(df.duplicated().sum() / len(df) * 100):.1f}%")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    else:
        # Show sample data option
        if st.button("🧪 Load Sample Dataset"):
            df = create_sample_dataset()
            st.session_state.uploaded_data = df
            st.success("✅ Sample dataset loaded!")
            st.rerun()

def data_cleaning_section():
    """New: AI-powered data cleaning section"""
    st.header("🧹 AI-Powered Data Cleaning")

    if st.session_state.uploaded_data is None:
        st.info("📤 Please upload a dataset first in the Data Upload tab.")
        return

    df = st.session_state.uploaded_data

    st.info("🧠 **Gemini will analyze your data quality and suggest intelligent cleaning operations!**")

    # Data quality analysis
    st.subheader("📊 Data Quality Analysis")

    if st.button("🔍 Analyze Data Quality", type="primary"):
        with st.spinner("🧠 Gemini is analyzing your data quality..."):
            try:
                cleaning_agent = create_data_cleaning_agent()
                quality_analysis = cleaning_agent.analyze_data_quality(df)

                # Display quality analysis
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Quality Score", f"{quality_analysis['quality_score']:.1f}/100")
                with col2:
                    st.metric("Missing Values", f"{quality_analysis['missing_values']['total_missing_percentage']:.1f}%")
                with col3:
                    st.metric("Duplicates", f"{quality_analysis['duplicates']['count']:,}")

                # Get AI cleaning suggestions
                cleaning_suggestions = cleaning_agent.suggest_cleaning_operations(df, quality_analysis)
                st.session_state.cleaning_suggestions = cleaning_suggestions

                # Display suggestions
                st.subheader("🤖 AI Cleaning Suggestions")

                for i, suggestion in enumerate(cleaning_suggestions):
                    with st.expander(f"{suggestion['operation'].title()} - {suggestion['priority'].title()} Priority"):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**Description:** {suggestion['description']}")
                            st.write(f"**Rationale:** {suggestion['rationale']}")
                            st.write(f"**Parameters:** {suggestion['parameters']}")

                        with col2:
                            # Toggle to enable/disable operation
                            suggestion['apply'] = st.checkbox(
                                "Apply", 
                                value=suggestion.get('apply', False),
                                key=f"apply_{i}"
                            )

                st.success("✅ Data quality analysis completed!")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

    # Apply cleaning operations
    if st.session_state.cleaning_suggestions:
        st.subheader("🛠️ Apply Cleaning Operations")

        if st.button("🚀 Apply Selected Operations", type="primary"):
            with st.spinner("🧹 Applying cleaning operations..."):
                try:
                    cleaning_agent = create_data_cleaning_agent()

                    # Get selected operations
                    selected_ops = [op for op in st.session_state.cleaning_suggestions if op.get('apply', False)]

                    if selected_ops:
                        cleaned_df, cleaning_report = cleaning_agent.apply_cleaning_operations(df, selected_ops)
                        st.session_state.cleaned_data = cleaned_df

                        st.success(f"✅ Applied {len(selected_ops)} cleaning operations!")

                        # Show cleaning results
                        st.subheader("📈 Cleaning Results")

                        col1, col2, col3, col4 = st.columns(4)
                        changes = cleaning_report.get('changes_summary', {})

                        with col1:
                            st.metric("Rows Removed", changes.get('rows_removed', 0))
                        with col2:
                            st.metric("Missing Values Handled", changes.get('missing_values_handled', 0))
                        with col3:
                            st.metric("Duplicates Removed", changes.get('duplicates_removed', 0))
                        with col4:
                            st.metric("Memory Saved", f"{changes.get('memory_reduction_mb', 0):.1f} MB")

                        # Show before/after comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Before Cleaning:**")
                            st.write(f"Shape: {cleaning_report['before_stats']['shape']}")
                            st.write(f"Missing: {cleaning_report['before_stats']['missing_values']:,}")
                            st.write(f"Duplicates: {cleaning_report['before_stats']['duplicates']:,}")

                        with col2:
                            st.markdown("**After Cleaning:**")
                            st.write(f"Shape: {cleaning_report['after_stats']['shape']}")
                            st.write(f"Missing: {cleaning_report['after_stats']['missing_values']:,}")
                            st.write(f"Duplicates: {cleaning_report['after_stats']['duplicates']:,}")

                    else:
                        st.warning("No operations selected for application.")

                except Exception as e:
                    st.error(f"❌ Cleaning failed: {str(e)}")

    # Show cleaned data preview
    if st.session_state.cleaned_data is not None:
        st.subheader("✨ Cleaned Data Preview")
        st.dataframe(st.session_state.cleaned_data.head(), use_container_width=True)

def data_analysis_section():
    """Enhanced data analysis section"""
    st.header("🔍 Enhanced Data Analysis")

    if st.session_state.uploaded_data is None:
        st.info("📤 Please upload a dataset first in the Data Upload tab.")
        return

    # Use cleaned data if available
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.uploaded_data

    if st.session_state.cleaned_data is not None:
        st.success("✅ Using cleaned data for analysis")

    # Target column selection
    st.subheader("🎯 Target Column Selection")
    target_column = st.selectbox(
        "Select target column (or leave empty for auto-detection):",
        ["Auto-detect"] + list(df.columns),
        help="The column you want to predict"
    )

    if target_column == "Auto-detect":
        target_column = None

    # Run analysis button
    if st.button("🔍 Analyze Dataset", type="primary"):
        with st.spinner("Analyzing dataset..."):
            try:
                from components.data_analyzer import analyze_uploaded_data
                analysis_results = analyze_uploaded_data(df, target_column)
                st.session_state.analysis_results = analysis_results

                # Display results
                display_analysis_results(analysis_results)

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

    # Display cached results if available
    if hasattr(st.session_state, 'analysis_results'):
        display_analysis_results(st.session_state.analysis_results)

def display_analysis_results(analysis_results):
    """Display enhanced data analysis results"""
    metadata = analysis_results["metadata"]
    recommendations = analysis_results["preprocessing_recommendations"]

    # Basic dataset info
    st.subheader("📋 Dataset Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Task Type", metadata["task_type"].title())
        st.metric("Target Column", metadata["target_column"])

    with col2:
        st.metric("Data Quality Score", f"{metadata.get('data_quality_score', 0):.1f}/100")
        st.metric("Duplicate Rows", f"{metadata.get('duplicate_rows', 0):,}")

    with col3:
        st.metric("Numeric Features", metadata["n_numeric"])
        st.metric("Categorical Features", metadata["n_categorical"])

    # Feature distribution
    if metadata["n_numeric"] > 0 or metadata["n_categorical"] > 0:
        st.subheader("📊 Feature Distribution")

        # Create pie chart for feature types
        labels = []
        values = []
        if metadata["n_numeric"] > 0:
            labels.append("Numeric")
            values.append(metadata["n_numeric"])
        if metadata["n_categorical"] > 0:
            labels.append("Categorical")
            values.append(metadata["n_categorical"])
        if metadata.get("n_text", 0) > 0:
            labels.append("Text")
            values.append(metadata["n_text"])
        if metadata.get("n_datetime", 0) > 0:
            labels.append("DateTime")
            values.append(metadata["n_datetime"])

        fig = px.pie(values=values, names=labels, title="Feature Types Distribution")
        st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("feature_types_pie"))

    # Target analysis
    if metadata["task_type"] == "classification" and "class_distribution" in metadata:
        st.subheader("🎯 Target Distribution")
        class_dist = metadata["class_distribution"]

        fig = px.bar(
            x=list(class_dist.keys()), 
            y=list(class_dist.values()),
            title="Class Distribution",
            labels={'x': 'Classes', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("target_distribution_bar"))

        # Class balance info
        balance_info = metadata.get("class_balance", {})
        if balance_info:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Balance Status", balance_info.get("balance_status", "Unknown").replace("_", " ").title())
            with col2:
                st.metric("Imbalance Ratio", f"{balance_info.get('imbalance_ratio', 0):.2f}")

    # Missing values heatmap
    if metadata.get("missing_values_summary"):
        st.subheader("❓ Missing Values Analysis")
        missing_data = metadata["missing_values_summary"]

        if missing_data:
            missing_df = pd.DataFrame([
                {"Column": col, "Missing Count": info["count"], "Missing %": info["percentage"]}
                for col, info in missing_data.items()
            ])

            fig = px.bar(
                missing_df, 
                x="Column", 
                y="Missing %",
                title="Missing Values by Column",
                text="Missing Count"
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("missing_values_bar"))
        else:
            st.success("✅ No missing values detected!")

    # Enhanced preprocessing recommendations
    st.subheader("💡 AI-Enhanced Preprocessing Recommendations")

    for category, recs in recommendations.items():
        if recs:
            with st.expander(f"{category.title()} Recommendations"):
                for rec in recs:
                    st.write(f"• {rec}")

def training_section():
    """Enhanced model training section"""
    st.header("🏋️ Enhanced Model Training")

    if st.session_state.uploaded_data is None:
        st.info("📤 Please upload a dataset first.")
        return

    if not hasattr(st.session_state, 'analysis_results'):
        st.info("🔍 Please run data analysis first.")
        return

    # Use cleaned data if available
    df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.uploaded_data
    metadata = st.session_state.analysis_results["metadata"]

    # Display enhanced training info
    st.info("🧠 **Gemini will analyze your dataset and suggest the best models automatically!**")

    if st.session_state.cleaned_data is not None:
        st.success("✅ Using cleaned data for training")

    # Enhanced training configuration
    st.subheader("⚙️ Training Configuration")

    col1, col2, col3 = st.columns(3)
    with col1:
        time_limit = st.slider(
            "Training Time Limit (minutes)",
            min_value=1,
            max_value=15,
            value=DEFAULT_TIME_LIMIT_MINUTES,
            help="Maximum time to spend training models"
        )

    with col2:
        interpretability = st.selectbox(
            "Interpretability Preference",
            ["low", "medium", "high"],
            index=1,
            help="Higher interpretability may sacrifice some accuracy"
        )

    with col3:
        enable_cleaning = st.checkbox(
            "Enable Data Cleaning",
            value=True,
            help="Use AI data cleaning before training"
        )

    # User constraints
    user_constraints = {
        "time_limit_minutes": time_limit,
        "interpretability": interpretability
    }

    # Start enhanced training button
    if st.button("🚀 Start Enhanced Gemini AutoML v2", type="primary"):
        with st.spinner("🧠 Gemini v2 is analyzing your data and training models..."):
            try:
                # Create progress placeholder
                progress_placeholder = st.empty()

                # Run enhanced AutoML v2
                automl_system = AutoMLSystemV2()
                results = automl_system.run_full_automl_pipeline(
                    df=st.session_state.uploaded_data,  # Always start with original data
                    target_column=metadata["target_column"],
                    user_constraints=user_constraints,
                    enable_data_cleaning=enable_cleaning,
                    time_limit_minutes=time_limit
                )

                st.session_state.automl_results = results
                st.session_state.training_completed = True

                # Initialize chat agent with final data
                final_data = results.get('v2_features', {}).get('cleaned_shape') or df
                st.session_state.chat_agent = automl_system.chat_agent

                st.success("🎉 Enhanced Gemini AutoML v2 completed successfully!")

                # Display quick results
                best_model = results["training_results"]["best_model"]
                if best_model:
                    st.success(f"🏆 Best Model: {best_model['name']} (CV Score: {best_model['cv_score']:.4f})")

                # Display v2 enhancements
                v2_features = results.get("v2_features", {})
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"📊 Original Shape: {v2_features.get('original_shape')}")
                with col2:
                    st.info(f"✨ Cleaned Shape: {v2_features.get('cleaned_shape')}")

                # Display feature contributions
                cleaning_applied = results["summary"]["data_cleaning"]["operations_applied"]
                if cleaning_applied > 0:
                    st.success(f"🧹 Data Cleaning: {cleaning_applied} operations applied")

                if results["summary"]["chat_agent"]["enabled"]:
                    st.success("💬 Chat Agent: Ready for data conversations")

            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)

    # Display training status
    if st.session_state.training_completed and st.session_state.automl_results:
        display_enhanced_training_progress(st.session_state.automl_results)

def display_enhanced_training_progress(results):
    """Display enhanced training progress and results"""
    training_results = results["training_results"]

    # Training summary
    st.subheader("📊 Enhanced Training Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Models Trained", training_results["n_models_trained"])
    with col2:
        st.metric("Models Failed", training_results["n_models_failed"])
    with col3:
        st.metric("Training Time", f"{training_results['training_time_seconds']:.1f}s")
    with col4:
        st.metric("CV Folds", training_results["cv_folds"])

    # v2 enhancements summary
    st.subheader("🚀 v2 Enhancements Applied")
    col1, col2 = st.columns(2)

    with col1:
        cleaning_ops = results["summary"]["data_cleaning"]["operations_applied"]
        st.metric("Cleaning Operations", cleaning_ops)

    with col2:
        chat_status = "✅ Ready" if results["summary"]["chat_agent"]["enabled"] else "❌ Not Available"
        st.metric("Chat Agent", chat_status)

    # Model comparison
    if "model_comparison" in training_results:
        st.subheader("🔄 Model Comparison (Enhanced)")

        comparison_data = []
        for model in training_results["model_comparison"]:
            if model.get("status") != "failed":
                comparison_data.append({
                    "Model": model["model_name"],
                    "CV Score": model["cv_score"],
                    "CV Std": model["cv_std"]
                })

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values("CV Score", ascending=False)

            # Bar chart
            fig = px.bar(
                comparison_df, 
                x="Model", 
                y="CV Score",
                error_y="CV Std",
                title="Model Performance Comparison (Enhanced with v2)"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("model_comparison_bar"))

            # Table
            st.dataframe(comparison_df, use_container_width=True)

def chat_with_data_section():
    """New: Chat with data section"""
    st.header("💬 Chat with Your Data")

    if st.session_state.uploaded_data is None:
        st.info("📤 Please upload a dataset first to start chatting with your data.")
        return

    # Initialize chat agent if not already done
    if st.session_state.chat_agent is None:
        data_to_use = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.uploaded_data

        with st.spinner("🧠 Initializing chat agent..."):
            st.session_state.chat_agent = create_chat_agent(data_to_use)

        st.success("✅ Chat agent ready! Ask me anything about your data.")

    st.info("🧠 **Ask questions about your data in natural language. Gemini will analyze and respond!**")

    # Chat interface
    st.subheader("🗨️ Conversation")

    # Display chat history
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                with st.chat_message("user"):
                    st.write(chat["message"])
            else:
                with st.chat_message("assistant"):
                    st.write(chat["message"])

    # Chat input
    user_message = st.chat_input("Ask about your data...")

    if user_message:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "message": user_message})

        with st.chat_message("user"):
            st.write(user_message)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Gemini is analyzing your question..."):
                try:
                    response = st.session_state.chat_agent.chat(user_message)

                    # Display response
                    st.write(response["text"])

                    # Show chart if suggested
                    if response.get("chart_data"):
                        chart_data = response["chart_data"]
                        if chart_data["type"] == "histogram":
                            fig = px.histogram(x=chart_data["data"], title=chart_data["title"])
                            st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("chat_histogram"))
                        elif chart_data["type"] == "scatter":
                            fig = px.scatter(x=chart_data["x_data"], y=chart_data["y_data"], title=chart_data["title"])
                            st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("chat_scatter"))
                        elif chart_data["type"] == "bar":
                            fig = px.bar(x=chart_data["categories"], y=chart_data["values"], title=chart_data["title"])
                            st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("chat_bar"))

                    # Show code if suggested
                    if response.get("code_suggestion"):
                        with st.expander("💻 Code Suggestion"):
                            st.code(response["code_suggestion"], language="python")

                    # Add to history
                    st.session_state.chat_history.append({"role": "assistant", "message": response["text"]})

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "message": error_msg})

    # Quick question buttons
    st.subheader("🚀 Quick Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Dataset Summary"):
            user_message = "Give me a comprehensive summary of this dataset"
            st.session_state.chat_history.append({"role": "user", "message": user_message})
            st.rerun()

    with col2:
        if st.button("🔍 Missing Values"):
            user_message = "Tell me about missing values in the data"
            st.session_state.chat_history.append({"role": "user", "message": user_message})
            st.rerun()

    with col3:
        if st.button("📈 Correlations"):
            user_message = "What are the strongest correlations in the data?"
            st.session_state.chat_history.append({"role": "user", "message": user_message})
            st.rerun()

    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        if st.session_state.chat_agent:
            st.session_state.chat_agent.reset_conversation()
        st.rerun()

    # Data insights
    if st.button("🧠 Get AI Data Insights"):
        with st.spinner("🧠 Generating comprehensive insights..."):
            try:
                insights = st.session_state.chat_agent.get_data_insights()

                st.subheader("🔍 AI-Generated Data Insights")

                # Basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dataset Size", f"{insights['basic_info']['shape'][0]:,} × {insights['basic_info']['shape'][1]}")
                with col2:
                    st.metric("Memory Usage", f"{insights['basic_info']['memory_mb']:.1f} MB")
                with col3:
                    st.metric("Missing %", f"{insights['basic_info']['missing_percentage']:.1f}%")

                # Correlations
                if insights["correlations"]:
                    st.subheader("🔗 Strong Correlations Found")
                    for corr in insights["correlations"][:5]:
                        st.write(f"• **{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f}")

                # Recommendations
                if insights["recommendations"]:
                    st.subheader("💡 AI Recommendations")
                    for rec in insights["recommendations"]:
                        st.write(f"• {rec}")

            except Exception as e:
                st.error(f"❌ Failed to generate insights: {str(e)}")

def results_section():
    """Enhanced results and model artifacts section"""
    st.header("📈 Enhanced Results & Model Artifacts")

    if not st.session_state.training_completed or not st.session_state.automl_results:
        st.info("🏋️ Please complete model training first.")
        return

    results = st.session_state.automl_results
    best_model = results["training_results"]["best_model"]

    # Enhanced best model details
    st.subheader("🏆 Best Model (Enhanced by Gemini v2)")
    if best_model:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Name", best_model["name"])
        with col2:
            st.metric("CV Score", f"{best_model['cv_score']:.4f}")
        with col3:
            st.metric("CV Std", f"{best_model['cv_std']:.4f}")

        # v2 Enhancement info
        st.info("✨ This model was trained on AI-cleaned data with Gemini-powered feature selection")

        # Detailed metrics
        if "detailed_metrics" in best_model and best_model["detailed_metrics"]:
            st.subheader("📊 Detailed Performance Metrics")
            metrics_df = pd.DataFrame([
                {"Metric": metric, "Value": f"{value:.4f}"}
                for metric, value in best_model["detailed_metrics"].items()
            ])
            st.dataframe(metrics_df, use_container_width=True)

    # Enhanced feature importance
    st.subheader("🔍 Feature Importance Analysis")
    if st.session_state.automl_results.get("best_pipeline"):
        try:
            automl_system = AutoMLSystemV2()
            automl_system.best_pipeline = st.session_state.automl_results["best_pipeline"]
            importance_df = automl_system.get_feature_importance()

            if importance_df is not None:
                # Plot top 15 features
                top_features = importance_df.head(15)
                fig = px.bar(
                    top_features, 
                    x="importance", 
                    y="feature",
                    orientation='h',
                    title="Top 15 Feature Importances (Enhanced Pipeline)"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("feature_importance_bar"))

                # Show full table
                with st.expander("📋 Full Feature Importance Table"):
                    st.dataframe(importance_df, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
        except Exception as e:
            st.warning(f"Could not extract feature importance: {str(e)}")

    # v2 Results Summary
    st.subheader("🚀 v2 Enhancement Summary")
    summary = results["summary"]

    col1, col2, col3 = st.columns(3)
    with col1:
        cleaning_ops = summary["data_cleaning"]["operations_applied"]
        st.metric("Cleaning Operations Applied", cleaning_ops)

    with col2:
        chat_status = "✅ Active" if summary["chat_agent"]["enabled"] else "❌ Inactive"
        st.metric("Chat Agent Status", chat_status)

    with col3:
        version_info = summary.get("version", "v2")
        st.metric("AutoML Version", version_info)

    # Enhanced download artifacts
    st.subheader("💾 Download Enhanced Model Artifacts")

    artifacts = results.get("artifacts_saved")
    if artifacts:
        # Create download buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📦 Download Preprocessor"):
                download_file(artifacts["preprocessor_path"], "preprocessor.pkl")

        with col2:
            if st.button("🤖 Download Model"):
                download_file(artifacts["model_path"], "model.pkl")

        with col3:
            if st.button("📋 Download Metadata"):
                download_file(artifacts["metadata_path"], "metadata.json")

        # Download all as zip
        if st.button("📁 Download All v2 Artifacts (ZIP)", type="primary"):
            create_and_download_zip(artifacts)

    # Enhanced prediction interface
    st.subheader("🔮 Make Predictions (Enhanced)")

    if st.session_state.automl_results.get("best_pipeline"):
        prediction_interface()

def prediction_interface():
    """Enhanced interface for making predictions"""

    # File upload for predictions
    pred_file = st.file_uploader(
        "Upload CSV file for predictions",
        type=['csv'],
        key="prediction_file"
    )

    if pred_file is not None:
        try:
            pred_df = pd.read_csv(pred_file)
            st.write("📊 Prediction Data Preview:")
            st.dataframe(pred_df.head(), use_container_width=True)

            if st.button("🔮 Generate Enhanced Predictions"):
                with st.spinner("Generating predictions with enhanced model..."):
                    automl_system = AutoMLSystemV2()
                    automl_system.best_pipeline = st.session_state.automl_results["best_pipeline"]

                    predictions, probabilities = automl_system.predict(pred_df)

                    # Create results dataframe
                    results_df = pred_df.copy()
                    results_df['Prediction'] = predictions

                    if probabilities is not None:
                        # Add probability columns
                        prob_cols = [f'Probability_Class_{i}' for i in range(probabilities.shape[1])]
                        prob_df = pd.DataFrame(probabilities, columns=prob_cols)
                        results_df = pd.concat([results_df, prob_df], axis=1)

                    st.success("✅ Enhanced predictions generated!")
                    st.dataframe(results_df, use_container_width=True)

                    # Download predictions
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Enhanced Predictions",
                        data=csv,
                        file_name="enhanced_predictions_v2.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

def download_file(file_path, filename):
    """Create download for a file"""
    try:
        with open(file_path, "rb") as f:
            st.download_button(
                label=f"Download {filename}",
                data=f.read(),
                file_name=filename,
                mime="application/octet-stream"
            )
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")

def create_and_download_zip(artifacts):
    """Create and download ZIP file with all artifacts"""
    try:
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for artifact_type, file_path in artifacts.items():
                if Path(file_path).exists():
                    filename = Path(file_path).name
                    zip_file.write(file_path, filename)

        zip_buffer.seek(0)

        st.download_button(
            label="📁 Download All v2 Artifacts (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="automl_v2_artifacts.zip",
            mime="application/zip"
        )
    except Exception as e:
        st.error(f"❌ ZIP creation failed: {str(e)}")

def create_sample_dataset():
    """Create an enhanced sample dataset for demonstration"""
    from sklearn.datasets import make_classification

    # Generate sample classification data with quality issues
    X, y = make_classification(
        n_samples=1200,
        n_features=10,
        n_informative=7,
        n_redundant=3,
        n_classes=3,
        random_state=42
    )

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y

    # Add categorical features
    df['category_A'] = np.random.choice(['High', 'Medium', 'Low'], size=1200, p=[0.3, 0.4, 0.3])
    df['category_B'] = np.random.choice(['TypeX', 'TypeY', 'TypeZ'], size=1200)
    df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=1200)

    # Add data quality issues for demonstration
    # Missing values
    missing_indices = np.random.choice(df.index, size=150, replace=False)
    df.loc[missing_indices[:50], 'feature_0'] = np.nan
    df.loc[missing_indices[50:100], 'category_A'] = np.nan
    df.loc[missing_indices[100:], 'feature_3'] = np.nan

    # Duplicates
    duplicate_rows = df.sample(80)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df

if __name__ == "__main__":
    main()
