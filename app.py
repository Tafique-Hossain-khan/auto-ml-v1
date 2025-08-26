"""
Streamlit Frontend for AutoML System v2 - Clean & Modern UI
Enhanced with 3 Gemini Features: Model Training + Data Cleaning + Chat with Data
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

# Import AutoML components
from main import AutoMLSystemV2, run_automl_pipeline_v2, quick_chat_with_data
from components.utils import validate_dataframe, ModelPredictor
from components.model_advisor import get_gemini_advisor_info
from components.data_cleaner import create_data_cleaning_agent
from components.chat_agent import create_chat_agent
from components.data_analyzer import analyze_uploaded_data, DataAnalyzer
from config.settings import (
    MAX_FILE_SIZE_MB, DEFAULT_TIME_LIMIT_MINUTES,
    MODEL_TRAINING_CONFIG, DATA_CLEANING_CONFIG, CHAT_WITH_DATA_CONFIG
)

# Page configuration
st.set_page_config(
    page_title="AutoML System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .status-good {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .tab-content {
        padding: 2rem 1rem;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        # Original variables
        'automl_results': None,
        'uploaded_data': None,
        'cleaned_data': None,
        'training_completed': False,
        'chart_counter': 0,
        'cleaning_suggestions': None,
        'chat_history': [],
        'chat_agent': None,
        'data_analyzer': None,
        'current_visualizations': None,
        'current_step': 1,
        'analysis_results': None,

        # NEW: Data cleaning workflow variables (FIX for AttributeError)
        'cleaning_analyzed': False,
        'cleaning_quality_analysis': None,
        'cleaning_report': None,

        # NEW: AI visualization workflow variables (FIX for nested buttons)
        'ai_suggestion_response': None,
        'ai_chart_generated': False,
        'generate_viz_requested': False,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def get_unique_chart_key(base_name):
    """Generate unique key for charts"""
    st.session_state.chart_counter += 1
    return f"{base_name}_{st.session_state.chart_counter}"

def main():
    """Main application"""
    st.cache_data.clear()
    st.cache_resource.clear()
    initialize_session_state()

    # Header
    display_header()

    # Quick Status Bar
    display_status_bar()

    # Main Navigation
    display_main_navigation()

def display_header():
    """Clean header with branding"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AutoML System v2</h1>
        <p style="font-size: 1.2em; color: #666; margin-top: 0.5rem;">
            Powered by Google Gemini • Intelligent Machine Learning Made Simple
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_status_bar():
    """Simplified status indicator"""
    col1, col2, col3 = st.columns(3)

    with col1:
        status = "✅ Ready" if MODEL_TRAINING_CONFIG["api_key"] else "⚙️ Setup"
        st.markdown(f"""
        <div class="feature-card">
            <h4>🎯 Model Training</h4>
            <p>{status}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        status = "✅ Ready" if DATA_CLEANING_CONFIG["api_key"] else "⚙️ Setup"
        st.markdown(f"""
        <div class="feature-card">
            <h4>🧹 Data Cleaning</h4>
            <p>{status}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        status = "✅ Ready" if CHAT_WITH_DATA_CONFIG["api_key"] else "⚙️ Setup"
        st.markdown(f"""
        <div class="feature-card">
            <h4>💬 Chat with Data</h4>
            <p>{status}</p>
        </div>
        """, unsafe_allow_html=True)

    # Setup reminder if needed
    missing_keys = []
    if not MODEL_TRAINING_CONFIG["api_key"]:
        missing_keys.append("GOOGLE_API_KEY1")
    if not DATA_CLEANING_CONFIG["api_key"]:
        missing_keys.append("GOOGLE_API_KEY2") 
    if not CHAT_WITH_DATA_CONFIG["api_key"]:
        missing_keys.append("GOOGLE_API_KEY3")

    if missing_keys:
        st.warning(f"💡 **Quick Setup**: Add {', '.join(missing_keys)} to your environment. [Get API Keys →](https://makersuite.google.com/app/apikey)")

def display_main_navigation():
    """Simplified tab navigation"""

    # Progress indicator
    steps = ["📊 Upload", "🔍 Explore", "📈 Visualize", "🤖 AutoML", "💬 Chat", "📈 Results"]
    current = st.session_state.get('current_step', 1)

    progress_html = '<div style="display: flex; justify-content: space-between; margin: 2rem 0;">'
    for i, step in enumerate(steps, 1):
        if i <= current:
            progress_html += f'<div style="color: #28a745; font-weight: bold;">{step}</div>'
        else:
            progress_html += f'<div style="color: #6c757d;">{step}</div>'
    progress_html += '</div>'
    st.markdown(progress_html, unsafe_allow_html=True)

    # Main tabs - enhanced with visualization
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Data", "🔍 Explore", "📈 Visualize", "🤖 AutoML", "💬 Chat", "📈 Results"])

    with tab1:
        data_section()

    with tab2:
        explore_section()

    with tab3:
        visualize_section()

    with tab4:
        automl_section()

    with tab5:
        print("➡️Getting into the chat section")
        chat_section()
        

    with tab6:
        results_section()

def data_section():
    """Simplified data upload and preview"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    # Upload Section
    st.subheader("📊 Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help=f"Maximum size: {MAX_FILE_SIZE_MB}MB"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_data = df
            st.session_state.current_step = 2

            # Quick overview
            st.success("✅ Dataset uploaded successfully!")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            with col4:
                duplicate_pct = (df.duplicated().sum() / len(df) * 100)
                st.metric("Duplicates", f"{duplicate_pct:.1f}%")

            # Data preview
            with st.expander("📋 Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            # Data quality insights
            if missing_pct > 10 or duplicate_pct > 5:
                st.info("💡 Your dataset could benefit from AI-powered cleaning. Check the Explore tab!")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    else:
        # Sample data option
        if st.button("🧪 Try with Sample Dataset", type="secondary"):
            df = create_sample_dataset()
            st.session_state.uploaded_data = df
            st.session_state.current_step = 2
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def explore_section():
    """Simplified exploration with cleaning and analysis"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.info("📊 Please upload a dataset first.")
        return

    df = st.session_state.uploaded_data

    # Two column layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🧹 Smart Data Cleaning")
        if not st.session_state.cleaning_analyzed:
            if st.button("🚀 Analyze & Clean Data", type="primary"):
                with st.spinner("🧠 AI is analyzing your data..."):
                    try:
                        cleaning_agent = create_data_cleaning_agent()
                        quality_analysis = cleaning_agent.analyze_data_quality(df)
                        suggestions = cleaning_agent.suggest_cleaning_operations(df, quality_analysis)


                        # Store results in session state
                        st.session_state.cleaning_suggestions = suggestions
                        st.session_state.cleaning_quality_analysis = quality_analysis
                        st.session_state.cleaning_analyzed = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Cleaning failed: {str(e)}")

        # Show cleaning status
        if st.session_state.cleaning_analyzed and st.session_state.cleaning_quality_analysis:
            quality_analysis = st.session_state.cleaning_quality_analysis
            suggestions = st.session_state.cleaning_suggestions

            # Show quality score
            score = quality_analysis['quality_score']
            if score >= 80:
                st.success(f"✅ Data Quality: {score:.0f}/100 - Excellent!")
            elif score >= 60:
                st.warning(f"⚠️ Data Quality: {score:.0f}/100 - Good with improvements")
            else:
                st.error(f"❌ Data Quality: {score:.0f}/100 - Needs attention")

            # Show suggestions summary
            if suggestions:
                st.info(f"Found {len(suggestions)} cleaning suggestions")

                # Show detailed suggestions
                with st.expander("📋 View Cleaning Suggestions"):
                    for i, suggestion in enumerate(suggestions):
                        priority_emoji = "🔴" if suggestion.get('priority') == 'high' else "🟡"
                        st.write(f"{priority_emoji} **{suggestion['operation']}**: {suggestion['description']}")
                        st.write(f"   *Rationale: {suggestion['rationale']}*")

            # Apply high-priority operations
            high_priority = [s for s in suggestions if s.get('priority') == 'high'] if suggestions else []

            if high_priority and st.session_state.cleaned_data is None:
                if st.button("✨ Apply AI Recommendations", type="secondary"):
                    with st.spinner("🧹 Applying cleaning operations..."):
                        try:
                            cleaning_agent = create_data_cleaning_agent()
                            cleaned_df, report = cleaning_agent.apply_cleaning_operations(df, high_priority)
                            st.session_state.cleaned_data = cleaned_df
                            st.session_state.cleaning_report = report
                            st.success(f"🎉 Applied {len(high_priority)} cleaning operations!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Cleaning application failed: {str(e)}")

            # Reset button to start over
            if st.button("🔄 Reset Analysis", type="secondary"):
                st.session_state.cleaning_analyzed = False
                st.session_state.cleaning_quality_analysis = None
                st.session_state.cleaning_suggestions = None
                if 'cleaning_report' in st.session_state:
                    del st.session_state.cleaning_report
                st.rerun()

        # Show cleaning status
        if st.session_state.cleaned_data is not None:
            st.success("✅ Data has been cleaned!")
            original_shape = st.session_state.uploaded_data.shape
            cleaned_shape = st.session_state.cleaned_data.shape
            st.info(f"Shape: {original_shape} → {cleaned_shape}")

            # Show cleaning report if available
            if 'cleaning_report' in st.session_state and st.session_state.cleaning_report:
                report = st.session_state.cleaning_report
                changes = report.get('changes_summary', {})

                if changes:
                    with st.expander("📊 Cleaning Report"):
                        if changes.get('rows_removed', 0) > 0:
                            st.write(f"• Rows removed: {changes['rows_removed']}")
                        if changes.get('missing_values_handled', 0) > 0:
                            st.write(f"• Missing values handled: {changes['missing_values_handled']}")
                        if changes.get('duplicates_removed', 0) > 0:
                            st.write(f"• Duplicates removed: {changes['duplicates_removed']}")
                        if changes.get('memory_reduction_mb', 0) > 0:
                            st.write(f"• Memory reduced: {changes['memory_reduction_mb']:.1f} MB")

    with col2:
        st.subheader("🔍 Quick Analysis & Visualization")

        # Use cleaned data if available
        analysis_df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else df

        # Target column selection
        target_column = st.selectbox(
            "Select target column:",
            ["Auto-detect"] + list(analysis_df.columns)
        )

        if target_column == "Auto-detect":
            target_column = None

        # Analysis and visualization options
        analysis_type = st.selectbox(
            "Analysis type:",
            ["Basic Analysis", "Generate Visualizations", "Chat with Data"]
        )

        if st.button("🔍 Analyze Dataset"):
            with st.spinner("Analyzing..."):
                try:
                    # Initialize analyzer
                    analyzer = DataAnalyzer()
                    analysis_results = analyze_uploaded_data(analysis_df, target_column)
                    st.session_state.analysis_results = analysis_results
                    st.session_state.data_analyzer = analyzer
                    st.session_state.current_step = 3

                    metadata = analysis_results["metadata"]
                    st.success("✅ Analysis complete!")

                    # Key insights
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Task Type", metadata["task_type"].title())
                        st.metric("Target", metadata["target_column"])
                    with col_b:
                        st.metric("Features", f"{metadata['n_numeric']} num, {metadata['n_categorical']} cat")
                        st.metric("Quality Score", f"{metadata.get('data_quality_score', 0):.0f}/100")

                    # Generate visualizations if requested
                    if analysis_type == "Generate Visualizations":
                        st.subheader("📊 Generated Visualizations")
                        
                        # Chart type selection
                        chart_type = st.selectbox(
                            "Select visualization type:",
                            ["auto", "distribution", "correlation", "missing", "outliers"]
                        )
                        
                        if st.button("📈 Generate Charts"):
                            with st.spinner("Generating visualizations..."):
                                try:
                                    visualizations = analyzer.generate_visualizations(chart_type)
                                    
                                    # Display charts
                                    for chart_name, fig in visualizations.items():
                                        if isinstance(fig, dict) and "charts" in visualizations:
                                            # Handle multiple charts
                                            for sub_chart_name, sub_fig in fig.items():
                                                st.plotly_chart(sub_fig, use_container_width=True, key=f"{chart_name}_{sub_chart_name}")
                                        else:
                                            st.plotly_chart(fig, use_container_width=True, key=chart_name)
                                    
                                    st.success("✅ Visualizations generated!")
                                    
                                except Exception as e:
                                    st.error(f"Visualization failed: {str(e)}")

                    # Chat with data if requested
                    elif analysis_type == "Chat with Data":
                        st.subheader("💬 Chat with Your Data")
                        
                        # Chat input
                        user_question = st.text_input("Ask anything about your data:")
                        
                        if st.button("🤖 Ask AI") and user_question:
                            with st.spinner("AI is analyzing..."):
                                try:
                                    chat_response = analyzer.chat_with_data(user_question)
                                    
                                    # Display response
                                    st.markdown(f"**AI Response:** {chat_response['text']}")
                                    
                                    # Show confidence
                                    confidence = chat_response.get('confidence', 'medium')
                                    if confidence == 'high':
                                        st.success("✅ High confidence response")
                                    elif confidence == 'medium':
                                        st.info("ℹ️ Medium confidence response")
                                    else:
                                        st.warning("⚠️ Low confidence response")
                                    
                                    # Show suggested visualization if available
                                    if chat_response.get('chart_suggestion'):
                                        st.info("📊 Suggested visualization available")
                                        
                                except Exception as e:
                                    st.error(f"Chat failed: {str(e)}")

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
def chat_section():
    """Chat with Data section - MISSING INTEGRATION ADDED 🟥🟥🟥
    """
    try:
        # st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.write("## Chat Section Loaded")
        print("➡️inside the chat function")
    except Exception as e:
        # this will force any error to your console so you can see it
        import traceback
        print("❌ Exception in chat_section():", e)
        print(traceback.format_exc())
        # re‐raise so Streamlit shows an error in the UI
        raise

    if st.session_state.uploaded_data is None:
        st.info("📊 Please upload a dataset first to start chatting with your data!")
        return
    
    df = st.session_state.uploaded_data
    print(df.head(3))
    
    # Initialize chat agent if not exists - THIS WAS MISSING!
    print("➡️ Before initializing the chat agent")
    if st.session_state.chat_agent is None:
        try:
            with st.spinner("🤖 Initializing AI Chat Agent..."):
                print("➡️ creating the chat agent")
                st.session_state.chat_agent = create_chat_agent(df)
                st.success("✅ Chat Agent Ready! Ask me anything about your data.")
        except Exception as e:
            st.error(f"❌ Failed to initialize chat agent: {str(e)}")
            return
    
    # Chat interface header
    st.subheader("💬 Chat with Your Data")
    st.markdown("Ask me anything about your dataset! I can analyze, visualize, and provide insights.")
    
    ######### Quick action buttons
    # st.markdown("**Quick Questions:**")
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     if st.button("📊 Summarize dataset", use_container_width=True):
    #         st.session_state.chat_history.append({
    #             "role": "user", 
    #             "message": "Summarize this dataset"
    #         })
    #         st.rerun()
    
    # with col2:
    #     if st.button("📈 Show distributions", use_container_width=True):
    #         st.session_state.chat_history.append({
    #             "role": "user", 
    #             "message": "Show me the distribution of numerical columns"
    #         })
    #         st.rerun()
    
    # with col3:
    #     if st.button("🔍 Find correlations", use_container_width=True):
    #         st.session_state.chat_history.append({
    #             "role": "user", 
    #             "message": "What are the strongest correlations in the data?"
    #         })
    #         st.rerun()
    
    st.markdown("---")
    
    # Chat input
    user_input = st.text_input(
        "💭 Ask me anything about your data:",
        placeholder="e.g., 'Find outliers', 'Create a scatter plot', 'What insights can you find?'",
        key="chat_input"
    )
    
    # Send message button
    col_send, col_clear = st.columns([3, 1])
    with col_send:
        send_message = st.button("🚀 Send Message", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Process new message - THIS HANDLES THE ACTUAL CHAT!
    if send_message and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "message": user_input
        })
        
        # Get AI response - THIS IS WHERE THE MAGIC HAPPENS!
        with st.spinner("🤖 AI is analyzing..."):
            try:
                response = st.session_state.chat_agent.chat(user_input)
                
                # Add AI response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "message": response.get("text", "I couldn't process your request."),
                    "chart": response.get("chart"),  # CHARTS ARE DISPLAYED HERE!
                    "code": response.get("code")
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Chat failed: {str(e)}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "message": f"I encountered an error: {str(e)}. Please try rephrasing your question."
                })
    
    # Display chat history with charts
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        
        for i, message in enumerate(reversed(st.session_state.chat_history[-10:])):
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🙋‍♂️ You:</strong> {message["message"]}
                </div>
                """, unsafe_allow_html=True)
                
            else:  # assistant
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>🤖 AI Assistant:</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Display text response
                st.markdown(message["message"])
                
                # Display chart if available - CHARTS DISPLAYED HERE!
                if message.get("chart"):
                    try:
                        st.plotly_chart(
                            message["chart"], 
                            use_container_width=True, 
                            key=f"chat_chart_{i}"
                        )
                    except Exception as e:
                        st.error(f"Could not display chart: {str(e)}")
                
                # Display code if available
                if message.get("code"):
                    st.code(message["code"], language="python")
    
    else:
        st.info("👋 Start a conversation by asking a question about your data!")
    
    st.markdown('</div>', unsafe_allow_html=True)
def visualize_section():
    """Dedicated visualization section with comprehensive chart generation"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if st.session_state.uploaded_data is None:
        st.info("📊 Please upload a dataset first.")
        return

    st.subheader("📈 Advanced Data Visualization")

    # Use cleaned data if available
    data_to_use = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.uploaded_data

    # Initialize analyzer if needed
    if not hasattr(st.session_state, 'data_analyzer') or st.session_state.data_analyzer is None:
        with st.spinner("🧠 Initializing data analyzer..."):
            st.session_state.data_analyzer = DataAnalyzer()
            st.session_state.data_analyzer.analyze_dataset(data_to_use)

    analyzer = st.session_state.data_analyzer

    # Visualization options
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎨 Chart Types")
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select visualization type:",
            ["auto", "distribution", "correlation", "missing", "outliers", "summary"]
        )

        # Column selection for specific charts
        if chart_type in ["distribution", "outliers"]:
            available_columns = data_to_use.columns.tolist()
            selected_columns = st.multiselect(
                "Select columns to visualize:",
                available_columns,
                default=available_columns[:3] if len(available_columns) >= 3 else available_columns
            )
        else:
            selected_columns = None

        # Generate button
        if st.button("📊 Generate Visualizations", type="primary", use_container_width=True):
            with st.spinner("Creating beautiful visualizations..."):
                try:
                    if selected_columns:
                        # Filter data for selected columns
                        filtered_data = data_to_use[selected_columns].copy()
                        temp_analyzer = DataAnalyzer()
                        temp_analyzer.analyze_dataset(filtered_data)
                        visualizations = temp_analyzer.generate_visualizations(chart_type, selected_columns)
                    else:
                        visualizations = analyzer.generate_visualizations(chart_type)

                    # Store visualizations in session state
                    st.session_state.current_visualizations = visualizations
                    st.success("✅ Visualizations generated successfully!")

                except Exception as e:
                    st.error(f"Visualization failed: {str(e)}")

    with col2:
        st.subheader("📋 Quick Actions")
        
        # Quick visualization buttons
        if st.button("📊 Distribution Analysis", use_container_width=True):
            with st.spinner("Generating distribution plots..."):
                try:
                    visualizations = analyzer.generate_visualizations("distribution")
                    st.session_state.current_visualizations = visualizations
                    st.success("✅ Distribution plots ready!")
                except Exception as e:
                    st.error(f"Failed: {str(e)}")

        if st.button("🔗 Correlation Analysis", use_container_width=True):
            with st.spinner("Generating correlation matrix..."):
                try:
                    visualizations = analyzer.generate_visualizations("correlation")
                    st.session_state.current_visualizations = visualizations
                    st.success("✅ Correlation analysis ready!")
                except Exception as e:
                    st.error(f"Failed: {str(e)}")

        if st.button("❓ Missing Data Analysis", use_container_width=True):
            with st.spinner("Analyzing missing data..."):
                try:
                    visualizations = analyzer.generate_visualizations("missing")
                    st.session_state.current_visualizations = visualizations
                    st.success("✅ Missing data analysis ready!")
                except Exception as e:
                    st.error(f"Failed: {str(e)}")

        if st.button("📈 Summary Dashboard", use_container_width=True):
            with st.spinner("Creating summary dashboard..."):
                try:
                    visualizations = analyzer.generate_visualizations("summary")
                    st.session_state.current_visualizations = visualizations
                    st.success("✅ Summary dashboard ready!")
                except Exception as e:
                    st.error(f"Failed: {str(e)}")

    # Display visualizations
    if hasattr(st.session_state, 'current_visualizations') and st.session_state.current_visualizations:
        st.subheader("📊 Generated Visualizations")
        
        visualizations = st.session_state.current_visualizations
        
        # Display each visualization
        for chart_name, fig in visualizations.items():
            try:
                if isinstance(fig, dict):
                    # Handle nested visualizations
                    for sub_name, sub_fig in fig.items():
                        st.markdown(f"**{sub_name.replace('_', ' ').title()}**")
                        st.plotly_chart(sub_fig, use_container_width=True, key=f"viz_{chart_name}_{sub_name}")
                        st.markdown("---")
                else:
                    # Single visualization
                    st.markdown(f"**{chart_name.replace('_', ' ').title()}**")
                    st.plotly_chart(fig, use_container_width=True, key=f"viz_{chart_name}")
                    st.markdown("---")
            except Exception as e:
                st.warning(f"Could not display {chart_name}: {str(e)}")

    # AI-powered visualization suggestions
    st.subheader("🤖 AI Visualization Suggestions")
    
    user_query = st.text_input("Describe what you want to visualize:", 
                              placeholder="e.g., 'Show me the relationship between age and income'")
    
    if st.button("🤖 Get AI Suggestions") and user_query:
        with st.spinner("AI is analyzing your request..."):
            try:
                response = analyzer.chat_with_data(f"Suggest visualizations for: {user_query}")
                
                st.markdown(f"**AI Suggestion:** {response['text']}")
                
                if response.get('chart_suggestion'):
                    suggestion = response['chart_suggestion']
                    st.info(f"📊 Suggested chart: {suggestion.get('type', 'chart')} with {suggestion.get('x', 'x-axis')} and {suggestion.get('y', 'y-axis')}")
                    
                    # Auto-generate the suggested chart
                    if st.button("🎨 Generate Suggested Chart"):
                        try:
                            chart_data = analyzer._create_suggested_visualization(suggestion)
                            if chart_data and chart_data is not None:
                                if chart_data["type"] == "histogram":
                                    fig = px.histogram(x=chart_data["data"], title=chart_data["title"])
                                    st.plotly_chart(fig, use_container_width=True)
                                elif chart_data["type"] == "scatter":
                                    fig = px.scatter(x=chart_data["x_data"], y=chart_data["y_data"], title=chart_data["title"])
                                    st.plotly_chart(fig, use_container_width=True)
                                elif chart_data["type"] == "bar":
                                    fig = px.bar(x=chart_data["categories"], y=chart_data["values"], title=chart_data["title"])
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("No chart data available for this suggestion.")
                        except Exception as e:
                            st.error(f"Failed to generate suggested chart: {str(e)}")
                
            except Exception as e:
                st.error(f"AI suggestion failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

def automl_section():
    """Simplified AutoML training"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if not hasattr(st.session_state, 'analysis_results') or st.session_state.analysis_results is None:
        st.info("🔍 Please analyze your dataset first in the Explore tab.")
        return

    st.subheader("🤖 AI-Powered Model Training")

    # Simple configuration
    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("🧠 **Gemini will automatically select the best models for your data**")

        # Training options
        time_limit = st.slider("Training time (minutes)", 1, 10, 3)
        interpretability = st.selectbox("Model complexity", ["Simple & Fast", "Balanced", "Advanced"])

        # Map to internal values
        interpretability_map = {"Simple & Fast": "high", "Balanced": "medium", "Advanced": "low"}
        interpretability_value = interpretability_map[interpretability]

    with col2:
        # Current settings summary
        st.markdown("""
        **Training Setup:**
        - AI model selection ✅
        - Data cleaning ✅  
        - Cross-validation ✅
        - Feature engineering ✅
        """)

    # Start training
    if st.button("🚀 Start AutoML Training", type="primary", use_container_width=True):
        with st.spinner("🧠 Training models with Gemini..."):
            try:
                # Use cleaned data if available
                training_data = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.uploaded_data

                automl_system = AutoMLSystemV2()
                results = automl_system.run_full_automl_pipeline(
                    df=training_data,
                    target_column=st.session_state.analysis_results["metadata"]["target_column"],
                    user_constraints={"interpretability": interpretability_value},
                    enable_data_cleaning=False,  # Already cleaned
                    time_limit_minutes=time_limit
                )

                st.session_state.automl_results = results
                st.session_state.training_completed = True
                st.session_state.chat_agent = automl_system.chat_agent
                st.session_state.current_step = 4

                # Show results
                best_model = results["training_results"]["best_model"]
                if best_model:
                    st.success(f"🎉 Training Complete!")

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Best Model", best_model["name"])
                    with col_b:
                        st.metric("Accuracy Score", f"{best_model['cv_score']:.3f}")
                    with col_c:
                        models_trained = results["training_results"]["n_models_trained"]
                        st.metric("Models Tested", models_trained)

            except Exception as e:
                st.error(f"Training failed: {str(e)}")

    # Show training results if available
    if st.session_state.training_completed:
        with st.expander("📊 Training Details", expanded=False):
            results = st.session_state.automl_results
            training_results = results["training_results"]

            if "model_comparison" in training_results:
                comparison_data = []
                for model in training_results["model_comparison"]:
                    if model.get("status") != "failed":
                        comparison_data.append({
                            "Model": model["model_name"],
                            "Score": model["cv_score"],
                            "Std": model["cv_std"]
                        })

                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data).sort_values("Score", ascending=False)

                    fig = px.bar(
                        comparison_df, 
                        x="Model", 
                        y="Score",
                        title="Model Performance Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("model_comparison"))

    st.markdown('</div>', unsafe_allow_html=True)


def results_section():
    """Simplified results and download"""
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)

    if not st.session_state.training_completed:
        st.info("🤖 Please complete AutoML training first.")
        return

    st.subheader("📈 Your AutoML Results")

    results = st.session_state.automl_results
    best_model = results["training_results"]["best_model"]

    # Key results
    if best_model:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏆 Best Model</h3>
                <p style="font-size: 1.5em; font-weight: bold;">{best_model['name']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Performance</h3>
                <p style="font-size: 1.5em; font-weight: bold;">{best_model['cv_score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            training_time = results["training_results"]["training_time_seconds"]
            st.markdown(f"""
            <div class="metric-card">
                <h3>⏱️ Training Time</h3>
                <p style="font-size: 1.5em; font-weight: bold;">{training_time:.1f}s</p>
            </div>
            """, unsafe_allow_html=True)

    # Feature importance
    if st.session_state.automl_results.get("best_pipeline"):
        with st.expander("🔍 Feature Importance", expanded=True):
            try:
                automl_system = AutoMLSystemV2()
                automl_system.best_pipeline = st.session_state.automl_results["best_pipeline"]
                importance_df = automl_system.get_feature_importance()

                if importance_df is not None:
                    # Top 10 features
                    top_features = importance_df.head(10)
                    fig = px.bar(
                        top_features,
                        x="importance",
                        y="feature",
                        orientation='h',
                        title="Top 10 Most Important Features"
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True, key=get_unique_chart_key("importance"))

            except Exception as e:
                st.warning("Feature importance not available for this model.")

    # Download section
    st.subheader("💾 Download Your Model")

    artifacts = results.get("artifacts_saved")
    if artifacts:
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📦 Download Complete Model Package",
                data=create_model_package(artifacts),
                file_name="automl_model_v2.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )

        with col2:
            # Prediction interface
            pred_file = st.file_uploader("Upload data for predictions", type=['csv'])

            if pred_file is not None:
                try:
                    pred_df = pd.read_csv(pred_file)

                    if st.button("🔮 Generate Predictions", use_container_width=True):
                        with st.spinner("Generating predictions..."):
                            automl_system = AutoMLSystemV2()
                            automl_system.best_pipeline = st.session_state.automl_results["best_pipeline"]
                            predictions, probabilities = automl_system.predict(pred_df)

                            results_df = pred_df.copy()
                            results_df['Prediction'] = predictions

                            st.success("✅ Predictions generated!")
                            st.dataframe(results_df, use_container_width=True)

                            # Download predictions
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Predictions",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

def create_model_package(artifacts):
    """Create a zip package of all model artifacts"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for artifact_type, file_path in artifacts.items():
            if Path(file_path).exists():
                filename = Path(file_path).name
                zip_file.write(file_path, filename)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_sample_dataset():
    """Create sample dataset for demo"""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=800, n_features=8, n_classes=2, random_state=42)

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
    df['target'] = y

    # Add some categorical features
    df['category'] = np.random.choice(['A', 'B', 'C'], size=800)
    df['region'] = np.random.choice(['North', 'South'], size=800)

    # Add some data quality issues
    missing_idx = np.random.choice(df.index, 60, replace=False)
    df.loc[missing_idx[:30], 'feature_0'] = np.nan
    df.loc[missing_idx[30:], 'category'] = np.nan

    # Add duplicates
    duplicates = df.sample(40)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df

if __name__ == "__main__":
    main()
