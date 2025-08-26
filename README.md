# 🚀 AutoML System v2 - Enhanced with 3 Gemini Features

**Powered by Google Gemini AI • Intelligent Automated Machine Learning**

## ✨ What's New in v2

AutoML System v2 introduces **3 separate Gemini-powered features**, each using its own API key for optimal performance and cost management:

### 🎯 Feature 1: Model Training & Selection (GOOGLE_API_KEY1)
- AI-powered model recommendations
- Intelligent hyperparameter optimization  
- Context-aware algorithm selection
- Production-ready pipeline generation

### 🧹 Feature 2: Data Cleaning Agent (GOOGLE_API_KEY2)  
- Automated data quality analysis
- Intelligent cleaning suggestions
- AI-powered preprocessing recommendations
- Quality scoring and improvement tracking

### 💬 Feature 3: Chat with Data (GOOGLE_API_KEY3)
- Natural language data queries
- Interactive data exploration
- AI-generated insights and summaries
- Conversational data analysis
- **NEW**: AI-powered visualization suggestions
- **NEW**: Automatic chart generation from chat responses

## 🚀 Quick Start

### 1. Installation

```bash
# Extract the system
unzip automl_system_gemini_v2.zip
cd automl_system_gemini_v2

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Setup

Get your **3 separate API keys** from [Google AI Studio](https://makersuite.google.com/app/apikey):

```bash
# Option 1: Environment variables
export GOOGLE_API_KEY1="your_model_training_key"
export GOOGLE_API_KEY2="your_data_cleaning_key" 
export GOOGLE_API_KEY3="your_chat_with_data_key"

# Option 2: .env file
echo "GOOGLE_API_KEY1=your_model_training_key" > .env
echo "GOOGLE_API_KEY2=your_data_cleaning_key" >> .env
echo "GOOGLE_API_KEY3=your_chat_with_data_key" >> .env

# Option 3: Use single key for all features (fallback)
echo "GOOGLE_API_KEY=your_single_key" > .env
```

### 3. Launch the Enhanced Web Interface

```bash
# Enhanced interface with 6 tabs including visualization
streamlit run app.py

# Or use the original interface
streamlit run streamlit_app.py
```

## 🧠 Core Features

### 📊 **Smart Data Upload & Validation**
- Comprehensive data profiling
- Quality assessment and scoring
- Automated validation with warnings
- Enhanced column analysis

### 🧹 **AI-Powered Data Cleaning**
- Quality analysis with scoring (0-100)
- Intelligent cleaning operation suggestions
- Automated preprocessing with rationale
- Before/after comparison tracking

### 🔍 **Enhanced Data Analysis**  
- Automatic task type detection
- Target column auto-discovery
- Feature type classification
- Missing value pattern analysis
- Class imbalance detection

### 🏋️ **Gemini Model Training**
- AI-powered model selection
- Intelligent hyperparameter tuning
- Cross-validation with detailed metrics
- Automated pipeline optimization

### 💬 **Chat with Your Data**
- Natural language queries
- Real-time data insights
- Interactive exploration
- AI-generated recommendations
- **NEW**: AI-powered visualization suggestions
- **NEW**: Automatic chart generation from chat responses

### 📊 **Advanced Data Visualization**
- **NEW**: Interactive chart generation for numerical and categorical data
- **NEW**: Multiple chart types (histograms, scatter plots, bar charts, correlation matrices)
- **NEW**: AI-powered visualization recommendations
- **NEW**: Real-time chart generation with one-click buttons
- **NEW**: Dedicated visualization tab with comprehensive options

### 📈 **Comprehensive Results**
- Model performance comparison
- Feature importance analysis
- Production-ready artifacts
- Enhanced prediction interface

## 💻 Python API Usage

### Basic AutoML Pipeline

```python
import pandas as pd
from main import AutoMLSystemV2

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize enhanced AutoML system
automl = AutoMLSystemV2()

# Run complete v2 pipeline with all features
results = automl.run_full_automl_pipeline(
    df=df,
    target_column='target',  # or None for auto-detection
    enable_data_cleaning=True,  # Use AI cleaning
    time_limit_minutes=5
)

# Access results
print(f"Best Model: {results['summary']['best_model']['name']}")
print(f"CV Score: {results['summary']['best_model']['score']:.4f}")
print(f"Cleaning Operations: {results['summary']['data_cleaning']['operations_applied']}")

# Make predictions
predictions, probabilities = automl.predict(new_data)
```

### Data Cleaning Only

```python
from components.data_cleaner import create_data_cleaning_agent

# Initialize cleaning agent
cleaner = create_data_cleaning_agent()

# Analyze data quality
quality_analysis = cleaner.analyze_data_quality(df)
print(f"Quality Score: {quality_analysis['quality_score']:.1f}/100")

# Get AI suggestions
suggestions = cleaner.suggest_cleaning_operations(df, quality_analysis)

# Apply cleaning operations
cleaned_df, report = cleaner.apply_cleaning_operations(df, suggestions)
```

### Chat with Data Only

```python
from components.chat_agent import create_chat_agent

# Initialize chat agent
chat_agent = create_chat_agent(df)

# Ask questions
response = chat_agent.chat("What are the main characteristics of this dataset?")
print(response['text'])

# Get comprehensive insights
insights = chat_agent.get_data_insights()
```

### Advanced Data Visualization

```python
from components.data_analyzer import DataAnalyzer

# Initialize enhanced data analyzer
analyzer = DataAnalyzer()
analyzer.analyze_dataset(df, target_column='target')

# Generate visualizations
visualizations = analyzer.generate_visualizations("distribution")
correlation_charts = analyzer.generate_visualizations("correlation")
missing_data_charts = analyzer.generate_visualizations("missing")

# Chat with data and get visualization suggestions
response = analyzer.chat_with_data("Show me the relationship between age and income")
if response.get('chart_suggestion'):
    chart_data = analyzer._create_suggested_visualization(response['chart_suggestion'])
```

## 🛠️ Configuration

### API Key Management

The system supports **flexible API key configuration**:

```python
# config/settings.py

# Separate keys for each feature (recommended)
GOOGLE_API_KEY1 = "key_for_model_training"
GOOGLE_API_KEY2 = "key_for_data_cleaning"  
GOOGLE_API_KEY3 = "key_for_chat_with_data"

# Fallback to single key
GOOGLE_API_KEY = "single_key_for_all_features"
```

### Model Configuration

Each feature uses optimized Gemini settings:

```python
MODEL_TRAINING_CONFIG = {
    "model": "gemini-1.5-flash",
    "temperature": 0.1,
    "max_output_tokens": 2048
}

DATA_CLEANING_CONFIG = {
    "model": "gemini-1.5-flash", 
    "temperature": 0.2,
    "max_output_tokens": 3072
}

CHAT_WITH_DATA_CONFIG = {
    "model": "gemini-1.5-flash",
    "temperature": 0.3,
    "max_output_tokens": 4096
}
```

## 📁 Project Structure

```
automl_system_gemini_v2/
├── components/
│   ├── data_analyzer.py      # Dataset analysis & profiling
│   ├── model_advisor.py      # 🧠 Model training (KEY1)
│   ├── data_cleaner.py       # 🧹 Data cleaning (KEY2)
│   ├── chat_agent.py         # 💬 Chat with data (KEY3)
│   ├── pipeline_builder.py   # Pipeline construction
│   ├── trainer.py           # Model training & evaluation
│   └── utils.py             # Utilities & deployment
├── config/
│   └── settings.py          # 🔑 Enhanced configuration
├── main.py                  # 🚀 Enhanced AutoML orchestration
├── app.py                   # 🖥️ Enhanced web interface (6 tabs with visualization)
├── streamlit_app.py         # 🖥️ Original web interface (5 tabs)
├── requirements.txt         # 📦 Enhanced dependencies
└── README.md               # 📖 This documentation
```

## 🎯 Use Cases

### 1. **Business Analytics**
- Upload sales data, get AI cleaning suggestions
- Chat: "What are the seasonal trends in sales?"
- Auto-train models for sales forecasting

### 2. **Research & Development**
- Upload experiment data with quality issues
- AI cleans and suggests improvements
- Train models with Gemini optimization

### 3. **Data Science Prototyping**
- Quick data exploration with chat interface
- Intelligent cleaning for messy datasets
- Rapid model development and comparison

### 4. **Educational & Learning**
- Interactive data analysis for students
- Learn ML concepts through conversation
- Hands-on experience with AI-powered tools

## 🔧 Advanced Usage

### Custom Data Cleaning

```python
# Get detailed cleaning suggestions
suggestions = automl.suggest_additional_cleaning(df)

# Apply custom operations
custom_operations = [
    {
        "operation": "handle_missing_values",
        "description": "Fill missing values intelligently", 
        "parameters": {"strategy": "advanced"},
        "priority": "high",
        "apply": True,
        "rationale": "High missing value percentage detected"
    }
]

cleaned_df, report = automl.apply_custom_cleaning(custom_operations)
```

### Extended Conversations

```python
# Multi-turn conversation
chat_agent = create_chat_agent(df)

response1 = chat_agent.chat("Summarize this dataset")
response2 = chat_agent.chat("What columns have the most missing values?")
response3 = chat_agent.chat("Suggest visualizations for the target variable")

# Get chat history
history = chat_agent.get_chat_history()
```

### Model Deployment

```python
# Save enhanced artifacts
artifacts = results["artifacts_saved"]

# Deploy with ModelPredictor
from components.utils import ModelPredictor

predictor = ModelPredictor(
    model_path=artifacts["model_path"],
    preprocessor_path=artifacts["preprocessor_path"], 
    metadata_path=artifacts["metadata_path"]
)

predictions, probabilities = predictor.predict(new_data)
```

## 🆓 Free Tier Optimization

All features use **gemini-1.5-flash** for optimal free tier usage:

- **Cost-effective**: Lower token costs than GPT models
- **Fast**: Quick response times for interactive use
- **Capable**: Advanced reasoning for all AutoML tasks
- **Reliable**: Robust fallback systems included

## 🔍 Troubleshooting

### Common Issues

1. **Import Error: SimpleImputer**
   ```python
   # Fixed in v2 - now imports from sklearn.impute
   from sklearn.impute import SimpleImputer
   ```

2. **Duplicate Chart Keys** 
   ```python
   # Fixed in v2 - automatic unique key generation
   st.plotly_chart(fig, key=get_unique_chart_key("chart_name"))
   ```

3. **API Key Issues**
   ```bash
   # Check if keys are loaded
   python -c "from config.settings import *; print(f'Key1: {bool(GOOGLE_API_KEY1)}')"
   ```

### Performance Tips

- Use separate API keys for better rate limiting
- Enable data cleaning for improved model performance  
- Use chat agent for quick data exploration
- Cache is enabled by default for repeated operations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for powerful language model capabilities
- **Streamlit** for the interactive web framework
- **scikit-learn** for robust machine learning algorithms
- **Plotly** for interactive visualizations

## 🚀 What's Next

- **v3 Planning**: Advanced feature engineering agent
- **Integration**: Direct cloud deployment options  
- **Visualization**: Enhanced chart recommendations
- **Collaboration**: Multi-user workspace features

---

**🧠 Powered by Google Gemini • AutoML Made Intelligent**

For questions or support, please open an issue or contact the development team.
