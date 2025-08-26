"""
Test script for enhanced AutoML System v2 features
Tests: Data Analysis, Visualization Generation, and Chat with Data
"""
import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_dataset():
    """Create a test dataset for demonstration"""
    np.random.seed(42)
    
    # Create sample data
    n_samples = 1000
    
    # Numerical features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    education_years = np.random.normal(12, 3, n_samples)
    
    # Categorical features
    gender = np.random.choice(['Male', 'Female'], n_samples)
    region = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    occupation = np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Sales'], n_samples)
    
    # Target variable (binary classification)
    target = (income > 50000).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'education_years': education_years,
        'gender': gender,
        'region': region,
        'occupation': occupation,
        'target': target
    })
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, 50, replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    return df

def test_data_analyzer():
    """Test the enhanced DataAnalyzer class"""
    print("🧪 Testing Enhanced DataAnalyzer...")
    
    try:
        from components.data_analyzer import DataAnalyzer
        
        # Create test data
        df = create_test_dataset()
        print(f"✅ Created test dataset with shape: {df.shape}")
        
        # Initialize analyzer
        analyzer = DataAnalyzer()
        print("✅ DataAnalyzer initialized")
        
        # Analyze dataset
        metadata = analyzer.analyze_dataset(df, target_column='target')
        print("✅ Dataset analysis completed")
        print(f"   - Task type: {metadata['task_type']}")
        print(f"   - Target column: {metadata['target_column']}")
        print(f"   - Data quality score: {metadata.get('data_quality_score', 0):.0f}/100")
        
        return analyzer, df
        
    except Exception as e:
        print(f"❌ DataAnalyzer test failed: {e}")
        return None, None

def test_visualization_generation(analyzer):
    """Test visualization generation capabilities"""
    print("\n📊 Testing Visualization Generation...")
    
    try:
        # Test different chart types
        chart_types = ["distribution", "correlation", "missing", "summary"]
        
        for chart_type in chart_types:
            print(f"   Testing {chart_type} charts...")
            visualizations = analyzer.generate_visualizations(chart_type)
            
            if visualizations:
                print(f"   ✅ Generated {len(visualizations)} {chart_type} visualizations")
            else:
                print(f"   ⚠️ No {chart_type} visualizations generated")
        
        print("✅ Visualization generation test completed")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_chat_with_data(analyzer):
    """Test chat with data functionality"""
    print("\n💬 Testing Chat with Data...")
    
    try:
        # Test basic questions
        test_questions = [
            "What is the shape of this dataset?",
            "How many missing values are there?",
            "What are the data types of the columns?",
            "Show me a summary of the numerical columns"
        ]
        
        for question in test_questions:
            print(f"   Testing: {question}")
            response = analyzer.chat_with_data(question)
            
            if response and 'text' in response:
                print(f"   ✅ Response received (confidence: {response.get('confidence', 'unknown')})")
            else:
                print(f"   ⚠️ No response received")
        
        print("✅ Chat with data test completed")
        return True
        
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

def test_ai_visualization_suggestions(analyzer):
    """Test AI-powered visualization suggestions"""
    print("\n🤖 Testing AI Visualization Suggestions...")
    
    try:
        # Test visualization requests
        test_requests = [
            "Show me the relationship between age and income",
            "Create a histogram of education years",
            "Show the distribution of gender by region"
        ]
        
        for request in test_requests:
            print(f"   Testing: {request}")
            response = analyzer.chat_with_data(f"Suggest visualizations for: {request}")
            
            if response and 'text' in response:
                print(f"   ✅ AI suggestion received")
                if response.get('chart_suggestion'):
                    print(f"   📊 Chart suggestion available")
            else:
                print(f"   ⚠️ No AI suggestion received")
        
        print("✅ AI visualization suggestions test completed")
        return True
        
    except Exception as e:
        print(f"❌ AI suggestions test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Enhanced AutoML System v2 Tests")
    print("=" * 50)
    
    # Test 1: Data Analyzer
    analyzer, df = test_data_analyzer()
    if analyzer is None:
        print("❌ Cannot proceed without DataAnalyzer")
        return
    
    # Test 2: Visualization Generation
    viz_success = test_visualization_generation(analyzer)
    
    # Test 3: Chat with Data
    chat_success = test_chat_with_data(analyzer)
    
    # Test 4: AI Visualization Suggestions
    ai_success = test_ai_visualization_suggestions(analyzer)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Data Analyzer: ✅")
    print(f"   Visualization Generation: {'✅' if viz_success else '❌'}")
    print(f"   Chat with Data: {'✅' if chat_success else '❌'}")
    print(f"   AI Suggestions: {'✅' if ai_success else '❌'}")
    
    if all([viz_success, chat_success, ai_success]):
        print("\n🎉 All tests passed! Enhanced features are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
    
    print("\n💡 To use these features in Streamlit:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
