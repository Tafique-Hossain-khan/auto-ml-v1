"""
Test script to verify the DataAnalyzer import fix
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_import_fix():
    """Test that DataAnalyzer can be imported and used correctly"""
    print("🧪 Testing DataAnalyzer import fix...")
    
    try:
        # Test the import
        from components.data_analyzer import DataAnalyzer, analyze_uploaded_data
        print("✅ DataAnalyzer imported successfully")
        
        # Test creating an instance
        analyzer = DataAnalyzer()
        print("✅ DataAnalyzer instance created successfully")
        
        # Test basic functionality
        import pandas as pd
        import numpy as np
        
        # Create test data
        df = pd.DataFrame({
            'age': np.random.normal(35, 10, 50),
            'income': np.random.normal(50000, 15000, 50),
            'target': np.random.choice([0, 1], 50)
        })
        
        print(f"✅ Created test dataset with shape: {df.shape}")
        
        # Test analyze_dataset method
        metadata = analyzer.analyze_dataset(df, target_column='target')
        if 'task_type' in metadata:
            print(f"✅ analyze_dataset works: task_type = {metadata['task_type']}")
        else:
            print("❌ analyze_dataset failed")
        
        # Test analyze_uploaded_data function
        analysis_results = analyze_uploaded_data(df, target_column='target')
        if 'metadata' in analysis_results:
            print("✅ analyze_uploaded_data works correctly")
        else:
            print("❌ analyze_uploaded_data failed")
        
        print("\n🎉 All import and functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_import_fix()
