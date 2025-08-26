"""
Test script to verify the metadata structure fix
"""
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_metadata_structure():
    """Test that the metadata structure is correct"""
    print("🧪 Testing metadata structure...")
    
    try:
        from components.data_analyzer import analyze_uploaded_data, DataAnalyzer
        
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.normal(35, 10, 100),
            'income': np.random.normal(50000, 15000, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        print(f"✅ Created test dataset with shape: {df.shape}")
        
        # Test the wrapped function
        analysis_results = analyze_uploaded_data(df, target_column='target')
        
        # Check structure
        if 'metadata' in analysis_results:
            print("✅ 'metadata' key found in analysis_results")
            
            metadata = analysis_results['metadata']
            required_keys = ['task_type', 'target_column', 'n_numeric', 'n_categorical']
            
            for key in required_keys:
                if key in metadata:
                    print(f"✅ '{key}' found in metadata: {metadata[key]}")
                else:
                    print(f"❌ '{key}' missing from metadata")
            
            if 'preprocessing_recommendations' in analysis_results:
                print("✅ 'preprocessing_recommendations' found in analysis_results")
            else:
                print("❌ 'preprocessing_recommendations' missing from analysis_results")
                
        else:
            print("❌ 'metadata' key missing from analysis_results")
            print(f"Available keys: {list(analysis_results.keys())}")
        
        # Test direct analyzer usage
        analyzer = DataAnalyzer()
        direct_metadata = analyzer.analyze_dataset(df, target_column='target')
        
        if 'task_type' in direct_metadata:
            print("✅ Direct analyzer usage works correctly")
        else:
            print("❌ Direct analyzer usage failed")
        
        print("\n🎉 Metadata structure test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_metadata_structure()
