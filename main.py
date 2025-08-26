"""
Main AutoML System Interface v2 - Enhanced with 3 Gemini Features
Features: Model Training + Data Cleaning + Chat with Data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
import os
from pathlib import Path

# Import components
from components.data_analyzer import analyze_uploaded_data
from components.model_advisor import get_model_suggestions, get_gemini_advisor_info
from components.trainer import train_automl_models
from components.data_cleaner import DataCleaningAgent, create_data_cleaning_agent
from components.chat_agent import ChatWithDataAgent, create_chat_agent
from components.utils import create_directories, validate_dataframe, ModelPredictor
from config.settings import DEFAULT_TIME_LIMIT_MINUTES

warnings.filterwarnings('ignore')

class AutoMLSystemV2:
    """
    Enhanced AutoML System v2 with 3 Gemini-powered features
    - Model Training (GOOGLE_API_KEY1)
    - Data Cleaning (GOOGLE_API_KEY2)  
    - Chat with Data (GOOGLE_API_KEY3)
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize AutoML System v2

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.data_analysis = None
        self.model_suggestions = None
        self.training_results = None
        self.best_pipeline = None

        # v2 New features
        self.data_cleaning_agent = None
        self.chat_agent = None
        self.cleaning_results = None
        self.original_data = None
        self.cleaned_data = None

        # Create necessary directories
        create_directories()

        # Display system info
        self._display_system_info()

    def _display_system_info(self):
        """Display information about v2 system capabilities"""
        try:
            advisor_info = get_gemini_advisor_info()
            print("🚀 AutoML System v2 - Enhanced with 3 Gemini Features")
            print("=" * 70)
            print("📊 Feature 1: Model Training & Selection")
            print(f"   Provider: {advisor_info['active_provider'].title()}")
            if advisor_info['gemini_available']:
                print("   Status: ✅ Gemini Available (GOOGLE_API_KEY1)")
            else:
                print("   Status: ❌ Gemini Not Available")

            print("\n🧹 Feature 2: Data Cleaning Agent")
            print("   Provider: Gemini Flash")
            print("   Status: ✅ Ready (GOOGLE_API_KEY2)")

            print("\n💬 Feature 3: Chat with Data Agent") 
            print("   Provider: Gemini Flash")
            print("   Status: ✅ Ready (GOOGLE_API_KEY3)")
            print()
        except Exception as e:
            print(f"⚠️ Could not get system info: {e}")

    def run_full_automl_pipeline(self, df: pd.DataFrame, target_column: str = None,
                               user_constraints: Dict[str, Any] = None,
                               enable_data_cleaning: bool = True,
                               time_limit_minutes: int = DEFAULT_TIME_LIMIT_MINUTES) -> Dict[str, Any]:
        """
        Run complete AutoML pipeline v2 with all features

        Args:
            df: Input DataFrame
            target_column: Target column name (optional, will auto-detect if None)
            user_constraints: User constraints dict (optional)
            enable_data_cleaning: Whether to use data cleaning agent
            time_limit_minutes: Time limit for training

        Returns:
            Complete AutoML results with all v2 features
        """
        print("🚀 Starting Enhanced Gemini AutoML Pipeline v2...")
        print("=" * 70)

        # Store original data
        self.original_data = df.copy()

        # Step 1: Data Validation
        print("\n📊 Step 1: Data Validation")
        validation_results = validate_dataframe(df)
        if not validation_results["is_valid"]:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")

        if validation_results["warnings"]:
            for warning in validation_results["warnings"]:
                print(f"⚠️  Warning: {warning}")

        print(f"✅ Data validation passed")
        print(f"  - Dataset size: {len(df):,} rows × {len(df.columns)} columns")
        print(f"  - Memory usage: {validation_results['info'].get('memory_usage_mb', 0):.1f} MB")

        # Step 2: Data Cleaning (New in v2)
        if enable_data_cleaning:
            print("\n🧹 Step 2: AI-Powered Data Cleaning")
            cleaned_df, cleaning_report = self._run_data_cleaning(df)
            self.cleaned_data = cleaned_df
            self.cleaning_results = cleaning_report
            df = cleaned_df  # Use cleaned data for analysis

            print(f"✅ Data cleaning completed")
            print(f"  - Operations applied: {len(cleaning_report.get('operations_applied', []))}")
            if 'changes_summary' in cleaning_report:
                changes = cleaning_report['changes_summary']
                if changes.get('rows_removed', 0) > 0:
                    print(f"  - Rows removed: {changes['rows_removed']}")
                if changes.get('missing_values_handled', 0) > 0:
                    print(f"  - Missing values handled: {changes['missing_values_handled']}")
        else:
            self.cleaned_data = df.copy()
            print("\n🧹 Step 2: Data Cleaning (Skipped)")

        # Step 3: Data Analysis
        print("\n🔍 Step 3: Data Analysis")
        self.data_analysis = analyze_uploaded_data(df, target_column)
        metadata = self.data_analysis["metadata"]

        print(f"✅ Data analysis completed")
        print(f"  - Task type: {metadata['task_type']}")
        print(f"  - Target column: {metadata['target_column']}")
        print(f"  - Features: {metadata['n_numeric']} numeric, {metadata['n_categorical']} categorical")
        print(f"  - Data quality score: {metadata.get('data_quality_score', 0):.1f}/100")

        # Step 4: Initialize Chat Agent (New in v2)
        print("\n💬 Step 4: Initialize Chat Agent")
        self.chat_agent = create_chat_agent(df)
        print(f"✅ Chat agent ready for data conversations")

        # Step 5: Gemini Model Selection
        print("\n🧠 Step 5: Gemini Model Selection")
        self.model_suggestions = get_model_suggestions(metadata, user_constraints)

        print(f"✅ Model selection completed")
        print(f"  - Suggested models: {len(self.model_suggestions)}")
        for suggestion in self.model_suggestions:
            print(f"    🎯 {suggestion['name']} ({suggestion['model']})")

        # Step 6: Model Training
        print("\n🏋️ Step 6: Model Training & Evaluation")
        self.training_results = train_automl_models(
            df=df,
            target_column=metadata['target_column'],
            model_configs=self.model_suggestions,
            task_type=metadata['task_type'],
            metadata=metadata,
            time_limit_minutes=time_limit_minutes
        )

        self.best_pipeline = self.training_results.get('best_pipeline')

        print(f"\n✅ Training completed")
        print(f"  - Models trained: {self.training_results['n_models_trained']}")
        print(f"  - Models failed: {self.training_results['n_models_failed']}")
        print(f"  - Training time: {self.training_results['training_time_seconds']:.1f}s")

        if self.training_results.get('best_model'):
            best = self.training_results['best_model']
            print(f"  - 🏆 Best model: {best['name']} (Score: {best['cv_score']:.4f})")

        # Step 7: Summary
        print("\n🎉 Enhanced Gemini AutoML v2 Completed!")
        print("=" * 70)

        # Compile final results
        automl_results = {
            "data_analysis": self.data_analysis,
            "data_cleaning_results": self.cleaning_results,
            "model_suggestions": self.model_suggestions,
            "training_results": self.training_results,
            "best_pipeline": self.best_pipeline,
            "artifacts_saved": self.training_results.get('saved_artifacts'),
            "chat_agent_ready": self.chat_agent is not None,
            "summary": self._create_summary(),
            "v2_features": {
                "data_cleaning_enabled": enable_data_cleaning,
                "chat_agent_enabled": True,
                "original_shape": self.original_data.shape,
                "cleaned_shape": self.cleaned_data.shape
            }
        }

        return automl_results

    def _run_data_cleaning(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run AI-powered data cleaning"""
        try:
            # Initialize cleaning agent
            self.data_cleaning_agent = create_data_cleaning_agent()

            # Analyze data quality
            quality_analysis = self.data_cleaning_agent.analyze_data_quality(df)

            # Get AI suggestions
            cleaning_suggestions = self.data_cleaning_agent.suggest_cleaning_operations(df, quality_analysis)

            # Apply high-priority operations automatically
            high_priority_ops = [op for op in cleaning_suggestions if op.get("priority") == "high" and op.get("apply")]

            if high_priority_ops:
                cleaned_df, cleaning_report = self.data_cleaning_agent.apply_cleaning_operations(df, high_priority_ops)
                cleaning_report["quality_analysis"] = quality_analysis
                cleaning_report["all_suggestions"] = cleaning_suggestions
                return cleaned_df, cleaning_report
            else:
                return df, {
                    "message": "No high-priority cleaning operations needed",
                    "quality_analysis": quality_analysis,
                    "all_suggestions": cleaning_suggestions,
                    "operations_applied": [],
                    "before_stats": self.data_cleaning_agent._get_data_stats(df),
                    "after_stats": self.data_cleaning_agent._get_data_stats(df),
                    "changes_summary": {}
                }

        except Exception as e:
            print(f"⚠️ Data cleaning failed: {e}")
            return df, {"error": str(e), "operations_applied": []}

    def chat_with_data(self, message: str) -> Dict[str, Any]:
        """
        Chat with your data using natural language

        Args:
            message: Natural language question about the data

        Returns:
            AI response about the data
        """
        if not self.chat_agent:
            # Initialize with cleaned data if available
            data_to_use = self.cleaned_data if self.cleaned_data is not None else self.original_data
            if data_to_use is not None:
                self.chat_agent = create_chat_agent(data_to_use)
            else:
                return {"error": "No data available for chat. Please run AutoML pipeline first."}

        return self.chat_agent.chat(message)

    def get_data_insights(self) -> Dict[str, Any]:
        """Get comprehensive data insights"""
        if not self.chat_agent:
            return {"error": "Chat agent not initialized"}

        return self.chat_agent.get_data_insights()

    def suggest_additional_cleaning(self, df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Get additional data cleaning suggestions"""
        if df is None:
            df = self.cleaned_data if self.cleaned_data is not None else self.original_data

        if df is None:
            return []

        if not self.data_cleaning_agent:
            self.data_cleaning_agent = create_data_cleaning_agent()

        quality_analysis = self.data_cleaning_agent.analyze_data_quality(df)
        return self.data_cleaning_agent.suggest_cleaning_operations(df, quality_analysis)

    def apply_custom_cleaning(self, operations: List[Dict[str, Any]], df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply custom cleaning operations"""
        if df is None:
            df = self.cleaned_data if self.cleaned_data is not None else self.original_data

        if df is None:
            raise ValueError("No data available for cleaning")

        if not self.data_cleaning_agent:
            self.data_cleaning_agent = create_data_cleaning_agent()

        return self.data_cleaning_agent.apply_cleaning_operations(df, operations)

    def predict(self, X: pd.DataFrame, artifacts_path: Dict[str, str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions using the best trained model

        Args:
            X: Features DataFrame
            artifacts_path: Optional path to saved artifacts (if not using current pipeline)

        Returns:
            Tuple of (predictions, probabilities)
        """
        if artifacts_path:
            # Load from saved artifacts
            predictor = ModelPredictor(
                model_path=artifacts_path["model_path"],
                preprocessor_path=artifacts_path["preprocessor_path"],
                metadata_path=artifacts_path["metadata_path"]
            )
            return predictor.predict(X)
        elif self.best_pipeline:
            # Use current best pipeline
            predictions = self.best_pipeline.predict(X)
            probabilities = None

            if hasattr(self.best_pipeline, 'predict_proba'):
                probabilities = self.best_pipeline.predict_proba(X)

            return predictions, probabilities
        else:
            raise ValueError("No trained model available. Run AutoML first or provide artifacts_path.")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the best model"""
        if not self.best_pipeline:
            return None

        try:
            model = self.best_pipeline.named_steps['model']

            # Get feature names
            preprocessor = self.best_pipeline.named_steps['preprocessor']
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = [f'feature_{i}' for i in range(len(model.feature_importances_))]

            # Get importance values
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                return importance_df
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                coef = model.coef_
                if coef.ndim > 1:  # Multi-class
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    coef = np.abs(coef)

                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coef
                }).sort_values('importance', ascending=False)

                return importance_df
        except Exception as e:
            print(f"Could not extract feature importance: {e}")

        return None

    def _create_summary(self) -> Dict[str, Any]:
        """Create a summary of the AutoML v2 run"""
        if not all([self.data_analysis, self.training_results]):
            return {"status": "incomplete"}

        metadata = self.data_analysis["metadata"]
        best_model = self.training_results.get("best_model")

        summary = {
            "status": "completed",
            "version": "v2",
            "powered_by": "Google Gemini (3 Features)",
            "dataset": {
                "original_shape": self.original_data.shape if self.original_data is not None else None,
                "cleaned_shape": self.cleaned_data.shape if self.cleaned_data is not None else None,
                "task_type": metadata["task_type"],
                "target_column": metadata["target_column"],
                "data_quality": metadata.get("data_quality_score", 0)
            },
            "data_cleaning": {
                "enabled": self.cleaning_results is not None,
                "operations_applied": len(self.cleaning_results.get("operations_applied", [])) if self.cleaning_results else 0
            },
            "training": {
                "models_evaluated": self.training_results["n_models_trained"],
                "training_time_seconds": self.training_results["training_time_seconds"],
                "cv_folds": self.training_results["cv_folds"]
            },
            "best_model": {
                "name": best_model["name"] if best_model else None,
                "score": best_model["cv_score"] if best_model else None,
                "std": best_model["cv_std"] if best_model else None
            } if best_model else None,
            "chat_agent": {
                "enabled": self.chat_agent is not None,
                "ready_for_queries": True
            },
            "artifacts": self.training_results.get("saved_artifacts")
        }

        return summary

    def reset_chat_history(self):
        """Reset chat conversation history"""
        if self.chat_agent:
            self.chat_agent.reset_conversation()

# Convenience functions
def run_automl_pipeline_v2(df: pd.DataFrame, target_column: str = None,
                          user_constraints: Dict[str, Any] = None,
                          enable_data_cleaning: bool = True,
                          time_limit_minutes: int = DEFAULT_TIME_LIMIT_MINUTES) -> Dict[str, Any]:
    """
    Convenience function to run the complete enhanced AutoML pipeline v2

    Args:
        df: Input DataFrame
        target_column: Target column name (optional)
        user_constraints: User constraints (optional)
        enable_data_cleaning: Enable AI data cleaning
        time_limit_minutes: Training time limit

    Returns:
        Complete AutoML v2 results
    """
    automl = AutoMLSystemV2()
    return automl.run_full_automl_pipeline(df, target_column, user_constraints, enable_data_cleaning, time_limit_minutes)

def quick_chat_with_data(df: pd.DataFrame, message: str) -> Dict[str, Any]:
    """Quick chat with data without full pipeline"""
    chat_agent = create_chat_agent(df)
    return chat_agent.chat(message)

# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    print("🧪 Running Enhanced Gemini AutoML v2 Example...")

    # Create sample data with some quality issues
    from sklearn.datasets import make_classification

    # Classification example with data quality issues
    print("\n📊 Classification Example with Data Cleaning:")
    X_class, y_class = make_classification(n_samples=1000, n_features=10, n_classes=3, random_state=42)
    df_class = pd.DataFrame(X_class, columns=[f'feature_{i}' for i in range(10)])
    df_class['target'] = y_class

    # Add data quality issues
    # Missing values
    missing_indices = np.random.choice(df_class.index, size=100, replace=False)
    df_class.loc[missing_indices, 'feature_0'] = np.nan

    # Duplicates
    duplicated_rows = df_class.sample(50)
    df_class = pd.concat([df_class, duplicated_rows], ignore_index=True)

    # Categorical features
    df_class['category_A'] = np.random.choice(['High', 'Medium', 'Low'], size=len(df_class))
    df_class['category_B'] = np.random.choice(['TypeX', 'TypeY'], size=len(df_class))

    automl_v2 = AutoMLSystemV2()
    results = automl_v2.run_full_automl_pipeline(
        df_class, 
        target_column='target', 
        enable_data_cleaning=True,
        time_limit_minutes=2
    )

    print(f"\n🏆 Enhanced AutoML v2 Results:")
    print(f"Best Model: {results['summary']['best_model']['name']}")
    print(f"CV Score: {results['summary']['best_model']['score']:.4f}")
    print(f"Data Cleaning: {results['summary']['data_cleaning']['operations_applied']} operations applied")
    print(f"Original Shape: {results['v2_features']['original_shape']}")
    print(f"Cleaned Shape: {results['v2_features']['cleaned_shape']}")

    # Demo chat functionality
    print(f"\n💬 Chat with Data Demo:")
    chat_response1 = automl_v2.chat_with_data("What is the shape of this dataset?")
    print(f"Q: What is the shape of this dataset?")
    print(f"A: {chat_response1['text']}")

    chat_response2 = automl_v2.chat_with_data("Are there any missing values?")
    print(f"Q: Are there any missing values?")
    print(f"A: {chat_response2['text']}")

    print("\n✅ Enhanced Gemini AutoML v2 example completed!")
    print("\n💡 Next Steps:")
    print("1. Set GOOGLE_API_KEY1, GOOGLE_API_KEY2, GOOGLE_API_KEY3 for full functionality")
    print("2. Run 'streamlit run streamlit_app.py' for enhanced web interface")
    print("3. Try the new data cleaning and chat features!")
