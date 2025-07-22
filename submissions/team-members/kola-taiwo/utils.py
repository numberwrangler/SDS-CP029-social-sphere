import pandas as pd
import joblib
import os

def load_models(regressor_path, classifier_path, cluster_path):
    """Load models with simple, robust method"""
    try:
        # Check if files exist
        required_files = [regressor_path, classifier_path, cluster_path, 'models/preprocessor.pkl']
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Load with simple joblib
        regressor = joblib.load(regressor_path)
        classifier = joblib.load(classifier_path)
        cluster_model = joblib.load(cluster_path)
        preprocessor = joblib.load('models/preprocessor.pkl')
        
        return regressor, classifier, cluster_model, preprocessor
    
    except Exception as e:
        raise Exception(f"Error loading models: {str(e)}")

def validate_input(input_df):
    """Validate user input"""
    if input_df['Age'].values[0] < 15 or input_df['Age'].values[0] > 30:
        raise ValueError("Age must be between 15 and 30")
    if input_df['Avg_Daily_Usage_Hours'].values[0] > 15:
        raise ValueError("Usage hours cannot exceed 15")
    return True

def preprocess_input(input_df, preprocessor):
    """Preprocess user input"""
    return preprocessor.transform(input_df)

def test_models(regressor, classifier, cluster_model, preprocessor):
    """Test models with simple method"""
    try:
        # Create test data
        sample_data = pd.DataFrame({
            'Age': [20],
            'Gender': ['Male'],
            'Academic_Level': ['Undergraduate'],
            'Country': ['USA'],
            'Avg_Daily_Usage_Hours': [5.0],
            'Most_Used_Platform': ['Instagram'],
            'Affects_Academic_Performance': ['Yes'],
            'Sleep_Hours_Per_Night': [7.0],
            'Mental_Health_Score': [6],
            'Relationship_Status': ['Single']
        })
        
        # Test preprocessing
        processed = preprocessor.transform(sample_data)
        
        # Test predictions
        regressor.predict(processed)
        classifier.predict(processed)
        cluster_model.predict(processed)
        
        return True
    except Exception as e:
        raise Exception(f"Model test failed: {str(e)}")
