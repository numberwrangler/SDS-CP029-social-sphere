"""
Social Media Analysis Prediction Service

This module provides a production-ready service for making predictions
using MLflow-trained models for social media addiction analysis.
"""

import mlflow
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SocialMediaPredictionService:
    """
    A service class for making predictions on social media data using MLflow models.
    """
    
    def __init__(self, model_name: str = "social_media_best_model", model_version: str = "latest"):
        """
        Initialize the prediction service.
        
        Args:
            model_name: Name of the registered MLflow model
            model_version: Version of the model to load (default: "latest")
        """
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.model_metadata = None
        self.feature_columns = None
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the MLflow model and metadata."""
        try:
            # Load the model
            model_uri = f"models:/{self.model_name}/{self.model_version}"
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Model loaded successfully: {model_uri}")
            
            # Try to load model metadata
            self._load_metadata()
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_metadata(self):
        """Load model metadata if available."""
        try:
            # Look for metadata in the model artifacts
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name='{self.model_name}'")
            
            if model_versions:
                latest_version = max(model_versions, key=lambda x: x.version)
                run_id = latest_version.run_id
                
                # Try to load metadata from the run
                run = client.get_run(run_id)
                if run.data.artifacts:
                    # Look for metadata file
                    for artifact in run.data.artifacts:
                        if artifact.path.endswith('model_metadata.json'):
                            metadata_path = f"mlruns/{run.info.experiment_id}/{run_id}/artifacts/{artifact.path}"
                            if Path(metadata_path).exists():
                                with open(metadata_path, 'r') as f:
                                    self.model_metadata = json.load(f)
                                self.feature_columns = self.model_metadata.get('feature_columns', [])
                                logger.info("✅ Model metadata loaded successfully")
                                break
                                
        except Exception as e:
            logger.warning(f"⚠️ Could not load model metadata: {e}")
    
    def preprocess_data(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Preprocess input data to match the model's expected format.
        
        Args:
            data: Input data in various formats
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a DataFrame, dict, or list of dicts")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Handle missing columns
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"⚠️ Missing columns: {missing_cols}")
                # Fill missing columns with 0 or appropriate defaults
                for col in missing_cols:
                    df[col] = 0
        
        # Select only the required features
        if self.feature_columns:
            available_cols = [col for col in self.feature_columns if col in df.columns]
            df = df[available_cols]
        
        # Handle categorical variables (basic encoding)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).astype('category').cat.codes
        
        # Fill missing values
        df = df.fillna(0)
        
        logger.info(f"✅ Data preprocessed: {df.shape}")
        return df
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> Dict:
        """
        Make predictions on the input data.
        
        Args:
            data: Input data to predict on
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please initialize the service properly.")
        
        try:
            # Preprocess the data
            processed_data = self.preprocess_data(data)
            
            # Make predictions
            predictions = self.model.predict(processed_data)
            probabilities = self.model.predict_proba(processed_data)
            
            # Prepare results
            results = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'model_name': self.model_name,
                'model_version': self.model_version,
                'confidence_scores': np.max(probabilities, axis=1).tolist(),
                'prediction_classes': ['Low Risk' if p == 0 else 'High Risk' for p in predictions],
                'data_shape': processed_data.shape
            }
            
            # Add metadata if available
            if self.model_metadata:
                results['model_metadata'] = {
                    'training_date': self.model_metadata.get('training_date'),
                    'model_type': self.model_metadata.get('model_type'),
                    'performance_metrics': self.model_metadata.get('performance_metrics', {})
                }
            
            logger.info(f"✅ Predictions completed for {len(predictions)} samples")
            return results
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def predict_single(self, data: Dict) -> Dict:
        """
        Make a prediction for a single data point.
        
        Args:
            data: Single data point as a dictionary
            
        Returns:
            Dictionary containing single prediction result
        """
        results = self.predict(data)
        
        # Return single prediction result
        return {
            'prediction': results['predictions'][0],
            'probability': results['probabilities'][0],
            'confidence': results['confidence_scores'][0],
            'prediction_class': results['prediction_classes'][0],
            'model_name': results['model_name'],
            'model_version': results['model_version']
        }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'model_loaded': self.model is not None,
            'feature_columns': self.feature_columns,
            'model_type': type(self.model.named_steps['classifier']).__name__ if self.model else None
        }
        
        if self.model_metadata:
            info['metadata'] = self.model_metadata
        
        return info


def create_prediction_service(model_name: str = "social_media_best_model") -> SocialMediaPredictionService:
    """
    Factory function to create a prediction service.
    
    Args:
        model_name: Name of the MLflow model to load
        
    Returns:
        Initialized prediction service
    """
    return SocialMediaPredictionService(model_name=model_name)


# Example usage and testing functions
def test_prediction_service():
    """Test the prediction service with sample data."""
    try:
        # Create prediction service
        service = create_prediction_service()
        
        # Get model info
        model_info = service.get_model_info()
        print("📊 Model Information:")
        print(json.dumps(model_info, indent=2))
        
        # Create sample data (adjust based on your actual features)
        sample_data = {
            'feature1': 0.5,
            'feature2': -0.2,
            'feature3': 1.0
        }
        
        # Make prediction
        result = service.predict_single(sample_data)
        print("\n🎯 Prediction Result:")
        print(json.dumps(result, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    print("🧪 Testing Social Media Prediction Service...")
    success = test_prediction_service()
    
    if success:
        print("✅ Prediction service test completed successfully!")
    else:
        print("❌ Prediction service test failed!") 