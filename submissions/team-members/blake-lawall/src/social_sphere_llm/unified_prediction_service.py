"""
Unified Social Media Analysis Prediction Service

This module provides a production-ready service for making predictions
using all three MLflow-trained models:
1. Conflicts Prediction (Notebook 07)
2. Addicted Score Regression (Notebook 08) 
3. Clustering Analysis (Notebook 09)
"""

import mlflow
import pandas as pd
import numpy as np
import json
import logging
import joblib
from typing import Dict, List, Union, Optional
from pathlib import Path
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedSocialMediaPredictionService:
    """
    A unified service class for making predictions on social media data using all three models.
    """
    
    def __init__(self):
        """
        Initialize the unified prediction service with all three models.
        """
        self.conflicts_model = None
        self.addicted_model = None
        self.clustering_model = None
        self.conflicts_scaler = None
        self.addicted_scaler = None
        self.clustering_scaler = None
        self.cluster_labels = None
        self.feature_names = {}
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Load all models
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all three models and their associated files."""
        try:
            # Load Conflicts Prediction Model (Notebook 07)
            self._load_conflicts_model()
            
            # Load Addicted Score Model (Notebook 08)
            self._load_addicted_model()
            
            # Load Clustering Model (Notebook 09)
            self._load_clustering_model()
            
            logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def _load_conflicts_model(self):
        """Load the conflicts prediction model from Notebook 07."""
        try:
            # Try to load from different paths
            model_paths = [
                'models/conflicts_classifier_rf.joblib',
                '../models/conflicts_classifier_rf.joblib',
                'notebooks/models/conflicts_classifier_rf.joblib'
            ]
            
            for path in model_paths:
                try:
                    self.conflicts_model = joblib.load(path)
                    logger.info(f"✅ Loaded conflicts model from: {path}")
                    break
                except:
                    continue
            
            # Load scaler
            scaler_paths = [
                'models/conflicts_scaler.joblib',
                '../models/conflicts_scaler.joblib',
                'notebooks/models/conflicts_scaler.joblib'
            ]
            
            for path in scaler_paths:
                try:
                    self.conflicts_scaler = joblib.load(path)
                    logger.info(f"✅ Loaded conflicts scaler from: {path}")
                    break
                except:
                    continue
            
            # Load feature names
            feature_paths = [
                'models/conflicts_feature_names.joblib',
                '../models/conflicts_feature_names.joblib',
                'notebooks/models/conflicts_feature_names.joblib'
            ]
            
            for path in feature_paths:
                try:
                    self.feature_names['conflicts'] = joblib.load(path)
                    logger.info(f"✅ Loaded conflicts feature names from: {path}")
                    break
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not load conflicts model: {e}")
    
    def _load_addicted_model(self):
        """Load the addicted score regression model from Notebook 08."""
        try:
            # Only use local joblib files for Gradio Spaces compatibility
            model_paths = [
                'models/addicted_score_model.joblib',
                '../models/addicted_score_model.joblib',
                'notebooks/models/addicted_score_model.joblib'
            ]
            loaded = False
            for path in model_paths:
                try:
                    self.addicted_model = joblib.load(path)
                    logger.info(f"✅ Loaded addicted model from: {path}")
                    loaded = True
                    break
                except Exception as e2:
                    logger.warning(f"⚠️ Could not load addicted model from {path}: {e2}")
            if not loaded:
                logger.error("❌ Failed to load addicted score model from all known paths.")
            # Load scaler
            scaler_paths = [
                'models/addicted_score_scaler.joblib',
                '../models/addicted_score_scaler.joblib',
                'notebooks/models/addicted_score_scaler.joblib'
            ]
            for path in scaler_paths:
                try:
                    self.addicted_scaler = joblib.load(path)
                    logger.info(f"✅ Loaded addicted scaler from: {path}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Could not load addicted scaler from {path}: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load addicted model: {e}")
    
    def _load_clustering_model(self):
        """Load the clustering model from Notebook 09."""
        try:
            # Try to load from different paths
            model_paths = [
                'models/clustering_model.joblib',
                '../models/clustering_model.joblib',
                'notebooks/models/clustering_model.joblib'
            ]
            
            for path in model_paths:
                try:
                    self.clustering_model = joblib.load(path)
                    logger.info(f"✅ Loaded clustering model from: {path}")
                    break
                except:
                    continue
            
            # Load scaler
            scaler_paths = [
                'models/clustering_scaler.joblib',
                '../models/clustering_scaler.joblib',
                'notebooks/models/clustering_scaler.joblib'
            ]
            
            for path in scaler_paths:
                try:
                    self.clustering_scaler = joblib.load(path)
                    logger.info(f"✅ Loaded clustering scaler from: {path}")
                    break
                except:
                    continue
            
            # Load cluster labels
            labels_paths = [
                'models/cluster_labels.joblib',
                '../models/cluster_labels.joblib',
                'notebooks/models/cluster_labels.joblib'
            ]
            
            for path in labels_paths:
                try:
                    self.cluster_labels = joblib.load(path)
                    logger.info(f"✅ Loaded cluster labels from: {path}")
                    break
                except:
                    continue
            
            # Load feature names
            feature_paths = [
                'models/clustering_feature_names.joblib',
                '../models/clustering_feature_names.joblib',
                'notebooks/models/clustering_feature_names.joblib'
            ]
            
            for path in feature_paths:
                try:
                    self.feature_names['clustering'] = joblib.load(path)
                    logger.info(f"✅ Loaded clustering feature names from: {path}")
                    break
                except:
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not load clustering model: {e}")
    
    def predict_conflicts(self, data: Dict) -> Dict:
        """
        Predict conflicts over social media using Notebook 07 model.
        """
        if self.conflicts_model is None or self.conflicts_scaler is None:
            return {
                "error": "Conflicts model not loaded. Please run notebook 07 first.",
                "timestamp": datetime.now().isoformat()
            }
        try:
            features = {}
            if 'Mental_Health_Score' in data:
                features['Mental_Health_Score'] = float(data['Mental_Health_Score'])
            if 'Age' in data:
                features['Age'] = float(data['Age'])
            if 'Gender' in data:
                gender = data['Gender'].lower()
                if gender in ['male', 'm']:
                    features['Gender_Male'] = True
                    features['Gender_Female'] = False
                elif gender in ['female', 'f']:
                    features['Gender_Male'] = False
                    features['Gender_Female'] = True
                else:
                    features['Gender_Male'] = False
                    features['Gender_Female'] = False
            scaler_features = ['Mental_Health_Score', 'Age']
            feature_vector = [features.get(f, 0) for f in scaler_features]
            feature_vector_scaled = self.conflicts_scaler.transform([feature_vector])[0]
            model_features = ['Mental_Health_Score', 'Age', 'Gender_Female', 'Gender_Male']
            # Use scaled for first two, raw for last two
            full_feature_vector = list(feature_vector_scaled) + [features.get('Gender_Female', False), features.get('Gender_Male', False)]
            # Build DataFrame with correct column names
            df = pd.DataFrame([dict(zip(model_features, full_feature_vector))])
            df['Gender_Female'] = df['Gender_Female'].astype(bool)
            df['Gender_Male'] = df['Gender_Male'].astype(bool)
            prediction = self.conflicts_model.predict(df)[0]
            # Handle predict_proba if available
            if hasattr(self.conflicts_model, 'predict_proba'):
                probability = self.conflicts_model.predict_proba(df)[0]
                confidence = max(probability)
            else:
                probability = None
                confidence = None
            if prediction == 1:
                conflict_level = 'High Risk'
                recommendation = 'Immediate intervention needed: Conflict resolution training, communication skills'
            else:
                conflict_level = 'Low Risk'
                recommendation = 'Monitor and provide resources: Healthy communication guidelines'
            result = {
                'predicted_conflicts': int(prediction),
                'conflict_level': conflict_level,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'conflicts_prediction'
            }
            if probability is not None:
                result['confidence'] = float(confidence)
                result['probability'] = probability.tolist()
            return result
        except Exception as e:
            logger.error(f"Exception in predict_conflicts: {e}\n{traceback.format_exc()}")
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_addicted_score(self, data: Dict) -> Dict:
        """
        Predict addicted score using Notebook 08 model.
        """
        if self.addicted_model is None or self.addicted_scaler is None:
            return {
                "error": "Addicted score model not loaded. Please run notebook 08 first.",
                "timestamp": datetime.now().isoformat()
            }
        try:
            features = {}
            if 'Age' in data:
                features['Age'] = float(data['Age'])
            if 'Mental_Health_Score' in data:
                features['Mental_Health_Score'] = float(data['Mental_Health_Score'])
                features['mental_health_squared'] = features['Mental_Health_Score'] ** 2
            if 'Conflicts_Over_Social_Media' in data:
                features['Conflicts_Over_Social_Media'] = float(data['Conflicts_Over_Social_Media'])
            if 'Gender' in data:
                gender = data['Gender'].lower()
                if gender in ['male', 'm']:
                    features['Gender_Male'] = True
                    features['Gender_Female'] = False
                elif gender in ['female', 'f']:
                    features['Gender_Male'] = False
                    features['Gender_Female'] = True
                else:
                    features['Gender_Male'] = False
                    features['Gender_Female'] = False
            scaler_features = ['Mental_Health_Score', 'Age', 'Conflicts_Over_Social_Media']
            feature_vector = [features.get(f, 0) for f in scaler_features]
            feature_vector_scaled = self.addicted_scaler.transform([feature_vector])[0]
            model_features = ['Mental_Health_Score', 'Age', 'Conflicts_Over_Social_Media', 'mental_health_squared', 'Gender_Female', 'Gender_Male']
            # Use scaled for first three, raw for last three
            full_feature_vector = list(feature_vector_scaled) + [features.get('mental_health_squared', 0), features.get('Gender_Female', False), features.get('Gender_Male', False)]
            df = pd.DataFrame([dict(zip(model_features, full_feature_vector))])
            df['Gender_Female'] = df['Gender_Female'].astype(bool)
            df['Gender_Male'] = df['Gender_Male'].astype(bool)
            prediction = self.addicted_model.predict(df)[0]
            if prediction >= 8:
                addiction_level = 'Very High'
            elif prediction >= 6:
                addiction_level = 'High'
            elif prediction >= 4:
                addiction_level = 'Moderate'
            else:
                addiction_level = 'Low'
            confidence = 0.8
            return {
                'predicted_score': float(prediction),
                'addiction_level': addiction_level,
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'addicted_score_regression'
            }
        except Exception as e:
            logger.error(f"Exception in predict_addicted_score: {e}\n{traceback.format_exc()}")
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_cluster(self, data: Dict) -> Dict:
        """
        Predict cluster assignment using Notebook 09 model.
        """
        if self.clustering_model is None or self.clustering_scaler is None:
            return {
                "error": "Clustering model not loaded. Please run notebook 09 first.",
                "timestamp": datetime.now().isoformat()
            }
        try:
            features = {}
            if 'Age' in data:
                features['Age'] = float(data['Age'])
            if 'Avg_Daily_Usage_Hours' in data:
                features['Avg_Daily_Usage_Hours'] = float(data['Avg_Daily_Usage_Hours'])
            if 'Sleep_Hours_Per_Night' in data:
                features['Sleep_Hours_Per_Night'] = float(data['Sleep_Hours_Per_Night'])
            if 'Mental_Health_Score' in data:
                features['Mental_Health_Score'] = float(data['Mental_Health_Score'])
            if 'Conflicts_Over_Social_Media' in data:
                features['Conflicts_Over_Social_Media'] = float(data['Conflicts_Over_Social_Media'])
            if 'Addicted_Score' in data:
                features['Addicted_Score'] = float(data['Addicted_Score'])
            if 'Gender' in data:
                gender = data['Gender'].lower()
                if gender in ['male', 'm']:
                    features['Is_Female'] = 0
                elif gender in ['female', 'f']:
                    features['Is_Female'] = 1
                else:
                    features['Is_Female'] = 0
            if 'Academic_Level' in data:
                level = data['Academic_Level'].lower()
                if 'undergraduate' in level:
                    features['Is_Undergraduate'] = 1
                    features['Is_Graduate'] = 0
                    features['Is_High_School'] = 0
                elif 'graduate' in level:
                    features['Is_Undergraduate'] = 0
                    features['Is_Graduate'] = 1
                    features['Is_High_School'] = 0
                elif 'high school' in level:
                    features['Is_Undergraduate'] = 0
                    features['Is_Graduate'] = 0
                    features['Is_High_School'] = 1
                else:
                    features['Is_Undergraduate'] = 0
                    features['Is_Graduate'] = 0
                    features['Is_High_School'] = 0
            if 'Avg_Daily_Usage_Hours' in features:
                features['High_Usage'] = 1 if features['Avg_Daily_Usage_Hours'] >= 6 else 0
            if 'Sleep_Hours_Per_Night' in features:
                features['Low_Sleep'] = 1 if features['Sleep_Hours_Per_Night'] <= 6 else 0
            if 'Mental_Health_Score' in features:
                features['Poor_Mental_Health'] = 1 if features['Mental_Health_Score'] <= 5 else 0
            if 'Conflicts_Over_Social_Media' in features:
                features['High_Conflict'] = 1 if features['Conflicts_Over_Social_Media'] >= 3 else 0
            if 'Addicted_Score' in features:
                features['High_Addiction'] = 1 if features['Addicted_Score'] >= 7 else 0
            if 'Avg_Daily_Usage_Hours' in features and 'Sleep_Hours_Per_Night' in features:
                features['Usage_Sleep_Ratio'] = features['Avg_Daily_Usage_Hours'] / features['Sleep_Hours_Per_Night']
            if 'Mental_Health_Score' in features and 'Avg_Daily_Usage_Hours' in features:
                features['Mental_Health_Usage_Ratio'] = features['Mental_Health_Score'] / features['Avg_Daily_Usage_Hours']
            feature_names = self.feature_names.get('clustering', [])
            feature_vector = [features.get(f, 0) for f in feature_names]
            feature_vector_scaled = self.clustering_scaler.transform([feature_vector])[0]
            # Build DataFrame with correct column names
            df = pd.DataFrame([dict(zip(feature_names, feature_vector_scaled))])
            cluster_prediction = self.clustering_model.predict(df)[0]
            cluster_label = self.cluster_labels.get(cluster_prediction, f'Cluster_{cluster_prediction}') if self.cluster_labels else f'Cluster_{cluster_prediction}'
            if 'High-Usage' in cluster_label and 'High-Addiction' in cluster_label:
                risk_level = 'High Risk'
                recommendation = 'Intensive intervention needed: Digital detox programs, counseling, parental monitoring'
            elif 'High-Usage' in cluster_label or 'Poor-Health' in cluster_label:
                risk_level = 'Moderate Risk'
                recommendation = 'Targeted intervention recommended: Screen time limits, mental health support, sleep hygiene'
            else:
                risk_level = 'Low Risk'
                recommendation = 'Monitor and provide resources: Educational materials, healthy usage guidelines'
            try:
                cluster_center = self.clustering_model.cluster_centers_[cluster_prediction]
                distance = np.linalg.norm(feature_vector_scaled - cluster_center)
                confidence = max(0.1, 1 - distance/10)
            except:
                confidence = 0.8
            return {
                'cluster_id': int(cluster_prediction),
                'cluster_label': cluster_label,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'clustering_analysis'
            }
        except Exception as e:
            logger.error(f"Exception in predict_cluster: {e}\n{traceback.format_exc()}")
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_all(self, data: Dict) -> Dict:
        """
        Make predictions using all three models.
        
        Args:
            data: Dictionary containing student data
            
        Returns:
            Dictionary containing all prediction results
        """
        results = {
            'conflicts_prediction': self.predict_conflicts(data),
            'addicted_score_prediction': self.predict_addicted_score(data),
            'clustering_prediction': self.predict_cluster(data),
            'timestamp': datetime.now().isoformat(),
            'student_data': data
        }
        
        return results
    
    def get_model_status(self) -> Dict:
        """
        Get status of all models.
        
        Returns:
            Dictionary containing model status information
        """
        return {
            'conflicts_model_loaded': self.conflicts_model is not None,
            'addicted_model_loaded': self.addicted_model is not None,
            'clustering_model_loaded': self.clustering_model is not None,
            'conflicts_scaler_loaded': self.conflicts_scaler is not None,
            'addicted_scaler_loaded': self.addicted_scaler is not None,
            'clustering_scaler_loaded': self.clustering_scaler is not None,
            'cluster_labels_loaded': self.cluster_labels is not None,
            'feature_names_loaded': len(self.feature_names) > 0,
            'timestamp': datetime.now().isoformat()
        }


def create_unified_prediction_service() -> UnifiedSocialMediaPredictionService:
    """
    Factory function to create a unified prediction service.
    
    Returns:
        Initialized unified prediction service
    """
    return UnifiedSocialMediaPredictionService()


# Example usage and testing functions
def test_unified_prediction_service():
    """Test the unified prediction service with sample data."""
    try:
        # Create prediction service
        service = create_unified_prediction_service()
        
        # Get model status
        status = service.get_model_status()
        print("📊 Model Status:")
        print(json.dumps(status, indent=2))
        
        # Test with sample data
        sample_data = {
            'Age': 20,
            'Gender': 'Female',
            'Academic_Level': 'Undergraduate',
            'Avg_Daily_Usage_Hours': 6.5,
            'Sleep_Hours_Per_Night': 7.0,
            'Mental_Health_Score': 7,
            'Conflicts_Over_Social_Media': 2,
            'Addicted_Score': 6,
            'Relationship_Status': 'Single',
            'Affects_Academic_Performance': 'Yes',
            'Most_Used_Platform': 'Instagram'
        }
        
        # Make all predictions
        results = service.predict_all(sample_data)
        
        print("\n📊 Unified Prediction Results:")
        print(json.dumps(results, indent=2))
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    test_unified_prediction_service() 