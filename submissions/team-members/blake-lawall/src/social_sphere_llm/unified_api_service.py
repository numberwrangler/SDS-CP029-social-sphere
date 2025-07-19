"""
Unified Social Media Analysis API Service

A FastAPI web service for serving all three MLflow-trained social media analysis models:
1. Conflicts Prediction (Notebook 07)
2. Addicted Score Regression (Notebook 08) 
3. Clustering Analysis (Notebook 09)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn
import json
import logging
from datetime import datetime
import pandas as pd

from .unified_prediction_service import UnifiedSocialMediaPredictionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unified Social Media Analysis API",
    description="API for predicting social media addiction, conflicts, and clustering using MLflow models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global prediction service
prediction_service = None


class StudentDataRequest(BaseModel):
    """Request model for student data."""
    age: int = Field(..., ge=10, le=100, description="Student age")
    gender: str = Field(..., description="Student gender (Male/Female)")
    academic_level: str = Field(..., description="Academic level (High School/Undergraduate/Graduate)")
    avg_daily_usage_hours: float = Field(..., ge=0, le=24, description="Average daily social media usage hours")
    sleep_hours_per_night: float = Field(..., ge=0, le=24, description="Sleep hours per night")
    mental_health_score: int = Field(..., ge=1, le=10, description="Mental health score (1-10)")
    conflicts_over_social_media: int = Field(..., ge=0, le=10, description="Number of conflicts over social media")
    addicted_score: int = Field(..., ge=1, le=10, description="Addiction score (1-10)")
    relationship_status: str = Field(..., description="Relationship status")
    affects_academic_performance: str = Field(..., description="Whether social media affects academic performance")
    most_used_platform: str = Field(..., description="Most used social media platform")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 20,
                "gender": "Female",
                "academic_level": "Undergraduate",
                "avg_daily_usage_hours": 6.5,
                "sleep_hours_per_night": 7.0,
                "mental_health_score": 7,
                "conflicts_over_social_media": 2,
                "addicted_score": 6,
                "relationship_status": "Single",
                "affects_academic_performance": "Yes",
                "most_used_platform": "Instagram"
            }
        }


class ConflictsPredictionResponse(BaseModel):
    """Response model for conflicts predictions."""
    predicted_conflicts: int = Field(..., description="Predicted conflicts (0: Low, 1: High)")
    conflict_level: str = Field(..., description="Conflict risk level")
    recommendation: str = Field(..., description="Intervention recommendation")
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_type: str = Field(..., description="Model type")


class AddictedScoreResponse(BaseModel):
    """Response model for addicted score predictions."""
    predicted_score: float = Field(..., description="Predicted addiction score")
    addiction_level: str = Field(..., description="Addiction level category")
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_type: str = Field(..., description="Model type")


class ClusteringResponse(BaseModel):
    """Response model for clustering predictions."""
    cluster_id: int = Field(..., description="Assigned cluster ID")
    cluster_label: str = Field(..., description="Cluster label")
    risk_level: str = Field(..., description="Risk level")
    recommendation: str = Field(..., description="Intervention recommendation")
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_type: str = Field(..., description="Model type")


class UnifiedPredictionResponse(BaseModel):
    """Response model for unified predictions."""
    conflicts_prediction: ConflictsPredictionResponse = Field(..., description="Conflicts prediction results")
    addicted_score_prediction: AddictedScoreResponse = Field(..., description="Addicted score prediction results")
    clustering_prediction: ClusteringResponse = Field(..., description="Clustering prediction results")
    timestamp: str = Field(..., description="Prediction timestamp")
    student_data: Dict[str, Any] = Field(..., description="Input student data")


class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    conflicts_model_loaded: bool = Field(..., description="Whether conflicts model is loaded")
    addicted_model_loaded: bool = Field(..., description="Whether addicted model is loaded")
    clustering_model_loaded: bool = Field(..., description="Whether clustering model is loaded")
    conflicts_scaler_loaded: bool = Field(..., description="Whether conflicts scaler is loaded")
    addicted_scaler_loaded: bool = Field(..., description="Whether addicted scaler is loaded")
    clustering_scaler_loaded: bool = Field(..., description="Whether clustering scaler is loaded")
    cluster_labels_loaded: bool = Field(..., description="Whether cluster labels are loaded")
    feature_names_loaded: bool = Field(..., description="Whether feature names are loaded")
    timestamp: str = Field(..., description="Status timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    models_loaded: bool = Field(..., description="Whether all models are loaded")
    uptime: str = Field(..., description="Service uptime")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the unified prediction service on startup."""
    global prediction_service
    try:
        prediction_service = UnifiedSocialMediaPredictionService()
        logger.info("✅ Unified prediction service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize unified prediction service: {e}")
        prediction_service = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🔄 Shutting down Unified Social Media Analysis API")


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of the API service."""
    models_loaded = (
        prediction_service and 
        prediction_service.conflicts_model and 
        prediction_service.addicted_model and 
        prediction_service.clustering_model
    )
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        uptime="running"
    )


# Model status endpoint
@app.get("/models/status", response_model=ModelStatusResponse, tags=["Models"])
async def get_model_status():
    """Get status of all models."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        status = prediction_service.get_model_status()
        return ModelStatusResponse(**status)
    except Exception as e:
        logger.error(f"❌ Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


# Conflicts prediction endpoint
@app.post("/predict/conflicts", response_model=ConflictsPredictionResponse, tags=["Predictions"])
async def predict_conflicts(request: StudentDataRequest):
    """Make a conflicts prediction for student data."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert request to dictionary
        data = request.dict()
        
        # Make prediction
        result = prediction_service.predict_conflicts(data)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return ConflictsPredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Conflicts prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conflicts prediction failed: {str(e)}")


# Addicted score prediction endpoint
@app.post("/predict/addicted-score", response_model=AddictedScoreResponse, tags=["Predictions"])
async def predict_addicted_score(request: StudentDataRequest):
    """Make an addicted score prediction for student data."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert request to dictionary
        data = request.dict()
        
        # Make prediction
        result = prediction_service.predict_addicted_score(data)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return AddictedScoreResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Addicted score prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Addicted score prediction failed: {str(e)}")


# Clustering prediction endpoint
@app.post("/predict/clustering", response_model=ClusteringResponse, tags=["Predictions"])
async def predict_clustering(request: StudentDataRequest):
    """Make a clustering prediction for student data."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert request to dictionary
        data = request.dict()
        
        # Make prediction
        result = prediction_service.predict_cluster(data)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return ClusteringResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Clustering prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering prediction failed: {str(e)}")


# Unified prediction endpoint
@app.post("/predict/all", response_model=UnifiedPredictionResponse, tags=["Predictions"])
async def predict_all(request: StudentDataRequest):
    """Make predictions using all three models."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Convert request to dictionary
        data = request.dict()
        
        # Make all predictions
        results = prediction_service.predict_all(data)
        
        # Check for errors in any prediction
        for key, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                raise HTTPException(status_code=500, detail=f"{key} failed: {result['error']}")
        
        return UnifiedPredictionResponse(**results)
        
    except Exception as e:
        logger.error(f"❌ Unified prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Unified prediction failed: {str(e)}")


# Model reload endpoint
@app.post("/models/reload", tags=["Models"])
async def reload_models(background_tasks: BackgroundTasks):
    """Reload all models in the background."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    def reload_models_task():
        """Background task to reload all models."""
        global prediction_service
        try:
            prediction_service = UnifiedSocialMediaPredictionService()
            logger.info("✅ All models reloaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to reload models: {e}")
    
    background_tasks.add_task(reload_models_task)
    
    return {
        "message": "Model reload initiated",
        "timestamp": datetime.now().isoformat()
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Unified Social Media Analysis API",
        "version": "2.0.0",
        "description": "API for predicting social media addiction, conflicts, and clustering",
        "docs": "/docs",
        "health": "/health",
        "model_status": "/models/status",
        "endpoints": {
            "conflicts_prediction": "/predict/conflicts",
            "addicted_score_prediction": "/predict/addicted-score",
            "clustering_prediction": "/predict/clustering",
            "unified_prediction": "/predict/all"
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {
        "error": "Not found",
        "message": "The requested endpoint does not exist",
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }


def start_unified_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the unified API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Whether to enable auto-reload
    """
    uvicorn.run(
        "src.social_sphere_llm.unified_api_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_unified_api_server() 