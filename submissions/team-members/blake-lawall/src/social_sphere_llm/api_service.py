"""
Social Media Analysis API Service

A FastAPI web service for serving MLflow-trained social media analysis models.
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

from .prediction_service import SocialMediaPredictionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Social Media Analysis API",
    description="API for predicting social media addiction using MLflow models",
    version="1.0.0",
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


class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    data: Dict[str, Any] = Field(..., description="Input features for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "feature1": 0.5,
                    "feature2": -0.2,
                    "feature3": 1.0
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    data: List[Dict[str, Any]] = Field(..., description="List of input features for predictions")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {"feature1": 0.5, "feature2": -0.2, "feature3": 1.0},
                    {"feature1": -0.1, "feature2": 0.8, "feature3": -0.5}
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Predicted class (0: Low Risk, 1: High Risk)")
    probability: List[float] = Field(..., description="Class probabilities")
    confidence: float = Field(..., description="Confidence score")
    prediction_class: str = Field(..., description="Human-readable prediction class")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[int] = Field(..., description="List of predicted classes")
    probabilities: List[List[float]] = Field(..., description="List of class probabilities")
    confidence_scores: List[float] = Field(..., description="List of confidence scores")
    prediction_classes: List[str] = Field(..., description="List of human-readable prediction classes")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model used")
    timestamp: str = Field(..., description="Prediction timestamp")
    total_predictions: int = Field(..., description="Total number of predictions made")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Version of the model")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    feature_columns: Optional[List[str]] = Field(None, description="Required feature columns")
    model_type: Optional[str] = Field(None, description="Type of the model")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Model metadata")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    uptime: str = Field(..., description="Service uptime")


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup."""
    global prediction_service
    try:
        prediction_service = SocialMediaPredictionService()
        logger.info("✅ Prediction service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize prediction service: {e}")
        prediction_service = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🔄 Shutting down Social Media Analysis API")


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of the API service."""
    return HealthResponse(
        status="healthy" if prediction_service and prediction_service.model else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=prediction_service is not None and prediction_service.model is not None,
        uptime="running"
    )


# Model information endpoint
@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        model_info = prediction_service.get_model_info()
        return ModelInfoResponse(**model_info)
    except Exception as e:
        logger.error(f"❌ Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(request: PredictionRequest):
    """Make a prediction for a single data point."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Make prediction
        result = prediction_service.predict_single(request.data)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple data points."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    try:
        # Make batch predictions
        results = prediction_service.predict(request.data)
        
        # Add timestamp and total count
        results['timestamp'] = datetime.now().isoformat()
        results['total_predictions'] = len(results['predictions'])
        
        return BatchPredictionResponse(**results)
        
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# Model reload endpoint
@app.post("/model/reload", tags=["Model"])
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model in the background."""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not available")
    
    def reload_model_task():
        """Background task to reload the model."""
        global prediction_service
        try:
            prediction_service = SocialMediaPredictionService()
            logger.info("✅ Model reloaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to reload model: {e}")
    
    background_tasks.add_task(reload_model_task)
    
    return {
        "message": "Model reload initiated",
        "timestamp": datetime.now().isoformat()
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Social Media Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
        "predict": "/predict",
        "batch_predict": "/predict/batch"
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return {
        "error": "Not found",
        "message": "The requested resource was not found",
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    return {
        "error": "Internal server error",
        "message": "An internal server error occurred",
        "timestamp": datetime.now().isoformat()
    }


def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the FastAPI server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        reload: Whether to enable auto-reload
    """
    uvicorn.run(
        "social_sphere_llm.api_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # Start the API server
    print("🚀 Starting Social Media Analysis API...")
    start_api_server(host="0.0.0.0", port=8000, reload=True) 