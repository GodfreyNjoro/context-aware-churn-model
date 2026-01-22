"""
FastAPI application for churn prediction service.

Provides RESTful endpoints for:
- Health checks and service status
- Churn probability predictions with explanations
- Feature importance retrieval
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.churn_model import ExplainableChurnModel
from src.config import API_CONFIG, MODEL_CONFIG, RISK_THRESHOLDS

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================
# Pydantic Models
# ============================================
class UserFeatures(BaseModel):
    """
    Input features for churn prediction.
    
    Contains user demographics and interaction metrics
    required for generating a churn probability.
    """
    age: int = Field(..., ge=18, le=100, description="User age in years")
    tenure_days: int = Field(..., ge=0, description="Days since signup")
    subscription_tier: str = Field(
        ...,
        pattern="^(basic|premium|enterprise)$",
        description="Subscription level"
    )
    monthly_spend: float = Field(..., ge=0, description="Monthly spend in USD")
    total_interactions: int = Field(..., ge=0, description="Total support interactions")
    avg_duration: Optional[float] = Field(None, description="Avg interaction duration (sec)")
    resolution_rate: Optional[float] = Field(None, ge=0, le=1, description="Issue resolution rate")
    recency_days: Optional[int] = Field(None, ge=0, description="Days since last interaction")
    interaction_frequency: Optional[float] = Field(None, ge=0, description="Interactions per day")
    sentiment_positive: Optional[int] = Field(0, ge=0, description="Positive sentiment count")
    sentiment_neutral: Optional[int] = Field(0, ge=0, description="Neutral sentiment count")
    sentiment_negative: Optional[int] = Field(0, ge=0, description="Negative sentiment count")


class PredictionResponse(BaseModel):
    """
    Churn prediction response with explanation.
    
    Includes probability, risk level, top contributing
    factors, and actionable recommendations.
    """
    user_id: Optional[str] = Field(None, description="User identifier if provided")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: bool = Field(..., description="Binary churn prediction")
    risk_level: str = Field(..., description="Risk category: low/medium/high")
    top_risk_factors: List[Dict[str, Any]] = Field(..., description="Top SHAP features")
    recommendation: str = Field(..., description="Retention recommendation")


# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
)

# global model instance - loaded on startup
model: Optional[ExplainableChurnModel] = None


# ============================================
# Startup/Shutdown Events
# ============================================
@app.on_event("startup")
async def load_model() -> None:
    """Load the trained churn model on application startup."""
    global model
    
    model_path = Path(MODEL_CONFIG["model_path"])
    
    logger.info("=" * 50)
    logger.info("STARTING CHURN PREDICTION API")
    logger.info("=" * 50)
    
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        model = ExplainableChurnModel()
        model.load_model(str(model_path))
        logger.info("✓ Model loaded successfully")
    else:
        logger.warning(f"Model file not found: {model_path}")
        logger.warning("Please train the model first using scripts/train_model.py")


# ============================================
# API Endpoints
# ============================================
@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint with API information.
    
    Returns service name, version, and model status.
    """
    return {
        "service": API_CONFIG["title"],
        "version": API_CONFIG["version"],
        "status": "active" if model is not None else "model_not_loaded",
        "endpoints": ["/health", "/predict", "/feature-importance"],
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.
    
    Returns service health status and model availability.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model.model_type if model else None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(features: UserFeatures) -> PredictionResponse:
    """
    Predict churn probability for a user.
    
    Takes user features and returns churn probability,
    risk level, top contributing factors (SHAP), and
    a recommended retention action.
    
    Args:
        features: User demographics and interaction metrics
        
    Returns:
        Prediction with explanation and recommendation
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        # convert input to DataFrame
        feature_dict = features.dict()
        X = pd.DataFrame([feature_dict])
        
        logger.info(f"Predicting churn for user with tenure={feature_dict['tenure_days']} days")
        
        # make prediction
        prediction = model.model.predict(X)[0]
        probability = model.model.predict_proba(X)[0][1]
        
        # get SHAP explanation
        explanation = model.explain_prediction(X, sample_idx=0)
        
        # determine risk level and recommendation
        risk_level, recommendation = _get_risk_assessment(probability)
        
        logger.info(f"Prediction: churn_prob={probability:.2f}, risk={risk_level}")
        
        return PredictionResponse(
            churn_probability=float(probability),
            churn_prediction=bool(prediction),
            risk_level=risk_level,
            top_risk_factors=explanation["top_features"][:5],
            recommendation=recommendation,
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/feature-importance")
async def get_feature_importance(top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Get feature importance from the trained model.
    
    Args:
        top_n: Number of top features to return (default: 20)
        
    Returns:
        List of feature names with importance scores
        
    Raises:
        HTTPException: If model not loaded
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        importance = model.get_feature_importance(top_n=top_n)
        return importance.to_dict("records")
    
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ============================================
# Helper Functions
# ============================================
def _get_risk_assessment(probability: float) -> tuple:
    """
    Determine risk level and recommendation based on churn probability.
    
    Args:
        probability: Churn probability (0-1)
        
    Returns:
        Tuple of (risk_level, recommendation)
    """
    if probability < RISK_THRESHOLDS["low"]:
        return (
            "low",
            "User is engaged. Continue regular engagement strategies."
        )
    elif probability < RISK_THRESHOLDS["medium"]:
        return (
            "medium",
            "Monitor user activity. Consider targeted retention campaigns."
        )
    else:
        return (
            "high",
            "High churn risk. Immediate intervention recommended with personalized offers."
        )


# ============================================
# Main entry point
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"],
    )
