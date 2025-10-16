"""
FastAPI application for churn prediction service
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.churn_model import ExplainableChurnModel

app = FastAPI(
    title="Churn Prediction API",
    description="Explainable Context-Aware NLP Churn Prediction Service",
    version="1.0.0"
)

# Global model instance
model = None

class UserFeatures(BaseModel):
    """User features for prediction"""
    age: int = Field(..., ge=18, le=100)
    tenure_days: int = Field(..., ge=0)
    subscription_tier: str = Field(..., pattern="^(basic|premium|enterprise)$")
    monthly_spend: float = Field(..., ge=0)
    total_interactions: int = Field(..., ge=0)
    avg_duration: Optional[float] = None
    resolution_rate: Optional[float] = Field(None, ge=0, le=1)
    recency_days: Optional[int] = Field(None, ge=0)
    interaction_frequency: Optional[float] = Field(None, ge=0)
    sentiment_positive: Optional[int] = Field(0, ge=0)
    sentiment_neutral: Optional[int] = Field(0, ge=0)
    sentiment_negative: Optional[int] = Field(0, ge=0)

class PredictionResponse(BaseModel):
    """Prediction response with explanation"""
    user_id: Optional[str] = None
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    top_risk_factors: List[Dict[str, float]]
    recommendation: str

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    model_path = Path(__file__).parent.parent.parent / "models" / "saved" / "churn_model.pkl"
    
    if model_path.exists():
        model = ExplainableChurnModel()
        model.load_model(str(model_path))
        print("Model loaded successfully")
    else:
        print("Warning: Model file not found. Please train the model first.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "status": "active" if model is not None else "model_not_loaded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(features: UserFeatures):
    """Predict churn probability for a user"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        feature_dict = features.dict()
        X = pd.DataFrame([feature_dict])
        
        # Make prediction
        prediction = model.model.predict(X)[0]
        probability = model.model.predict_proba(X)[0][1]
        
        # Get explanation
        explanation = model.explain_prediction(X, sample_idx=0)
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "low"
            recommendation = "User is engaged. Continue regular engagement strategies."
        elif probability < 0.6:
            risk_level = "medium"
            recommendation = "Monitor user activity. Consider targeted retention campaigns."
        else:
            risk_level = "high"
            recommendation = "High churn risk. Immediate intervention recommended with personalized offers."
        
        return PredictionResponse(
            churn_probability=float(probability),
            churn_prediction=bool(prediction),
            risk_level=risk_level,
            top_risk_factors=explanation['top_features'][:5],
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/feature-importance")
async def get_feature_importance(top_n: int = 20):
    """Get feature importance from the model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        importance = model.get_feature_importance(top_n=top_n)
        return importance.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
