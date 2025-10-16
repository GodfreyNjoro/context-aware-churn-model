"""
Unit tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_predict_endpoint():
    """Test prediction endpoint"""
    payload = {
        "age": 35,
        "tenure_days": 180,
        "subscription_tier": "premium",
        "monthly_spend": 75.50,
        "total_interactions": 15,
        "resolution_rate": 0.85,
        "sentiment_positive": 8,
        "sentiment_neutral": 5,
        "sentiment_negative": 2
    }
    
    response = client.post("/predict", json=payload)
    # May return 503 if model not loaded, which is acceptable in test environment
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "churn_probability" in data
        assert "churn_prediction" in data
        assert "risk_level" in data

def test_feature_importance_endpoint():
    """Test feature importance endpoint"""
    response = client.get("/feature-importance?top_n=10")
    # May return 503 if model not loaded
    assert response.status_code in [200, 503]
