"""
Unit tests for FastAPI churn prediction endpoints.

Tests health checks, root endpoint, prediction endpoint,
and feature importance endpoint functionality.
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


# create test client
client = TestClient(app)


class TestRootEndpoint:
    """Tests for the root (/) endpoint."""
    
    def test_returns_200_status(self):
        """Root endpoint should return 200 OK."""
        response = client.get("/")
        
        assert response.status_code == 200
    
    def test_returns_service_info(self):
        """Root endpoint should return service information."""
        response = client.get("/")
        data = response.json()
        
        assert "service" in data
        assert "version" in data
        assert "status" in data
    
    def test_returns_available_endpoints(self):
        """Root endpoint should list available endpoints."""
        response = client.get("/")
        data = response.json()
        
        assert "endpoints" in data
        assert "/health" in data["endpoints"]
        assert "/predict" in data["endpoints"]


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_returns_200_status(self):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        
        assert response.status_code == 200
    
    def test_returns_healthy_status(self):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
    
    def test_includes_model_loaded_flag(self):
        """Health endpoint should indicate if model is loaded."""
        response = client.get("/health")
        data = response.json()
        
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""
    
    @pytest.fixture
    def valid_payload(self):
        """Valid prediction request payload."""
        return {
            "age": 35,
            "tenure_days": 180,
            "subscription_tier": "premium",
            "monthly_spend": 75.50,
            "total_interactions": 15,
            "resolution_rate": 0.85,
            "sentiment_positive": 8,
            "sentiment_neutral": 5,
            "sentiment_negative": 2,
        }
    
    def test_accepts_valid_payload(self, valid_payload):
        """Predict endpoint should accept valid payload."""
        response = client.post("/predict", json=valid_payload)
        
        # 200 if model loaded, 503 if not - both acceptable
        assert response.status_code in [200, 503]
    
    def test_returns_prediction_when_model_loaded(self, valid_payload):
        """Predict endpoint should return prediction if model available."""
        response = client.post("/predict", json=valid_payload)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "churn_probability" in data
            assert "churn_prediction" in data
            assert "risk_level" in data
            assert "recommendation" in data
    
    def test_returns_top_risk_factors(self, valid_payload):
        """Predict endpoint should return SHAP-based risk factors."""
        response = client.post("/predict", json=valid_payload)
        
        if response.status_code == 200:
            data = response.json()
            
            assert "top_risk_factors" in data
            assert isinstance(data["top_risk_factors"], list)
    
    def test_probability_in_valid_range(self, valid_payload):
        """Churn probability should be between 0 and 1."""
        response = client.post("/predict", json=valid_payload)
        
        if response.status_code == 200:
            data = response.json()
            
            assert 0 <= data["churn_probability"] <= 1
    
    def test_risk_level_is_valid_category(self, valid_payload):
        """Risk level should be low, medium, or high."""
        response = client.post("/predict", json=valid_payload)
        
        if response.status_code == 200:
            data = response.json()
            
            assert data["risk_level"] in ["low", "medium", "high"]
    
    def test_rejects_invalid_subscription_tier(self, valid_payload):
        """Should reject invalid subscription tier."""
        valid_payload["subscription_tier"] = "invalid_tier"
        
        response = client.post("/predict", json=valid_payload)
        
        assert response.status_code == 422  # validation error
    
    def test_rejects_negative_age(self, valid_payload):
        """Should reject negative age value."""
        valid_payload["age"] = -5
        
        response = client.post("/predict", json=valid_payload)
        
        assert response.status_code == 422
    
    def test_rejects_missing_required_fields(self):
        """Should reject payload missing required fields."""
        incomplete_payload = {"age": 35}
        
        response = client.post("/predict", json=incomplete_payload)
        
        assert response.status_code == 422


class TestFeatureImportanceEndpoint:
    """Tests for the feature importance endpoint."""
    
    def test_returns_list_when_model_loaded(self):
        """Feature importance should return list of features."""
        response = client.get("/feature-importance")
        
        # 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
    
    def test_respects_top_n_parameter(self):
        """Should return at most top_n features."""
        response = client.get("/feature-importance?top_n=5")
        
        if response.status_code == 200:
            data = response.json()
            assert len(data) <= 5
    
    def test_features_have_correct_structure(self):
        """Each feature should have feature name and importance."""
        response = client.get("/feature-importance?top_n=3")
        
        if response.status_code == 200:
            data = response.json()
            
            if len(data) > 0:
                first = data[0]
                assert "feature" in first
                assert "importance" in first
