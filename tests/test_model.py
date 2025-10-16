"""
Unit tests for churn prediction model
"""
import pytest
import pandas as pd
import numpy as np
from src.models.churn_model import ExplainableChurnModel

@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    X = pd.DataFrame({
        'age': np.random.randint(18, 75, 100),
        'tenure_days': np.random.randint(1, 1000, 100),
        'monthly_spend': np.random.uniform(10, 200, 100),
        'total_interactions': np.random.randint(1, 50, 100),
    })
    y = pd.Series(np.random.choice([0, 1], 100))
    return X, y

def test_model_initialization():
    """Test model initialization"""
    model = ExplainableChurnModel(model_type="random_forest")
    assert model.model is not None
    assert model.model_type == "random_forest"

def test_model_training(sample_data):
    """Test model training"""
    X, y = sample_data
    model = ExplainableChurnModel(model_type="random_forest")
    results = model.train(X, y, test_size=0.2)
    
    assert 'train_accuracy' in results
    assert 'test_accuracy' in results
    assert 'roc_auc' in results
    assert results['train_accuracy'] > 0
    assert results['test_accuracy'] > 0

def test_model_prediction(sample_data):
    """Test model prediction"""
    X, y = sample_data
    model = ExplainableChurnModel(model_type="random_forest")
    model.train(X, y, test_size=0.2)
    
    predictions = model.model.predict(X[:5])
    assert len(predictions) == 5
    assert all(p in [0, 1] for p in predictions)

def test_feature_importance(sample_data):
    """Test feature importance extraction"""
    X, y = sample_data
    model = ExplainableChurnModel(model_type="random_forest")
    model.train(X, y, test_size=0.2)
    
    importance = model.get_feature_importance(top_n=4)
    assert len(importance) == 4
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns

def test_explain_prediction(sample_data):
    """Test prediction explanation"""
    X, y = sample_data
    model = ExplainableChurnModel(model_type="random_forest")
    model.train(X, y, test_size=0.2)
    
    explanation = model.explain_prediction(X, sample_idx=0)
    assert 'prediction' in explanation
    assert 'probability' in explanation
    assert 'shap_values' in explanation
    assert 'top_features' in explanation
