"""
Unit tests for churn prediction model.

Tests model initialization, training, prediction, feature importance,
and SHAP-based explanation functionality.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.churn_model import ExplainableChurnModel


class TestModelInitialization:
    """Tests for model initialization."""
    
    def test_creates_random_forest_model(self):
        """Random forest model should be created with correct type."""
        model = ExplainableChurnModel(model_type="random_forest")
        
        assert model.model is not None
        assert model.model_type == "random_forest"
        assert model.explainer is None  # not trained yet
    
    def test_creates_gradient_boosting_model(self):
        """Gradient boosting model should be created with correct type."""
        model = ExplainableChurnModel(model_type="gradient_boosting")
        
        assert model.model is not None
        assert model.model_type == "gradient_boosting"
    
    def test_raises_error_for_unknown_model_type(self):
        """Unknown model type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ExplainableChurnModel(model_type="unknown_model")
    
    def test_accepts_custom_random_state(self):
        """Model should use provided random state."""
        model = ExplainableChurnModel(random_state=123)
        
        assert model.random_state == 123


class TestModelTraining:
    """Tests for model training functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            "age": np.random.randint(18, 75, n_samples),
            "tenure_days": np.random.randint(1, 1000, n_samples),
            "monthly_spend": np.random.uniform(10, 200, n_samples),
            "total_interactions": np.random.randint(1, 50, n_samples),
        })
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
        
        return X, y
    
    def test_train_returns_metrics_dict(self, sample_data):
        """Training should return dictionary with required metrics."""
        X, y = sample_data
        model = ExplainableChurnModel()
        
        results = model.train(X, y, test_size=0.2, run_cv=False)
        
        assert "train_accuracy" in results
        assert "test_accuracy" in results
        assert "roc_auc" in results
        assert "X_test" in results
        assert "y_test" in results
    
    def test_train_accuracy_is_reasonable(self, sample_data):
        """Trained model should have reasonable accuracy."""
        X, y = sample_data
        model = ExplainableChurnModel()
        
        results = model.train(X, y, test_size=0.2, run_cv=False)
        
        # accuracy should be above random (50%)
        assert results["train_accuracy"] > 0.5
        assert results["test_accuracy"] > 0.3  # test may vary more
    
    def test_train_stores_feature_names(self, sample_data):
        """Training should store feature names."""
        X, y = sample_data
        model = ExplainableChurnModel()
        
        model.train(X, y, run_cv=False)
        
        assert model.feature_names == list(X.columns)
    
    def test_train_initializes_explainer(self, sample_data):
        """Training should initialize SHAP explainer."""
        X, y = sample_data
        model = ExplainableChurnModel()
        
        model.train(X, y, run_cv=False)
        
        # explainer should be initialized after training
        assert model.explainer is not None


class TestModelPrediction:
    """Tests for model prediction functionality."""
    
    @pytest.fixture
    def trained_model(self):
        """Create and train a model for testing predictions."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            "age": np.random.randint(18, 75, n_samples),
            "tenure_days": np.random.randint(1, 1000, n_samples),
            "monthly_spend": np.random.uniform(10, 200, n_samples),
            "total_interactions": np.random.randint(1, 50, n_samples),
        })
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
        
        model = ExplainableChurnModel()
        model.train(X, y, run_cv=False)
        
        return model, X
    
    def test_predict_returns_binary_values(self, trained_model):
        """Predictions should be binary (0 or 1)."""
        model, X = trained_model
        
        predictions = model.model.predict(X[:5])
        
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba_returns_probabilities(self, trained_model):
        """Probability predictions should be in [0, 1]."""
        model, X = trained_model
        
        probas = model.model.predict_proba(X[:5])
        
        assert probas.shape == (5, 2)  # 2 classes
        assert np.all(probas >= 0) and np.all(probas <= 1)
        assert np.allclose(probas.sum(axis=1), 1.0)  # rows sum to 1


class TestFeatureImportance:
    """Tests for feature importance extraction."""
    
    @pytest.fixture
    def trained_model(self):
        """Create trained model for testing."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            "age": np.random.randint(18, 75, n_samples),
            "tenure_days": np.random.randint(1, 1000, n_samples),
            "monthly_spend": np.random.uniform(10, 200, n_samples),
            "total_interactions": np.random.randint(1, 50, n_samples),
        })
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
        
        model = ExplainableChurnModel()
        model.train(X, y, run_cv=False)
        
        return model
    
    def test_returns_dataframe_with_correct_columns(self, trained_model):
        """Feature importance should return DataFrame with feature and importance."""
        importance = trained_model.get_feature_importance(top_n=4)
        
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
    
    def test_returns_requested_number_of_features(self, trained_model):
        """Should return exactly top_n features."""
        importance = trained_model.get_feature_importance(top_n=2)
        
        assert len(importance) == 2
    
    def test_features_sorted_by_importance(self, trained_model):
        """Features should be sorted by importance descending."""
        importance = trained_model.get_feature_importance(top_n=4)
        
        values = importance["importance"].values
        assert all(values[i] >= values[i+1] for i in range(len(values)-1))
    
    def test_raises_error_if_not_trained(self):
        """Should raise error if model not trained."""
        model = ExplainableChurnModel()
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.get_feature_importance()


class TestExplainPrediction:
    """Tests for SHAP-based prediction explanations."""
    
    @pytest.fixture
    def trained_model_with_data(self):
        """Create trained model with test data."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            "age": np.random.randint(18, 75, n_samples),
            "tenure_days": np.random.randint(1, 1000, n_samples),
            "monthly_spend": np.random.uniform(10, 200, n_samples),
            "total_interactions": np.random.randint(1, 50, n_samples),
        })
        y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.7, 0.3]))
        
        model = ExplainableChurnModel()
        model.train(X, y, run_cv=False)
        
        return model, X
    
    def test_returns_explanation_dict(self, trained_model_with_data):
        """Explanation should return dict with required keys."""
        model, X = trained_model_with_data
        
        explanation = model.explain_prediction(X, sample_idx=0)
        
        assert "prediction" in explanation
        assert "probability" in explanation
        assert "shap_values" in explanation
        assert "top_features" in explanation
    
    def test_prediction_is_binary(self, trained_model_with_data):
        """Explanation prediction should be 0 or 1."""
        model, X = trained_model_with_data
        
        explanation = model.explain_prediction(X, sample_idx=0)
        
        assert explanation["prediction"] in [0, 1]
    
    def test_probability_is_valid(self, trained_model_with_data):
        """Probability should be between 0 and 1."""
        model, X = trained_model_with_data
        
        explanation = model.explain_prediction(X, sample_idx=0)
        
        assert 0 <= explanation["probability"] <= 1
    
    def test_top_features_has_correct_structure(self, trained_model_with_data):
        """Top features should be list of dicts with feature and shap_value."""
        model, X = trained_model_with_data
        
        explanation = model.explain_prediction(X, sample_idx=0)
        
        assert isinstance(explanation["top_features"], list)
        assert len(explanation["top_features"]) > 0
        
        first_feature = explanation["top_features"][0]
        assert "feature" in first_feature
        assert "shap_value" in first_feature
