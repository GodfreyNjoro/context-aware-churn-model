"""
Configuration settings for the Context-Aware Churn Prediction System.

Centralizes all project paths, model hyperparameters, feature engineering
settings, API configuration, and logging setup for consistent behavior
across all modules.
"""
import os
from pathlib import Path
from typing import Dict, Any, List

# ============================================
# Project Paths
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# Data Configuration
# ============================================
DATA_CONFIG: Dict[str, Any] = {
    "users_file": str(RAW_DATA_DIR / "users.csv"),
    "interactions_file": str(RAW_DATA_DIR / "interactions.csv"),
    "features_file": str(PROCESSED_DATA_DIR / "features.csv"),
    "target_file": str(PROCESSED_DATA_DIR / "target.csv"),
    "n_users": 10000,           # default users for synthetic data
    "n_interactions": 200000,   # default interactions for synthetic data
    "churn_rate": 0.25,         # target churn rate (25%)
}

# ============================================
# Model Configuration
# ============================================
MODEL_CONFIG: Dict[str, Any] = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.1,
    "cv_folds": 5,
    "scoring_metric": "roc_auc",
    "model_path": str(MODELS_DIR / "churn_model.pkl"),
}

# Random Forest configuration
RF_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",  # helps with class imbalance
    "n_jobs": -1,                # use all CPU cores
}

# Gradient Boosting configuration
GB_CONFIG: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
}

# Hyperparameter tuning grids
RF_PARAM_GRID: Dict[str, List] = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
}

# ============================================
# NLP Configuration
# ============================================
NLP_CONFIG: Dict[str, Any] = {
    "max_sequence_length": 512,
    "embedding_dim": 768,
    "pretrained_model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32,
    "sample_size": 1000,  # max interactions to embed per batch
}

# ============================================
# Feature Engineering
# ============================================
FEATURE_CONFIG: Dict[str, Any] = {
    "interaction_window_days": 30,
    "min_interactions": 5,
    "sentiment_threshold": 0.5,
    "create_embeddings": False,  # disable by default for speed
    
    # categorical features to encode
    "categorical_features": ["subscription_tier"],
    
    # numerical features to scale
    "numerical_features": [
        "age", "tenure_days", "monthly_spend", "total_interactions",
        "avg_duration", "total_duration", "resolution_rate",
        "recency_days", "interaction_frequency",
    ],
    
    # sentiment features
    "sentiment_features": ["sentiment_positive", "sentiment_neutral", "sentiment_negative"],
}

# ============================================
# API Configuration
# ============================================
API_CONFIG: Dict[str, Any] = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "reload": os.getenv("API_RELOAD", "true").lower() == "true",
    "title": "Churn Prediction API",
    "description": "Explainable Context-Aware NLP Churn Prediction Service",
    "version": "1.0.0",
}

# Risk thresholds for predictions
RISK_THRESHOLDS: Dict[str, float] = {
    "low": 0.3,      # probability below this = low risk
    "medium": 0.6,   # probability between low and medium = medium risk
    # above medium = high risk
}

# ============================================
# Logging Configuration
# ============================================
LOGGING_CONFIG: Dict[str, Any] = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "log_file": str(LOGS_DIR / "churn_model.log"),
    "rotation": "500 MB",
    "retention": "10 days",
}

# ============================================
# Output Configuration
# ============================================
OUTPUT_CONFIG: Dict[str, Any] = {
    "predictions_file": str(OUTPUT_DIR / "predictions.csv"),
    "feature_importance_file": str(OUTPUT_DIR / "feature_importance.csv"),
    "model_metrics_file": str(OUTPUT_DIR / "model_metrics.json"),
}
