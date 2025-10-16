"""
Configuration settings for the churn prediction system
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

# Model configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.1,
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.1,
}

# NLP configuration
NLP_CONFIG = {
    "max_sequence_length": 512,
    "embedding_dim": 768,
    "pretrained_model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32,
}

# Feature engineering
FEATURE_CONFIG = {
    "interaction_window_days": 30,
    "min_interactions": 5,
    "sentiment_threshold": 0.5,
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
}

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
