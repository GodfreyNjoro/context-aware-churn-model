#!/usr/bin/env python3
"""
Training script for the churn prediction model.

Orchestrates data loading, preprocessing, model training,
evaluation, and model persistence. Supports command-line
arguments for flexible configuration.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --model-type gradient_boosting
    python scripts/train_model.py --tune --cv-folds 10
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_CONFIG, MODEL_CONFIG, LOGGING_CONFIG
from src.data.preprocessor import ChurnPreprocessor
from src.models.churn_model import ExplainableChurnModel

# ============================================
# Logging setup
# ============================================
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    datefmt=LOGGING_CONFIG["datefmt"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["log_file"]),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_data(users_path: str, interactions_path: str) -> tuple:
    """
    Load raw user and interaction data.
    
    Args:
        users_path: Path to users CSV file
        interactions_path: Path to interactions CSV file
        
    Returns:
        Tuple of (users DataFrame, interactions DataFrame)
    """
    logger.info(f"Loading users from {users_path}")
    users = pd.read_csv(users_path)
    
    logger.info(f"Loading interactions from {interactions_path}")
    interactions = pd.read_csv(interactions_path)
    
    logger.info(f"Loaded {len(users):,} users and {len(interactions):,} interactions")
    
    return users, interactions


def main(
    model_type: str = "random_forest",
    cv_folds: int = 5,
    include_embeddings: bool = False,
    tune_hyperparams: bool = False
) -> None:
    """
    Main training pipeline.
    
    Args:
        model_type: Type of model to train
        cv_folds: Number of cross-validation folds
        include_embeddings: Whether to include text embeddings
        tune_hyperparams: Whether to tune hyperparameters (not yet implemented)
    """
    # ----------------------------------------
    # header
    # ----------------------------------------
    logger.info("=" * 60)
    logger.info("CHURN PREDICTION MODEL - TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model type: {model_type}")
    logger.info(f"CV folds: {cv_folds}")
    logger.info(f"Include embeddings: {include_embeddings}")
    
    # ----------------------------------------
    # load data
    # ----------------------------------------
    data_dir = Path(DATA_CONFIG.get("users_file")).parent
    users, interactions = load_data(
        DATA_CONFIG["users_file"],
        DATA_CONFIG["interactions_file"]
    )
    
    # ----------------------------------------
    # preprocess data
    # ----------------------------------------
    logger.info("\nPreprocessing data...")
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.prepare_features(
        users, interactions,
        include_embeddings=include_embeddings
    )
    
    # save processed data
    processed_dir = Path(DATA_CONFIG["features_file"]).parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    X.to_csv(DATA_CONFIG["features_file"], index=False)
    y.to_csv(DATA_CONFIG["target_file"], index=False)
    logger.info(f"Processed data saved to {processed_dir}")
    
    # ----------------------------------------
    # train model
    # ----------------------------------------
    logger.info("\nTraining model...")
    model = ExplainableChurnModel(model_type=model_type)
    results = model.train(X, y, cv_folds=cv_folds)
    
    # ----------------------------------------
    # save model
    # ----------------------------------------
    model_path = MODEL_CONFIG["model_path"]
    model.save_model(model_path)
    
    # ----------------------------------------
    # display feature importance
    # ----------------------------------------
    logger.info("\n" + "=" * 50)
    logger.info("TOP 10 MOST IMPORTANT FEATURES")
    logger.info("=" * 50)
    
    importance = model.get_feature_importance(top_n=10)
    for i, row in importance.iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    # ----------------------------------------
    # summary
    # ----------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"  ROC-AUC Score: {results['roc_auc']:.4f}")
    logger.info(f"  Model saved to: {model_path}")
    logger.info("\n✓ Training pipeline finished successfully!")


# ============================================
# Command-line interface
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train churn prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_model.py
  python scripts/train_model.py --model-type gradient_boosting
  python scripts/train_model.py --cv-folds 10 --embeddings
        """
    )
    
    parser.add_argument(
        "--model-type",
        default="random_forest",
        choices=["random_forest", "gradient_boosting"],
        help="Type of model to train (default: random_forest)"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Include text embeddings (slower but may improve accuracy)"
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune hyperparameters (not yet implemented)"
    )
    
    args = parser.parse_args()
    
    if args.tune:
        logger.warning("Hyperparameter tuning not yet implemented. Using defaults.")
    
    main(
        model_type=args.model_type,
        cv_folds=args.cv_folds,
        include_embeddings=args.embeddings,
        tune_hyperparams=args.tune
    )
