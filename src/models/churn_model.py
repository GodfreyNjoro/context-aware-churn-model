"""
Explainable churn prediction model with SHAP interpretability.

This module provides the core ExplainableChurnModel class that handles:
- Model initialization (Random Forest or Gradient Boosting)
- Training with cross-validation and performance metrics
- SHAP-based prediction explanations
- Feature importance extraction
- Model persistence (save/load)
"""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

# optional SHAP dependency
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ExplainableChurnModel:
    """
    Churn prediction model with SHAP explainability.
    
    Supports Random Forest and Gradient Boosting classifiers with
    built-in SHAP integration for model interpretability.
    
    Attributes:
        model_type: Type of model ('random_forest' or 'gradient_boosting')
        random_state: Random seed for reproducibility
        model: Underlying sklearn classifier
        explainer: SHAP TreeExplainer instance
        feature_names: List of feature column names
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42,
        **model_params: Any
    ) -> None:
        """
        Initialize the churn prediction model.
        
        Args:
            model_type: Model type - 'random_forest' or 'gradient_boosting'
            random_state: Random seed for reproducibility
            **model_params: Additional parameters passed to the classifier
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.explainer: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        
        # initialize the model based on type
        self._initialize_model(model_params)
    
    def _initialize_model(self, params: Dict[str, Any]) -> None:
        """Initialize the underlying classifier based on model_type."""
        if self.model_type == "random_forest":
            # default RF config - good for interpretability
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "class_weight": "balanced",  # helps with imbalanced classes
                "n_jobs": -1,
            }
            default_params.update(params)
            self.model = RandomForestClassifier(
                random_state=self.random_state,
                **default_params
            )
            
        elif self.model_type == "gradient_boosting":
            # default GB config
            default_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
            }
            default_params.update(params)
            self.model = GradientBoostingClassifier(
                random_state=self.random_state,
                **default_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Supported: 'random_forest', 'gradient_boosting'")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        cv_folds: int = 5,
        run_cv: bool = True
    ) -> Dict[str, Any]:
        """
        Train the churn prediction model.
        
        Args:
            X: Feature DataFrame
            y: Target Series (0=retained, 1=churned)
            test_size: Fraction of data for testing
            cv_folds: Number of cross-validation folds
            run_cv: Whether to run cross-validation
            
        Returns:
            Dictionary containing training metrics and test data
        """
        logger.info("=" * 60)
        logger.info("TRAINING CHURN PREDICTION MODEL")
        logger.info("=" * 60)
        
        # store feature names
        self.feature_names = X.columns.tolist()
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Churn rate (train): {y_train.mean():.2%}")
        
        # cross-validation
        if run_cv:
            logger.info(f"\nRunning {cv_folds}-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=cv_folds, scoring="roc_auc"
            )
            logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # train model
        logger.info(f"\nTraining {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # evaluate on train and test
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # log performance metrics
        logger.info(f"\n{'='*50}")
        logger.info(f"MODEL PERFORMANCE")
        logger.info(f"{'='*50}")
        logger.info(f"  Train Accuracy: {train_score:.4f}")
        logger.info(f"  Test Accuracy:  {test_score:.4f}")
        logger.info(f"  ROC-AUC Score:  {roc_auc:.4f}")
        
        # classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        # initialize SHAP explainer
        self._initialize_explainer()
        
        return {
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "roc_auc": roc_auc,
            "cv_scores": cv_scores.tolist() if run_cv else None,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }
    
    def _initialize_explainer(self) -> None:
        """Initialize SHAP TreeExplainer for model interpretability."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return
        
        logger.info("Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        logger.info("SHAP explainer ready")
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            X: Feature DataFrame containing the sample(s)
            sample_idx: Index of the sample to explain
            
        Returns:
            Dictionary with prediction, probability, SHAP values, and top features
        """
        if self.explainer is None:
            if not SHAP_AVAILABLE:
                raise ValueError("SHAP is not installed. Cannot explain predictions.")
            raise ValueError("Model must be trained before explaining predictions")
        
        # get SHAP values for the sample
        sample = X.iloc[[sample_idx]]
        shap_values = self.explainer.shap_values(sample)
        
        # for binary classification, get values for positive class (churned)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # get prediction and probability
        prediction = self.model.predict(sample)[0]
        prediction_proba = self.model.predict_proba(sample)[0]
        
        # build feature importance ranking
        feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": shap_values[0]
        }).sort_values("shap_value", key=abs, ascending=False)
        
        return {
            "prediction": int(prediction),
            "probability": float(prediction_proba[1]),  # churn probability
            "shap_values": shap_values[0].tolist(),
            "top_features": feature_importance.head(10).to_dict("records"),
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Extract feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).head(top_n)
        
        return importance_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and metadata to disk.
        
        Args:
            filepath: Path to save the model pickle file
        """
        # ensure parent directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "explainer": self.explainer,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "random_state": self.random_state,
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the model pickle file
        """
        data = joblib.load(filepath)
        
        self.model = data["model"]
        self.explainer = data.get("explainer")
        self.feature_names = data["feature_names"]
        self.model_type = data["model_type"]
        self.random_state = data.get("random_state", 42)
        
        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Model type: {self.model_type}, Features: {len(self.feature_names)}")


# ============================================
# Main entry point for standalone execution
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--model-type", default="random_forest",
                       choices=["random_forest", "gradient_boosting"],
                       help="Type of model to train")
    parser.add_argument("--no-cv", action="store_true",
                       help="Skip cross-validation")
    parser.add_argument("--features", default="data/processed/features.csv",
                       help="Path to features CSV")
    parser.add_argument("--target", default="data/processed/target.csv",
                       help="Path to target CSV")
    parser.add_argument("--output", default="models/saved/churn_model.pkl",
                       help="Path to save trained model")
    
    args = parser.parse_args()
    
    # load data
    logger.info(f"Loading features from {args.features}")
    X = pd.read_csv(args.features)
    y = pd.read_csv(args.target).squeeze()
    
    # train model
    model = ExplainableChurnModel(model_type=args.model_type)
    results = model.train(X, y, run_cv=not args.no_cv)
    
    # save model
    model.save_model(args.output)
    
    # display feature importance
    logger.info("\nTop 10 Features:")
    importance = model.get_feature_importance(top_n=10)
    for _, row in importance.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    logger.info("\n✓ Training complete!")
