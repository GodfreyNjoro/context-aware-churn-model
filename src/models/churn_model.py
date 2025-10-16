"""
Churn prediction model with explainability
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from typing import Tuple, Dict
import shap
import matplotlib.pyplot as plt
import seaborn as sns

class ExplainableChurnModel:
    """Churn prediction model with SHAP explainability"""
    
    def __init__(self, model_type: str = "random_forest", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.explainer = None
        self.feature_names = None
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """Train the churn prediction model"""
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training {self.model_type} model...")
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nModel Performance:")
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Initialize SHAP explainer
        print("\nInitializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'roc_auc': roc_auc,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def explain_prediction(self, X: pd.DataFrame, sample_idx: int = 0) -> Dict:
        """Explain a single prediction using SHAP"""
        if self.explainer is None:
            raise ValueError("Model must be trained before explaining predictions")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X.iloc[[sample_idx]])
        
        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get prediction
        prediction = self.model.predict(X.iloc[[sample_idx]])[0]
        prediction_proba = self.model.predict_proba(X.iloc[[sample_idx]])[0]
        
        # Get top features
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return {
            'prediction': int(prediction),
            'probability': float(prediction_proba[1]),
            'shap_values': shap_values[0].tolist(),
            'top_features': feature_importance.head(10).to_dict('records')
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'explainer': self.explainer,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.explainer = data['explainer']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Load processed data
    X = pd.read_csv("data/processed/features.csv")
    y = pd.read_csv("data/processed/target.csv").squeeze()
    
    # Train model
    model = ExplainableChurnModel(model_type="random_forest")
    results = model.train(X, y)
    
    # Save model
    model.save_model("models/saved/churn_model.pkl")
    
    # Get feature importance
    importance = model.get_feature_importance()
    print("\nTop Features:")
    print(importance)
