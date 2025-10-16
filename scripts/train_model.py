#!/usr/bin/env python3
"""
Script to train churn prediction model
"""
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import ChurnPreprocessor
from src.models.churn_model import ExplainableChurnModel

def main():
    print("=" * 60)
    print("Training Churn Prediction Model")
    print("=" * 60)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    print(f"\nLoading data from {data_dir}...")
    
    users = pd.read_csv(data_dir / "users.csv")
    interactions = pd.read_csv(data_dir / "interactions.csv")
    
    print(f"Users: {len(users)}, Interactions: {len(interactions)}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.prepare_features(users, interactions, include_embeddings=False)
    
    # Save processed data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    X.to_csv(processed_dir / "features.csv", index=False)
    y.to_csv(processed_dir / "target.csv", index=False)
    
    # Train model
    print("\nTraining model...")
    model = ExplainableChurnModel(model_type="random_forest")
    results = model.train(X, y)
    
    # Save model
    model_dir = Path(__file__).parent.parent / "models" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / "churn_model.pkl"))
    
    # Display feature importance
    print("\nTop 10 Most Important Features:")
    importance = model.get_feature_importance(top_n=10)
    print(importance.to_string(index=False))
    
    print("\n✓ Model training complete!")

if __name__ == "__main__":
    main()
