"""
Data preprocessing and feature engineering for churn prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class ChurnPreprocessor:
    """Preprocess and engineer features for churn prediction"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_embedding_model(self):
        """Load sentence transformer model for text embeddings"""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
    
    def engineer_interaction_features(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from interaction data"""
        # Convert timestamp to datetime
        interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
        
        # Aggregate features per user
        user_features = interactions.groupby('user_id').agg({
            'interaction_id': 'count',  # Total interactions
            'timestamp': ['min', 'max'],  # First and last interaction
            'duration_seconds': ['mean', 'sum'],  # Average and total duration
            'resolved': 'mean',  # Resolution rate
        }).reset_index()
        
        user_features.columns = ['user_id', 'total_interactions', 'first_interaction', 
                                 'last_interaction', 'avg_duration', 'total_duration', 'resolution_rate']
        
        # Calculate recency (days since last interaction)
        user_features['recency_days'] = (pd.Timestamp.now() - user_features['last_interaction']).dt.days
        
        # Calculate frequency (interactions per day)
        user_features['tenure_days'] = (user_features['last_interaction'] - user_features['first_interaction']).dt.days + 1
        user_features['interaction_frequency'] = user_features['total_interactions'] / user_features['tenure_days']
        
        # Sentiment features
        sentiment_features = interactions.groupby(['user_id', 'sentiment']).size().unstack(fill_value=0)
        sentiment_features.columns = [f'sentiment_{col}' for col in sentiment_features.columns]
        sentiment_features['sentiment_ratio'] = (sentiment_features.get('sentiment_positive', 0) - 
                                                  sentiment_features.get('sentiment_negative', 0)) / user_features['total_interactions']
        
        # Interaction type features
        type_features = interactions.groupby(['user_id', 'interaction_type']).size().unstack(fill_value=0)
        type_features.columns = [f'type_{col}' for col in type_features.columns]
        
        # Merge all features
        user_features = user_features.merge(sentiment_features, on='user_id', how='left')
        user_features = user_features.merge(type_features, on='user_id', how='left')
        
        return user_features
    
    def generate_text_embeddings(self, interactions: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
        """Generate text embeddings for interaction texts"""
        self.load_embedding_model()
        
        # Sample interactions per user for efficiency
        sampled = interactions.groupby('user_id').apply(
            lambda x: x.sample(min(len(x), sample_size // len(interactions['user_id'].unique())))
        ).reset_index(drop=True)
        
        print(f"Generating embeddings for {len(sampled)} interactions...")
        embeddings = self.embedding_model.encode(sampled['text'].tolist(), show_progress_bar=True)
        
        # Average embeddings per user
        sampled['embedding'] = list(embeddings)
        user_embeddings = sampled.groupby('user_id')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0))
        
        # Convert to DataFrame
        embedding_df = pd.DataFrame(
            user_embeddings.tolist(),
            index=user_embeddings.index,
            columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
        ).reset_index()
        
        return embedding_df
    
    def prepare_features(self, users: pd.DataFrame, interactions: pd.DataFrame, 
                        include_embeddings: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare final feature set for modeling"""
        print("Engineering interaction features...")
        interaction_features = self.engineer_interaction_features(interactions)
        
        # Merge with user data
        features = users.merge(interaction_features, on='user_id', how='left')
        
        # Add text embeddings if requested
        if include_embeddings:
            print("Generating text embeddings...")
            embeddings = self.generate_text_embeddings(interactions)
            features = features.merge(embeddings, on='user_id', how='left')
        
        # Fill missing values
        features = features.fillna(0)
        
        # Separate target variable
        target = features['churned']
        features = features.drop(['churned', 'user_id', 'first_interaction', 'last_interaction'], axis=1, errors='ignore')
        
        # Encode categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features[col] = self.label_encoders[col].fit_transform(features[col])
            else:
                features[col] = self.label_encoders[col].transform(features[col])
        
        # Scale numerical features
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        
        print(f"Final feature shape: {features.shape}")
        return features, target

if __name__ == "__main__":
    # Load data
    users = pd.read_csv("data/raw/users.csv")
    interactions = pd.read_csv("data/raw/interactions.csv")
    
    # Preprocess
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.prepare_features(users, interactions, include_embeddings=False)
    
    # Save processed data
    X.to_csv("data/processed/features.csv", index=False)
    y.to_csv("data/processed/target.csv", index=False)
    print("Processed data saved!")
