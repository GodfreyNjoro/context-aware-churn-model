"""
Data preprocessing and feature engineering for churn prediction.

This module provides the ChurnPreprocessor class responsible for:
- Engineering behavioral features from user interaction data
- Generating text embeddings using sentence transformers
- Encoding categorical features and scaling numerical features
- Preparing final feature sets for model training and prediction
"""
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# optional sentence-transformers dependency
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings("ignore")

# setup logging
logger = logging.getLogger(__name__)


class ChurnPreprocessor:
    """
    Preprocess and engineer features for churn prediction.
    
    Handles interaction data aggregation, sentiment feature extraction,
    text embedding generation, and feature scaling/encoding.
    
    Attributes:
        embedding_model_name: Name of the sentence transformer model
        embedding_model: Loaded SentenceTransformer instance
        scaler: StandardScaler for numerical features
        label_encoders: Dict of LabelEncoders for categorical features
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        """
        Initialize the preprocessor.
        
        Args:
            embedding_model: HuggingFace model name for text embeddings
        """
        self.embedding_model_name = embedding_model
        self.embedding_model: Optional[Any] = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
    
    def load_embedding_model(self) -> None:
        """
        Load the sentence transformer model for text embeddings.
        
        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available. "
                        "Install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers not installed")
        
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
    
    def engineer_interaction_features(
        self,
        interactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer features from user interaction data.
        
        Extracts behavioral metrics like interaction frequency, recency,
        resolution rate, and sentiment distribution per user.
        
        Args:
            interactions: DataFrame with columns [user_id, timestamp, 
                         duration_seconds, resolved, sentiment, interaction_type]
        
        Returns:
            DataFrame with one row per user and engineered features
        """
        logger.info("Engineering interaction features...")
        
        # convert timestamp to datetime
        interactions = interactions.copy()
        interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])
        
        # ----------------------------------------
        # aggregate basic features per user
        # ----------------------------------------
        user_features = interactions.groupby("user_id").agg({
            "interaction_id": "count",           # total interactions
            "timestamp": ["min", "max"],         # first and last interaction
            "duration_seconds": ["mean", "sum"], # avg and total duration
            "resolved": "mean",                  # resolution rate
        }).reset_index()
        
        # flatten multi-level columns
        user_features.columns = [
            "user_id", "total_interactions", "first_interaction",
            "last_interaction", "avg_duration", "total_duration", "resolution_rate"
        ]
        
        # ----------------------------------------
        # calculate recency and frequency
        # ----------------------------------------
        # days since last interaction
        user_features["recency_days"] = (
            pd.Timestamp.now() - user_features["last_interaction"]
        ).dt.days
        
        # interaction frequency (per day)
        user_features["tenure_days"] = (
            user_features["last_interaction"] - user_features["first_interaction"]
        ).dt.days + 1  # +1 to avoid division by zero
        
        user_features["interaction_frequency"] = (
            user_features["total_interactions"] / user_features["tenure_days"]
        )
        
        # ----------------------------------------
        # sentiment features
        # ----------------------------------------
        sentiment_features = (
            interactions.groupby(["user_id", "sentiment"])
            .size()
            .unstack(fill_value=0)
        )
        sentiment_features.columns = [f"sentiment_{col}" for col in sentiment_features.columns]
        
        # calculate sentiment ratio: (positive - negative) / total
        positive = sentiment_features.get("sentiment_positive", 0)
        negative = sentiment_features.get("sentiment_negative", 0)
        sentiment_features["sentiment_ratio"] = (
            (positive - negative) / user_features.set_index("user_id")["total_interactions"]
        )
        
        # ----------------------------------------
        # interaction type features
        # ----------------------------------------
        type_features = (
            interactions.groupby(["user_id", "interaction_type"])
            .size()
            .unstack(fill_value=0)
        )
        type_features.columns = [f"type_{col}" for col in type_features.columns]
        
        # ----------------------------------------
        # merge all features
        # ----------------------------------------
        user_features = user_features.merge(
            sentiment_features.reset_index(), on="user_id", how="left"
        )
        user_features = user_features.merge(
            type_features.reset_index(), on="user_id", how="left"
        )
        
        logger.info(f"Engineered {user_features.shape[1]} features for "
                   f"{len(user_features)} users")
        
        return user_features
    
    def generate_text_embeddings(
        self,
        interactions: pd.DataFrame,
        sample_size: int = 1000
    ) -> pd.DataFrame:
        """
        Generate text embeddings for user interactions.
        
        Uses sentence transformers to create dense vector representations
        of interaction texts, averaged per user.
        
        Args:
            interactions: DataFrame with 'user_id' and 'text' columns
            sample_size: Max interactions to embed (for efficiency)
        
        Returns:
            DataFrame with user_id and embedding columns
        """
        self.load_embedding_model()
        
        # sample interactions for efficiency
        n_users = interactions["user_id"].nunique()
        samples_per_user = max(1, sample_size // n_users)
        
        sampled = (
            interactions.groupby("user_id")
            .apply(lambda x: x.sample(min(len(x), samples_per_user)))
            .reset_index(drop=True)
        )
        
        logger.info(f"Generating embeddings for {len(sampled)} interactions...")
        
        # generate embeddings
        embeddings = self.embedding_model.encode(
            sampled["text"].tolist(),
            show_progress_bar=True
        )
        
        # average embeddings per user
        sampled["embedding"] = list(embeddings)
        user_embeddings = (
            sampled.groupby("user_id")["embedding"]
            .apply(lambda x: np.mean(np.vstack(x), axis=0))
        )
        
        # convert to DataFrame
        embedding_dim = embeddings.shape[1]
        embedding_df = pd.DataFrame(
            user_embeddings.tolist(),
            index=user_embeddings.index,
            columns=[f"embedding_{i}" for i in range(embedding_dim)]
        ).reset_index()
        
        logger.info(f"Generated {embedding_dim}-dim embeddings for {len(embedding_df)} users")
        
        return embedding_df
    
    def prepare_features(
        self,
        users: pd.DataFrame,
        interactions: pd.DataFrame,
        include_embeddings: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final feature set for model training or prediction.
        
        Combines user demographics with engineered interaction features,
        optionally including text embeddings. Handles encoding and scaling.
        
        Args:
            users: DataFrame with user demographics (must have 'user_id', 'churned')
            interactions: DataFrame with user interactions
            include_embeddings: Whether to generate and include text embeddings
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("=" * 50)
        logger.info("PREPARING FEATURES")
        logger.info("=" * 50)
        
        # engineer interaction features
        interaction_features = self.engineer_interaction_features(interactions)
        
        # merge with user demographics
        features = users.merge(interaction_features, on="user_id", how="left")
        logger.info(f"Merged user data: {features.shape}")
        
        # add text embeddings if requested
        if include_embeddings:
            logger.info("Generating text embeddings...")
            embeddings = self.generate_text_embeddings(interactions)
            features = features.merge(embeddings, on="user_id", how="left")
            logger.info(f"After embeddings: {features.shape}")
        
        # fill missing values
        features = features.fillna(0)
        
        # separate target variable
        target = features["churned"].copy()
        
        # drop non-feature columns
        drop_cols = ["churned", "user_id", "first_interaction", "last_interaction"]
        features = features.drop(columns=[c for c in drop_cols if c in features.columns])
        
        # ----------------------------------------
        # encode categorical variables
        # ----------------------------------------
        categorical_cols = features.select_dtypes(include=["object"]).columns.tolist()
        if categorical_cols:
            logger.info(f"Encoding categorical features: {categorical_cols}")
            for col in categorical_cols:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features[col] = self.label_encoders[col].fit_transform(features[col])
                else:
                    features[col] = self.label_encoders[col].transform(features[col])
        
        # ----------------------------------------
        # scale numerical features
        # ----------------------------------------
        numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Scaling {len(numerical_cols)} numerical features")
        features[numerical_cols] = self.scaler.fit_transform(features[numerical_cols])
        
        logger.info(f"\nFinal feature shape: {features.shape}")
        logger.info(f"Target distribution: {target.value_counts().to_dict()}")
        
        return features, target


# ============================================
# Main entry point for standalone execution
# ============================================
if __name__ == "__main__":
    import argparse
    
    # setup logging for standalone run
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    parser = argparse.ArgumentParser(description="Preprocess churn data")
    parser.add_argument("--users", default="data/raw/users.csv",
                       help="Path to users CSV")
    parser.add_argument("--interactions", default="data/raw/interactions.csv",
                       help="Path to interactions CSV")
    parser.add_argument("--output-dir", default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--embeddings", action="store_true",
                       help="Include text embeddings (slow)")
    
    args = parser.parse_args()
    
    # load data
    logger.info(f"Loading users from {args.users}")
    users = pd.read_csv(args.users)
    logger.info(f"Loading interactions from {args.interactions}")
    interactions = pd.read_csv(args.interactions)
    
    # preprocess
    preprocessor = ChurnPreprocessor()
    X, y = preprocessor.prepare_features(
        users, interactions,
        include_embeddings=args.embeddings
    )
    
    # save processed data
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X.to_csv(output_dir / "features.csv", index=False)
    y.to_csv(output_dir / "target.csv", index=False)
    
    logger.info(f"\n✓ Processed data saved to {output_dir}")
