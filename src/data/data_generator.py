"""
Generate synthetic customer interaction data for churn prediction.

This module provides the ChurnDataGenerator class that creates realistic
synthetic datasets including:
- User demographic data with subscription tiers
- Customer interaction history with sentiment labels
- Simulated text content for NLP features
"""
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# setup logging
logger = logging.getLogger(__name__)


class ChurnDataGenerator:
    """
    Generate realistic synthetic customer churn data.
    
    Creates correlated user demographics and interaction histories
    where churned users tend to have more negative interactions.
    
    Attributes:
        n_users: Number of users to generate
        n_interactions: Total number of interactions to generate
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_users: int = 10000,
        n_interactions: int = 200000,
        random_state: int = 42
    ) -> None:
        """
        Initialize the data generator.
        
        Args:
            n_users: Number of unique users to create
            n_interactions: Total interactions across all users
            random_state: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_interactions = n_interactions
        self.random_state = random_state
        
        # set random seeds for reproducibility
        np.random.seed(random_state)
        random.seed(random_state)
        
        # ----------------------------------------
        # interaction configuration
        # ----------------------------------------
        self.interaction_types: List[str] = [
            "support_ticket", "chat", "email", "phone_call", "feedback"
        ]
        self.sentiment_labels: List[str] = ["positive", "neutral", "negative"]
        
        # ----------------------------------------
        # sample text templates by sentiment
        # ----------------------------------------
        self.positive_texts: List[str] = [
            "Great service! Very satisfied with the product.",
            "Excellent support team, resolved my issue quickly.",
            "Love the new features, keep up the good work!",
            "Outstanding experience, highly recommend.",
            "Thank you for the prompt response!",
            "Your team went above and beyond to help me.",
        ]
        
        self.neutral_texts: List[str] = [
            "Just checking on my account status.",
            "Need information about billing.",
            "How do I update my profile?",
            "General inquiry about services.",
            "Can you explain this feature?",
            "Looking for documentation.",
        ]
        
        self.negative_texts: List[str] = [
            "Very disappointed with the service quality.",
            "Having issues with the platform, not working properly.",
            "Considering canceling my subscription.",
            "Poor customer support, waited too long.",
            "This is frustrating, please fix ASAP.",
            "Not happy with recent changes.",
        ]
    
    def generate_users(self) -> pd.DataFrame:
        """
        Generate user demographic data.
        
        Creates users with age, tenure, subscription tier, spend,
        and churn labels. Churn probability is adjusted based on
        tenure and spend (short tenure + low spend = higher churn).
        
        Returns:
            DataFrame with user demographics and churn labels
        """
        logger.info(f"Generating {self.n_users} users...")
        
        # base user attributes
        users = pd.DataFrame({
            "user_id": range(1, self.n_users + 1),
            "age": np.random.randint(18, 75, self.n_users),
            "tenure_days": np.random.randint(1, 1825, self.n_users),  # up to 5 years
            "subscription_tier": np.random.choice(
                ["basic", "premium", "enterprise"],
                self.n_users,
                p=[0.5, 0.35, 0.15]
            ),
            "monthly_spend": np.random.gamma(2, 50, self.n_users).round(2),
        })
        
        # ----------------------------------------
        # base churn rate: 25%
        # ----------------------------------------
        users["churned"] = np.random.choice(
            [0, 1], self.n_users, p=[0.75, 0.25]
        )
        
        # ----------------------------------------
        # adjust churn based on risk factors
        # ----------------------------------------
        # new users (< 90 days) churn more often
        new_user_mask = users["tenure_days"] < 90
        users.loc[new_user_mask, "churned"] = np.random.choice(
            [0, 1], new_user_mask.sum(), p=[0.5, 0.5]
        )
        
        # low spenders (< $20/month) churn more often
        low_spend_mask = users["monthly_spend"] < 20
        users.loc[low_spend_mask, "churned"] = np.random.choice(
            [0, 1], low_spend_mask.sum(), p=[0.6, 0.4]
        )
        
        # log statistics
        churn_rate = users["churned"].mean()
        logger.info(f"Users generated - Churn rate: {churn_rate:.1%}")
        
        return users
    
    def generate_interactions(self, users: pd.DataFrame) -> pd.DataFrame:
        """
        Generate customer interaction data.
        
        Creates interactions with sentiment correlated to churn status:
        - Churned users: more negative sentiment
        - Retained users: more positive sentiment
        
        Args:
            users: DataFrame with user_id and churned columns
        
        Returns:
            DataFrame with interaction records
        """
        logger.info(f"Generating {self.n_interactions} interactions...")
        
        # create lookup for churned status
        churn_lookup = users.set_index("user_id")["churned"].to_dict()
        
        interactions = []
        
        for i in range(self.n_interactions):
            # select random user
            user_id = np.random.choice(users["user_id"])
            user_churned = churn_lookup[user_id]
            
            # ----------------------------------------
            # sentiment depends on churn status
            # ----------------------------------------
            if user_churned:
                # churned users: more negative interactions
                sentiment = np.random.choice(
                    self.sentiment_labels, p=[0.2, 0.3, 0.5]
                )
            else:
                # retained users: more positive interactions
                sentiment = np.random.choice(
                    self.sentiment_labels, p=[0.5, 0.35, 0.15]
                )
            
            # select text based on sentiment
            if sentiment == "positive":
                text = random.choice(self.positive_texts)
            elif sentiment == "neutral":
                text = random.choice(self.neutral_texts)
            else:
                text = random.choice(self.negative_texts)
            
            # create interaction record
            interaction = {
                "interaction_id": i + 1,
                "user_id": user_id,
                "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 365)),
                "interaction_type": np.random.choice(self.interaction_types),
                "text": text,
                "sentiment": sentiment,
                "duration_seconds": (
                    np.random.randint(30, 3600) if np.random.random() > 0.3 else None
                ),
                "resolved": np.random.choice([True, False], p=[0.8, 0.2]),
            }
            interactions.append(interaction)
            
            # progress logging every 50k interactions
            if (i + 1) % 50000 == 0:
                logger.info(f"  Generated {i + 1:,} interactions...")
        
        logger.info(f"Interactions generated: {len(interactions):,}")
        
        return pd.DataFrame(interactions)
    
    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete synthetic dataset.
        
        Returns:
            Tuple of (users DataFrame, interactions DataFrame)
        """
        logger.info("=" * 50)
        logger.info("GENERATING SYNTHETIC CHURN DATASET")
        logger.info("=" * 50)
        
        # generate users
        users = self.generate_users()
        
        # generate interactions
        interactions = self.generate_interactions(users)
        
        # summary statistics
        logger.info("\n" + "=" * 50)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  Users: {len(users):,}")
        logger.info(f"  Churned: {users['churned'].sum():,} ({users['churned'].mean():.1%})")
        logger.info(f"  Interactions: {len(interactions):,}")
        logger.info(f"  Avg interactions/user: {len(interactions)/len(users):.1f}")
        
        return users, interactions


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
    
    parser = argparse.ArgumentParser(description="Generate synthetic churn data")
    parser.add_argument("--users", type=int, default=10000,
                       help="Number of users to generate")
    parser.add_argument("--interactions", type=int, default=200000,
                       help="Number of interactions to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="data/raw",
                       help="Output directory for generated data")
    
    args = parser.parse_args()
    
    # generate data
    generator = ChurnDataGenerator(
        n_users=args.users,
        n_interactions=args.interactions,
        random_state=args.seed
    )
    users, interactions = generator.generate_dataset()
    
    # save datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    users.to_csv(output_dir / "users.csv", index=False)
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    
    logger.info(f"\n✓ Datasets saved to {output_dir}")
