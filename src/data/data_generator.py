"""
Generate synthetic customer interaction data for churn prediction
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Tuple

class ChurnDataGenerator:
    """Generate realistic synthetic customer churn data"""
    
    def __init__(self, n_users: int = 10000, n_interactions: int = 200000, random_state: int = 42):
        self.n_users = n_users
        self.n_interactions = n_interactions
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Interaction types and sentiments
        self.interaction_types = ['support_ticket', 'chat', 'email', 'phone_call', 'feedback']
        self.sentiment_labels = ['positive', 'neutral', 'negative']
        
        # Sample text templates
        self.positive_texts = [
            "Great service! Very satisfied with the product.",
            "Excellent support team, resolved my issue quickly.",
            "Love the new features, keep up the good work!",
            "Outstanding experience, highly recommend.",
        ]
        
        self.neutral_texts = [
            "Just checking on my account status.",
            "Need information about billing.",
            "How do I update my profile?",
            "General inquiry about services.",
        ]
        
        self.negative_texts = [
            "Very disappointed with the service quality.",
            "Having issues with the platform, not working properly.",
            "Considering canceling my subscription.",
            "Poor customer support, waited too long.",
        ]
    
    def generate_users(self) -> pd.DataFrame:
        """Generate user demographic data"""
        users = pd.DataFrame({
            'user_id': range(1, self.n_users + 1),
            'age': np.random.randint(18, 75, self.n_users),
            'tenure_days': np.random.randint(1, 1825, self.n_users),  # Up to 5 years
            'subscription_tier': np.random.choice(['basic', 'premium', 'enterprise'], self.n_users, p=[0.5, 0.35, 0.15]),
            'monthly_spend': np.random.gamma(2, 50, self.n_users).round(2),
            'churned': np.random.choice([0, 1], self.n_users, p=[0.75, 0.25])  # 25% churn rate
        })
        
        # Adjust churn probability based on features
        users.loc[users['tenure_days'] < 90, 'churned'] = np.random.choice([0, 1], sum(users['tenure_days'] < 90), p=[0.5, 0.5])
        users.loc[users['monthly_spend'] < 20, 'churned'] = np.random.choice([0, 1], sum(users['monthly_spend'] < 20), p=[0.6, 0.4])
        
        return users
    
    def generate_interactions(self, users: pd.DataFrame) -> pd.DataFrame:
        """Generate customer interaction data"""
        interactions = []
        
        for _ in range(self.n_interactions):
            user_id = np.random.choice(users['user_id'])
            user_churned = users[users['user_id'] == user_id]['churned'].values[0]
            
            # Churned users tend to have more negative interactions
            if user_churned:
                sentiment = np.random.choice(self.sentiment_labels, p=[0.2, 0.3, 0.5])
            else:
                sentiment = np.random.choice(self.sentiment_labels, p=[0.5, 0.35, 0.15])
            
            # Select text based on sentiment
            if sentiment == 'positive':
                text = random.choice(self.positive_texts)
            elif sentiment == 'neutral':
                text = random.choice(self.neutral_texts)
            else:
                text = random.choice(self.negative_texts)
            
            interaction = {
                'interaction_id': len(interactions) + 1,
                'user_id': user_id,
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'interaction_type': np.random.choice(self.interaction_types),
                'text': text,
                'sentiment': sentiment,
                'duration_seconds': np.random.randint(30, 3600) if np.random.random() > 0.3 else None,
                'resolved': np.random.choice([True, False], p=[0.8, 0.2])
            }
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset"""
        print(f"Generating {self.n_users} users...")
        users = self.generate_users()
        
        print(f"Generating {self.n_interactions} interactions...")
        interactions = self.generate_interactions(users)
        
        print("Dataset generation complete!")
        print(f"Users: {len(users)}, Churned: {users['churned'].sum()} ({users['churned'].mean()*100:.1f}%)")
        print(f"Interactions: {len(interactions)}")
        
        return users, interactions

if __name__ == "__main__":
    generator = ChurnDataGenerator(n_users=10000, n_interactions=200000)
    users, interactions = generator.generate_dataset()
    
    # Save datasets
    users.to_csv("data/raw/users.csv", index=False)
    interactions.to_csv("data/raw/interactions.csv", index=False)
    print("Datasets saved to data/raw/")
