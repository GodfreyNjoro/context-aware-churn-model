#!/usr/bin/env python3
"""
Script to generate synthetic churn dataset
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_generator import ChurnDataGenerator

def main():
    print("=" * 60)
    print("Generating Synthetic Churn Dataset")
    print("=" * 60)
    
    # Generate data
    generator = ChurnDataGenerator(n_users=10000, n_interactions=200000)
    users, interactions = generator.generate_dataset()
    
    # Save datasets
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    users.to_csv(output_dir / "users.csv", index=False)
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    
    print(f"\nDatasets saved to {output_dir}")
    print("✓ Data generation complete!")

if __name__ == "__main__":
    main()
