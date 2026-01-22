#!/usr/bin/env python3
"""
Script to generate synthetic churn dataset.

Creates realistic user demographics and customer interaction
data with correlated churn labels for model training and testing.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --users 5000 --interactions 100000
    python scripts/generate_data.py --seed 123 --output-dir data/custom
"""
import argparse
import logging
import sys
from pathlib import Path

# add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_CONFIG, LOGGING_CONFIG
from src.data.data_generator import ChurnDataGenerator

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


def main(
    n_users: int,
    n_interactions: int,
    random_seed: int,
    output_dir: str
) -> None:
    """
    Generate synthetic churn dataset.
    
    Args:
        n_users: Number of users to generate
        n_interactions: Number of interactions to generate
        random_seed: Random seed for reproducibility
        output_dir: Directory to save generated data
    """
    # ----------------------------------------
    # header
    # ----------------------------------------
    logger.info("=" * 60)
    logger.info("SYNTHETIC CHURN DATA GENERATION")
    logger.info("=" * 60)
    logger.info(f"Users: {n_users:,}")
    logger.info(f"Interactions: {n_interactions:,}")
    logger.info(f"Random seed: {random_seed}")
    logger.info(f"Output directory: {output_dir}")
    
    # ----------------------------------------
    # generate data
    # ----------------------------------------
    generator = ChurnDataGenerator(
        n_users=n_users,
        n_interactions=n_interactions,
        random_state=random_seed
    )
    users, interactions = generator.generate_dataset()
    
    # ----------------------------------------
    # save datasets
    # ----------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    users_file = output_path / "users.csv"
    interactions_file = output_path / "interactions.csv"
    
    users.to_csv(users_file, index=False)
    interactions.to_csv(interactions_file, index=False)
    
    # ----------------------------------------
    # summary
    # ----------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("DATA GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Users file: {users_file}")
    logger.info(f"  Interactions file: {interactions_file}")
    logger.info(f"  Total users: {len(users):,}")
    logger.info(f"  Total interactions: {len(interactions):,}")
    logger.info(f"  Churn rate: {users['churned'].mean():.1%}")
    logger.info("\n✓ Data generation finished successfully!")


# ============================================
# Command-line interface
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic churn dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_data.py
  python scripts/generate_data.py --users 5000 --interactions 100000
  python scripts/generate_data.py --seed 123
        """
    )
    
    parser.add_argument(
        "--users",
        type=int,
        default=DATA_CONFIG.get("n_users", 10000),
        help=f"Number of users to generate (default: {DATA_CONFIG.get('n_users', 10000)})"
    )
    parser.add_argument(
        "--interactions",
        type=int,
        default=DATA_CONFIG.get("n_interactions", 200000),
        help=f"Number of interactions to generate (default: {DATA_CONFIG.get('n_interactions', 200000)})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(DATA_CONFIG.get("users_file", "data/raw/users.csv")).parent),
        help="Output directory for generated data"
    )
    
    args = parser.parse_args()
    
    main(
        n_users=args.users,
        n_interactions=args.interactions,
        random_seed=args.seed,
        output_dir=args.output_dir
    )
