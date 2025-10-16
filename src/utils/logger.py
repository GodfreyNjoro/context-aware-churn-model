"""
Logging configuration
"""
from loguru import logger
import sys

def setup_logger(log_level: str = "INFO"):
    """Setup logger configuration"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level
    )
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        level=log_level
    )
    return logger
