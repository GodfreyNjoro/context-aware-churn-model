"""
Logging configuration utilities for the churn prediction system.

Provides centralized logging setup with dual handlers (console + file)
and customizable formatting for consistent logging across all modules.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

# optional loguru dependency (for enhanced logging)
try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False


def setup_logger(
    name: str = "churn_model",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    use_loguru: bool = False
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Creates a logger with consistent formatting across the application.
    Supports both standard logging and loguru (if available).
    
    Args:
        name: Logger name (typically __name__ of calling module)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        use_loguru: Use loguru instead of standard logging (if available)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(__name__, "DEBUG", "logs/app.log")
        >>> logger.info("Application started")
    """
    # use loguru if requested and available
    if use_loguru and LOGURU_AVAILABLE:
        return _setup_loguru_logger(log_level, log_file)
    
    # standard library logging
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # ----------------------------------------
    # log format with timestamp and context
    # ----------------------------------------
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # ----------------------------------------
    # console handler (stdout)
    # ----------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # ----------------------------------------
    # file handler (optional)
    # ----------------------------------------
    if log_file:
        # ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger


def _setup_loguru_logger(log_level: str, log_file: Optional[str]):
    """
    Configure loguru logger with enhanced formatting.
    
    Args:
        log_level: Logging level string
        log_file: Optional path to log file
        
    Returns:
        Configured loguru logger
    """
    if not LOGURU_AVAILABLE:
        raise ImportError("loguru is not installed. pip install loguru")
    
    # remove default handler
    loguru_logger.remove()
    
    # console handler with colors
    loguru_logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level.upper(),
        colorize=True,
    )
    
    # file handler with rotation
    if log_file:
        loguru_logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=log_level.upper(),
            rotation="500 MB",
            retention="10 days",
            compression="zip",
        )
    
    return loguru_logger


def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Simple wrapper for getting a logger with the module's name.
    Use setup_logger() first to configure handlers.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================
# Convenience function for quick setup
# ============================================
def configure_root_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure the root logger for the entire application.
    
    Call this once at application startup to set up logging
    for all modules that use the standard logging module.
    
    Args:
        log_level: Logging level string
        log_file: Optional path to log file
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # override any existing configuration
    )
