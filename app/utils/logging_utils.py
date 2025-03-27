"""
Logging utilities for SagaX1
"""

import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """Set up logging for the application
    
    Args:
        log_level: Logging level
        log_dir: Directory for log files
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler for debug logs
    debug_file_handler = RotatingFileHandler(
        os.path.join(log_dir, "debug.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    debug_file_handler.setFormatter(debug_file_formatter)
    root_logger.addHandler(debug_file_handler)
    
    # File handler for error logs
    error_file_handler = RotatingFileHandler(
        os.path.join(log_dir, "error.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    error_file_handler.setFormatter(error_file_formatter)
    root_logger.addHandler(error_file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Create application logger
    logger = logging.getLogger("sagax1")
    logger.info("Logging initialized")
    
    return logger