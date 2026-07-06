# jarvis/utils/logging_utils.py

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Optional logging level override
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level from parameter, environment, or default to INFO
    log_level = (level or "INFO").upper()
    logger.setLevel(log_level)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = RotatingFileHandler(
            log_dir / f"{name.split('.')[-1]}.log",
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logger.addHandler(file_handler)
    
    return logger