"""
Logging utilities for sports data processing.

Provides structured logging with organized directory structure:
- logs/{sport}/scraping/{type}/
- logs/{sport}/fitting/
- logs/{sport}/simulation/
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_timestamp() -> str:
    """Get current timestamp in format YYYY-MM-DD_HH."""
    return datetime.now().strftime("%Y-%m-%d_%H")


def get_log_base_dir() -> Path:
    """Get base log directory (project root / logs)."""
    script_dir = Path(__file__).resolve().parent.parent.parent
    return script_dir / "logs"


def get_scraping_log_path(sport: str, scraper_type: str) -> Path:
    """
    Get log path for scraping operations.
    
    Args:
        sport: Sport name (e.g., 'ncaab', 'nba', 'wnba')
        scraper_type: Type of scraper ('game', 'pbp', 'player', 'schedule')
    
    Returns:
        Path to log file: logs/{sport}/scraping/{scraper_type}/{scraper_type}_{timestamp}.log
    """
    base_dir = get_log_base_dir()
    log_dir = base_dir / sport / "scraping" / scraper_type
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp()
    log_file = log_dir / f"{scraper_type}_{timestamp}.log"
    return log_file


def get_fitting_log_path(sport: str, model_name: str) -> Path:
    """
    Get log path for model fitting operations.
    
    Args:
        sport: Sport name (e.g., 'ncaab', 'nba', 'wnba')
        model_name: Name of the model (e.g., 'Vanilla', 'TeamVol')
    
    Returns:
        Path to log file: logs/{sport}/fitting/fitting_{model_name}_{timestamp}.log
    """
    base_dir = get_log_base_dir()
    log_dir = base_dir / sport / "fitting"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp()
    log_file = log_dir / f"fitting_{model_name}_{timestamp}.log"
    return log_file


def get_simulation_log_path(sport: str, model_name: str) -> Path:
    """
    Get log path for simulation operations.
    
    Args:
        sport: Sport name (e.g., 'ncaab', 'nba', 'wnba')
        model_name: Name of the model (e.g., 'Vanilla', 'TeamVol')
    
    Returns:
        Path to log file: logs/{sport}/simulation/simulation_{model_name}_{timestamp}.log
    """
    base_dir = get_log_base_dir()
    log_dir = base_dir / sport / "simulation"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_timestamp()
    log_file = log_dir / f"simulation_{model_name}_{timestamp}.log"
    return log_file


def setup_logger(log_file: Path, logger_name: Optional[str] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        log_file: Path to log file
        logger_name: Name for the logger (default: root logger)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def setup_scraping_logger(sport: str, scraper_type: str, 
                          level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger for scraping operations.
    
    Args:
        sport: Sport name (e.g., 'ncaab', 'nba', 'wnba')
        scraper_type: Type of scraper ('game', 'pbp', 'player', 'schedule')
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    log_file = get_scraping_log_path(sport, scraper_type)
    return setup_logger(log_file, level=level)


def setup_fitting_logger(sport: str, model_name: str, 
                         level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger for model fitting operations.
    
    Args:
        sport: Sport name (e.g., 'ncaab', 'nba', 'wnba')
        model_name: Name of the model (e.g., 'Vanilla', 'TeamVol')
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    log_file = get_fitting_log_path(sport, model_name)
    return setup_logger(log_file, level=level)


def setup_simulation_logger(sport: str, model_name: str, 
                            level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger for simulation operations.
    
    Args:
        sport: Sport name (e.g., 'ncaab', 'nba', 'wnba')
        model_name: Name of the model (e.g., 'Vanilla', 'TeamVol')
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    log_file = get_simulation_log_path(sport, model_name)
    return setup_logger(log_file, level=level)
