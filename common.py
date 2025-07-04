#!/usr/bin/env python3
"""
Common utilities for XAI Deep Learning Logic analysis pipeline.
Centralizes shared functionality to eliminate code duplication.
"""

import pickle
import hashlib
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "10 MB",
    retention: str = "1 week"
) -> None:
    """
    Configure loguru logging with structured format and optional file output.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for persistent logging
        rotation: Log rotation size (e.g., "10 MB", "1 GB")
        retention: Log retention period (e.g., "1 week", "30 days")
    """
    # Remove default handler to avoid duplicate logs
    logger.remove()
    
    # Console handler with colors and emojis
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True
    )
    
    # File handler with rotation if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
        logger.info(f"📄 Logging to file: {log_path}")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with comprehensive error handling.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Dictionary containing configuration settings
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration file {config_path}: {e}")


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Generate MD5 hash of a file for cache validation.
    
    Args:
        file_path: Path to the file to hash
        
    Returns:
        MD5 hash string of the file contents
    """
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"❌ Error hashing file {file_path}: {e}")
        raise


def cache_exists(cache_dir: Path, cache_name: str) -> bool:
    """
    Check if a cache file exists.
    
    Args:
        cache_dir: Directory containing cache files
        cache_name: Name of the cache file (without .pkl extension)
        
    Returns:
        True if cache file exists, False otherwise
    """
    cache_file = cache_dir / f"{cache_name}.pkl"
    return cache_file.exists()


def save_to_cache(
    data: Any, 
    cache_dir: Path, 
    cache_name: str, 
    compress: bool = True
) -> None:
    """
    Save data to cache using pickle.
    
    Args:
        data: Data to cache (any pickle-serializable object)
        cache_dir: Directory to store cache files
        cache_name: Name of the cache file (without .pkl extension)
        compress: Whether to use highest compression protocol
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_name}.pkl"
        
        protocol = pickle.HIGHEST_PROTOCOL if compress else pickle.DEFAULT_PROTOCOL
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
        
        # Log cache info
        file_size = cache_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        logger.info(f"💾 Cached {cache_name} ({size_mb:.1f} MB)")
        
    except Exception as e:
        logger.error(f"❌ Error saving to cache {cache_name}: {e}")
        raise


def load_from_cache(cache_dir: Path, cache_name: str) -> Any:
    """
    Load data from cache using pickle.
    
    Args:
        cache_dir: Directory containing cache files
        cache_name: Name of the cache file (without .pkl extension)
        
    Returns:
        Loaded data object
    """
    try:
        cache_file = cache_dir / f"{cache_name}.pkl"
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        # Log cache info
        file_size = cache_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        logger.info(f"📂 Loaded {cache_name} from cache ({size_mb:.1f} MB)")
        return data
        
    except Exception as e:
        logger.error(f"❌ Error loading from cache {cache_name}: {e}")
        raise


def create_timestamped_filename(prefix: str, extension: str = "csv") -> str:
    """
    Create a timestamped filename for output files.
    
    Args:
        prefix: Prefix for the filename
        extension: File extension (without dot)
        
    Returns:
        Timestamped filename string
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def ensure_output_directories(base_dir: Union[str, Path], subdirs: list = None) -> Path:
    """
    Ensure output directories exist, creating them if necessary.
    
    Args:
        base_dir: Base output directory
        subdirs: List of subdirectories to create
        
    Returns:
        Path object for the base directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    if subdirs:
        for subdir in subdirs:
            (base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Output directories ready: {base_path}")
    return base_path


def validate_file_exists(file_path: Union[str, Path], description: str = "File") -> Path:
    """
    Validate that a file exists and is readable.
    
    Args:
        file_path: Path to the file to validate
        description: Description of the file for error messages
        
    Returns:
        Path object for the validated file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file isn't readable
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {file_path}")
    
    if not path.stat().st_size > 0:
        raise ValueError(f"{description} is empty: {file_path}")
    
    logger.info(f"✅ {description} validated: {file_path}")
    return path
