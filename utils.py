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


# === Config Helper Functions ===

def get_cache_dir(config: Dict[str, Any]) -> Path:
    """Get cache directory path from config."""
    return Path(config['output']['cache_dir'])


def get_results_dir(config: Dict[str, Any]) -> Path:
    """Get results directory path from config."""
    return Path(config['output']['results_dir'])


def get_models_dir(config: Dict[str, Any]) -> Path:
    """Get models directory path from config."""
    return Path(config['output']['models_dir'])


def get_plots_dir(config: Dict[str, Any]) -> Path:
    """Get plots directory path from config."""
    return Path(config['output']['plots_dir'])


def get_embedding_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get embedding model configuration section."""
    return config['embedding_model']


def get_umap_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get UMAP parameters configuration section."""
    return config['umap_params']


def get_hdbscan_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get HDBSCAN parameters configuration section."""
    return config['hdbscan_params']


def get_visualization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get visualization configuration section."""
    return config['visualization']


def get_systematic_review_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get systematic review configuration section."""
    return config['systematic_review']


def get_outlier_reduction_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get outlier reduction configuration section."""
    return config.get('outlier_reduction', {})


def get_domain_guidance_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get domain guidance configuration section."""
    return config.get('domain_guidance', {})


def get_random_seed(config: Dict[str, Any]) -> int:
    """Get random seed for reproducibility."""
    return config.get('random_seed', 42)


def get_timestamp_format(config: Dict[str, Any]) -> str:
    """Get timestamp format string."""
    return config.get('timestamp_format', '%Y%m%d_%H%M%S')


def get_output_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get output configuration section from config."""
    return config.get('output', {})


def ensure_output_dirs(config: Dict[str, Any], base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
    """
    Ensure all output directories exist and return path dictionary.
    
    Args:
        config: Configuration dictionary
        base_dir: Optional base directory to prepend to all paths
        
    Returns:
        Dictionary mapping directory names to Path objects
    """
    base_path = Path(base_dir) if base_dir else Path.cwd()
    
    dirs = {
        'cache': get_cache_dir(config),
        'results': get_results_dir(config),
        'models': get_models_dir(config),
        'plots': get_plots_dir(config)
    }
    
    # Create directories if they don't exist
    for name, path in dirs.items():
        full_path = base_path / path
        full_path.mkdir(parents=True, exist_ok=True)
        dirs[name] = full_path
    
    logger.info(f"📁 Output directories ready: {', '.join(dirs.keys())}")
    return dirs


def validate_config_sections(config: Dict[str, Any], required_sections: list) -> None:
    """
    Validate that required configuration sections exist.
    
    Args:
        config: Configuration dictionary
        required_sections: List of required section names
        
    Raises:
        ValueError: If required sections are missing
    """
    missing_sections = []
    
    for section in required_sections:
        if '.' in section:
            # Handle nested sections like 'output.cache_dir'
            keys = section.split('.')
            current = config
            try:
                for key in keys:
                    current = current[key]
            except (KeyError, TypeError):
                missing_sections.append(section)
        else:
            # Handle top-level sections
            if section not in config:
                missing_sections.append(section)
    
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {', '.join(missing_sections)}")
    
    logger.info(f"✅ Configuration validation passed for: {', '.join(required_sections)}")


# === End Config Helper Functions ===


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
