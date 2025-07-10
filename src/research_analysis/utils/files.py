#!/usr/bin/env python3
"""
File and cache-related utilities.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Union

from loguru import logger


def get_file_hash(file_path: Union[str, Path]) -> str:
    """
    Computes the SHA256 hash of a file for integrity checking.

    Args:
        file_path: Path to the file.

    Returns:
        The hex digest of the file's hash.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_cache_path(cache_dir: Union[str, Path], file_name: str) -> Path:
    """
    Constructs the full path for a cache file.

    Args:
        cache_dir: The directory where caches are stored.
        file_name: The name of the cache file (e.g., 'parsed_bib.pkl').

    Returns:
        The full Path object for the cache file.
    """
    return Path(cache_dir) / file_name


def cache_exists(cache_path: Path) -> bool:
    """
    Check if a given cache file exists.

    Args:
        cache_path: The full path to the cache file.

    Returns:
        True if the cache file exists, False otherwise.
    """
    return cache_path.exists()


def save_to_cache(data: Any, cache_path: Path, compress: bool = True) -> None:
    """
    Save data to a pickle file at the specified path.

    Args:
        data: The Python object to cache.
        cache_path: The full path where the cache will be stored.
        compress: Whether to use pickle's compression.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL if compress else 4)
        logger.info(f"Saved to cache: {cache_path} ({cache_path.stat().st_size / 1e6:.2f} MB)")
    except Exception as e:
        logger.error(f"Failed to save to cache: {cache_path}: {e}")
        raise


def load_from_cache(cache_path: Path) -> Any:
    """
    Load data from a pickle file from the specified path.

    Args:
        cache_path: The full path of the cache file to load.

    Returns:
        The deserialized Python object from the cache.
    """
    if not cache_path.exists():
        logger.warning(f"Cache file not found: {cache_path}")
        return None

    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded from cache: {cache_path} ({cache_path.stat().st_size / 1e6:.2f} MB)")
        return data
    except Exception as e:
        logger.error(f"Failed to load from cache: {cache_path}: {e}")
        raise 