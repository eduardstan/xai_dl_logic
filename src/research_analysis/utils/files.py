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


def cache_exists(cache_dir: Path, cache_name: str) -> bool:
    """
    Check if a given cache file exists.

    Args:
        cache_dir: The directory where caches are stored.
        cache_name: The name of the specific cache file.

    Returns:
        True if the cache file exists, False otherwise.
    """
    return (cache_dir / f"{cache_name}.pkl").exists()


def save_to_cache(
    data: Any, cache_dir: Path, cache_name: str, compress: bool = True
) -> None:
    """
    Save data to a pickle file in the specified cache directory.

    Args:
        data: The Python object to cache.
        cache_dir: The directory where the cache will be stored.
        cache_name: The name for the cache file (without extension).
        compress: Whether to use pickle's compression.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_name}.pkl"
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL if compress else 4)
        logger.info(f"💾 Saved {cache_name} to cache ({cache_path.stat().st_size / 1e6:.2f} MB)")
    except Exception as e:
        logger.error(f"❌ Failed to save {cache_name} to cache: {e}")
        raise


def load_from_cache(cache_dir: Path, cache_name: str) -> Any:
    """
    Load data from a pickle file in the specified cache directory.

    Args:
        cache_dir: The directory where caches are stored.
        cache_name: The name of the cache file to load.

    Returns:
        The deserialized Python object from the cache.
    """
    cache_path = cache_dir / f"{cache_name}.pkl"
    if not cache_path.exists():
        logger.warning(f"Cache file not found: {cache_path}")
        return None

    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"📂 Loaded {cache_name} from cache ({cache_path.stat().st_size / 1e6:.2f} MB)")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to load {cache_name} from cache: {e}")
        raise 