#!/usr/bin/env python3
"""
Embeddings module for the research analysis framework.
Handles text embedding generation with GPU acceleration and caching.
"""

from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from research_analysis.config.models import AppConfig
from research_analysis.utils.files import (
    cache_exists,
    get_cache_path,
    load_from_cache,
    save_to_cache,
)
from research_analysis.utils.logging import get_logger

logger = get_logger()


def prepare_embeddings(docs: List[str], config: AppConfig) -> np.ndarray:
    """
    Generate or load cached text embeddings using sentence transformers.

    This function checks for a cached version of the embeddings first.
    If not found, it computes them using a specified sentence-transformer
    model, with GPU acceleration if available, and then saves them to the cache.

    Args:
        docs: A list of text documents to embed.
        config: The application's configuration object.

    Returns:
        A numpy array containing the document embeddings.

    Raises:
        RuntimeError: If the embedding model fails to load or computation fails.
    """
    model_name = config.stage_1.embedding_model.name
    cache_file_name = f"embeddings_{model_name.replace('/', '_')}.pkl"
    cache_path = get_cache_path(
        cache_dir=config.pipeline.paths.cache,
        file_name=cache_file_name,
    )

    if cache_exists(cache_path):
        logger.info(f"Loading embeddings from cache: {cache_path}")
        embeddings = load_from_cache(cache_path)
        if embeddings is not None and isinstance(embeddings, np.ndarray):
            logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
            return embeddings
        logger.warning("Cache file was found but contained no data. Recomputing.")

    logger.info(f"No valid cache found. Computing embeddings using {model_name}.")
    embeddings = _compute_embeddings(docs, config.stage_1.embedding_model.dict())

    logger.info(f"Saving new embeddings to cache: {cache_path}")
    save_to_cache(embeddings, cache_path)

    return embeddings


def _compute_embeddings(
    docs: List[str], model_config: Dict
) -> np.ndarray:
    """
    Internal function to compute embeddings with GPU optimization.

    Args:
        docs: A list of text documents to embed.
        model_config: A dictionary with embedding model settings.

    Returns:
        A numpy array of the computed embeddings.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = model_config.get("batch_size", 64)
    if device == "cuda":
        batch_size *= 2
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} with batch size {batch_size}")
    else:
        logger.info(f"Using CPU with batch size {batch_size}")

    try:
        model = SentenceTransformer(model_config["name"], device=device)
        embeddings = model.encode(
            docs,
            batch_size=batch_size,
            show_progress_bar=model_config.get("show_progress", True),
            convert_to_numpy=True,
        )
    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}", exc_info=True)
        raise RuntimeError("Embedding computation failed") from e

    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory cache.")

    logger.info(f"Successfully computed embeddings with shape: {embeddings.shape}")
    return embeddings 