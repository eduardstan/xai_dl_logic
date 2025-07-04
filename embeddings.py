"""
Embeddings module for XAI Deep Learning Logic Bibliography Analysis.

This module provides text embedding generation using sentence transformers
with GPU acceleration and intelligent caching capabilities.
"""

import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from utils import cache_exists, load_from_cache, save_to_cache


def prepare_embeddings(texts: List[str], config: Dict) -> np.ndarray:
    """
    Generate embeddings using sentence transformers with intelligent caching.
    
    This function automatically detects GPU availability and optimizes batch size
    for maximum performance. Results are cached to avoid recomputation.
    
    Args:
        texts: List of text documents to embed
        config: Configuration dictionary with embedding model settings
        
    Returns:
        Numpy array of embeddings with shape (n_documents, embedding_dim)
        
    Raises:
        RuntimeError: If model loading or embedding computation fails
    """
    model_config = config['embedding_model']
    
    # Create embedding cache key based on model and documents
    docs_hash = hashlib.md5(str(texts).encode()).hexdigest()
    model_name = model_config['name']
    cache_dir = Path(config['output']['cache_dir'])
    embedding_cache_name = f"embeddings_{model_name.replace('/', '_')}_{docs_hash[:8]}"
    
    # Check for cached embeddings
    if cache_exists(cache_dir, embedding_cache_name):
        logger.info("🚀 Loading embeddings from cache (super fast!)...")
        embeddings = load_from_cache(cache_dir, embedding_cache_name)
        logger.info(f"✅ Loaded embeddings shape: {embeddings.shape}")
        return embeddings
    
    logger.info(f"🤖 Computing embeddings using {model_config['name']}")
    
    # Generate embeddings with optimized settings
    embeddings = _compute_embeddings(texts, model_config)
    
    # Cache the embeddings for future use
    save_to_cache(embeddings, cache_dir, embedding_cache_name)
    
    logger.info(f"✅ Generated embeddings shape: {embeddings.shape}")
    return embeddings


def _compute_embeddings(texts: List[str], model_config: Dict) -> np.ndarray:
    """
    Compute embeddings with GPU optimization and memory management.
    
    Args:
        texts: List of text documents
        model_config: Model configuration dictionary
        
    Returns:
        Numpy array of embeddings
    """
    # Detect and configure device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = _get_optimal_batch_size(model_config, device)
    
    if device == "cuda":
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("💻 Using CPU")
    
    # Initialize sentence transformer model
    try:
        sentence_model = SentenceTransformer(model_config['name'], device=device)
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_config['name']}: {e}")
        raise RuntimeError(f"Could not load embedding model: {e}")
    
    # Generate embeddings with progress tracking
    logger.info(f"⏳ Computing embeddings for {len(texts):,} documents...")
    if len(texts) > 1000:
        logger.info("   This may take several minutes...")
    
    try:
        embeddings = sentence_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=model_config.get('show_progress', True),
            convert_to_numpy=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to compute embeddings: {e}")
        raise RuntimeError(f"Embedding computation failed: {e}")
    
    # Clean up GPU memory if using CUDA
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("🧹 GPU memory cleared")
    
    return embeddings


def _get_optimal_batch_size(model_config: Dict, device: str) -> int:
    """
    Determine optimal batch size based on device and configuration.
    
    Args:
        model_config: Model configuration dictionary
        device: Device type ('cuda' or 'cpu')
        
    Returns:
        Optimal batch size
    """
    base_batch_size = model_config.get('batch_size', 64)
    
    if device == "cuda":
        # Increase batch size for GPU with memory safety
        return min(base_batch_size * 2, 128)
    else:
        # Use conservative batch size for CPU
        return base_batch_size


def get_embedding_cache_info(texts: List[str], config: Dict) -> Dict:
    """
    Get information about embedding cache status.
    
    Args:
        texts: List of text documents
        config: Configuration dictionary
        
    Returns:
        Dictionary with cache information
    """
    model_config = config['embedding_model']
    docs_hash = hashlib.md5(str(texts).encode()).hexdigest()
    model_name = model_config['name']
    cache_dir = Path(config['output']['cache_dir'])
    embedding_cache_name = f"embeddings_{model_name.replace('/', '_')}_{docs_hash[:8]}"
    
    cache_path = cache_dir / f"{embedding_cache_name}.pkl"
    
    return {
        'cache_name': embedding_cache_name,
        'cache_path': str(cache_path),
        'exists': cache_exists(cache_dir, embedding_cache_name),
        'model_name': model_name,
        'docs_hash': docs_hash[:8],
        'num_documents': len(texts)
    }


def validate_embeddings(embeddings: np.ndarray, texts: List[str]) -> bool:
    """
    Validate that embeddings are properly formed.
    
    Args:
        embeddings: Numpy array of embeddings
        texts: Original text documents
        
    Returns:
        True if embeddings are valid, False otherwise
    """
    try:
        # Check basic properties
        if not isinstance(embeddings, np.ndarray):
            logger.error("❌ Embeddings must be numpy array")
            return False
        
        if len(embeddings) != len(texts):
            logger.error(f"❌ Embeddings length ({len(embeddings)}) != texts length ({len(texts)})")
            return False
        
        if embeddings.ndim != 2:
            logger.error(f"❌ Embeddings must be 2D array, got {embeddings.ndim}D")
            return False
        
        if embeddings.shape[1] == 0:
            logger.error("❌ Embeddings have zero dimensions")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(embeddings)):
            logger.error("❌ Embeddings contain NaN values")
            return False
        
        if np.any(np.isinf(embeddings)):
            logger.error("❌ Embeddings contain infinite values")
            return False
        
        logger.info(f"✅ Embeddings validation passed: {embeddings.shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding validation failed: {e}")
        return False


def get_device_info() -> Dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'memory_allocated': None,
        'memory_reserved': None
    }
    
    if info['cuda_available']:
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['memory_allocated'] = torch.cuda.memory_allocated(0)
        info['memory_reserved'] = torch.cuda.memory_reserved(0)
    
    return info 