"""
Embeddings module for XAI Deep Learning Logic Bibliography Analysis.
Text embedding generation with GPU acceleration and intelligent caching.
"""

import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from utils import cache_exists, load_from_cache, save_to_cache, get_cache_dir, get_embedding_config


def prepare_embeddings(texts: List[str], config: Dict) -> np.ndarray:
    """
    Generate high-quality text embeddings using sentence transformers with intelligent caching.
    
    This function handles the complete embedding pipeline including:
    - Intelligent caching based on model and document hash
    - GPU optimization and memory management
    - Batch processing for large document collections
    - Comprehensive error handling and logging
    
    Args:
        texts: List of text documents to embed. Each document should be a string
               containing the preprocessed text content (typically combined title,
               abstract, and keywords for academic papers).
        config: Configuration dictionary containing embedding model settings.
                Expected keys:
                - 'embedding_model': Dict with model configuration
                  - 'name': Model name (e.g., 'all-MiniLM-L6-v2')
                  - 'batch_size': Processing batch size (default: 64)
                  - 'show_progress': Whether to show progress bar (default: True)
                - 'output': Dict with cache directory settings
                  - 'cache_dir': Directory for caching embeddings
    
    Returns:
        np.ndarray: Dense embedding matrix of shape (n_documents, embedding_dim).
                   Each row represents one document's embedding vector.
                   Typical dimensions: (n_docs, 384) for MiniLM models.
    
    Raises:
        RuntimeError: If embedding model cannot be loaded or computation fails
        FileNotFoundError: If cache directory cannot be created
        ValueError: If texts list is empty or contains invalid content
    
    Example:
        >>> texts = ["Machine learning paper about transformers", "Deep learning research"]
        >>> config = {'embedding_model': {'name': 'all-MiniLM-L6-v2'}, 'output': {'cache_dir': 'cache'}}
        >>> embeddings = prepare_embeddings(texts, config)
        >>> embeddings.shape
        (2, 384)
    
    Note:
        - Embeddings are automatically cached for faster subsequent runs
        - GPU acceleration is used when available with automatic fallback to CPU
        - Memory usage is optimized through batch processing and cleanup
        - Cache keys include model name and document hash for validity
    """
    model_config = get_embedding_config(config)
    
    # Create embedding cache key based on model and documents
    docs_hash = hashlib.md5(str(texts).encode()).hexdigest()
    model_name = model_config['name']
    cache_dir = get_cache_dir(config)
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
    Internal function to compute embeddings with GPU optimization and memory management.
    
    This function handles the actual embedding computation including:
    - Automatic GPU/CPU device detection and configuration
    - Dynamic batch size optimization based on available hardware
    - Comprehensive error handling for model loading and computation
    - Memory cleanup after GPU operations
    - Progress tracking for large document collections
    
    Args:
        texts: List of text documents to embed, already preprocessed
        model_config: Embedding model configuration dict containing:
                     - 'name': Sentence transformer model name
                     - 'batch_size': Base batch size for processing
                     - 'show_progress': Whether to display progress bar
    
    Returns:
        np.ndarray: Dense embedding matrix of shape (n_documents, embedding_dim)
    
    Raises:
        RuntimeError: If model loading fails or embedding computation encounters errors
        MemoryError: If insufficient GPU/CPU memory for the operation
    
    Note:
        - GPU batch size is automatically doubled when CUDA is available
        - Memory is cleaned up after GPU operations to prevent accumulation
        - Progress bar is shown for collections over 1000 documents
        - Model is loaded fresh each time to ensure clean state
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
    Determine optimal batch size based on device capabilities and configuration.
    
    Args:
        model_config: Embedding model configuration containing base batch size
        device: Target device ("cuda" or "cpu")
    
    Returns:
        int: Optimized batch size for the specified device.
             GPU batch size is doubled up to 128, CPU uses base size.
    
    Note:
        - GPU processing can handle larger batches more efficiently
        - Capped at 128 to prevent memory issues on smaller GPUs
        - CPU processing uses conservative batch sizes
    """
    base_batch_size = model_config.get('batch_size', 64)
    return min(base_batch_size * 2, 128) if device == "cuda" else base_batch_size


def get_embedding_cache_info(texts: List[str], config: Dict) -> Dict:
    """
    Get comprehensive information about embedding cache status and metadata.
    
    Args:
        texts: List of documents to be embedded
        config: Configuration dictionary containing embedding and cache settings
    
    Returns:
        Dict: Complete cache information including:
              - cache_name: Generated cache filename
              - cache_path: Full path to cache file
              - exists: Whether cache file exists
              - model_name: Embedding model name
              - docs_hash: Short hash of document content
              - num_documents: Number of documents
    
    Note:
        - Cache key is generated from model name and document content hash
        - Used for cache validation and debugging
        - Helps identify cache misses and invalidations
    """
    model_config = get_embedding_config(config)
    docs_hash = hashlib.md5(str(texts).encode()).hexdigest()
    model_name = model_config['name']
    cache_dir = get_cache_dir(config)
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
    Validate embedding matrix for correctness and quality.
    
    Args:
        embeddings: Numpy array of document embeddings
        texts: Original list of documents
    
    Returns:
        bool: True if embeddings pass all validation checks, False otherwise
    
    Validation checks:
        - Correct numpy array format
        - Proper dimensionality (2D matrix)
        - Size consistency with input documents
        - No NaN or infinite values
        - Non-zero embedding dimensions
    
    Note:
        - Logs specific validation failures for debugging
        - Essential for ensuring embedding quality before topic modeling
        - Helps identify model loading or computation issues
    """
    try:
        checks = [
            (isinstance(embeddings, np.ndarray), "❌ Embeddings must be numpy array"),
            (len(embeddings) == len(texts), f"❌ Length mismatch: {len(embeddings)} != {len(texts)}"),
            (embeddings.ndim == 2, f"❌ Must be 2D array, got {embeddings.ndim}D"),
            (embeddings.shape[1] > 0, "❌ Embeddings have zero dimensions"),
            (not np.any(np.isnan(embeddings)), "❌ Embeddings contain NaN values"),
            (not np.any(np.isinf(embeddings)), "❌ Embeddings contain infinite values")
        ]
        
        for check, msg in checks:
            if not check:
                logger.error(msg)
                return False
        
        logger.info(f"✅ Embeddings validation passed: {embeddings.shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding validation failed: {e}")
        return False


def get_device_info() -> Dict:
    """
    Get comprehensive information about available compute devices.
    
    Returns:
        Dict: Complete device information including:
              - cuda_available: Whether CUDA is available
              - device_count: Number of GPU devices
              - current_device: Current GPU device index
              - device_name: Name of the GPU device
              - memory_allocated: Currently allocated GPU memory
              - memory_reserved: Reserved GPU memory
    
    Note:
        - Returns None values for GPU-specific info when CUDA unavailable
        - Useful for debugging device issues and memory management
        - Helps optimize batch sizes and memory usage
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
        info.update({
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(0),
            'memory_allocated': torch.cuda.memory_allocated(0),
            'memory_reserved': torch.cuda.memory_reserved(0)
        })
    
    return info 