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
    """Generate embeddings using sentence transformers with caching and GPU optimization."""
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
    """Compute embeddings with GPU optimization and memory management."""
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
    """Determine optimal batch size based on device and configuration."""
    base_batch_size = model_config.get('batch_size', 64)
    return min(base_batch_size * 2, 128) if device == "cuda" else base_batch_size


def get_embedding_cache_info(texts: List[str], config: Dict) -> Dict:
    """Get information about embedding cache status."""
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
    """Validate that embeddings are properly formed."""
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
    """Get information about available compute devices."""
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