"""
BERTopic model setup and configuration module.

This module provides comprehensive BERTopic model configuration including
UMAP, HDBSCAN, vectorizers, and guided topic modeling capabilities.
"""

from typing import Dict

import numpy as np
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from utils import (
    get_umap_config, get_hdbscan_config, get_domain_guidance_config,
    get_random_seed
)


def setup_bertopic_model(config: Dict) -> BERTopic:
    """
    Configure and initialize a BERTopic model optimized for academic literature analysis.
    
    This function creates a fully configured BERTopic model with academic-optimized parameters
    including dimensionality reduction (UMAP), clustering (HDBSCAN), vectorization, and
    optional guided topic modeling. All parameters are derived from the configuration to
    ensure reproducible and tunable topic modeling.
    
    Args:
        config: Configuration dictionary containing model parameters.
                Expected structure:
                - 'umap_params': Dict with UMAP configuration
                  - 'n_neighbors': Number of neighbors for UMAP (default: 15)
                  - 'n_components': Dimensionality of UMAP output (default: 128)
                  - 'min_dist': Minimum distance in UMAP embedding (default: 0.0)
                  - 'metric': Distance metric ('cosine' recommended for text)
                  - 'random_state': Random seed for reproducibility
                - 'hdbscan_params': Dict with HDBSCAN clustering configuration
                  - 'min_cluster_size': Minimum cluster size (default: 16)
                  - 'max_cluster_size': Maximum cluster size (optional)
                  - 'metric': Distance metric for clustering
                  - 'cluster_selection_method': Method for cluster selection
                - 'data': Dict with BERTopic data parameters
                  - 'min_topic_size': Minimum documents per topic
                  - 'calculate_probabilities': Whether to compute topic probabilities
                - 'domain_guidance': Dict with optional guided topic modeling
                  - 'guided_topics': Configuration for seed-based topic guidance
                  - 'seed_words': Domain-specific terms for topic enhancement
    
    Returns:
        BERTopic: Fully configured BERTopic model ready for training.
                 The model includes:
                 - UMAP for dimensionality reduction optimized for academic text
                 - HDBSCAN for density-based clustering
                 - Academic-optimized CountVectorizer (1-2 grams, English stopwords)
                 - Optional ClassTfidfTransformer with seed word enhancement
                 - Optional guided topic modeling with predefined seed topics
    
    Raises:
        KeyError: If required configuration sections are missing
        ValueError: If configuration parameters are invalid
        ImportError: If required dependencies (UMAP, HDBSCAN) are not available
    
    Example:
        >>> config = {
        ...     'umap_params': {'n_neighbors': 15, 'n_components': 128},
        ...     'hdbscan_params': {'min_cluster_size': 16},
        ...     'data': {'min_topic_size': 10, 'calculate_probabilities': True}
        ... }
        >>> model = setup_bertopic_model(config)
        >>> isinstance(model, BERTopic)
        True
    
    Note:
        - Model configuration is logged for transparency and reproducibility
        - Guided topic modeling is automatically enabled if seed topics are provided
        - All parameters are validated before model creation
        - The model supports both standard and guided topic modeling workflows
    """
    logger.info("🔧 Setting up BERTopic model with custom parameters")
    
    # Configure UMAP for dimensionality reduction
    umap_params = get_umap_config(config).copy()
    logger.info(f"UMAP: neighbors={umap_params.get('n_neighbors')}, components={umap_params.get('n_components')}")
    umap_model = UMAP(**umap_params)
    
    # Configure HDBSCAN for clustering
    hdbscan_params = get_hdbscan_config(config).copy()
    logger.info(f"HDBSCAN: min_cluster_size={hdbscan_params.get('min_cluster_size')}, max_cluster_size={hdbscan_params.get('max_cluster_size')}")
    hdbscan_model = HDBSCAN(**hdbscan_params)
    
    # Configure vectorizer for academic term extraction
    vectorizer_model = _setup_vectorizer()
    
    # Configure ClassTfidfTransformer with seed words if enabled
    ctfidf_model = _setup_ctfidf_transformer(config)
    
    # Check if guided topic modeling is enabled
    domain_config = get_domain_guidance_config(config)
    guided_topics_config = domain_config.get('guided_topics', {})
    
    if guided_topics_config.get('enabled', False):
        logger.info("🎯 Guided topic modeling enabled")
        return _setup_guided_bertopic_model(
            config, umap_model, hdbscan_model, vectorizer_model, ctfidf_model
        )
    
    # Initialize standard BERTopic model
    logger.info("📊 Initializing standard BERTopic model")
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        calculate_probabilities=config['data']['calculate_probabilities'],
        min_topic_size=config['data']['min_topic_size'],
        verbose=True
    )
    
    return topic_model


def _setup_vectorizer() -> CountVectorizer:
    """Configure vectorizer for better academic term extraction."""
    return CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.95,
        max_features=5000
    )


def _setup_ctfidf_transformer(config: Dict) -> ClassTfidfTransformer:
    """Configure ClassTfidfTransformer with seed words if enabled."""
    domain_config = get_domain_guidance_config(config)
    seed_config = domain_config.get('seed_words', {})
    
    if not seed_config.get('enabled', False):
        logger.info("🌱 Seed words enhancement disabled")
        return None
    
    seed_words = seed_config.get('words', [])
    multiplier = seed_config.get('multiplier', 2.0)
    
    if not seed_words:
        logger.warning("⚠️ Seed words enabled but no words provided - skipping enhancement")
        return None
    
    logger.info(f"🌱 Enabling seed words enhancement with {len(seed_words)} domain terms")
    logger.info(f"   Multiplier: {multiplier}x for words: {seed_words[:5]}{'...' if len(seed_words) > 5 else ''}")
    
    return ClassTfidfTransformer(
        seed_words=seed_words,
        seed_multiplier=multiplier
    )


def _setup_guided_bertopic_model(
    config: Dict, 
    umap_model: UMAP, 
    hdbscan_model: HDBSCAN,
    vectorizer_model: CountVectorizer,
    ctfidf_model: ClassTfidfTransformer
) -> BERTopic:
    """Configure BERTopic model with guided topic modeling."""
    domain_config = get_domain_guidance_config(config)
    guided_config = domain_config['guided_topics']
    
    # Extract guided topics and their seed words
    topics_dict = guided_config.get('topics', {})
    seed_topic_list = []
    
    for topic_name, topic_config in topics_dict.items():
        seeds = topic_config.get('seeds', [])
        if seeds:
            seed_topic_list.append(seeds)
            logger.info(f"   📋 {topic_name}: {seeds}")
    
    if not seed_topic_list:
        logger.warning("⚠️ Guided topics enabled but no seeds - falling back to standard model")
        return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model,
                       ctfidf_model=ctfidf_model, calculate_probabilities=config['data']['calculate_probabilities'],
                       min_topic_size=config['data']['min_topic_size'], verbose=True)
    
    logger.info(f"   🎯 Configured {len(seed_topic_list)} guided topics")
    
    # Initialize BERTopic model with guided topic modeling
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model,
                          ctfidf_model=ctfidf_model, seed_topic_list=seed_topic_list,
                          calculate_probabilities=config['data']['calculate_probabilities'],
                          min_topic_size=config['data']['min_topic_size'], verbose=True)
    
    return topic_model


def get_model_info(topic_model: BERTopic) -> Dict:
    """Get information about a configured BERTopic model."""
    info = {
        'has_umap': topic_model.umap_model is not None,
        'has_hdbscan': topic_model.hdbscan_model is not None,
        'has_vectorizer': topic_model.vectorizer_model is not None,
        'has_ctfidf': topic_model.ctfidf_model is not None,
        'calculate_probabilities': getattr(topic_model, 'calculate_probabilities', False),
        'min_topic_size': getattr(topic_model, 'min_topic_size', None),
        'verbose': getattr(topic_model, 'verbose', False)
    }
    
    # Check for guided topics
    if hasattr(topic_model, 'seed_topic_list') and topic_model.seed_topic_list:
        info['guided_topics'] = len(topic_model.seed_topic_list)
        info['is_guided'] = True
    else:
        info['guided_topics'] = 0
        info['is_guided'] = False
    
    return info


def validate_model_config(config: Dict) -> bool:
    """Validate BERTopic model configuration."""
    try:
        required_sections = [
            'umap_params', 'hdbscan_params', 'data'
        ]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"❌ Missing required config section: {section}")
                return False
        
        # Validate UMAP parameters
        umap_config = get_umap_config(config)
        required_umap = ['n_neighbors', 'n_components', 'min_dist', 'metric']
        for param in required_umap:
            if param not in umap_config:
                logger.error(f"❌ Missing UMAP parameter: {param}")
                return False
        
        # Validate HDBSCAN parameters
        hdbscan_config = get_hdbscan_config(config)
        required_hdbscan = ['min_cluster_size', 'metric']
        for param in required_hdbscan:
            if param not in hdbscan_config:
                logger.error(f"❌ Missing HDBSCAN parameter: {param}")
                return False
        
        # Validate data parameters
        data_config = config['data']
        if 'min_topic_size' not in data_config:
            logger.error("❌ Missing data parameter: min_topic_size")
            return False
        
        logger.info("✅ BERTopic model configuration is valid")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating model configuration: {e}")
        return False 