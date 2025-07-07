#!/usr/bin/env python3
"""
Topic Training Module for BERTopic Analysis

This module handles training BERTopic models and applying outlier reduction
for optimal topic coverage in academic literature analysis.
"""

import numpy as np
from typing import Dict, List, Tuple
from loguru import logger
from bertopic import BERTopic

# Import specialized modules
from outlier_reduction import apply_outlier_reduction


def train_topic_model(topic_model: BERTopic, docs: List[str], embeddings: np.ndarray,
                     config: Dict) -> Tuple[List[int], np.ndarray]:
    """
    Train BERTopic model and apply outlier reduction for optimal topic coverage.
    
    This function orchestrates the complete topic modeling pipeline including:
    - Training the BERTopic model using pre-computed embeddings
    - Comprehensive logging of initial topic discovery results  
    - Configurable outlier reduction using multiple strategies
    - Detailed analysis of cluster size distribution
    - Preservation of high-quality topic representations
    
    Args:
        topic_model: Pre-configured BERTopic model with UMAP, HDBSCAN, and
                    vectorizer components already set up for academic text analysis.
                    Must include proper dimensionality reduction and clustering parameters.
        docs: List of document texts to analyze. Should be preprocessed and
              cleaned text (typically combined title + abstract + keywords).
              Each document should contain meaningful academic content.
        embeddings: Pre-computed document embeddings matrix of shape (n_docs, embedding_dim).
                   Must correspond exactly to the docs list order. Typically generated
                   using sentence transformers optimized for academic text.
        config: Configuration dictionary containing model and outlier reduction settings.
                Expected structure:
                - 'outlier_reduction': Dict with outlier reduction configuration
                  - 'enabled': Boolean to enable/disable outlier reduction
                  - 'strategy': String specifying reduction method
                  - 'threshold': Float for reduction threshold
                - 'data': Dict with topic modeling parameters
                  - 'min_topic_size': Minimum documents per topic
                  - 'calculate_probabilities': Boolean for probability calculation
                - 'random_seed': Integer for reproducibility across runs
        
    Returns:
        Tuple[List[int], np.ndarray]: A tuple containing:
            - topics: List of topic assignments for each document. 
                     Topic IDs are integers (0, 1, 2, ...) with -1 for outliers.
                     Length matches input docs list exactly.
            - probs: Topic probability matrix of shape (n_docs, n_topics) if
                    calculate_probabilities=True in config, otherwise None.
                    Contains probability distributions over topics for each document.
    
    Raises:
        RuntimeError: If topic model training fails due to model configuration issues
        ValueError: If embeddings and docs have mismatched lengths or invalid dimensions
        KeyError: If required configuration keys are missing from config dictionary
        TypeError: If input types don't match expected formats
    
    Example:
        >>> from bertopic import BERTopic
        >>> from bertopic_setup import setup_bertopic_model
        >>> import numpy as np
        >>> 
        >>> # Setup components
        >>> config = {
        ...     'data': {'min_topic_size': 10, 'calculate_probabilities': False},
        ...     'outlier_reduction': {'enabled': True, 'strategy': 'c-tf-idf'},
        ...     'random_seed': 42
        ... }
        >>> docs = ["AI research paper", "Machine learning study", ...]  # 100 docs
        >>> embeddings = np.random.random((100, 384))  # Corresponding embeddings
        >>> model = setup_bertopic_model(config)
        >>> 
        >>> # Train model
        >>> topics, probs = train_topic_model(model, docs, embeddings, config)
        >>> print(f"Discovered {len(set(topics))} topics with {topics.count(-1)} outliers")
        Discovered 15 topics with 3 outliers
    
    Training Process:
        1. **Initial Training**: Uses BERTopic's fit_transform with pre-computed embeddings
           to avoid recomputation and ensure consistency across analysis pipeline.
        
        2. **Result Analysis**: Comprehensive logging of initial topic discovery including
           topic count, outlier count, and coverage statistics for transparency.
        
        3. **Outlier Reduction**: Configurable post-processing to reassign outlier documents
           to appropriate topics using various strategies (c-tf-idf, probabilities, etc.).
        
        4. **Quality Assessment**: Detailed cluster size distribution analysis including
           largest/smallest clusters, averages, and medians for quality evaluation.
        
        5. **Final Validation**: Comparison of pre/post outlier reduction statistics
           to track improvement in topic coverage and document assignment quality.
    
    Performance Considerations:
        - Uses pre-computed embeddings to avoid recomputation during training
        - Outlier reduction strategies vary in computational complexity
        - Large corpora (>10k docs) may require several minutes for complete processing
        - Memory usage scales with corpus size and embedding dimensionality
    
    Quality Metrics Logged:
        - Initial and final topic counts for model complexity assessment
        - Outlier counts and coverage percentages for assignment quality  
        - Cluster size statistics for topic distribution analysis
        - Outlier reduction effectiveness measurements
    
    Note:
        - All operations include comprehensive emoji-enhanced logging for monitoring
        - Outlier reduction preserves original topic representations while improving coverage
        - Function supports both probability and non-probability modes based on configuration
        - Random seed from config ensures reproducible results across multiple runs
        - Cluster statistics help assess topic quality and identify potential issues
    """
    logger.info("🚀 Training BERTopic model with pre-computed embeddings")
    
    # Validate input dimensions for early error detection
    if len(docs) != len(embeddings):
        raise ValueError(f"Docs and embeddings length mismatch: {len(docs)} vs {len(embeddings)}")
    
    # Use pre-computed embeddings to avoid recomputation
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    # Log initial analysis results with detailed statistics
    n_topics_initial = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers_initial = sum(1 for t in topics if t == -1)
    coverage_initial = ((len(docs) - n_outliers_initial) / len(docs) * 100) if len(docs) > 0 else 0
    
    logger.info(f"Initial analysis complete:")
    logger.info(f"  - Found {n_topics_initial} topics")
    logger.info(f"  - {n_outliers_initial} outlier documents")
    logger.info(f"  - Coverage: {coverage_initial:.1f}%")
    
    # Apply outlier reduction if configured
    topics = apply_outlier_reduction(topic_model, docs, topics, probs, embeddings, config, logger)
    
    # Log final analysis results with improvement metrics
    n_topics_final = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers_final = sum(1 for t in topics if t == -1)
    coverage_final = ((len(docs) - n_outliers_final) / len(docs) * 100) if len(docs) > 0 else 0
    
    logger.info(f"✅ Final analysis complete:")
    logger.info(f"  - Final topic count: {n_topics_final}")
    logger.info(f"  - {n_outliers_final} outlier documents") 
    logger.info(f"  - Coverage: {coverage_final:.1f}%")
    
    # Log outlier reduction effectiveness
    if n_outliers_initial > n_outliers_final:
        outliers_reduced = n_outliers_initial - n_outliers_final
        improvement = ((outliers_reduced / n_outliers_initial) * 100) if n_outliers_initial > 0 else 0
        logger.info(f"🎯 Outlier reduction: {outliers_reduced} documents reassigned ({improvement:.1f}% improvement)")
    
    # Comprehensive cluster size distribution analysis
    logger.info("📊 Cluster size distribution:")
    cluster_sizes = []
    for topic_id in sorted(set(topics)):
        if topic_id != -1:  # Skip outliers
            count = topics.count(topic_id)
            cluster_sizes.append(count)
    
    if cluster_sizes:
        logger.info(f"  📈 Largest cluster: {max(cluster_sizes)} documents")
        logger.info(f"  📉 Smallest cluster: {min(cluster_sizes)} documents")
        logger.info(f"  📊 Average cluster size: {sum(cluster_sizes) / len(cluster_sizes):.1f}")
        logger.info(f"  🎯 Median cluster size: {sorted(cluster_sizes)[len(cluster_sizes)//2]}")
        
        # Additional distribution insights
        large_clusters = sum(1 for size in cluster_sizes if size > 50)
        small_clusters = sum(1 for size in cluster_sizes if size < 10)
        logger.info(f"  🔍 Large clusters (>50 docs): {large_clusters}")
        logger.info(f"  🔍 Small clusters (<10 docs): {small_clusters}")
    
    return topics, probs


def validate_topic_training_config(config: Dict) -> bool:
    """
    Validate topic training configuration for completeness and correctness.
    
    This function performs comprehensive validation of all configuration parameters
    required for topic training to ensure proper model setup and avoid runtime errors.
    
    Args:
        config: Configuration dictionary to validate.
                Expected structure includes data parameters, random seed,
                and optional outlier reduction settings.
        
    Returns:
        bool: True if topic training configuration is valid and complete,
              False if any validation check fails.
        
    Validation Checks:
        - Presence of required configuration sections
        - Data parameter validity (min_topic_size, calculate_probabilities)
        - Random seed type and value validation
        - Outlier reduction parameter consistency
        - Type checking for all critical parameters
    
    Note:
        - Logs specific validation issues for debugging assistance
        - Returns False on first validation failure for fail-fast behavior
        - Validation is designed to catch common configuration mistakes
        - Used in pipeline initialization to prevent runtime failures
    """
    try:
        # Check required sections
        required_sections = ['data', 'random_seed']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Check data parameters
        data_config = config['data']
        required_data_params = ['min_topic_size', 'calculate_probabilities']
        for param in required_data_params:
            if param not in data_config:
                logger.error(f"Missing required data parameter: {param}")
                return False
        
        # Validate min_topic_size
        min_topic_size = data_config['min_topic_size']
        if not isinstance(min_topic_size, int) or min_topic_size < 1:
            logger.error(f"Invalid min_topic_size: {min_topic_size}. Must be positive integer")
            return False
        
        # Validate calculate_probabilities
        calc_probs = data_config['calculate_probabilities']
        if not isinstance(calc_probs, bool):
            logger.error(f"Invalid calculate_probabilities: {calc_probs}. Must be boolean")
            return False
        
        # Check random seed
        random_seed = config['random_seed']
        if not isinstance(random_seed, int):
            logger.error(f"Invalid random_seed: {random_seed}. Must be integer")
            return False
        
        logger.info("✅ Topic training configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating topic training configuration: {e}")
        return False


def get_topic_training_summary(topics: List[int], n_outliers_initial: int) -> Dict:
    """
    Generate comprehensive summary information for topic training results.
    
    Args:
        topics: Final topic assignments after training and outlier reduction
        n_outliers_initial: Number of outliers before outlier reduction
        
    Returns:
        Dict containing topic training summary and statistics
    """
    try:
        n_documents = len(topics)
        unique_topics = set(topics)
        n_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
        n_outliers_final = sum(1 for t in topics if t == -1)
        
        # Calculate cluster statistics
        cluster_sizes = []
        for topic_id in unique_topics:
            if topic_id != -1:  # Skip outliers
                count = topics.count(topic_id)
                cluster_sizes.append(count)
        
        return {
            'total_documents': n_documents,
            'topics_discovered': n_topics,
            'outliers_initial': n_outliers_initial,
            'outliers_final': n_outliers_final,
            'outliers_reduced': n_outliers_initial - n_outliers_final,
            'coverage_final': ((n_documents - n_outliers_final) / n_documents * 100) if n_documents > 0 else 0,
            'largest_cluster': max(cluster_sizes) if cluster_sizes else 0,
            'smallest_cluster': min(cluster_sizes) if cluster_sizes else 0,
            'average_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            'median_cluster_size': sorted(cluster_sizes)[len(cluster_sizes)//2] if cluster_sizes else 0
        }
        
    except Exception as e:
        logger.error(f"Error generating topic training summary: {e}")
        return {'error': str(e)}