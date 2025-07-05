#!/usr/bin/env python3
"""
Paper Metrics Calculation for Systematic Literature Reviews

This module provides comprehensive metrics calculation for academic papers
within their topic clusters, including similarity, diversity, and representativeness scores.
"""

import numpy as np
from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from utils import get_systematic_review_config


def compute_paper_metrics(paper_embeddings: np.ndarray, cluster_embeddings: np.ndarray, config: Dict) -> Dict[str, float]:
    """
    Compute comprehensive metrics for a paper within its cluster.
    
    This function calculates multiple metrics that characterize how a paper relates
    to its topic cluster, including centrality (representativeness), diversity
    (uniqueness), and overall representativeness combining both factors.
    
    Args:
        paper_embeddings: Single paper embedding vector of shape (embedding_dim,)
        cluster_embeddings: Embedding matrix for all papers in the cluster, 
                          shape (n_papers, embedding_dim)
        config: Configuration dictionary containing systematic review settings
                with diversity_weight parameter for representativeness calculation
    
    Returns:
        Dict[str, float]: Dictionary containing comprehensive paper metrics:
            - 'similarity_to_centroid': How similar the paper is to cluster center (0.0-1.0)
            - 'similarity_to_cluster_papers': Average similarity to all cluster papers (0.0-1.0)
            - 'diversity_score': How unique the paper is within cluster (0.0-1.0)
            - 'representativeness_score': Combined metric balancing centrality and diversity (0.0-1.0)
    
    Raises:
        ValueError: If embeddings have incompatible shapes or invalid dimensions
        KeyError: If required configuration parameters are missing
    
    Example:
        >>> import numpy as np
        >>> config = {'systematic_review': {'diversity_weight': 0.6}}
        >>> paper_emb = np.random.random(384)  # Single paper embedding
        >>> cluster_embs = np.random.random((20, 384))  # 20 papers in cluster
        >>> metrics = compute_paper_metrics(paper_emb, cluster_embs, config)
        >>> print(f"Representativeness: {metrics['representativeness_score']:.3f}")
        Representativeness: 0.742
    
    Metric Definitions:
        - **Similarity to Centroid**: Cosine similarity between paper and cluster centroid.
          Higher values indicate the paper is representative of the cluster's main theme.
        
        - **Similarity to Cluster Papers**: Average cosine similarity to all papers in cluster.
          Measures how well the paper fits with the overall cluster content.
        
        - **Diversity Score**: 1 - max_similarity to other papers in cluster.
          Higher values indicate the paper brings unique content to the cluster.
        
        - **Representativeness Score**: Weighted combination of centrality and diversity.
          Balances being representative (central) with being unique (diverse).
          Formula: (1 - diversity_weight) * centrality + diversity_weight * diversity
    
    Note:
        - All similarity calculations use cosine similarity for semantic comparison
        - Diversity score excludes self-similarity to avoid division issues
        - Max similarity is clamped to [0, 1] to handle floating-point precision
        - Representativeness weight is configurable for different analysis strategies
        - Single-paper clusters get diversity_score = 1.0 (maximally diverse)
    """
    if len(paper_embeddings.shape) != 1:
        raise ValueError(f"paper_embeddings must be 1D vector, got shape {paper_embeddings.shape}")
    
    if len(cluster_embeddings.shape) != 2:
        raise ValueError(f"cluster_embeddings must be 2D matrix, got shape {cluster_embeddings.shape}")
    
    if paper_embeddings.shape[0] != cluster_embeddings.shape[1]:
        raise ValueError(f"Embedding dimension mismatch: paper={paper_embeddings.shape[0]}, cluster={cluster_embeddings.shape[1]}")
    
    if len(cluster_embeddings) == 0:
        raise ValueError("cluster_embeddings cannot be empty")
    
    try:
        review_config = get_systematic_review_config(config)
        
        # 1. Similarity to cluster centroid (representativeness/centrality)
        centroid = np.mean(cluster_embeddings, axis=0)
        similarity_to_centroid = cosine_similarity([paper_embeddings], [centroid])[0][0]
        
        # 2. Average similarity to all cluster papers (cluster coherence)
        similarities = cosine_similarity([paper_embeddings], cluster_embeddings)[0]
        avg_similarity_to_cluster = np.mean(similarities)
        
        # 3. Diversity score (uniqueness within cluster)
        other_similarities = similarities[similarities != 1.0]  # Exclude self-similarity
        if len(other_similarities) > 0:
            max_similarity = np.max(other_similarities)
            # Clamp max_similarity to [0, 1] to avoid floating point precision issues
            max_similarity = np.clip(max_similarity, 0.0, 1.0)
            diversity_score = 1.0 - max_similarity
        else:
            # Single paper in cluster - maximally diverse
            diversity_score = 1.0
        
        # 4. Representativeness score (combination of centrality and diversity)
        diversity_weight = review_config.get('diversity_weight', 0.6)
        representativeness_score = (
            (1 - diversity_weight) * similarity_to_centroid + 
            diversity_weight * diversity_score
        )
        
        return {
            'similarity_to_centroid': float(similarity_to_centroid),
            'similarity_to_cluster_papers': float(avg_similarity_to_cluster),
            'diversity_score': float(diversity_score),
            'representativeness_score': float(representativeness_score)
        }
    
    except Exception as e:
        logger.error(f"Error computing paper metrics: {e}")
        raise ValueError(f"Failed to compute paper metrics: {e}")


def compute_cluster_metrics(cluster_embeddings: np.ndarray, config: Dict) -> Dict[str, float]:
    """
    Compute comprehensive metrics for an entire cluster/topic.
    
    This function analyzes the internal structure and coherence of a topic cluster
    by computing aggregate statistics across all papers in the cluster.
    
    Args:
        cluster_embeddings: Embedding matrix for all papers in the cluster,
                          shape (n_papers, embedding_dim)
        config: Configuration dictionary containing systematic review settings
    
    Returns:
        Dict[str, float]: Dictionary containing cluster-level metrics:
            - 'cluster_coherence': Average pairwise similarity within cluster (0.0-1.0)
            - 'cluster_diversity': Average diversity of papers within cluster (0.0-1.0)
            - 'cluster_size': Number of papers in the cluster
            - 'cluster_compactness': Standard deviation of distances to centroid
            - 'cluster_representativeness': Average representativeness of all papers
    
    Note:
        - Coherence measures how similar papers are to each other
        - Diversity measures how unique papers are within the cluster
        - Compactness measures how tightly papers cluster around centroid
        - Single-paper clusters get special handling for edge cases
    """
    if len(cluster_embeddings.shape) != 2:
        raise ValueError(f"cluster_embeddings must be 2D matrix, got shape {cluster_embeddings.shape}")
    
    if len(cluster_embeddings) == 0:
        raise ValueError("cluster_embeddings cannot be empty")
    
    try:
        n_papers = len(cluster_embeddings)
        
        # Single paper cluster - special case
        if n_papers == 1:
            return {
                'cluster_coherence': 1.0,
                'cluster_diversity': 1.0,
                'cluster_size': float(n_papers),
                'cluster_compactness': 0.0,
                'cluster_representativeness': 1.0
            }
        
        # Compute pairwise similarities
        pairwise_similarities = cosine_similarity(cluster_embeddings)
        
        # Cluster coherence (average pairwise similarity, excluding diagonal)
        upper_triangle = pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)]
        cluster_coherence = np.mean(upper_triangle)
        
        # Cluster diversity (average of individual diversity scores)
        diversity_scores = []
        for i in range(n_papers):
            other_similarities = pairwise_similarities[i][pairwise_similarities[i] != 1.0]
            if len(other_similarities) > 0:
                max_sim = np.clip(np.max(other_similarities), 0.0, 1.0)
                diversity = 1.0 - max_sim
                diversity_scores.append(diversity)
            else:
                diversity_scores.append(1.0)
        
        cluster_diversity = np.mean(diversity_scores)
        
        # Cluster compactness (std dev of distances to centroid)
        centroid = np.mean(cluster_embeddings, axis=0)
        distances_to_centroid = []
        for embedding in cluster_embeddings:
            similarity = cosine_similarity([embedding], [centroid])[0][0]
            distance = 1.0 - similarity  # Convert similarity to distance
            distances_to_centroid.append(distance)
        
        cluster_compactness = np.std(distances_to_centroid)
        
        # Cluster representativeness (average of individual representativeness scores)
        representativeness_scores = []
        for i, embedding in enumerate(cluster_embeddings):
            metrics = compute_paper_metrics(embedding, cluster_embeddings, config)
            representativeness_scores.append(metrics['representativeness_score'])
        
        cluster_representativeness = np.mean(representativeness_scores)
        
        return {
            'cluster_coherence': float(cluster_coherence),
            'cluster_diversity': float(cluster_diversity), 
            'cluster_size': float(n_papers),
            'cluster_compactness': float(cluster_compactness),
            'cluster_representativeness': float(cluster_representativeness)
        }
    
    except Exception as e:
        logger.error(f"Error computing cluster metrics: {e}")
        raise ValueError(f"Failed to compute cluster metrics: {e}")


def validate_paper_metrics_config(config: Dict) -> bool:
    """
    Validate paper metrics configuration for completeness and correctness.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if paper metrics configuration is valid
        
    Note:
        - Validates diversity_weight range (0.0 to 1.0)
        - Checks systematic review configuration structure
        - Logs specific validation issues for debugging
    """
    try:
        review_config = get_systematic_review_config(config)
        
        # Check diversity weight
        diversity_weight = review_config.get('diversity_weight', 0.6)
        if not isinstance(diversity_weight, (int, float)) or not (0.0 <= diversity_weight <= 1.0):
            logger.error(f"Invalid diversity_weight: {diversity_weight}. Must be between 0.0 and 1.0")
            return False
        
        logger.info("✅ Paper metrics configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating paper metrics configuration: {e}")
        return False


def get_paper_metrics_summary(metrics: Dict[str, float], cluster_size: int) -> Dict:
    """
    Generate summary information for paper metrics analysis.
    
    Args:
        metrics: Paper metrics dictionary from compute_paper_metrics
        cluster_size: Number of papers in the cluster
        
    Returns:
        Dict containing paper metrics summary and interpretations
    """
    try:
        # Interpret scores based on common thresholds
        centrality_level = "High" if metrics['similarity_to_centroid'] > 0.7 else "Medium" if metrics['similarity_to_centroid'] > 0.4 else "Low"
        diversity_level = "High" if metrics['diversity_score'] > 0.6 else "Medium" if metrics['diversity_score'] > 0.3 else "Low"
        representativeness_level = "High" if metrics['representativeness_score'] > 0.7 else "Medium" if metrics['representativeness_score'] > 0.4 else "Low"
        
        return {
            'paper_metrics': metrics,
            'cluster_size': cluster_size,
            'centrality_level': centrality_level,
            'diversity_level': diversity_level,
            'representativeness_level': representativeness_level,
            'centrality_percentile': min(metrics['similarity_to_centroid'] * 100, 100),
            'diversity_percentile': min(metrics['diversity_score'] * 100, 100),
            'representativeness_percentile': min(metrics['representativeness_score'] * 100, 100),
            'balanced_metrics': abs(metrics['similarity_to_centroid'] - metrics['diversity_score']) < 0.3
        }
        
    except Exception as e:
        logger.error(f"Error generating paper metrics summary: {e}")
        return {'error': str(e)} 