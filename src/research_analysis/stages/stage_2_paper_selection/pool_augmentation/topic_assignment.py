#!/usr/bin/env python3
"""
Topic assignment for pool augmentation.
Assigns new papers to existing topics via centroid similarity.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from research_analysis.utils.logging import get_logger
from research_analysis.utils.math import compute_cluster_centroids

logger = get_logger()

def assign_to_topics(
    augmentation_df: pd.DataFrame,
    augmentation_embeddings: np.ndarray,
    frozen_embeddings: np.ndarray,
    frozen_topics: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """
    Assign augmentation papers to the most similar existing topic centroids.
    
    Args:
        augmentation_df: DataFrame of new papers.
        augmentation_embeddings: Embeddings for the new papers.
        frozen_embeddings: Embeddings from the frozen Stage 1 run.
        frozen_topics: Topic labels from the frozen Stage 1 run.
        
    Returns:
        - augmentation_df with 'topic' and 'similarity_to_centroid' columns.
        - Dict mapping topic_id to its centroid embedding.
    """
    logger.info("Computing centroids for frozen topics...")
    # Filter out outliers (-1) for centroid calculation
    valid_mask = frozen_topics != -1
    centroids = compute_cluster_centroids(
        frozen_embeddings[valid_mask], 
        frozen_topics[valid_mask]
    )
    
    topic_ids = sorted(centroids.keys())
    centroid_matrix = np.array([centroids[tid] for tid in topic_ids])
    
    logger.info(f"Assigning {len(augmentation_df)} papers to {len(topic_ids)} topics...")
    
    # Compute similarity to all centroids
    similarities = cosine_similarity(augmentation_embeddings, centroid_matrix)
    
    # Find best match
    best_match_idx = np.argmax(similarities, axis=1)
    best_similarities = np.max(similarities, axis=1)
    assigned_topics = [topic_ids[idx] for idx in best_match_idx]
    
    augmentation_df["topic"] = assigned_topics
    augmentation_df["similarity_to_centroid"] = best_similarities
    
    return augmentation_df, centroids
