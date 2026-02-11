#!/usr/bin/env python3
"""
Analysis module for comparing centroid-based vs medoid-based paper selection.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

from research_analysis.utils.logging import get_logger
from research_analysis.utils.math import compute_cluster_centroid, compute_cluster_medoid

logger = get_logger()

def calculate_rank_correlation(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    medoid: np.ndarray
) -> float:
    """
    Calculates the Spearman rank correlation between distances to centroid and medoid.
    """
    sim_to_centroid = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    sim_to_medoid = cosine_similarity(embeddings, medoid.reshape(1, -1)).flatten()
    
    rho, _ = spearmanr(sim_to_centroid, sim_to_medoid)
    return float(rho)

def compare_representative_sets(
    centroid_reps: List[str],
    medoid_reps: List[str]
) -> Dict[str, float]:
    """
    Compares two sets of representative paper IDs.
    Returns intersection size and Jaccard similarity.
    """
    set1 = set(centroid_reps)
    set2 = set(medoid_reps)
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    jaccard = len(intersection) / len(union) if len(union) > 0 else 1.0
    
    return {
        "intersection_count": len(intersection),
        "jaccard_similarity": jaccard,
        "overlap_percentage": (len(intersection) / len(centroid_reps) * 100) if len(centroid_reps) > 0 else 100.0
    }

def run_centroid_vs_medoid_analysis(
    topic_id: int,
    cluster_embeddings: np.ndarray,
    paper_ids: List[str],
    top_n: int = 5
) -> Dict:
    """
    Runs a full comparison for a single topic.
    """
    if len(cluster_embeddings) < 2:
        return {
            "topic_id": topic_id,
            "spearman_rho": 1.0,
            "jaccard_similarity": 1.0,
            "status": "too_small"
        }
        
    centroid = compute_cluster_centroid(cluster_embeddings)
    medoid, _ = compute_cluster_medoid(cluster_embeddings)
    
    # 1. Rank correlation
    rho = calculate_rank_correlation(cluster_embeddings, centroid, medoid)
    
    # 2. Top-N intersection
    sim_to_centroid = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
    sim_to_medoid = cosine_similarity(cluster_embeddings, medoid.reshape(1, -1)).flatten()
    
    top_idx_centroid = np.argsort(sim_to_centroid)[-top_n:][::-1]
    top_idx_medoid = np.argsort(sim_to_medoid)[-top_n:][::-1]
    
    centroid_reps = [paper_ids[i] for i in top_idx_centroid]
    medoid_reps = [paper_ids[i] for i in top_idx_medoid]
    
    comparison = compare_representative_sets(centroid_reps, medoid_reps)
    
    return {
        "topic_id": topic_id,
        "spearman_rho": rho,
        "jaccard_similarity": comparison["jaccard_similarity"],
        "overlap_percentage": comparison["overlap_percentage"],
        "cluster_size": len(cluster_embeddings)
    }
