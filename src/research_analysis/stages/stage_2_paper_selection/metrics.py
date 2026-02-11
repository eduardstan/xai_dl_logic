#!/usr/bin/env python3
"""
Paper metrics calculation for the research analysis framework.
"""

from typing import Dict

import numpy as np
import pandas as pd

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger
from research_analysis.utils.math import (
    compute_cosine_similarity_to_centroid,
    compute_diversity_score,
    compute_pairwise_similarities,
    compute_representativeness_score,
)

logger = get_logger()


def compute_paper_metrics(
    paper_embedding: np.ndarray,
    cluster_embeddings: np.ndarray,
    config: AppConfig,
    centroid_embedding: np.ndarray = None,
    medoid_embedding: np.ndarray = None,
) -> Dict[str, float]:
    """
    Computes a set of metrics for a single paper relative to its cluster.

    Args:
        paper_embedding: The embedding of the paper to analyze.
        cluster_embeddings: Embeddings of all papers in the same cluster.
        config: The application configuration.
        centroid_embedding: Precomputed centroid (optional).
        medoid_embedding: Precomputed medoid (optional).

    Returns:
        A dictionary containing the calculated metrics for the paper.
    """
    if paper_embedding.ndim != 1:
        raise ValueError(
            f"paper_embedding must be a 1D vector, but got shape {paper_embedding.shape}"
        )
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    try:
        # Avoid redundant computations if passed from outer loop
        if centroid_embedding is None:
            from research_analysis.utils.math import compute_cluster_centroid
            centroid_embedding = compute_cluster_centroid(cluster_embeddings)
        
        # 1. Similarity to cluster centroid (centrality)
        similarity_to_centroid = float(cosine_similarity(
            paper_embedding.reshape(1, -1), centroid_embedding.reshape(1, -1)
        )[0, 0])

        # 2. Similarity to cluster medoid
        similarity_to_medoid = 0.0
        if medoid_embedding is not None:
            similarity_to_medoid = float(cosine_similarity(
                paper_embedding.reshape(1, -1), medoid_embedding.reshape(1, -1)
            )[0, 0])

        # 3. Average similarity to all other papers in the cluster
        pairwise_sims = compute_pairwise_similarities(
            paper_embedding, cluster_embeddings
        )
        avg_similarity_to_cluster = float(np.mean(pairwise_sims))

        # 4. Diversity score (uniqueness)
        diversity = compute_diversity_score(paper_embedding, cluster_embeddings)

        # 5. Final representativeness score (weighted combination)
        representativeness = compute_representativeness_score(
            centrality_score=similarity_to_centroid,
            diversity_score=diversity,
            diversity_weight=config.stage_2.metrics.diversity_weight,
        )

        return {
            "similarity_to_centroid": similarity_to_centroid,
            "similarity_to_medoid": similarity_to_medoid,
            "avg_similarity_to_cluster": avg_similarity_to_cluster,
            "diversity_score": diversity,
            "representativeness_score": representativeness,
        }
    except Exception as e:
        logger.error(f"Failed to compute metrics for paper: {e}")
        return {
            "similarity_to_centroid": 0.0,
            "similarity_to_medoid": 0.0,
            "avg_similarity_to_cluster": 0.0,
            "diversity_score": 0.0,
            "representativeness_score": 0.0,
        }


def calculate_all_metrics_for_topic(
    topic_id: int,
    df_topic: pd.DataFrame,
    embeddings_map: Dict[str, np.ndarray],
    config: AppConfig,
) -> pd.DataFrame:
    """
    Calculates metrics for all papers within a single topic.

    Args:
        topic_id: The ID of the current topic.
        df_topic: DataFrame containing all papers for the topic.
        embeddings_map: A dictionary mapping paper IDs to their embeddings.
        config: The application configuration.

    Returns:
        A DataFrame with the calculated metrics appended as columns.
    """
    logger.debug(f"Calculating metrics for Topic {topic_id} ({len(df_topic)} papers)")
    cluster_embeddings = np.array([embeddings_map[pid] for pid in df_topic["id"]])

    if cluster_embeddings.size == 0:
        logger.warning(f"No embeddings found for Topic {topic_id}. Skipping metrics.")
        return df_topic

    # Precompute centroid and medoid for efficiency
    from research_analysis.utils.math import compute_cluster_centroid, compute_cluster_medoid
    centroid_embedding = compute_cluster_centroid(cluster_embeddings)
    medoid_embedding = None
    if config.stage_2.augmentation.run_medoid_analysis:
        medoid_embedding, _ = compute_cluster_medoid(cluster_embeddings)

    metrics_list = []
    id_col = "id" if "id" in df_topic.columns else "ID"
    for paper_id in df_topic[id_col]:
        paper_embedding = embeddings_map[paper_id]
        metrics = compute_paper_metrics(
            paper_embedding, 
            cluster_embeddings, 
            config,
            centroid_embedding=centroid_embedding,
            medoid_embedding=medoid_embedding
        )
        metrics["id"] = paper_id
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    return pd.merge(df_topic, metrics_df, on="id", how="left")
 