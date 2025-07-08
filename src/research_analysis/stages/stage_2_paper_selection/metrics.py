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
) -> Dict[str, float]:
    """
    Computes a set of metrics for a single paper relative to its cluster.

    Args:
        paper_embedding: The embedding of the paper to analyze.
        cluster_embeddings: Embeddings of all papers in the same cluster.
        config: The application configuration.

    Returns:
        A dictionary containing the calculated metrics for the paper.
    """
    if paper_embedding.ndim != 1:
        raise ValueError(
            f"paper_embedding must be a 1D vector, but got shape {paper_embedding.shape}"
        )
    if cluster_embeddings.ndim != 2:
        raise ValueError(
            "cluster_embeddings must be a 2D matrix, but got shape "
            f"{cluster_embeddings.shape}"
        )

    try:
        # 1. Similarity to cluster centroid (centrality)
        similarity_to_centroid = compute_cosine_similarity_to_centroid(
            paper_embedding, cluster_embeddings
        )

        # 2. Average similarity to all other papers in the cluster
        pairwise_sims = compute_pairwise_similarities(
            paper_embedding, cluster_embeddings
        )
        avg_similarity_to_cluster = float(np.mean(pairwise_sims))

        # 3. Diversity score (uniqueness)
        diversity = compute_diversity_score(paper_embedding, cluster_embeddings)

        # 4. Final representativeness score (weighted combination)
        representativeness = compute_representativeness_score(
            centrality_score=similarity_to_centroid,
            diversity_score=diversity,
            diversity_weight=config.stage_2.metrics.diversity_weight,
        )

        return {
            "similarity_to_centroid": similarity_to_centroid,
            "avg_similarity_to_cluster": avg_similarity_to_cluster,
            "diversity_score": diversity,
            "representativeness_score": representativeness,
        }
    except Exception as e:
        logger.error(f"Failed to compute metrics for paper: {e}")
        # Return default values in case of an error to avoid crashing the pipeline
        return {
            "similarity_to_centroid": 0.0,
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

    metrics_list = []
    for paper_id in df_topic["id"]:
        paper_embedding = embeddings_map[paper_id]
        metrics = compute_paper_metrics(paper_embedding, cluster_embeddings, config)
        metrics["id"] = paper_id
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    return pd.merge(df_topic, metrics_df, on="id", how="left") 