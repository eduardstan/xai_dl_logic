#!/usr/bin/env python3
"""
Paper selection algorithms for the research analysis framework.
"""

from typing import List

import numpy as np

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger
from research_analysis.utils.math import (
    compute_cluster_centroid,
    compute_diversity_score,
    compute_pairwise_similarities,
    compute_representativeness_score,
)

logger = get_logger()


def select_diverse_representatives(
    cluster_embeddings: np.ndarray, n_select: int, config: AppConfig
) -> List[int]:
    """
    Selects a set of diverse and representative papers from a cluster.

    This function acts as a dispatcher, currently supporting a greedy
    selection algorithm.

    Args:
        cluster_embeddings: Embeddings for all papers in the cluster.
        n_select: The number of papers to select.
        config: The application configuration.

    Returns:
        A list of local indices for the selected papers.
    """
    if n_select <= 0:
        logger.warning(f"n_select={n_select} is not positive; returning empty list.")
        return []
    if len(cluster_embeddings) == 0:
        logger.warning("cluster_embeddings is empty; returning empty list.")
        return []
    if n_select >= len(cluster_embeddings):
        return list(range(len(cluster_embeddings)))

    return _greedy_selection(cluster_embeddings, n_select, config)


def _greedy_selection(
    cluster_embeddings: np.ndarray, n_select: int, config: AppConfig
) -> List[int]:
    """
    Selects papers using a greedy algorithm that balances centrality and diversity.

    Args:
        cluster_embeddings: Embeddings for all papers in the cluster.
        n_select: The number of papers to select.
        config: The application configuration.

    Returns:
        A list of local indices for the selected papers.
    """
    try:
        centroid = compute_cluster_centroid(cluster_embeddings)
        similarities_to_centroid = compute_pairwise_similarities(
            centroid, cluster_embeddings
        )
        diversity_scores = np.array(
            [
                compute_diversity_score(emb, cluster_embeddings)
                for emb in cluster_embeddings
            ]
        )

        selected_indices = []
        candidate_indices = list(range(len(cluster_embeddings)))

        # First, select the most representative paper
        initial_scores = compute_representativeness_score(
            similarities_to_centroid,
            diversity_scores,
            config.stage_2.metrics.diversity_weight,
        )
        best_initial_idx = np.argmax(initial_scores)
        selected_indices.append(best_initial_idx)
        candidate_indices.remove(best_initial_idx)

        # Select the rest of the papers
        while len(selected_indices) < n_select and candidate_indices:
            best_candidate_idx = -1
            max_score = -1

            for idx in candidate_indices:
                sim_to_selected = compute_pairwise_similarities(
                    cluster_embeddings[idx], cluster_embeddings[selected_indices]
                )
                diversity_from_selected = 1.0 - np.mean(sim_to_selected)

                score = compute_representativeness_score(
                    similarities_to_centroid[idx],
                    diversity_from_selected,
                    config.stage_2.metrics.diversity_weight,
                )

                if score > max_score:
                    max_score = score
                    best_candidate_idx = idx

            if best_candidate_idx != -1:
                selected_indices.append(best_candidate_idx)
                candidate_indices.remove(best_candidate_idx)
            else:
                break  # No more valid candidates

        return selected_indices

    except Exception as e:
        logger.error(f"Greedy selection failed: {e}", exc_info=True)
        # Fallback: return top N by centrality if greedy fails
        return list(
            np.argsort(similarities_to_centroid)[-n_select:][::-1]
        ) 