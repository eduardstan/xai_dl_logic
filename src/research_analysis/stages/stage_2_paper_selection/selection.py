#!/usr/bin/env python3
"""
Paper selection algorithms for the research analysis framework.

This module provides both iterative and greedy algorithms for selecting
a diverse and representative set of papers from a topic cluster.
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

    This function acts as a dispatcher for different selection algorithms
    based on the application configuration. It first pre-filters candidates
    based on a minimum diversity threshold and then uses either an iterative
    or a greedy selection method.

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

    # Pre-filter papers with sufficient diversity potential
    min_diversity = config.stage_2.metrics.min_diversity_threshold
    diversity_scores = np.array(
        [compute_diversity_score(emb, cluster_embeddings) for emb in cluster_embeddings]
    )
    valid_indices = [i for i, score in enumerate(diversity_scores) if score >= min_diversity]

    # If not enough valid papers, fall back to all papers
    if len(valid_indices) < n_select:
        logger.debug(
            f"Only {len(valid_indices)} papers passed diversity threshold "
            f"(need {n_select}). Using all papers as candidates."
        )
        valid_indices = list(range(len(cluster_embeddings)))

    if config.stage_2.metrics.use_iterative_selection:
        logger.debug("Using iterative selection algorithm.")
        return _iterative_selection(
            cluster_embeddings, n_select, valid_indices, diversity_scores, config
        )
    else:
        logger.debug("Using greedy selection algorithm.")
        return _greedy_selection(
            cluster_embeddings, n_select, valid_indices, diversity_scores, config
        )


def _evaluate_selection_quality(
    cluster_embeddings: np.ndarray,
    selected_indices: List[int],
    similarities_to_centroid: np.ndarray,
    diversity_weight: float,
) -> float:
    """Evaluates the quality of a selected set of papers."""
    if not selected_indices:
        return 0.0

    # Overall centrality
    avg_centrality = np.mean(similarities_to_centroid[selected_indices])

    # Overall diversity (1 - avg intra-cluster similarity)
    selected_embeddings = cluster_embeddings[selected_indices]
    if len(selected_embeddings) > 1:
        # Correctly compute the full pairwise matrix for the selection
        from research_analysis.utils.math import compute_pairwise_similarity_matrix
        pairwise_sim_matrix = compute_pairwise_similarity_matrix(selected_embeddings)
        # Get the upper triangle, excluding the diagonal
        indices = np.triu_indices(len(selected_embeddings), k=1)
        avg_intra_similarity = np.mean(pairwise_sim_matrix[indices])
        overall_diversity = 1.0 - avg_intra_similarity
    else:
        overall_diversity = 1.0  # Max diversity for a single paper

    return compute_representativeness_score(
        avg_centrality, overall_diversity, diversity_weight
    )


def _iterative_selection(
    cluster_embeddings: np.ndarray,
    n_select: int,
    valid_indices: List[int],
    diversity_scores: np.ndarray,
    config: AppConfig,
) -> List[int]:
    """
    Uses a multi-start iterative optimization to find the best set of papers.
    """
    try:
        centroid = compute_cluster_centroid(cluster_embeddings)
        similarities_to_centroid = compute_pairwise_similarities(
            centroid, cluster_embeddings
        )
        diversity_weight = config.stage_2.metrics.diversity_weight
        iterations = config.stage_2.metrics.selection_iterations

        best_selection = []
        best_score = -1

        for i in range(iterations):
            seed = config.pipeline.reproducibility.random_seed + i
            np.random.seed(seed)

            if i == 0:
                # Start with the most central paper
                valid_centralities = {
                    idx: similarities_to_centroid[idx] for idx in valid_indices
                }
                start_node = max(valid_centralities, key=valid_centralities.get)
                selected = [start_node]
            else:
                # Start with a random paper
                selected = [np.random.choice(valid_indices)]

            current_candidates = list(set(valid_indices) - set(selected))

            while len(selected) < n_select and current_candidates:
                best_candidate = -1
                max_gain = -1

                for candidate_idx in current_candidates:
                    # Score is the marginal gain of adding this candidate
                    temp_selection = selected + [candidate_idx]
                    score = _evaluate_selection_quality(
                        cluster_embeddings,
                        temp_selection,
                        similarities_to_centroid,
                        diversity_weight,
                    )
                    gain = score - _evaluate_selection_quality(
                        cluster_embeddings,
                        selected,
                        similarities_to_centroid,
                        diversity_weight,
                    )
                    if gain > max_gain:
                        max_gain = gain
                        best_candidate = candidate_idx

                if best_candidate != -1:
                    selected.append(best_candidate)
                    current_candidates.remove(best_candidate)
                else:
                    break  # No candidate improves the score

            current_score = _evaluate_selection_quality(
                cluster_embeddings, selected, similarities_to_centroid, diversity_weight
            )

            if current_score > best_score:
                best_score = current_score
                best_selection = selected

        return best_selection

    except Exception as e:
        logger.error(f"Iterative selection failed: {e}", exc_info=True)
        return _greedy_selection(
            cluster_embeddings, n_select, valid_indices, diversity_scores, config
        )


def _greedy_selection(
    cluster_embeddings: np.ndarray,
    n_select: int,
    valid_indices: List[int],
    diversity_scores: np.ndarray,
    config: AppConfig,
) -> List[int]:
    """
    Selects papers using a greedy algorithm that balances centrality and diversity.
    """
    try:
        centroid = compute_cluster_centroid(cluster_embeddings)
        similarities_to_centroid = compute_pairwise_similarities(
            centroid, cluster_embeddings
        )

        selected_indices = []
        candidate_indices = valid_indices.copy()

        # First, select the most representative paper among valid candidates
        initial_scores = {
            idx: compute_representativeness_score(
                similarities_to_centroid[idx],
                diversity_scores[idx],
                config.stage_2.metrics.diversity_weight,
            )
            for idx in candidate_indices
        }
        best_initial_idx = max(initial_scores, key=initial_scores.get)
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
                break

        return selected_indices

    except Exception as e:
        logger.error(f"Greedy selection failed: {e}", exc_info=True)
        # Fallback: return top N by centrality if greedy fails
        return list(np.argsort(similarities_to_centroid)[-n_select:][::-1]) 