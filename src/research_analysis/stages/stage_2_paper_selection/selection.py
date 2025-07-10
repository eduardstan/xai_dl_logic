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
    
    This implementation exactly matches the legacy algorithm for faithful reproduction.
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

        # Try multiple random initializations - exactly like legacy
        for iteration in range(iterations):
            if iteration == 0:
                # First iteration: start with highest centrality from valid papers (like legacy)
                valid_centralities = [(i, similarities_to_centroid[i]) for i in valid_indices]
                valid_centralities.sort(key=lambda x: x[1], reverse=True)
                selected = [valid_centralities[0][0]]
            else:
                # Random initialization for other iterations (like legacy)
                base_seed = config.pipeline.reproducibility.random_seed
                np.random.seed(base_seed + iteration)  # Reproducible randomness with config seed
                selected = [np.random.choice(valid_indices)]
            
            # Build selection iteratively - exactly like legacy algorithm
            for _ in range(n_select - 1):
                remaining = [i for i in valid_indices if i not in selected]
                if not remaining:
                    break
                
                best_candidate = None
                best_candidate_score = -1
                
                for candidate in remaining:
                    # Calculate diversity to already selected papers - exactly like legacy
                    if len(selected) > 0:
                        similarities_to_selected = compute_pairwise_similarities(
                            cluster_embeddings[candidate], 
                            cluster_embeddings[selected]
                        )
                        avg_similarity_to_selected = np.mean(similarities_to_selected)
                        diversity_to_selected = 1.0 - avg_similarity_to_selected
                    else:
                        diversity_to_selected = 1.0
                    
                    # Combined score using utility function - exactly like legacy
                    centrality_score = similarities_to_centroid[candidate]
                    combined_score = compute_representativeness_score(
                        centrality_score, diversity_to_selected, diversity_weight
                    )
                    
                    if combined_score > best_candidate_score:
                        best_candidate_score = combined_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    selected.append(best_candidate)
            
            # Evaluate this selection - exactly like legacy
            selection_score = _evaluate_selection_quality(
                cluster_embeddings, selected, similarities_to_centroid, diversity_weight
            )
            
            if selection_score > best_score:
                best_score = selection_score
                best_selection = selected.copy()

        return best_selection if best_selection else _greedy_selection(
            cluster_embeddings, n_select, valid_indices, diversity_scores, config
        )

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
    Fallback greedy selection with improved scoring.
    
    This implementation exactly matches the legacy algorithm for faithful reproduction.
    """
    try:
        centroid = compute_cluster_centroid(cluster_embeddings)
        similarities_to_centroid = compute_pairwise_similarities(centroid, cluster_embeddings)
        diversity_weight = config.stage_2.metrics.diversity_weight
        
        # Start with the best combination of centrality and individual diversity - exactly like legacy
        initial_scores = []
        for idx in valid_indices:
            centrality = similarities_to_centroid[idx]
            diversity = diversity_scores[idx]
            score = compute_representativeness_score(centrality, diversity, diversity_weight)
            initial_scores.append((idx, score))
        
        initial_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [initial_scores[0][0]]
        
        # Greedily add remaining papers - exactly like legacy
        for _ in range(n_select - 1):
            remaining_indices = [i for i in valid_indices if i not in selected_indices]
            
            if not remaining_indices:
                break
            
            best_score = -1
            best_idx = -1
            
            for idx in remaining_indices:
                # Calculate average similarity to already selected papers - exactly like legacy
                similarities_to_selected = compute_pairwise_similarities(
                    cluster_embeddings[idx], 
                    cluster_embeddings[selected_indices]
                )
                
                avg_similarity = np.mean(similarities_to_selected)
                diversity_to_selected = 1.0 - avg_similarity
                
                # Enhanced scoring using utility functions - exactly like legacy
                centrality_score = similarities_to_centroid[idx]
                individual_diversity = diversity_scores[idx]
                
                # Combine individual diversity and diversity to selected - exactly like legacy (60%/40%)
                combined_diversity = 0.6 * diversity_to_selected + 0.4 * individual_diversity
                combined_score = compute_representativeness_score(
                    centrality_score, combined_diversity, diversity_weight
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
        
        return selected_indices
    
    except Exception as e:
        logger.error(f"Error in greedy selection: {e}")
        # Fallback to simple selection
        return valid_indices[:n_select] if len(valid_indices) >= n_select else valid_indices


def assign_non_selected_papers(
    cluster_embeddings: np.ndarray, selected_indices: List[int], config: AppConfig
) -> dict:
    """
    Assigns each non-selected paper to its most similar representative.

    Args:
        cluster_embeddings: Embeddings for all papers in the cluster.
        selected_indices: A list of local indices for the selected papers.
        config: The application configuration.

    Returns:
        A dictionary mapping non-selected paper indices to their assignment info,
        which includes the representative's index and the similarity score.
    """
    if not selected_indices:
        logger.warning("No representatives selected, cannot assign papers.")
        return {}
    if len(cluster_embeddings) == 0:
        logger.warning("Cluster embeddings are empty, cannot assign papers.")
        return {}

    assignments = {}
    similarity_threshold = config.stage_2.metrics.similarity_threshold
    selected_embeddings = cluster_embeddings[selected_indices]

    all_indices = set(range(len(cluster_embeddings)))
    non_selected_indices = list(all_indices - set(selected_indices))

    if not non_selected_indices:
        return {}

    non_selected_embeddings = cluster_embeddings[non_selected_indices]

    # Compute similarities in a batch
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(
        non_selected_embeddings, selected_embeddings
    )

    # Find the best representative for each non-selected paper
    best_rep_indices = np.argmax(similarity_matrix, axis=1)
    best_similarities = np.max(similarity_matrix, axis=1)

    for i, original_idx in enumerate(non_selected_indices):
        rep_local_idx = best_rep_indices[i]
        assignments[original_idx] = {
            "representative_idx": selected_indices[rep_local_idx],
            "similarity": float(best_similarities[i]),
            "is_similar": bool(best_similarities[i] >= similarity_threshold),
        }

    return assignments 