#!/usr/bin/env python3
"""
Mathematical and vector-based utility functions for the research analysis framework.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_cluster_centroid(cluster_embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the centroid of a cluster of embeddings.

    Args:
        cluster_embeddings: A 2D NumPy array where each row is an embedding.

    Returns:
        A 1D NumPy array representing the centroid of the cluster.
    """
    if cluster_embeddings.ndim != 2:
        raise ValueError("Input embeddings must be a 2D array.")
    return np.mean(cluster_embeddings, axis=0)


def compute_cosine_similarity_to_centroid(
    paper_embedding: np.ndarray, cluster_embeddings: np.ndarray
) -> float:
    """
    Computes the cosine similarity between a paper's embedding and the
    centroid of its cluster.

    Args:
        paper_embedding: The 1D embedding of the paper.
        cluster_embeddings: The 2D array of embeddings for all papers in the cluster.

    Returns:
        The cosine similarity as a float.
    """
    centroid = compute_cluster_centroid(cluster_embeddings)
    similarity = cosine_similarity(
        paper_embedding.reshape(1, -1), centroid.reshape(1, -1)
    )
    # Return raw similarity without clipping to match legacy behavior exactly
    return float(similarity[0, 0])


def compute_pairwise_similarities(
    paper_embedding: np.ndarray, cluster_embeddings: np.ndarray
) -> np.ndarray:
    """
    Computes the cosine similarities between a single paper's embedding and all
    embeddings in its cluster.

    Args:
        paper_embedding: The 1D embedding of the paper.
        cluster_embeddings: The 2D array of embeddings for all papers in the cluster.

    Returns:
        A 1D NumPy array of cosine similarities.
    """
    similarities = cosine_similarity(
        paper_embedding.reshape(1, -1), cluster_embeddings
    )
    # Return raw similarities without clipping to match legacy behavior exactly
    return similarities.flatten()


def compute_diversity_score(
    paper_embedding: np.ndarray, cluster_embeddings: np.ndarray
) -> float:
    """
    Computes the diversity score for a paper within its cluster.

    The score is defined as 1 minus the highest similarity to any other paper
    in the cluster, ignoring self-similarity.

    Args:
        paper_embedding: The 1D embedding of the paper.
        cluster_embeddings: The 2D array of embeddings for all papers in the cluster.

    Returns:
        The diversity score as a float between 0.0 and 1.0.
    """
    similarities = compute_pairwise_similarities(paper_embedding, cluster_embeddings)

    # Exclude self-similarity (similarity = 1.0) - exact match with legacy implementation
    other_similarities = similarities[similarities != 1.0]

    if other_similarities.size > 0:
        max_similarity = np.max(other_similarities)
        # Clamp max_similarity to [0, 1] to avoid floating point precision issues - exactly like legacy
        max_similarity = np.clip(max_similarity, 0.0, 1.0)
        return 1.0 - max_similarity
    else:
        # Single paper in cluster - maximally diverse (exactly like legacy)
        return 1.0


def compute_pairwise_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise cosine similarity matrix for a set of embeddings.

    Args:
        embeddings: A 2D NumPy array where each row is an embedding.

    Returns:
        A 2D NumPy array representing the pairwise cosine similarity matrix.
    """
    if embeddings.ndim != 2:
        raise ValueError("Input embeddings must be a 2D array.")
    return cosine_similarity(embeddings)

def compute_representativeness_score(
    centrality_score: float, diversity_score: float, diversity_weight: float = 0.6
) -> float:
    """
    Computes a weighted score combining centrality and diversity.

    Args:
        centrality_score: The paper's similarity to the cluster centroid.
        diversity_score: The paper's diversity score.
        diversity_weight: The weight to give to the diversity score (0.0 to 1.0).

    Returns:
        The combined representativeness score.
    """
    return ((1 - diversity_weight) * centrality_score) + (
        diversity_weight * diversity_score
    ) 