#!/usr/bin/env python3
"""
Paper Selection Algorithms for Systematic Literature Reviews

This module provides advanced algorithms for selecting diverse representative papers
from academic clusters using optimization techniques that balance centrality and diversity.
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from utils import get_systematic_review_config


def select_diverse_representatives(cluster_embeddings: np.ndarray, n_select: int, config: Dict) -> List[int]:
    """
    Select diverse representative papers using advanced optimization algorithm.
    
    This function implements a sophisticated paper selection strategy that balances
    centrality (representativeness) and diversity (coverage) within academic clusters.
    It supports both iterative optimization and greedy selection approaches.
    
    Args:
        cluster_embeddings: Embedding matrix of shape (n_papers, embedding_dim) for papers
                          in the current cluster/topic
        n_select: Number of representative papers to select from the cluster
        config: Configuration dictionary containing systematic review settings
                with diversity_weight, min_diversity_threshold, and selection parameters
    
    Returns:
        List[int]: Indices of selected representative papers within the cluster
                  (local indices relative to cluster_embeddings)
    
    Raises:
        ValueError: If n_select is invalid or cluster_embeddings is malformed
        KeyError: If required configuration parameters are missing
    
    Example:
        >>> config = {'systematic_review': {'diversity_weight': 0.6, 'use_iterative_selection': True}}
        >>> embeddings = np.random.random((20, 384))  # 20 papers, 384-dim embeddings
        >>> selected = select_diverse_representatives(embeddings, 5, config)
        >>> len(selected)
        5
    
    Algorithm Details:
        1. Pre-filters papers with sufficient diversity potential
        2. Chooses between iterative optimization or greedy selection
        3. Iterative: Multiple random initializations with quality evaluation
        4. Greedy: Fast fallback with enhanced scoring
        5. Always returns at least 1 paper and at most n_select papers
    
    Note:
        - Diversity threshold filtering can be disabled by setting min_diversity_threshold=0.0
        - Iterative selection is computationally expensive but often yields better results
        - Greedy selection provides fast, reliable results for large clusters
        - Selection algorithms are designed for academic literature clustering
    """
    if n_select <= 0:
        raise ValueError(f"Invalid n_select: {n_select}. Must be positive integer.")
    
    if len(cluster_embeddings) == 0:
        raise ValueError("Empty cluster_embeddings provided.")
    
    if n_select >= len(cluster_embeddings):
        return list(range(len(cluster_embeddings)))
    
    try:
        review_config = get_systematic_review_config(config)
        
        # Pre-filter papers with sufficient diversity potential
        min_diversity = review_config.get('min_diversity_threshold', 0.0)
        diversity_scores = []
        valid_indices = []
        
        for i in range(len(cluster_embeddings)):
            similarities = cosine_similarity([cluster_embeddings[i]], cluster_embeddings)[0]
            other_similarities = similarities[similarities != 1.0]  # Exclude self-similarity
            if len(other_similarities) > 0:
                max_similarity = np.clip(np.max(other_similarities), 0.0, 1.0)
                diversity = 1.0 - max_similarity
                diversity_scores.append(diversity)
                if diversity >= min_diversity:
                    valid_indices.append(i)
            else:
                diversity_scores.append(0.0)
        
        # If not enough valid papers, fall back to all papers
        if len(valid_indices) < n_select:
            valid_indices = list(range(len(cluster_embeddings)))
        
        # Choose selection algorithm based on configuration
        use_iterative = review_config.get('use_iterative_selection', False)
        if use_iterative:
            return _iterative_selection(cluster_embeddings, n_select, valid_indices, diversity_scores, config)
        else:
            return _greedy_selection(cluster_embeddings, n_select, valid_indices, diversity_scores, config)
    
    except Exception as e:
        logger.error(f"Error in diverse representative selection: {e}")
        raise ValueError(f"Failed to select diverse representatives: {e}")


def _iterative_selection(cluster_embeddings: np.ndarray, n_select: int, 
                        valid_indices: List[int], diversity_scores: List[float], config: Dict) -> List[int]:
    """
    Use iterative optimization to find the best diverse representative set.
    
    This function implements a multi-start optimization approach that tries multiple
    random initializations to find the optimal combination of centrality and diversity.
    It's more computationally expensive but often yields better results than greedy selection.
    
    Args:
        cluster_embeddings: Embedding matrix for the cluster
        n_select: Number of papers to select
        valid_indices: List of valid paper indices after diversity filtering
        diversity_scores: Pre-computed individual diversity scores for each paper
        config: Configuration dictionary with selection parameters
    
    Returns:
        List[int]: Indices of selected papers using iterative optimization
    
    Algorithm:
        1. Multiple random initializations (default: 5 iterations)
        2. For each iteration: start with best centrality or random selection
        3. Iteratively build selection by adding papers with best combined score
        4. Evaluate final selection quality and keep the best across iterations
        5. Fallback to greedy selection if optimization fails
    
    Note:
        - Uses reproducible randomness with configurable seed
        - Combines centrality and diversity scores using diversity_weight
        - Evaluates selection quality using pairwise diversity metrics
        - Computationally intensive for large clusters (>100 papers)
    """
    try:
        review_config = get_systematic_review_config(config)
        centroid = np.mean(cluster_embeddings, axis=0)
        similarities_to_centroid = cosine_similarity(cluster_embeddings, [centroid]).flatten()
        diversity_weight = review_config['diversity_weight']
        iterations = review_config.get('selection_iterations', 5)
        
        best_selection = None
        best_score = -1
        
        # Try multiple random initializations
        for iteration in range(iterations):
            if iteration == 0:
                # First iteration: start with highest centrality from valid papers
                valid_centralities = [(i, similarities_to_centroid[i]) for i in valid_indices]
                valid_centralities.sort(key=lambda x: x[1], reverse=True)
                selected = [valid_centralities[0][0]]
            else:
                # Random initialization for other iterations
                np.random.seed(42 + iteration)  # Reproducible randomness
                selected = [np.random.choice(valid_indices)]
            
            # Build selection iteratively
            for _ in range(n_select - 1):
                remaining = [i for i in valid_indices if i not in selected]
                if not remaining:
                    break
                
                best_candidate = None
                best_candidate_score = -1
                
                for candidate in remaining:
                    # Calculate diversity to already selected papers
                    if len(selected) > 0:
                        similarities_to_selected = cosine_similarity(
                            [cluster_embeddings[candidate]], 
                            cluster_embeddings[selected]
                        ).flatten()
                        avg_similarity_to_selected = np.mean(similarities_to_selected)
                        diversity_to_selected = 1.0 - avg_similarity_to_selected
                    else:
                        diversity_to_selected = 1.0
                    
                    # Combined score
                    centrality_score = similarities_to_centroid[candidate]
                    combined_score = (
                        (1 - diversity_weight) * centrality_score + 
                        diversity_weight * diversity_to_selected
                    )
                    
                    if combined_score > best_candidate_score:
                        best_candidate_score = combined_score
                        best_candidate = candidate
                
                if best_candidate is not None:
                    selected.append(best_candidate)
            
            # Evaluate this selection
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
        logger.error(f"Error in iterative selection: {e}")
        # Fallback to greedy selection
        return _greedy_selection(cluster_embeddings, n_select, valid_indices, diversity_scores, config)


def _greedy_selection(cluster_embeddings: np.ndarray, n_select: int,
                     valid_indices: List[int], diversity_scores: List[float], config: Dict) -> List[int]:
    """
    Fallback greedy selection with improved scoring.
    
    This function implements a fast, reliable greedy selection algorithm that builds
    the representative set incrementally by choosing the best candidate at each step.
    It's computationally efficient and provides good results for most use cases.
    
    Args:
        cluster_embeddings: Embedding matrix for the cluster
        n_select: Number of papers to select
        valid_indices: List of valid paper indices after diversity filtering
        diversity_scores: Pre-computed individual diversity scores for each paper
        config: Configuration dictionary with selection parameters
    
    Returns:
        List[int]: Indices of selected papers using greedy selection
    
    Algorithm:
        1. Start with the paper having the best combination of centrality and individual diversity
        2. For each remaining selection: choose the paper with highest combined score
        3. Combined score includes: centrality, individual diversity, and diversity to selected papers
        4. Uses weighted combination of diversity metrics (60% to selected, 40% individual)
        5. Ensures no duplicate selections and handles edge cases
    
    Note:
        - Much faster than iterative selection, suitable for large clusters
        - Provides consistent, reliable results across different datasets
        - Enhanced scoring considers both individual and collective diversity
        - Handles edge cases like insufficient valid papers gracefully
    """
    try:
        review_config = get_systematic_review_config(config)
        centroid = np.mean(cluster_embeddings, axis=0)
        similarities_to_centroid = cosine_similarity(cluster_embeddings, [centroid]).flatten()
        diversity_weight = review_config['diversity_weight']
        
        # Start with the best combination of centrality and individual diversity
        initial_scores = []
        for idx in valid_indices:
            centrality = similarities_to_centroid[idx]
            diversity = diversity_scores[idx]
            score = (1 - diversity_weight) * centrality + diversity_weight * diversity
            initial_scores.append((idx, score))
        
        initial_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [initial_scores[0][0]]
        
        # Greedily add remaining papers
        for _ in range(n_select - 1):
            remaining_indices = [i for i in valid_indices if i not in selected_indices]
            
            if not remaining_indices:
                break
            
            best_score = -1
            best_idx = -1
            
            for idx in remaining_indices:
                # Calculate average similarity to already selected papers (better than min)
                similarities_to_selected = cosine_similarity(
                    [cluster_embeddings[idx]], 
                    cluster_embeddings[selected_indices]
                ).flatten()
                
                avg_similarity = np.mean(similarities_to_selected)
                diversity_to_selected = 1.0 - avg_similarity
                
                # Enhanced scoring
                centrality_score = similarities_to_centroid[idx]
                individual_diversity = diversity_scores[idx]
                
                # Combine individual diversity and diversity to selected
                combined_diversity = 0.6 * diversity_to_selected + 0.4 * individual_diversity
                combined_score = (1 - diversity_weight) * centrality_score + diversity_weight * combined_diversity
                
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


def _evaluate_selection_quality(cluster_embeddings: np.ndarray, selected_indices: List[int],
                               similarities_to_centroid: np.ndarray, diversity_weight: float) -> float:
    """
    Evaluate the quality of a selection based on centrality and diversity.
    
    This function computes a comprehensive quality score for a set of selected papers
    by combining centrality (representativeness) and pairwise diversity measures.
    
    Args:
        cluster_embeddings: Embedding matrix for the cluster
        selected_indices: List of selected paper indices
        similarities_to_centroid: Pre-computed similarities to cluster centroid
        diversity_weight: Weight for diversity vs centrality trade-off (0.0 to 1.0)
    
    Returns:
        float: Quality score between 0.0 and 1.0, where higher is better
    
    Quality Metrics:
        - Centrality Score: Average similarity to cluster centroid
        - Diversity Score: 1 - average pairwise similarity between selected papers
        - Combined Score: Weighted combination using diversity_weight
    
    Note:
        - Single paper selections get maximum diversity score (1.0)
        - Pairwise diversity uses upper triangle of similarity matrix
        - Returns 0.0 for empty selections (edge case handling)
    """
    if not selected_indices:
        return 0.0
    
    try:
        # Average centrality
        centrality_score = np.mean([similarities_to_centroid[i] for i in selected_indices])
        
        # Pairwise diversity
        if len(selected_indices) > 1:
            pairwise_similarities = cosine_similarity(cluster_embeddings[selected_indices])
            # Get upper triangle (excluding diagonal)
            upper_triangle = pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)]
            avg_pairwise_similarity = np.mean(upper_triangle)
            diversity_score = 1.0 - avg_pairwise_similarity
        else:
            diversity_score = 1.0
        
        # Combined quality score
        quality = (1 - diversity_weight) * centrality_score + diversity_weight * diversity_score
        return quality
    
    except Exception as e:
        logger.error(f"Error evaluating selection quality: {e}")
        return 0.0


def assign_non_selected_papers(cluster_embeddings: np.ndarray, 
                             selected_indices: List[int], config: Dict) -> Dict[int, Dict]:
    """
    Assign non-selected papers to their most similar representative.
    
    This function maps each non-selected paper in a cluster to its most similar
    representative paper, providing similarity scores and similarity assessments
    based on configured thresholds.
    
    Args:
        cluster_embeddings: Embedding matrix for all papers in the cluster
        selected_indices: List of indices for selected representative papers
        config: Configuration dictionary containing similarity_threshold
    
    Returns:
        Dict[int, Dict]: Mapping from non-selected paper indices to assignment info.
                        Each assignment contains:
                        - 'representative_idx': Index of most similar representative
                        - 'similarity': Cosine similarity score (0.0 to 1.0)
                        - 'is_similar': Boolean indicating if similarity >= threshold
    
    Raises:
        ValueError: If selected_indices is empty or contains invalid indices
        KeyError: If required configuration parameters are missing
    
    Example:
        >>> config = {'systematic_review': {'similarity_threshold': 0.7}}
        >>> embeddings = np.random.random((10, 384))
        >>> selected = [0, 3, 7]  # 3 representatives
        >>> assignments = assign_non_selected_papers(embeddings, selected, config)
        >>> len(assignments)  # Should be 7 (10 - 3 selected)
        7
    
    Note:
        - Only assigns non-selected papers (selected papers are skipped)
        - Similarity scores are computed using cosine similarity
        - Similarity threshold determines 'is_similar' flag for analysis
        - Used for systematic review paper mapping and coverage analysis
    """
    if not selected_indices:
        raise ValueError("Cannot assign papers: no representatives selected")
    
    if len(cluster_embeddings) == 0:
        raise ValueError("Cannot assign papers: empty cluster embeddings")
    
    try:
        review_config = get_systematic_review_config(config)
        assignments = {}
        similarity_threshold = review_config['similarity_threshold']
        
        selected_embeddings = cluster_embeddings[selected_indices]
        
        for i, embedding in enumerate(cluster_embeddings):
            if i in selected_indices:
                continue
            
            # Find most similar representative
            similarities = cosine_similarity([embedding], selected_embeddings)[0]
            best_rep_idx = np.argmax(similarities)
            best_similarity = similarities[best_rep_idx]
            
            assignments[i] = {
                'representative_idx': selected_indices[best_rep_idx],
                'similarity': float(best_similarity),
                'is_similar': bool(best_similarity >= similarity_threshold)
            }
        
        return assignments
    
    except Exception as e:
        logger.error(f"Error assigning non-selected papers: {e}")
        raise ValueError(f"Failed to assign non-selected papers: {e}")


def validate_paper_selection_config(config: Dict) -> bool:
    """
    Validate paper selection configuration for completeness and correctness.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if paper selection configuration is valid
        
    Note:
        - Validates diversity_weight range (0.0 to 1.0)
        - Checks selection algorithm parameters
        - Validates similarity threshold range
        - Logs specific validation issues for debugging
    """
    try:
        review_config = get_systematic_review_config(config)
        
        # Check diversity weight
        diversity_weight = review_config.get('diversity_weight', 0.6)
        if not isinstance(diversity_weight, (int, float)) or not (0.0 <= diversity_weight <= 1.0):
            logger.error(f"Invalid diversity_weight: {diversity_weight}. Must be between 0.0 and 1.0")
            return False
        
        # Check similarity threshold
        similarity_threshold = review_config.get('similarity_threshold', 0.7)
        if not isinstance(similarity_threshold, (int, float)) or not (0.0 <= similarity_threshold <= 1.0):
            logger.error(f"Invalid similarity_threshold: {similarity_threshold}. Must be between 0.0 and 1.0")
            return False
        
        # Check min diversity threshold
        min_diversity = review_config.get('min_diversity_threshold', 0.0)
        if not isinstance(min_diversity, (int, float)) or not (0.0 <= min_diversity <= 1.0):
            logger.error(f"Invalid min_diversity_threshold: {min_diversity}. Must be between 0.0 and 1.0")
            return False
        
        # Check iterative selection parameters
        if review_config.get('use_iterative_selection', False):
            iterations = review_config.get('selection_iterations', 5)
            if not isinstance(iterations, int) or iterations < 1:
                logger.error(f"Invalid selection_iterations: {iterations}. Must be positive integer")
                return False
        
        logger.info("✅ Paper selection configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating paper selection configuration: {e}")
        return False


def get_paper_selection_summary(cluster_size: int, selected_count: int, 
                              selected_indices: List[int], assignments: Dict[int, Dict], 
                              config: Dict) -> Dict:
    """
    Generate summary information for paper selection decisions.
    
    Args:
        cluster_size: Number of papers in the cluster
        selected_count: Number of papers selected as representatives
        selected_indices: List of selected paper indices
        assignments: Assignment mapping from assign_non_selected_papers
        config: Configuration dictionary
        
    Returns:
        Dict containing paper selection summary and statistics
    """
    try:
        review_config = get_systematic_review_config(config)
        
        # Calculate assignment statistics
        if assignments:
            similarities = [assignment['similarity'] for assignment in assignments.values()]
            similar_count = sum(1 for assignment in assignments.values() if assignment['is_similar'])
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            min_similarity = np.min(similarities) if similarities else 0.0
            max_similarity = np.max(similarities) if similarities else 0.0
        else:
            avg_similarity = min_similarity = max_similarity = 0.0
            similar_count = 0
        
        return {
            'cluster_size': cluster_size,
            'selected_count': selected_count,
            'selection_ratio': selected_count / cluster_size,
            'non_selected_count': len(assignments),
            'avg_similarity_to_representatives': avg_similarity,
            'min_similarity_to_representatives': min_similarity,
            'max_similarity_to_representatives': max_similarity,
            'similar_papers_count': similar_count,
            'similar_papers_ratio': similar_count / len(assignments) if assignments else 0.0,
            'selection_algorithm': 'iterative' if review_config.get('use_iterative_selection', False) else 'greedy',
            'diversity_weight': review_config.get('diversity_weight', 0.6),
            'similarity_threshold': review_config.get('similarity_threshold', 0.7)
        }
        
    except Exception as e:
        logger.error(f"Error generating paper selection summary: {e}")
        return {'error': str(e)} 