#!/usr/bin/env python3
"""
Cluster analysis functions for the research analysis framework.

This module provides helpers for determining selection counts based on
cluster size and the configured strategy (ratio-based or threshold-based).
"""

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger

logger = get_logger()


def determine_papers_to_select(cluster_size: int, config: AppConfig) -> int:
    """
    Determines how many papers to select from a cluster based on its size
    and the configured selection strategy.

    Args:
        cluster_size: The number of papers in the cluster.
        config: The application configuration.

    Returns:
        The number of papers to select.
    """
    if cluster_size <= 0:
        logger.warning(f"Invalid cluster_size: {cluster_size}. Returning 0.")
        return 0

    strategy = config.stage_2.selection_strategy.count_method
    logger.debug(
        f"Determining selection count for cluster of size {cluster_size} "
        f"using '{strategy}' strategy."
    )

    if strategy == "ratio_based":
        return _determine_papers_ratio_based(cluster_size, config)
    elif strategy == "threshold_based":
        return _determine_papers_threshold_based(cluster_size, config)
    else:
        logger.error(f"Unknown selection count method: {strategy}. Defaulting to ratio-based.")
        return _determine_papers_ratio_based(cluster_size, config)


def _determine_papers_ratio_based(cluster_size: int, config: AppConfig) -> int:
    """Calculates selection count based on a fixed ratio."""
    strategy_config = config.stage_2.selection_strategy
    params = strategy_config.ratio_params
    target_ratio = params.target_selection_ratio
    min_papers = params.min_papers_per_cluster
    max_papers = strategy_config.max_papers_per_cluster

    calculated_papers = max(1, int(cluster_size * target_ratio))
    constrained_papers = min(max(calculated_papers, min_papers), max_papers)

    # Ensure we don't select more papers than exist
    return min(constrained_papers, cluster_size)


def _calculate_cluster_size_category(cluster_size: int, config: AppConfig) -> str:
    """Categorizes a cluster size based on configured thresholds."""
    thresholds = config.stage_2.selection_strategy.threshold_params.cluster_thresholds

    if cluster_size <= thresholds["small"]:
        return "small"
    if cluster_size <= thresholds["medium"]:
        return "medium"
    if cluster_size <= thresholds["large"]:
        return "large"
    if cluster_size <= thresholds["xlarge"]:
        return "xlarge"
    if cluster_size <= thresholds["xxlarge"]:
        return "xxlarge"
    return "xxxlarge"


def _determine_papers_threshold_based(cluster_size: int, config: AppConfig) -> int:
    """Calculates selection count based on discrete size thresholds."""
    strategy_config = config.stage_2.selection_strategy
    params = strategy_config.threshold_params
    base = params.base_papers_per_cluster
    max_papers = strategy_config.max_papers_per_cluster
    category = _calculate_cluster_size_category(cluster_size, config)

    size_mapping = {
        "small": base,
        "medium": base + 1,
        "large": base + 2,
        "xlarge": base + 3,
        "xxlarge": base + 4,
        "xxxlarge": max_papers,
    }

    selected_count = size_mapping.get(category, base)
    return min(selected_count, cluster_size) 