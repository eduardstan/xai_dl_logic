#!/usr/bin/env python3
"""
BERTopic outlier reduction module for the research analysis framework.
"""

from typing import List, Optional, Tuple

import numpy as np
from bertopic import BERTopic

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger

logger = get_logger()


def apply_outlier_reduction(
    model: BERTopic,
    docs: List[str],
    topics: List[int],
    probs: Optional[np.ndarray],
    embeddings: np.ndarray,
    config: AppConfig,
) -> Tuple[List[int], Optional[np.ndarray]]:
    """
    Apply a chain of outlier reduction strategies as defined in the config.

    Args:
        model: The trained BERTopic model.
        docs: The list of documents.
        topics: The initial topic assignments.
        probs: The initial topic probabilities.
        embeddings: The document embeddings.
        config: The application's configuration object.

    Returns:
        A tuple containing the updated topic assignments and probabilities.
    """
    reduction_config = config.stage_1.outlier_reduction
    if not reduction_config.enabled:
        logger.info("Outlier reduction is disabled by config.")
        return topics, probs

    logger.info("Starting outlier reduction process...")
    current_topics = topics
    current_probs = probs

    initial_outliers = sum(1 for t in topics if t == -1)

    for i, strategy_config in enumerate(reduction_config.strategies):
        strategy = strategy_config.strategy
        threshold = strategy_config.threshold
        current_outlier_count = sum(1 for t in current_topics if t == -1)

        if current_outlier_count == 0:
            logger.info("No outliers remaining. Stopping reduction chain.")
            break

        logger.info(
            f"Applying strategy {i+1}/{len(reduction_config.strategies)}: "
            f"'{strategy}' with threshold {threshold}. "
            f"({current_outlier_count} outliers remaining)"
        )

        try:
            new_topics = model.reduce_outliers(
                docs,
                current_topics,
                probabilities=current_probs if strategy == "probabilities" else None,
                strategy=strategy,
                threshold=threshold,
                embeddings=embeddings if strategy == "embeddings" else None,
            )

            outliers_reduced = current_outlier_count - sum(1 for t in new_topics if t == -1)
            if outliers_reduced > 0:
                logger.info(f"Strategy '{strategy}' reassigned {outliers_reduced} outliers.")
                current_topics = new_topics
            else:
                logger.info(f"Strategy '{strategy}' did not reduce further outliers.")

        except Exception as e:
            logger.error(f"Error applying outlier reduction strategy '{strategy}': {e}", exc_info=True)
            continue

    final_outliers = sum(1 for t in current_topics if t == -1)
    if final_outliers < initial_outliers:
        logger.info(
            "Outlier reduction complete. "
            f"Total reassigned: {initial_outliers - final_outliers}. "
            f"Final outlier count: {final_outliers}."
        )
        if not reduction_config.update_representations:
            logger.info("Preserving original topic representations as per config.")
            model.update_topics(docs, topics=current_topics)
        else:
            logger.warning("Updating topic representations. This may change topic meanings.")
            # Note: The original logic did not have an explicit update step.
            # This branch is for future use if needed, but default is to preserve.
            model.topics_ = current_topics


    else:
        logger.info("Outlier reduction strategies did not reassign any documents.")

    return current_topics, current_probs 