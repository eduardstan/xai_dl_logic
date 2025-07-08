#!/usr/bin/env python3
"""
BERTopic training module for the research analysis framework.
"""

from typing import List, Optional, Tuple

import numpy as np
from bertopic import BERTopic

from research_analysis.config.models import AppConfig
from research_analysis.stages.stage_1_topic_model.outlier_reduction import (
    apply_outlier_reduction,
)
from research_analysis.utils.logging import get_logger

logger = get_logger()


def train_topic_model(
    model: BERTopic,
    docs: List[str],
    embeddings: np.ndarray,
    config: AppConfig,
) -> Tuple[List[int], Optional[np.ndarray]]:
    """
    Train the BERTopic model and apply outlier reduction.

    Args:
        model: The configured BERTopic model instance.
        docs: A list of documents for training.
        embeddings: Pre-computed embeddings for the documents.
        config: The application's configuration object.

    Returns:
        A tuple containing the final topic assignments and probabilities.
    """
    logger.info("Starting BERTopic model training...")
    if len(docs) != len(embeddings):
        raise ValueError(
            f"Mismatch between number of documents ({len(docs)}) and "
            f"embeddings ({len(embeddings)})."
        )

    # Train the model
    topics, probs = model.fit_transform(docs, embeddings)
    _log_training_summary("Initial", docs, topics)

    # Apply outlier reduction
    final_topics, final_probs = apply_outlier_reduction(
        model, docs, topics, probs, embeddings, config
    )
    _log_training_summary("Final", docs, final_topics)

    return final_topics, final_probs


def _log_training_summary(
    stage: str, docs: List[str], topics: List[int]
) -> None:
    """Logs a summary of the topic modeling results at a given stage."""
    num_docs = len(docs)
    num_topics = len(set(topics)) - (1 if -1 in topics else 0)
    num_outliers = sum(1 for t in topics if t == -1)
    coverage = (num_docs - num_outliers) / num_docs * 100 if num_docs > 0 else 0

    logger.info(f"{stage} training summary:")
    logger.info(f"  - Topics discovered: {num_topics}")
    logger.info(f"  - Outlier documents: {num_outliers} ({100-coverage:.1f}%)")
    logger.info(f"  - Document coverage: {coverage:.1f}%") 