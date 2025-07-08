#!/usr/bin/env python3
"""
BERTopic model setup and configuration for the research analysis framework.
"""

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger

logger = get_logger()


def setup_bertopic_model(config: AppConfig) -> BERTopic:
    """
    Configure and initialize a BERTopic model from configuration settings.

    This function assembles the necessary components for the BERTopic model,
    including UMAP for dimensionality reduction, HDBSCAN for clustering,
    and a CountVectorizer for tokenization, all based on parameters
    defined in the application's configuration.

    Args:
        config: The application's configuration object.

    Returns:
        A fully configured BERTopic model instance, ready for training.
    """
    logger.info("Setting up BERTopic model with parameters from config.")

    stage_config = config.stage_1
    umap_params = stage_config.umap_params.dict()
    hdbscan_params = stage_config.hdbscan_params.dict()
    vectorizer_params = stage_config.vectorizer_params.dict()
    bertopic_params = stage_config.bertopic_params.dict()

    # Ensure reproducibility for UMAP
    umap_params["random_state"] = config.pipeline.reproducibility.random_seed
    logger.info(f"UMAP params: {umap_params}")
    umap_model = UMAP(**umap_params)

    logger.info(f"HDBSCAN params: {hdbscan_params}")
    hdbscan_model = HDBSCAN(**hdbscan_params)

    # The vectorizer in the legacy code had hardcoded stop words.
    # The new approach defines this in the config for flexibility.
    logger.info(f"Vectorizer params: {vectorizer_params}")
    vectorizer_model = CountVectorizer(**vectorizer_params)

    logger.info(f"BERTopic core params: {bertopic_params}")
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        **bertopic_params,
    )

    logger.info("BERTopic model setup complete.")
    return topic_model 