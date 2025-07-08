#!/usr/bin/env python3
"""
Temporary test script for the model_setup module.
"""
from bertopic import BERTopic

from research_analysis.config.loader import load_config
from research_analysis.stages.stage_1_topic_model.model_setup import setup_bertopic_model
from research_analysis.utils.logging import setup_logging, get_logger


def main():
    """
    Tests the model_setup module.
    """
    config = load_config()
    setup_logging(config.pipeline.logging)
    logger = get_logger()
    logger.info("Starting test for model_setup module.")

    try:
        topic_model = setup_bertopic_model(config)
        logger.info(f"Successfully set up BERTopic model of type: {type(topic_model)}")
        assert isinstance(topic_model, BERTopic), "Model is not a BERTopic instance."
        logger.info("Test passed: Model is a valid BERTopic instance.")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main() 