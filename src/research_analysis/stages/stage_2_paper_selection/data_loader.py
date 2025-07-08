#!/usr/bin/env python3
"""
Data loader for Stage 2 of the research analysis pipeline.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from research_analysis.utils.logging import get_logger

logger = get_logger()


def load_stage_1_artifacts(
    stage_1_output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load the key artifacts produced by Stage 1.

    Args:
        stage_1_output_dir: The directory containing the outputs of Stage 1.

    Returns:
        A tuple containing:
        - The topic information DataFrame.
        - The bibliography DataFrame with topic assignments.
        - The document embeddings NumPy array.
    """
    logger.info(f"Loading artifacts from Stage 1 output: {stage_1_output_dir}")

    topic_info_path = stage_1_output_dir / "topic_info.csv"
    bib_with_topics_path = stage_1_output_dir / "bibliography_with_topics.csv"
    embeddings_path = stage_1_output_dir / "embeddings.npy"

    for path in [topic_info_path, bib_with_topics_path, embeddings_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required Stage 1 artifact not found: {path}")

    topic_info_df = pd.read_csv(topic_info_path)
    logger.info(f"Loaded topic info: {topic_info_df.shape[0]} topics")

    bib_df = pd.read_csv(bib_with_topics_path)
    logger.info(f"Loaded bibliography with topics: {bib_df.shape[0]} documents")

    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")

    # Validate data consistency
    if bib_df.shape[0] != embeddings.shape[0]:
        raise ValueError(
            "Mismatch between number of documents in bibliography "
            f"({bib_df.shape[0]}) and embeddings ({embeddings.shape[0]})."
        )

    return topic_info_df, bib_df, embeddings 