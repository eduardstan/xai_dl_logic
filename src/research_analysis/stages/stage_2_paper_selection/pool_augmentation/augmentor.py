#!/usr/bin/env python3
"""
Orchestrator for paper pool augmentation.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger
from .bib_parser import parse_augmentation_bib
from .embeddings import compute_augmentation_embeddings
from .topic_assignment import assign_to_topics
from .eligibility import filter_eligible_papers

logger = get_logger()

def augment_pool(
    config: AppConfig,
    frozen_embeddings: np.ndarray,
    frozen_topics: np.ndarray,
    embedding_model: any = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, np.ndarray]]:
    """
    Refined main entry point for paper pool augmentation.
    
    This function now focuses on processing the NEW papers only,
    allowing the orchestrator to decide when and how to merge them.
    
    Returns:
        - augmentation_df: DataFrame of new papers with assigned topics and centrality.
        - augmentation_embeddings: Embeddings for these target papers.
        - centroids: The topic centroids used for assignment.
    """
    logger.info("--- Starting Paper Pool Augmentation (R1) ---")
    
    # 1. Parse new papers
    new_df = parse_augmentation_bib(config)
    
    # 2. Embed new papers
    new_embeddings = compute_augmentation_embeddings(
        new_df["combined_text"].tolist(), config, model=embedding_model
    )
    
    # 3. Assign to topics (computes centroids from frozen data)
    new_df, centroids = assign_to_topics(
        new_df, new_embeddings, frozen_embeddings, frozen_topics
    )
    
    # 4. Filter eligibility (e.g. duplicates against frozen pool)
    # We pass an empty existing_df if we just want the new ones processed for now,
    # or the frozen one to check for duplicates.
    # User said "Include all 11", so we mostly check for technical validity.
    eligible_df = filter_eligible_papers(new_df, pd.DataFrame(), config)
    
    # Sync embeddings
    eligible_indices = eligible_df.index
    eligible_embeddings = new_embeddings[eligible_indices]
    
    logger.info(f"Processed {len(eligible_df)} augmentation papers.")
    
    return eligible_df, eligible_embeddings, centroids
