#!/usr/bin/env python3
"""
Embedding generation for pool augmentation.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger

logger = get_logger()

def compute_augmentation_embeddings(
    texts: List[str], config: AppConfig, model: SentenceTransformer = None
) -> np.ndarray:
    """
    Compute Sentence-BERT embeddings for augmentation papers.
    
    If 'model' is provided (e.g., from a loaded BERTopic instance), uses it.
    Otherwise, initializes a new SentenceTransformer using config settings.
    """
    if model is not None:
        logger.info(f"Using provided pretrained model for {len(texts)} papers...")
        return model.encode(
            texts,
            batch_size=config.stage_1.embedding_model.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    model_name = config.stage_1.embedding_model.name
    batch_size = config.stage_1.embedding_model.batch_size
    device = config.stage_1.embedding_model.device
    
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Computing embeddings for {len(texts)} papers using {model_name} on {device}...")
    
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings
