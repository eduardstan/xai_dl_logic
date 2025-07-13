#!/usr/bin/env python3
"""
Orchestrator for Stage 1 of the research analysis pipeline: Topic Modeling.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from research_analysis.config.models import AppConfig
from research_analysis.stages.stage_1_topic_model import (
    bib_parser,
    embeddings,
    model_setup,
    outlier_reduction,
    report,
    training,
)
from research_analysis.utils.logging import get_logger

logger = get_logger()


def run_stage_1(config: AppConfig, output_dir: Path) -> None:
    """
    Executes the full pipeline for Stage 1: Topic Modeling.

    Args:
        config: The application's configuration object.
        output_dir: The *root* output directory for the entire pipeline run.
    """
    logger.info("--- Starting Stage 1: Topic Modeling ---")
    
    # Create a dedicated subdirectory for this stage's artifacts
    stage_output_dir = output_dir / config.pipeline.paths.stage_1_path
    stage_output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Parse bibliography data
    df = bib_parser.parse_bib_file(config)

    # 2. Prepare embeddings
    docs = df["combined_text"].tolist()
    embeddings_array = embeddings.prepare_embeddings(docs, config)

    # 3. Set up BERTopic model
    model = model_setup.setup_bertopic_model(config)

    # 4. Train the model (decoupled from outlier reduction)
    initial_topics, initial_probs = training.train_topic_model(
        model, docs, embeddings_array, config
    )

    # 5. Apply outlier reduction only if enabled
    if config.stage_1.outlier_reduction.enabled:
        final_topics, final_probs = outlier_reduction.apply_outlier_reduction(
            model, docs, initial_topics, initial_probs, embeddings_array, config
        )
    else:
        logger.info("Skipping outlier reduction as it is disabled in the config.")
        final_topics, final_probs = initial_topics, initial_probs


    # 6. Save all results and generate report
    report.save_results(
        model=model,
        df=df,
        docs=docs,
        topics=final_topics,
        probs=final_probs,
        embeddings=embeddings_array,
        config=config,
        output_dir=stage_output_dir,
    )

    logger.info("--- Stage 1: Topic Modeling Finished ---") 