#!/usr/bin/env python3
"""
Orchestrator for the Stage 2 paper selection pipeline.

This module loads the artifacts from Stage 1 (topic modeling) and executes
the paper selection, analysis, and reporting workflow.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from bertopic import BERTopic

from research_analysis.config.models import AppConfig
from research_analysis.stages.stage_2_paper_selection.topic_processor import (
    process_topics,
)
from research_analysis.stages.stage_2_paper_selection.report import (
    generate_selection_report,
)
from research_analysis.utils.logging import get_logger
from research_analysis.utils.files import load_from_cache

logger = get_logger()


def run_stage_2(config: AppConfig, output_dir: Path) -> None:
    """
    Executes the full Stage 2 paper selection and analysis pipeline.

    Args:
        config: The application configuration.
        output_dir: The *root* output directory for the entire pipeline run.
    """
    logger.info("Starting Stage 2: Paper Selection and Analysis")
    
    # Construct input and output paths from config
    stage_1_input_dir = output_dir / config.pipeline.paths.stage_1_path
    stage_2_output_dir = output_dir / config.pipeline.paths.stage_2_path
    stage_2_output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load artifacts from the specified Stage 1 directory
    logger.info(f"Loading artifacts from {stage_1_input_dir}...")

    model_path = stage_1_input_dir / "bertopic_model"
    if not model_path.exists():
        raise FileNotFoundError(f"BERTopic model not found at {model_path}")
    topic_model = BERTopic.load(model_path)

    embeddings_path = stage_1_input_dir / "embeddings.pkl"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    embeddings = load_from_cache(embeddings_path)

    bib_with_topics_path = stage_1_input_dir / "bibliography_with_topics.csv"
    if not bib_with_topics_path.exists():
        raise FileNotFoundError(f"Bibliography with topics not found at {bib_with_topics_path}")
    documents_df = pd.read_csv(bib_with_topics_path)

    # 2. Run the core topic processing and paper selection
    (
        results_df,
        summary_df,
        selected_df,
    ) = process_topics(
        config=config,
        topic_model=topic_model,
        embeddings=embeddings,
        documents_df=documents_df,
    )

    # 3. Save the core results
    logger.info(f"Saving analysis dataframes to {stage_2_output_dir}...")
    results_df.to_csv(stage_2_output_dir / "comprehensive_analysis.csv", index=False)
    summary_df.to_csv(stage_2_output_dir / "selection_summary.csv", index=False)
    selected_df.to_csv(stage_2_output_dir / "selected_representatives.csv", index=False)

    # 4. Save configuration used for this run
    config_path = stage_2_output_dir / "stage_2_config_used.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
    logger.info(f"Stage 2 configuration saved to: {config_path}")

    # 5. Generate the final report
    generate_selection_report(config, results_df, summary_df, selected_df, stage_2_output_dir)

    logger.info("Stage 2 completed successfully!") 