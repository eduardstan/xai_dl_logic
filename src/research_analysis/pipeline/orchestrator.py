#!/usr/bin/env python3
"""
Main pipeline orchestrator for the research analysis framework.

This module orchestrates the complete pipeline execution across all stages,
managing timestamped output directories and configuration persistence.
"""
from datetime import datetime
from pathlib import Path
import yaml

from research_analysis.config.models import AppConfig
from research_analysis.pipeline.stage_1 import run_stage_1
from research_analysis.pipeline.stage_2 import run_stage_2
from research_analysis.pipeline.stage_3 import run_stage_3
from research_analysis.utils.logging import get_logger

logger = get_logger()


def run_full_pipeline(config: AppConfig) -> Path:
    """
    Execute the complete research analysis pipeline across all stages.

    Args:
        config: The application configuration.

    Returns:
        The path to the timestamped output directory.
    """
    timestamp = datetime.now().strftime(config.pipeline.reproducibility.timestamp_format)
    output_dir = Path(config.pipeline.paths.outputs) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting full pipeline execution - Output: {output_dir}")

    # Save pipeline-level configuration
    _save_pipeline_config(config, output_dir)

    # Execute all stages sequentially
    logger.info("=== Stage 1: Topic Modeling ===")
    run_stage_1(config, output_dir)

    logger.info("=== Stage 2: Paper Selection ===") 
    stage_1_output = output_dir / "stage_1_topic_model"
    stage_2_output = output_dir / "stage_2_paper_selection"
    run_stage_2(config, stage_1_output, stage_2_output)

    logger.info("=== Stage 3: Visualization ===")
    stage_3_output = output_dir / "stage_3_visualization"
    run_stage_3(config, stage_2_output, stage_3_output)

    logger.info(f"Pipeline completed successfully! Results: {output_dir}")
    return output_dir


def _save_pipeline_config(config: AppConfig, output_dir: Path) -> None:
    """
    Save the complete pipeline configuration.

    Args:
        config: The application configuration.
        output_dir: The timestamped output directory.
    """
    config_path = output_dir / "pipeline_config_used.yaml"
    
    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
    
    logger.info(f"Pipeline configuration saved to: {config_path}") 