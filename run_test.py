#!/usr/bin/env python3
"""
Temporary test script to run the full pipeline orchestrator.
"""
import time
from pathlib import Path

from research_analysis.config.loader import load_config
from research_analysis.pipeline.orchestrator import run_full_pipeline
from research_analysis.utils.logging import setup_logging


def test_full_pipeline():
    """Test the full pipeline orchestrator with the new consistent API."""
    config = load_config(
        pipeline_config_path="configs/pipeline.yaml",
        stage_1_config_path="configs/stage_1_topic_model.yaml",
        stage_2_config_path="configs/stage_2_selection.yaml",
    )
    setup_logging(config.pipeline.logging)

    print(f"--- Testing Full Pipeline Orchestrator ---")
    print(f"--- Configuration Paths ---")
    print(f"  Stage 1: {config.pipeline.paths.stage_1_path}")
    print(f"  Stage 2: {config.pipeline.paths.stage_2_path}")
    print(f"  Stage 3: {config.pipeline.paths.stage_3_path}")
    print(f"--- Starting Pipeline ---")

    output_dir = run_full_pipeline(config)
    
    print(f"--- Pipeline Completed ---")
    print(f"--- Output Directory: {output_dir} ---")


if __name__ == "__main__":
    test_full_pipeline() 