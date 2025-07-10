#!/usr/bin/env python3
"""
Temporary test script to run pipeline stages.
"""
import time
from pathlib import Path

from research_analysis.config.loader import load_config
from research_analysis.pipeline.stage_2 import run_stage_2
from research_analysis.utils.logging import setup_logging

# --- Configuration for the test ---
STAGE_1_TIMESTAMP = "20250710_081548" # Manually set from the last Stage 1 run

def run_stage_2_task():
    """Runs Stage 2 in a detached manner, using output from a previous Stage 1 run."""
    config = load_config(
        pipeline_config_path="configs/pipeline.yaml",
        stage_1_config_path="configs/stage_1_topic_model.yaml",
        stage_2_config_path="configs/stage_2_selection.yaml",
    )
    setup_logging(config.pipeline.logging)

    # Define input from the specific Stage 1 run
    stage_1_input_dir = Path(config.pipeline.paths.outputs) / STAGE_1_TIMESTAMP / "stage_1_topic_model"
    
    # Define a new, separate output directory for Stage 2
    stage_2_output_dir = Path(config.pipeline.paths.outputs) / STAGE_1_TIMESTAMP / "stage_2_paper_selection"
    stage_2_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Running Stage 2 ---")
    print(f"--- Input from: {stage_1_input_dir} ---")
    print(f"--- Output to: {stage_2_output_dir} ---")

    run_stage_2(
        config=config,
        stage_1_input_dir=stage_1_input_dir,
        output_dir=stage_2_output_dir
    )
    print(f"--- Stage 2 Finished ---")


if __name__ == "__main__":
    run_stage_2_task() 