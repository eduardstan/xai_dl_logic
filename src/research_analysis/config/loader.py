#!/usr/bin/env python3
"""
Configuration loading and validation module.

This module is responsible for loading the staged YAML configuration
files, validating them against Pydantic models, and providing a single,
merged, and strongly-typed configuration object for the application.
"""

from pathlib import Path
import yaml
from .models import AppConfig, PipelineConfig, Stage1Config, Stage2Config

def load_config(
    pipeline_config_path: str = "configs/pipeline.yaml",
    stage_1_config_path: str = "configs/stage_1_topic_model.yaml",
    stage_2_config_path: str = "configs/stage_2_selection.yaml",
) -> AppConfig:
    """
    Loads all configuration files and merges them into a single AppConfig object.

    Args:
        pipeline_config_path: Path to the main pipeline configuration file.
        stage_1_config_path: Path to the Stage 1 (topic modeling) config file.
        stage_2_config_path: Path to the Stage 2 (selection) config file.

    Returns:
        An instance of AppConfig containing the validated and merged settings.
    """
    # Load raw dictionaries from YAML files
    raw_pipeline = _load_yaml(pipeline_config_path)
    raw_stage_1 = _load_yaml(stage_1_config_path)
    raw_stage_2 = _load_yaml(stage_2_config_path)

    # Validate and structure each configuration part
    pipeline_conf = PipelineConfig(**raw_pipeline)
    stage_1_conf = Stage1Config(**raw_stage_1)
    stage_2_conf = Stage2Config(**raw_stage_2)

    # Combine into the final AppConfig object
    app_config = AppConfig(
        pipeline=pipeline_conf,
        stage_1=stage_1_conf,
        stage_2=stage_2_conf,
    )

    return app_config

def _load_yaml(config_path: str) -> dict:
    """A helper function to load a single YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) 