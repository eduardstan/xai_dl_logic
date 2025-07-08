#!/usr/bin/env python3
"""
Pydantic models for configuration files.

These models define the structure and validation rules for the YAML
configuration files, providing a strongly-typed and self-documenting
way to manage pipeline parameters.
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field


# --- Models for pipeline.yaml ---

class PathsConfig(BaseModel):
    bibliography_file: str = "data/merged.bib"
    outputs: str = "outputs"
    logs: str = "logs"
    cache: str = "cache"

class LoggingConfig(BaseModel):
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "1 week"

class ReproducibilityConfig(BaseModel):
    random_seed: int = 42

class PipelineConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)


# --- Models for stage_1_topic_model.yaml ---

class EmbeddingModelConfig(BaseModel):
    name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    show_progress: bool = True
    device: str = "auto"
    cache_embeddings: bool = True

class UmapParams(BaseModel):
    n_neighbors: int = 15
    n_components: int = 128
    min_dist: float = 0.0
    metric: str = "cosine"

class HdbscanParams(BaseModel):
    min_cluster_size: int = 16
    max_cluster_size: int = 128
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.1
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True

class DataProcessingConfig(BaseModel):
    text_fields: List[str] = ["abstract", "title", "keywords"]

class OutlierReductionStrategy(BaseModel):
    strategy: Literal["c-tf-idf", "embeddings", "distributions"]
    threshold: float
    distributions_params: Dict[str, Any] = None

class OutlierReductionConfig(BaseModel):
    enabled: bool = True
    strategies: List[OutlierReductionStrategy]
    update_representations: bool = False

class SeedWordsConfig(BaseModel):
    enabled: bool = False
    multiplier: float = 2.0
    words: List[str] = []

class GuidedTopic(BaseModel):
    seeds: List[str]

class GuidedTopicsConfig(BaseModel):
    enabled: bool = False
    topics: Dict[str, GuidedTopic] = {}

class DomainGuidanceConfig(BaseModel):
    seed_words: SeedWordsConfig = Field(default_factory=SeedWordsConfig)
    guided_topics: GuidedTopicsConfig = Field(default_factory=GuidedTopicsConfig)

class Stage1Config(BaseModel):
    embedding_model: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    umap_params: UmapParams = Field(default_factory=UmapParams)
    hdbscan_params: HdbscanParams = Field(default_factory=HdbscanParams)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    outlier_reduction: OutlierReductionConfig
    domain_guidance: DomainGuidanceConfig = Field(default_factory=DomainGuidanceConfig)


# --- Models for stage_2_selection.yaml ---

class RatioParams(BaseModel):
    target_selection_ratio: float = 0.15
    min_papers_per_cluster: int = 2

class ThresholdParams(BaseModel):
    base_papers_per_cluster: int = 4
    max_papers_per_cluster: int = 200
    cluster_thresholds: Dict[str, int]

class SelectionStrategyConfig(BaseModel):
    method: Literal["diverse_representative", "centroid_based"] = "diverse_representative"
    count_method: Literal["ratio_based", "threshold_based"] = "ratio_based"
    ratio_params: RatioParams = Field(default_factory=RatioParams)
    threshold_params: ThresholdParams

class MetricsConfig(BaseModel):
    diversity_weight: float = 0.75
    similarity_threshold: float = 0.8
    use_iterative_selection: bool = True
    selection_iterations: int = 20
    compute: List[str]

class ResearchAlignmentConfig(BaseModel):
    xai: List[str]
    symbolic: List[str]
    subsymbolic: List[str]

class Stage2Config(BaseModel):
    selection_strategy: SelectionStrategyConfig
    metrics: MetricsConfig
    research_alignment: ResearchAlignmentConfig


# --- Top-level Application Configuration Model ---

class AppConfig(BaseModel):
    """The complete, validated, and merged configuration for the application."""
    pipeline: PipelineConfig
    stage_1: Stage1Config
    stage_2: Stage2Config 