#!/usr/bin/env python3
"""
Pydantic models for configuration files.

These models define the structure and validation rules for the YAML
configuration files, providing a strongly-typed and self-documenting
way to manage pipeline parameters.
"""

from typing import List, Dict, Any, Literal, Tuple
from pydantic import BaseModel, Field


# --- Models for pipeline.yaml ---

class PathsConfig(BaseModel):
    bibliography_file: str = "data/merged.bib"
    outputs: str = "outputs"
    logs: str = "logs"
    cache: str = "cache"
    stage_1_path: str = "stage_1_topic_model"
    stage_2_path: str = "stage_2_paper_selection"
    stage_3_path: str = "stage_3_visualization"

class LoggingConfig(BaseModel):
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "1 week"

class ReproducibilityConfig(BaseModel):
    random_seed: int = 42
    timestamp_format: str = "%Y%m%d_%H%M%S"

class PipelineConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)


# --- Models for stage_1_topic_model.yaml ---

class UMAPParams(BaseModel):
    n_neighbors: int = 15
    n_components: int = 128
    min_dist: float = 0.0
    metric: str = "cosine"

class HDBSCANParams(BaseModel):
    min_cluster_size: int = 16
    max_cluster_size: int = 128
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.1
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    prediction_data: bool = True

class VectorizerParams(BaseModel):
    stop_words: str = "english"
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    max_features: int = 5000

class BERTopicParams(BaseModel):
    min_topic_size: int = 15
    calculate_probabilities: bool = False
    verbose: bool = True

class EmbeddingModelConfig(BaseModel):
    name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    show_progress: bool = True
    device: str = "auto"
    cache_embeddings: bool = True

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
    save_intermediate_results: bool = True

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
    text_fields: List[str] = Field(default_factory=lambda: ["title", "abstract"])
    embedding_model: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    umap_params: UMAPParams = Field(default_factory=UMAPParams)
    hdbscan_params: HDBSCANParams = Field(default_factory=HDBSCANParams)
    vectorizer_params: VectorizerParams = Field(default_factory=VectorizerParams)
    bertopic_params: BERTopicParams = Field(default_factory=BERTopicParams)
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    outlier_reduction: OutlierReductionConfig
    domain_guidance: DomainGuidanceConfig = Field(default_factory=DomainGuidanceConfig)


# --- Models for stage_2_selection.yaml ---

class RatioParams(BaseModel):
    target_selection_ratio: float = 0.15
    min_papers_per_cluster: int = 2


class ThresholdParams(BaseModel):
    base_papers_per_cluster: int = 4
    cluster_thresholds: Dict[str, int]


class SelectionStrategyConfig(BaseModel):
    method: Literal["diverse_representative", "centroid_based"] = "diverse_representative"
    count_method: Literal["ratio_based", "threshold_based"] = "ratio_based"
    max_papers_per_cluster: int = Field(
        200, gt=0, description="Absolute maximum papers to select, regardless of strategy."
    )
    ratio_params: RatioParams = Field(default_factory=RatioParams)
    threshold_params: ThresholdParams


class MetricsConfig(BaseModel):
    diversity_weight: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description="Weight for diversity in the representativeness score (0.0=centrality, 1.0=diversity).",
    )
    min_diversity_threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum diversity score for a paper to be considered a candidate for selection.",
    )
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