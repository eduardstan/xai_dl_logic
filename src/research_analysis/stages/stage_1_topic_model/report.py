#!/usr/bin/env python3
"""
Results saving and reporting module for the research analysis framework.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml
from bertopic import BERTopic
import pickle

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger

logger = get_logger()


def save_results(
    model: BERTopic,
    df: pd.DataFrame,
    docs: List[str],
    topics: List[int],
    probs: Optional[np.ndarray],
    embeddings: np.ndarray,
    output_dir: Path,
    config: AppConfig,
) -> None:
    """
    Save all artifacts from the topic modeling stage.

    Args:
        model: The trained BERTopic model.
        df: The original DataFrame with bibliography data.
        docs: The list of processed documents.
        topics: The final topic assignments for each document.
        probs: The topic probabilities for each document.
        embeddings: The document embeddings.
        output_dir: The main output directory for the stage.
        config: The application's configuration object.
    """
    logger.info(f"Saving Stage 1 results to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the BERTopic model
    model_path = output_dir / "bertopic_model"
    model.save(str(model_path))
    logger.info(f"BERTopic model saved to: {model_path}")

    # Save embeddings as a pickle file for consistency
    embeddings_path = output_dir / "embeddings.pkl"
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"Embeddings saved to: {embeddings_path}")

    # Save topics as a pickle file
    topics_path = output_dir / "topics.pkl"
    with open(topics_path, "wb") as f:
        pickle.dump(topics, f)
    logger.info(f"Topics saved to: {topics_path}")

    # Save configuration used for this run
    config_path = output_dir / "config_used.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)
    logger.info(f"Run configuration saved to: {config_path}")

    # Create and save the results DataFrame
    df_results = df.copy()
    df_results["topic"] = topics
    if probs is not None and len(probs.shape) > 1:
        df_results["probability"] = probs.max(axis=1)
    else:
        df_results["probability"] = 0.0  # Assign a default value

    # Save the original documents DataFrame separately for Stage 2
    documents_path = output_dir / "documents.csv"
    df.to_csv(documents_path, index=False)
    logger.info(f"Original documents saved to: {documents_path}")

    results_path = output_dir / "bibliography_with_topics.csv"
    df_results.to_csv(results_path, index=False)
    logger.info(f"Bibliography with topics saved to: {results_path}")

    # Create and save the topic info DataFrame
    topic_info = model.get_topic_info()
    topic_counts = Counter(topics)
    topic_info["Count"] = topic_info["Topic"].map(topic_counts)
    topic_info = topic_info.sort_values("Count", ascending=False).reset_index(drop=True)
    topic_info_path = output_dir / "topic_info.csv"
    topic_info.to_csv(topic_info_path, index=False)
    logger.info(f"Topic info saved to: {topic_info_path}")

    # Generate and save the summary report, passing the corrected topic_info
    _generate_summary_report(
        topic_info=topic_info,
        num_docs=len(docs),
        topics=list(topics),
        output_dir=output_dir,
        config=config,
    )


def _generate_summary_report(
    topic_info: pd.DataFrame,
    num_docs: int,
    topics: List[int],
    output_dir: Path,
    config: AppConfig,
) -> None:
    """Generate and save a markdown summary report of the analysis."""
    num_topics = len(set(t for t in topics if t != -1))
    num_outliers = topics.count(-1)
    coverage = (num_docs - num_outliers) / num_docs * 100 if num_docs > 0 else 0

    report = f"""# Stage 1: Topic Modeling Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## 1. Overview
- **Documents Processed:** {num_docs:,}
- **Topics Discovered:** {num_topics}
- **Outlier Documents:** {num_outliers} ({100-coverage:.1f}%)
- **Document Coverage:** {coverage:.1f}%

## 2. Configuration
- **Embedding Model:** `{config.stage_1.embedding_model.name}`
- **UMAP Neighbors:** `{config.stage_1.umap_params.n_neighbors}`
- **HDBSCAN Min Cluster Size:** `{config.stage_1.hdbscan_params.min_cluster_size}`
- **Outlier Reduction Enabled:** `{config.stage_1.outlier_reduction.enabled}`

## 3. Top 10 Topics by Size
"""
    top_10 = topic_info[topic_info["Topic"] != -1].head(10)
    for _, row in top_10.iterrows():
        report += f"- **Topic {row['Topic']}**: {row['Count']} docs - _{row['Name']}_\n"

    report_path = output_dir / "stage_1_summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Summary report saved to: {report_path}") 