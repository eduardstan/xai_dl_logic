#!/usr/bin/env python3
"""
Stage 2 paper selection pipeline for the research analysis framework.

This module orchestrates the selection of representative papers from topic
clusters, including metric computation, diversity selection, and research
alignment analysis.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic

from research_analysis.config.models import AppConfig
from research_analysis.stages.stage_2_paper_selection.cluster_analysis import (
    determine_papers_to_select,
)
from research_analysis.stages.stage_2_paper_selection.metrics import (
    compute_paper_metrics,
)
from research_analysis.stages.stage_2_paper_selection.research_alignment import (
    analyze_research_alignment,
)
from research_analysis.stages.stage_2_paper_selection.selection import (
    assign_non_selected_papers,
    select_diverse_representatives,
)
from research_analysis.utils.logging import get_logger

logger = get_logger()


def run_stage_2_pipeline(
    config: AppConfig,
    topic_model: BERTopic,
    embeddings: np.ndarray,
    documents: pd.DataFrame,
    topics: list[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the paper selection and analysis pipeline for Stage 2.

    Args:
        config: The application configuration.
        topic_model: The trained BERTopic model.
        embeddings: The document embeddings.
        documents: The DataFrame of documents.
        topics: The list of topic assignments for each document.

    Returns:
        A tuple containing the comprehensive analysis DataFrame, the summary
        DataFrame, and the DataFrame of selected representative papers.
    """
    all_results = []
    topic_info = topic_model.get_topic_info()

    for topic_id in sorted(topic_info.Topic.unique()):
        if topic_id == -1:
            continue

        topic_mask = np.array(topics) == topic_id
        topic_indices = np.where(topic_mask)[0]
        topic_embeddings = embeddings[topic_mask]
        topic_documents = documents.iloc[topic_indices]

        cluster_size = len(topic_documents)
        n_select = determine_papers_to_select(cluster_size, config)
        logger.info(
            f"Topic {topic_id}: {cluster_size} papers -> selecting {n_select} reps"
        )

        selected_local_indices = select_diverse_representatives(
            topic_embeddings, n_select, config
        )
        assignments = assign_non_selected_papers(
            topic_embeddings, selected_local_indices, config
        )

        for local_idx, global_idx in enumerate(topic_indices):
            doc = documents.iloc[global_idx]
            full_text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('keywords', '')}"

            metrics = compute_paper_metrics(
                embeddings[global_idx], topic_embeddings, config
            )
            alignment = analyze_research_alignment(full_text, config)

            is_selected = local_idx in selected_local_indices
            rep_info = {}
            if not is_selected and local_idx in assignments:
                assignment = assignments[local_idx]
                rep_global_idx = topic_indices[assignment["representative_idx"]]
                rep_doc = documents.iloc[rep_global_idx]
                rep_info = {
                    "representative_title": rep_doc.get("title"),
                    "representative_authors": rep_doc.get("author"),
                    "similarity_to_representative": assignment["similarity"],
                }

            all_results.append(
                {
                    "topic_id": topic_id,
                    "is_selected_representative": is_selected,
                    "cluster_size": cluster_size,
                    "papers_selected_from_cluster": n_select,
                    **doc.to_dict(),
                    **metrics,
                    **alignment,
                    **rep_info,
                }
            )

    results_df = pd.DataFrame(all_results)
    summary_df = _create_summary_df(results_df, topic_info, config)
    selected_df = results_df[
        results_df["is_selected_representative"]
    ].reset_index(drop=True)

    return results_df, summary_df, selected_df


def _create_summary_df(
    results_df: pd.DataFrame, topic_info: pd.DataFrame, config: AppConfig
) -> pd.DataFrame:
    """Creates a summary DataFrame with statistics for each topic."""
    summary_data = []
    for topic_id, group in results_df.groupby("topic_id"):
        if topic_id == -1:
            continue

        selected_papers = group[group["is_selected_representative"]]
        topic_name = topic_info.loc[
            topic_info.Topic == topic_id, "Name"
        ].iloc[0]

        summary_data.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "cluster_size": len(group),
                "papers_selected": len(selected_papers),
                "selection_ratio": len(selected_papers) / len(group)
                if len(group) > 0
                else 0,
                "avg_centrality": selected_papers[
                    "similarity_to_centroid"
                ].mean(),
                "avg_diversity": selected_papers["diversity_score"].mean(),
            }
        )

    return pd.DataFrame(summary_data) 