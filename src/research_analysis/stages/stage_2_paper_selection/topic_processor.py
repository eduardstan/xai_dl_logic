#!/usr/bin/env python3
"""
Topic processing and paper selection for the research analysis pipeline.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from research_analysis.config.models import AppConfig
from research_analysis.stages.stage_2_paper_selection import (
    cluster_analysis,
    metrics,
    research_alignment,
    selection,
)
from research_analysis.utils.logging import get_logger

logger = get_logger()


def process_topics(
    config: AppConfig,
    topic_model: BERTopic,
    embeddings: np.ndarray,
    documents_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process topics to select diverse representative papers.

    Args:
        config: The application configuration.
        topic_model: The trained BERTopic model.
        embeddings: The document embeddings.
        documents_df: DataFrame containing documents and their assigned topics.

    Returns:
        A tuple containing the comprehensive analysis, summary, and selected papers DataFrames.
    """
    logger.info("Processing topics to select representative papers...")

    all_results = []
    topic_info = topic_model.get_topic_info()
    topics = documents_df["topic"].tolist()

    for topic_id in sorted(topic_info.Topic.unique()):
        if topic_id == -1:
            continue

        topic_mask = np.array(topics) == topic_id
        topic_indices = np.where(topic_mask)[0]
        topic_embeddings = embeddings[topic_mask]
        topic_documents = documents_df.iloc[topic_indices]

        if topic_documents.empty:
            continue

        n_select = cluster_analysis.determine_papers_to_select(
            len(topic_documents), config
        )
        logger.info(
            f"Topic {topic_id}: {len(topic_documents)} papers -> selecting {n_select} reps"
        )

        selected_local_indices = selection.select_diverse_representatives(
            topic_embeddings, n_select, config
        )
        assignments = selection.assign_non_selected_papers(
            topic_embeddings, selected_local_indices, config
        )

        # Precompute medoid if needed
        medoid_embedding = None
        if config.stage_2.augmentation.run_medoid_analysis:
            from research_analysis.utils.math import compute_cluster_medoid
            medoid_embedding, _ = compute_cluster_medoid(topic_embeddings)

        for local_idx, global_idx in enumerate(topic_indices):
            doc = documents_df.iloc[global_idx]
            
            # Create doc_info matching legacy format exactly
            doc_info = {
                'title': doc.get('title', 'Unknown'),
                'authors': doc.get('author', 'Unknown'),  # Legacy uses 'authors' but source has 'author'
                'year': doc.get('year', 'Unknown'),
                'journal': doc.get('journal', 'Unknown'),
                'doi': doc.get('doi', ''),
                'abstract': doc.get('abstract', ''),
                'keywords': doc.get('keywords', '')
            }
            
            paper_metrics = metrics.compute_paper_metrics(
                embeddings[global_idx], 
                topic_embeddings, 
                config,
                medoid_embedding=medoid_embedding
            )
            
            # Fix metrics key to match legacy naming
            if 'avg_similarity_to_cluster' in paper_metrics:
                paper_metrics['similarity_to_cluster_papers'] = paper_metrics.pop('avg_similarity_to_cluster')
            
            # Get research alignment
            full_text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('keywords', '')}"
            alignment = research_alignment.analyze_research_alignment(full_text, config)

            is_selected = local_idx in selected_local_indices
            representative_info = {}
            
            # Process non-selected papers assignment info to match legacy format
            if not is_selected and local_idx in assignments:
                assignment = assignments[local_idx]
                rep_global_idx = topic_indices[assignment["representative_idx"]]
                rep_doc = documents_df.iloc[rep_global_idx]
                representative_info = {
                    'representative_title': rep_doc.get("title", 'Unknown'),
                    'representative_authors': rep_doc.get("author", 'Unknown'),
                    'similarity_to_representative': assignment["similarity"],
                    'is_similar_to_representative': assignment.get("is_similar", False)
                }

            # Create result structure matching legacy exactly
            result = {
                'topic_id': topic_id,
                'global_index': global_idx,
                'local_index': local_idx,
                'is_selected_representative': is_selected,
                'cluster_size': len(topic_documents),
                'papers_selected_from_cluster': n_select,
                **doc_info,
                **paper_metrics,
                **alignment,
                **representative_info,
            }

            all_results.append(result)

    if not all_results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    summary_df = _create_summary_df(results_df)
    selected_df = results_df[results_df["is_selected_representative"]].reset_index(
        drop=True
    )

    logger.info("Successfully processed topics and selected representatives.")
    
    # Extract topic centers for reuse
    topic_centers = {}
    for topic_id in sorted(topic_info.Topic.unique()):
        if topic_id == -1: continue
        topic_mask = np.array(topics) == topic_id
        if not any(topic_mask): continue
        topic_embeddings = embeddings[topic_mask]
        from research_analysis.utils.math import compute_cluster_centroid, compute_cluster_medoid
        centroid = compute_cluster_centroid(topic_embeddings)
        medoid, _ = compute_cluster_medoid(topic_embeddings)
        topic_centers[topic_id] = {'centroid': centroid, 'medoid': medoid}

    return results_df, summary_df, selected_df, topic_centers


def _create_summary_df(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create a topic-level summary DataFrame."""
    if results_df.empty:
        return pd.DataFrame()

    summary = (
        results_df.groupby("topic_id")
        .agg(
            cluster_size=("cluster_size", "first"),
            papers_selected_from_cluster=("papers_selected_from_cluster", "first"),
            avg_centrality=("similarity_to_centroid", "mean"),
            avg_diversity=("diversity_score", "mean"),
        )
        .reset_index()
    )
    return summary