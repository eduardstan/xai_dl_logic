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

    # 2. Run the core topic processing and paper selection on the ORIGINAL pool
    (
        results_df,
        summary_df,
        selected_df,
        topic_centers,
    ) = process_topics(
        config=config,
        topic_model=topic_model,
        embeddings=embeddings,
        documents_df=documents_df,
    )

    # 3. Optional Centroid vs Medoid Analysis (on ORIGINAL pool)
    if config.stage_2.augmentation.run_medoid_analysis:
        from research_analysis.stages.stage_2_paper_selection.medoid_analysis import run_centroid_vs_medoid_analysis
        import json
        logger.info("Running Centroid vs Medoid comparison analysis (Original Pool)...")
        
        medoid_results = []
        unique_topics = sorted([t for t in documents_df["topic"].unique() if t != -1])
        paper_ids = documents_df["id"].tolist() if "id" in documents_df.columns else documents_df["ID"].tolist()
        
        for tid in unique_topics:
            topic_mask = documents_df["topic"].values == tid
            cluster_embeddings = embeddings[topic_mask]
            topic_paper_ids = [paper_ids[i] for i, mask in enumerate(topic_mask) if mask]
            
            analysis = run_centroid_vs_medoid_analysis(
                int(tid), cluster_embeddings, topic_paper_ids, top_n=5
            )
            medoid_results.append(analysis)
            
        medoid_path = stage_2_output_dir / "medoid_comparison.json"
        
        # Ensure all values are JSON serializable (numpy types to native)
        def _make_serializable(obj):
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_make_serializable(v) for v in obj]
            if hasattr(obj, "item"): # Numpy types
                return obj.item()
            return obj

        with open(medoid_path, "w") as f:
            json.dump(_make_serializable(medoid_results), f, indent=4)
        logger.info(f"Medoid comparison results saved to: {medoid_path}")

    # 4. Augment results with R1 papers if enabled
    if config.stage_2.augmentation.enabled:
        from research_analysis.stages.stage_2_paper_selection.pool_augmentation import augment_pool
        from research_analysis.stages.stage_2_paper_selection import metrics, research_alignment
        
        logger.info("Post-processing R1 augmentation papers...")
        r1_df, r1_embeddings, _ = augment_pool(
            config=config,
            frozen_embeddings=embeddings,
            frozen_topics=documents_df["topic"].values,
            embedding_model=topic_model.embedding_model
        )
        
        r1_entries = []
        for i, (local_idx, row) in enumerate(r1_df.iterrows()):
            tid = row["topic"]
            if tid not in topic_centers:
                continue # Skip outliers or unassigned
            
            centers = topic_centers[tid]
            
            # Compute metrics relative to original cluster
            topic_mask = documents_df["topic"].values == tid
            topic_embeddings = embeddings[topic_mask]
            
            r1_metrics = metrics.compute_paper_metrics(
                r1_embeddings[i],
                topic_embeddings,
                config,
                centroid_embedding=centers['centroid'],
                medoid_embedding=centers['medoid']
            )
            
            # Legacy mapping
            if 'avg_similarity_to_cluster' in r1_metrics:
                r1_metrics['similarity_to_cluster_papers'] = r1_metrics.pop('avg_similarity_to_cluster')
            
            # Alignment and other info
            full_text = f"{row.get('title', '')} {row.get('abstract', '')} {row.get('keywords', '')}"
            alignment = research_alignment.analyze_research_alignment(full_text, config)
            
            entry = {
                'topic_id': tid,
                'is_selected_representative': False,
                'cluster_size': len(topic_embeddings),
                'title': row.get('title', 'Unknown'),
                'authors': row.get('author', 'Unknown'),
                'year': row.get('year', 'Unknown'),
                'journal': row.get('journal', 'Unknown'),
                'doi': row.get('doi', ''),
                'source': 'r1_reviewer',
                'prisma_pathway': 'other_methods',
                **r1_metrics,
                **alignment
            }
            r1_entries.append(entry)
            
        if r1_entries:
            r1_results_df = pd.DataFrame(r1_entries)
            results_df = pd.concat([results_df, r1_results_df], ignore_index=True)
            logger.info(f"Augmented results with {len(r1_results_df)} R1 papers.")

    # 5. Save the core results
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