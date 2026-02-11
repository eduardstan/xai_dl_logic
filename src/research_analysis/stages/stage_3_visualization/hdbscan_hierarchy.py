#!/usr/bin/env python3
"""
HDBSCAN hierarchy visualization for the research analysis framework.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bertopic import BERTopic
from research_analysis.utils.logging import get_logger

logger = get_logger()

def visualize_hdbscan_structure(
    topic_model: BERTopic, 
    output_dir: Path,
    docs: list = None
) -> None:
    """
    Generates visualizations of the topic hierarchy using BERTopic's built-in tools.
    
    Args:
        topic_model: The trained BERTopic model.
        output_dir: Directory to save the plots.
        docs: List of documents used to compute the hierarchy labels.
    """
    logger.info("Generating BERTopic hierarchy visualizations...")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Precompute hierarchical topics if docs are provided
    hierarchical_topics = None
    if docs:
        try:
            logger.info(f"Computing hierarchical topics for {len(docs)} documents...")
            hierarchical_topics = topic_model.hierarchical_topics(docs)
        except Exception as e:
            logger.error(f"Failed to compute hierarchical topics: {e}")
    else:
        logger.warning("Docs missing. Skipping hierarchical topic computation.")

    # 1. Interactive Hierarchy (Dendrogram)
    try:
        fig = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        hier_html = output_dir / "topic_hierarchy.html"
        fig.write_html(str(hier_html))
        logger.info(f"Saved interactive hierarchy to {hier_html}")
        
        # Also try to save a static version if possible
        try:
            fig.write_image(str(output_dir / "topic_hierarchy.png"), engine="kaleido")
            logger.info(f"Saved static hierarchy to {output_dir / 'topic_hierarchy.png'}")
        except Exception:
            logger.debug("Static image export failed (Kaleido not installed), skipping .png hierarchy.")
            
    except Exception as e:
        logger.error(f"Failed to generate hierarchy visualization: {e}")

    # 2. Textual Topic Tree
    try:
        # get_topic_tree requires hierarchical_topics
        if hierarchical_topics is not None:
            tree = topic_model.get_topic_tree(hierarchical_topics)
            tree_path = output_dir / "topic_tree.txt"
            with open(tree_path, "w") as f:
                f.write(tree)
            logger.info(f"Saved textual topic tree to {tree_path}")
        else:
            logger.warning("Skipping topic_tree.txt as hierarchical_topics were not computed.")
    except Exception as e:
        logger.error(f"Failed to generate topic tree: {e}")

    # 3. Topic Similarity Matrix
    try:
        fig_sim = topic_model.visualize_heatmap()
        sim_html = output_dir / "topic_similarity_matrix.html"
        fig_sim.write_html(str(sim_html))
        logger.info(f"Saved topic similarity matrix to {sim_html}")
    except Exception as e:
        logger.error(f"Failed to generate similarity matrix: {e}")
