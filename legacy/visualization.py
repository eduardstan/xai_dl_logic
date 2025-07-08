#!/usr/bin/env python3
"""
BERTopic Visualization Module
Generates comprehensive visualizations following BERTopic best practices.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from bertopic import BERTopic
from umap import UMAP
from loguru import logger

from utils import get_plots_dir, get_visualization_config


def create_visualizations(topic_model: BERTopic, docs: List[str], 
                         embeddings: np.ndarray, topics: List[int], config: Dict,
                         output_dir: str) -> None:
    """
    Generate comprehensive visualizations following BERTopic best practices.
    Now supports updated topic assignments for consistent artifacts.
    
    Args:
        topic_model: Trained BERTopic model
        docs: List of documents
        embeddings: Document embeddings
        topics: Updated topic assignments (post-outlier reduction)
        config: Configuration dictionary
        output_dir: Output directory for plots
    """
    logger.info("🎨 Creating visualizations with updated topic assignments and preserved representations")
    
    viz_config = get_visualization_config(config)
    plots_dir = Path(output_dir) / get_plots_dir(config)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Model now has updated topic assignments with preserved original representations
    # No need for additional updates - visualizations will show correct assignments with quality names
    
    # Generate core visualizations
    _create_topic_overview(topic_model, plots_dir)
    _create_topic_hierarchy(topic_model, plots_dir)
    _create_topic_heatmap(topic_model, plots_dir)
    
    # Generate interactive document visualizations if enabled
    if viz_config['interactive_datamapplot']:
        _create_document_visualizations(
            topic_model, docs, embeddings, viz_config, config, plots_dir
        )
    
    # Generate topic terms visualization
    _create_topic_barchart(topic_model, plots_dir)
    
    logger.info(f"✅ All visualizations saved to: {plots_dir}")


def _create_topic_overview(topic_model: BERTopic, plots_dir: Path) -> None:
    """
    Create interactive topic overview visualization showing topic similarity and relationships.
    
    Args:
        topic_model: Trained BERTopic model with topic representations
        plots_dir: Directory path for saving plot files
    
    Note:
        - Generates 2D scatter plot of topics based on similarity
        - Shows topic clusters and relationships
        - Saves as interactive HTML file: topics_overview.html
        - Essential for understanding topic landscape
    """
    logger.info("Creating topic overview visualization")
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(str(plots_dir / "topics_overview.html"))


def _create_topic_hierarchy(topic_model: BERTopic, plots_dir: Path) -> None:
    """
    Create hierarchical topic visualization showing topic merging relationships.
    
    Args:
        topic_model: Trained BERTopic model with hierarchical topic information
        plots_dir: Directory path for saving plot files
    
    Note:
        - Generates dendrogram showing how topics merge hierarchically
        - Helps understand topic relationships and similarity
        - Saves as interactive HTML file: topics_hierarchy.html
        - Useful for identifying topic granularity and structure
    """
    logger.info("Creating hierarchical topic visualization")
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html(str(plots_dir / "topics_hierarchy.html"))


def _create_topic_heatmap(topic_model: BERTopic, plots_dir: Path) -> None:
    """
    Create topic similarity heatmap showing pairwise topic relationships.
    
    Args:
        topic_model: Trained BERTopic model with topic representations
        plots_dir: Directory path for saving plot files
    
    Note:
        - Generates heatmap matrix of topic-topic similarities
        - Color intensity indicates similarity strength
        - Saves as interactive HTML file: topics_heatmap.html
        - Helps identify related topics and potential redundancies
    """
    logger.info("Creating topic similarity heatmap")
    fig_heatmap = topic_model.visualize_heatmap()
    fig_heatmap.write_html(str(plots_dir / "topics_heatmap.html"))


def _create_document_visualizations(topic_model: BERTopic, docs: List[str], 
                                   embeddings: np.ndarray, viz_config: Dict,
                                   config: Dict, plots_dir: Path) -> None:
    """
    Create interactive document visualizations with DataMapPlot and standard plots.
    
    Args:
        topic_model: Trained BERTopic model
        docs: List of documents
        embeddings: Document embeddings
        viz_config: Visualization configuration
        config: Full configuration dictionary
        plots_dir: Directory path for saving plot files
    
    Note:
        - Generates both DataMapPlot and standard document visualizations
        - Uses UMAP for 2D dimensionality reduction if configured
        - Saves as interactive HTML files: documents_interactive_datamap.html, documents_plotly.html
        - Shows document distribution across topics with interactive hover details
    """
    logger.info("Creating interactive document visualization with DataMapPlot")
    
    # Reduce embeddings for visualization if configured
    if viz_config['reduce_embeddings_for_viz']:
        reduced_embeddings = UMAP(
            n_neighbors=10,
            n_components=2,
            min_dist=0.0,
            metric='cosine',
            random_state=config['random_seed']
        ).fit_transform(embeddings)
        
        # Interactive DataMapPlot - uses preserved topic names with updated assignments
        fig_datamap = topic_model.visualize_document_datamap(
            docs,
            reduced_embeddings=reduced_embeddings,
            interactive=True
        )
        
        # Regular document plot for comparison - uses preserved topic names with updated assignments
        fig_docs = topic_model.visualize_documents(
            docs,
            reduced_embeddings=reduced_embeddings,
            hide_document_hover=viz_config['hide_document_hover']
        )
        
    else:
        fig_datamap = topic_model.visualize_document_datamap(
            docs,
            embeddings=embeddings,
            interactive=True
        )
        
        fig_docs = topic_model.visualize_documents(
            docs,
            embeddings=embeddings,
            hide_document_hover=viz_config['hide_document_hover']
        )
    
    # Save interactive plots
    fig_datamap.save(str(plots_dir / "documents_interactive_datamap.html"))
    fig_docs.write_html(str(plots_dir / "documents_plotly.html"))


def _create_topic_barchart(topic_model: BERTopic, plots_dir: Path) -> None:
    """
    Create topic terms barchart visualization showing top words per topic.
    
    Args:
        topic_model: Trained BERTopic model with topic representations
        plots_dir: Directory path for saving plot files
    
    Note:
        - Generates horizontal bar charts for top 12 topics
        - Shows 8 most important words per topic with c-tf-idf scores
        - Saves as interactive HTML file: topics_barchart.html
        - Essential for understanding topic content and quality
    """
    logger.info("Creating topic terms barchart")
    fig_barchart = topic_model.visualize_barchart(top_n_topics=12, n_words=8)
    fig_barchart.write_html(str(plots_dir / "topics_barchart.html"))


def validate_visualization_config(config: Dict) -> bool:
    """
    Validate visualization configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    viz_config = config.get('visualization', {})
    
    required_keys = ['interactive_datamapplot', 'reduce_embeddings_for_viz', 'hide_document_hover']
    
    for key in required_keys:
        if key not in viz_config:
            return False
            
    return True


def get_visualization_info(topic_model: BERTopic) -> Dict:
    """
    Get information about available visualizations.
    
    Args:
        topic_model: Trained BERTopic model
        
    Returns:
        Dictionary with visualization capabilities and info
    """
    try:
        n_topics = len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)
        
        return {
            'total_topics': n_topics,
            'has_hierarchical_topics': n_topics > 1,
            'has_topic_info': hasattr(topic_model, 'topic_representations_'),
            'available_visualizations': [
                'topics_overview',
                'topics_hierarchy', 
                'topics_heatmap',
                'documents_interactive_datamap',
                'documents_plotly',
                'topics_barchart'
            ]
        }
    except Exception:
        return {
            'total_topics': 0,
            'has_hierarchical_topics': False,
            'has_topic_info': False,
            'available_visualizations': []
        } 