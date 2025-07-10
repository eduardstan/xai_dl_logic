#!/usr/bin/env python3
"""
Network visualization module for Stage 3.

This module creates network-based visualizations showing relationships
between representative papers and their assigned papers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_analysis.utils.logging import get_logger

logger = get_logger()


def create_network_visualizations(df_all: pd.DataFrame, output_dir: Path) -> None:
    """
    Create network-style visualizations showing paper assignment relationships.
    
    Generates visualizations for the top topics showing representatives
    as large nodes and assigned papers as smaller nodes with connections.
    
    Args:
        df_all: Complete analysis DataFrame with all papers
        output_dir: Directory to save the visualization
        
    Raises:
        ValueError: If required columns are missing
        RuntimeError: If visualization generation fails
    """
    logger.info("🕸️ Creating paper assignment network visualization...")
    
    # Validate required columns
    required_cols = ['topic_id', 'is_selected_representative', 'similarity_to_centroid', 'diversity_score']
    _validate_network_columns(df_all, required_cols)
    
    try:
        # Create network visualization
        _create_paper_assignment_network(df_all, output_dir)
        
        logger.info("✅ Network visualizations completed")
        
    except Exception as e:
        logger.error(f"Failed to create network visualizations: {e}", exc_info=True)
        raise RuntimeError(f"Network visualization failed: {e}")


def _validate_network_columns(df_all: pd.DataFrame, required_cols: list) -> None:
    """Validate that required columns exist in the DataFrame."""
    missing_cols = [col for col in required_cols if col not in df_all.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for network visualization: {missing_cols}")


def _create_paper_assignment_network(df_all: pd.DataFrame, output_dir: Path) -> None:
    """Create network visualization showing representatives and their assigned papers."""
    # Get top 5 topics by paper count
    topic_counts = df_all['topic_id'].value_counts().head(5)
    top_topics = topic_counts.index.tolist()
    
    if len(top_topics) == 0:
        logger.warning("No topics found for network visualization")
        return
    
    # Set up the figure
    fig, axes = plt.subplots(1, len(top_topics), figsize=(4*len(top_topics), 6))
    if len(top_topics) == 1:
        axes = [axes]  # Ensure axes is always a list
    
    fig.suptitle('Paper Assignment Networks: Top Topics', fontsize=16, fontweight='bold')
    
    for i, topic_id in enumerate(top_topics):
        ax = axes[i]
        _plot_topic_network(df_all, topic_id, ax)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = output_dir / "paper_assignment_networks.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Network visualization saved to: {output_path}")


def _plot_topic_network(df_all: pd.DataFrame, topic_id: int, ax) -> None:
    """Plot network for a single topic."""
    # Filter papers for this topic
    topic_papers = df_all[df_all['topic_id'] == topic_id].copy()
    
    if len(topic_papers) == 0:
        ax.text(0.5, 0.5, f'Topic {topic_id}\nNo Data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Topic {topic_id}')
        return
    
    # Separate representatives and non-representatives
    representatives = topic_papers[topic_papers['is_selected_representative'] == True]
    non_representatives = topic_papers[topic_papers['is_selected_representative'] == False]
    
    # Create layout based on metrics (use centrality and diversity as coordinates)
    rep_x = representatives['similarity_to_centroid'] if len(representatives) > 0 else []
    rep_y = representatives['diversity_score'] if len(representatives) > 0 else []
    
    non_rep_x = non_representatives['similarity_to_centroid'] if len(non_representatives) > 0 else []
    non_rep_y = non_representatives['diversity_score'] if len(non_representatives) > 0 else []
    
    # Plot non-representative papers as small blue circles
    if len(non_representatives) > 0:
        ax.scatter(non_rep_x, non_rep_y, 
                  s=20, c='lightblue', alpha=0.6, edgecolors='blue', linewidth=0.5,
                  label='Assigned Papers')
    
    # Plot representatives as large red circles
    if len(representatives) > 0:
        ax.scatter(rep_x, rep_y,
                  s=120, c='red', alpha=0.8, edgecolors='darkred', linewidth=2,
                  label='Representatives', marker='o')
    
    # Draw connections from non-representatives to nearest representative
    _draw_assignment_connections(ax, representatives, non_representatives)
    
    # Set labels and title
    ax.set_xlabel('Similarity to Centroid', fontsize=10)
    ax.set_ylabel('Diversity Score', fontsize=10)
    ax.set_title(f'Topic {topic_id}\n({len(topic_papers)} papers)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add legend only to the first subplot
    if topic_id == df_all['topic_id'].value_counts().index[0]:
        ax.legend(loc='upper right', fontsize=9)


def _draw_assignment_connections(ax, representatives: pd.DataFrame, non_representatives: pd.DataFrame) -> None:
    """Draw lines connecting non-representatives to their closest representatives."""
    if len(representatives) == 0 or len(non_representatives) == 0:
        return
    
    # Check if assignment information is available
    if 'representative_title' in non_representatives.columns and 'similarity_to_representative' in non_representatives.columns:
        # Use actual assignment data if available
        for _, paper in non_representatives.iterrows():
            if pd.notna(paper.get('representative_title')):
                # Find the representative by title
                rep_match = representatives[representatives['title'] == paper['representative_title']]
                if len(rep_match) > 0:
                    rep = rep_match.iloc[0]
                    ax.plot([paper['similarity_to_centroid'], rep['similarity_to_centroid']],
                           [paper['diversity_score'], rep['diversity_score']],
                           'gray', alpha=0.3, linewidth=0.5)
    else:
        # Fallback: connect to nearest representative by distance
        for _, paper in non_representatives.iterrows():
            # Find nearest representative
            distances = []
            for _, rep in representatives.iterrows():
                dist = np.sqrt((paper['similarity_to_centroid'] - rep['similarity_to_centroid'])**2 +
                              (paper['diversity_score'] - rep['diversity_score'])**2)
                distances.append(dist)
            
            if distances:
                nearest_idx = np.argmin(distances)
                nearest_rep = representatives.iloc[nearest_idx]
                
                # Draw connection line
                ax.plot([paper['similarity_to_centroid'], nearest_rep['similarity_to_centroid']],
                       [paper['diversity_score'], nearest_rep['diversity_score']],
                       'gray', alpha=0.3, linewidth=0.5)


def create_topic_similarity_network(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a topic similarity network visualization (alternative visualization).
    
    Args:
        df_summary: Topic-level summary DataFrame
        output_dir: Directory to save the visualization
    """
    logger.info("Creating topic similarity network...")
    
    if len(df_summary) == 0:
        logger.warning("Empty summary DataFrame for topic network")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Add missing column if needed
        if 'papers_selected' not in df_summary.columns:
            df_summary = df_summary.copy()
            df_summary['papers_selected'] = df_summary['papers_selected_from_cluster']
        
        # Create a simple layout based on cluster size and selection ratio
        x = df_summary['cluster_size']
        y = df_summary['papers_selected'] / df_summary['cluster_size']
        
        # Create scatter plot
        scatter = ax.scatter(x, y, 
                           s=df_summary['papers_selected']*5,
                           alpha=0.7, 
                           edgecolors='black', 
                           linewidth=0.5)
        
        # Add topic labels
        for i, (_, row) in enumerate(df_summary.iterrows()):
            if i < 20:  # Show labels for first 20 topics only
                ax.annotate(f"T{row['topic_id']}", 
                           (row['cluster_size'], y.iloc[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Selection Ratio')
        ax.set_title('Topic Size vs Selection Efficiency\n(bubble size = papers selected)')
        ax.grid(alpha=0.3)
        
        # Save
        output_path = output_dir / "topic_similarity_network.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Topic similarity network saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create topic similarity network: {e}")


def get_network_statistics(df_all: pd.DataFrame) -> dict:
    """
    Calculate network statistics for the paper assignment network.
    
    Args:
        df_all: Complete analysis DataFrame
        
    Returns:
        Dictionary with network statistics
    """
    if len(df_all) == 0:
        return {}
    
    total_papers = len(df_all)
    representatives = df_all[df_all['is_selected_representative'] == True]
    assigned_papers = df_all[df_all['is_selected_representative'] == False]
    
    # Calculate basic network metrics
    num_representatives = len(representatives)
    num_assigned = len(assigned_papers)
    
    # Calculate average connections per representative
    avg_connections = num_assigned / num_representatives if num_representatives > 0 else 0
    
    # Calculate topic-level statistics
    topics_with_reps = df_all[df_all['is_selected_representative'] == True]['topic_id'].nunique()
    total_topics = df_all['topic_id'].nunique()
    
    return {
        'total_papers': total_papers,
        'representatives': num_representatives,
        'assigned_papers': num_assigned,
        'avg_connections_per_rep': avg_connections,
        'topics_with_representatives': topics_with_reps,
        'total_topics': total_topics,
        'coverage_ratio': topics_with_reps / total_topics if total_topics > 0 else 0
    } 