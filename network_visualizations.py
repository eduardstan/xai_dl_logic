#!/usr/bin/env python3
"""
Network Visualization Module for Enhanced Research Visualization Pipeline

This module provides functions to create network-based visualizations showing
relationships between representative papers and their assigned papers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from loguru import logger


def create_paper_assignment_network(df_all: pd.DataFrame, output_dir: Path) -> None:
    """
    Visualize the paper assignment network showing representatives and their assigned papers.
    
    This function creates network-style visualizations for the top 5 topics,
    showing representative papers as large nodes and assigned papers as smaller
    nodes, with connecting lines indicating assignment relationships.
    
    Args:
        df_all: Complete analysis DataFrame containing all papers
        output_dir: Output directory path for saving visualization
    
    Raises:
        ValueError: If required columns are missing from input DataFrame
        RuntimeError: If visualization generation fails
    
    Generated visualization:
        - Multi-panel network plot (one panel per top topic)
        - Representatives shown as large red circles
        - Assigned papers shown as small blue circles
        - Gray lines connect assigned papers to their representatives
        - Scatter plot layout based on centrality vs diversity coordinates
    
    Example:
        >>> create_paper_assignment_network(df_all, Path("results"))
        # Creates and saves paper_assignment_networks.png
    
    Note:
        - Focuses on top 5 topics by paper count for clarity
        - Network layout uses similarity_to_centroid vs diversity_score coordinates
        - Representative-assignment connections shown when data is available
        - Professional formatting suitable for academic presentation
    """
    logger.info("🕸️ Creating paper assignment network visualization...")
    
    # Validate required columns
    required_columns = [
        'topic_id', 'similarity_to_centroid', 'diversity_score', 'is_selected_representative'
    ]
    
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for network visualization: {missing_columns}")
    
    try:
        # Focus on top 5 topics by paper count for clarity
        selected_topics = df_all['topic_id'].value_counts().head(5).index
        
        fig, axes = plt.subplots(1, len(selected_topics), figsize=(20, 4))
        fig.suptitle('Paper Assignment Networks (Top 5 Topics)', fontsize=16, fontweight='bold')
        
        # Handle single topic case
        if len(selected_topics) == 1:
            axes = [axes]
        
        for i, topic_id in enumerate(selected_topics):
            ax = axes[i]
            topic_data = df_all[df_all['topic_id'] == topic_id].copy()
            
            representatives = topic_data[topic_data['is_selected_representative']]
            non_selected = topic_data[~topic_data['is_selected_representative']]
            
            # Plot representatives as large circles
            if len(representatives) > 0:
                ax.scatter(representatives['similarity_to_centroid'], 
                          representatives['diversity_score'],
                          s=200, c='red', alpha=0.8, label='Representatives', 
                          edgecolors='black', linewidth=2, zorder=3)
            
            # Plot non-selected papers as smaller circles
            if len(non_selected) > 0:
                ax.scatter(non_selected['similarity_to_centroid'], 
                          non_selected['diversity_score'],
                          s=30, c='lightblue', alpha=0.6, label='Assigned Papers',
                          edgecolors='navy', linewidth=0.5, zorder=2)
                
                # Draw lines from non-selected to their representatives
                if 'representative_title' in non_selected.columns:
                    _draw_assignment_connections(ax, non_selected, representatives)
            
            # Format subplot
            ax.set_xlabel('Similarity to Centroid')
            ax.set_ylabel('Diversity Score')
            ax.set_title(f'Topic {topic_id}\n({len(topic_data)} papers)', fontweight='bold')
            ax.grid(alpha=0.3, zorder=1)
            ax.legend(framealpha=0.9)
            
            # Set consistent axis limits for comparison
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = output_dir / 'paper_assignment_networks.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"🕸️ Paper assignment network visualization saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error creating paper assignment network: {e}")
        raise RuntimeError(f"Failed to create paper assignment network: {e}")


def _draw_assignment_connections(ax, non_selected: pd.DataFrame, representatives: pd.DataFrame) -> None:
    """
    Draw connection lines between non-selected papers and their representatives.
    
    Args:
        ax: Matplotlib axis object
        non_selected: DataFrame of non-selected papers
        representatives: DataFrame of representative papers
    
    Note:
        - Draws gray lines with low opacity for subtle connection indication
        - Only draws connections when representative_title data is available
        - Handles missing or invalid representative assignments gracefully
    """
    connections_drawn = 0
    
    for _, paper in non_selected.iterrows():
        if pd.notna(paper.get('representative_title', '')):
            # Find the representative by title
            rep_title = paper['representative_title']
            
            if 'title' in representatives.columns:
                rep_row = representatives[representatives['title'] == rep_title]
                
                if not rep_row.empty:
                    rep = rep_row.iloc[0]
                    
                    # Draw connection line
                    ax.plot([paper['similarity_to_centroid'], rep['similarity_to_centroid']], 
                           [paper['diversity_score'], rep['diversity_score']], 
                           color='gray', alpha=0.3, linewidth=0.5, zorder=1)
                    connections_drawn += 1
    
    if connections_drawn > 0:
        logger.debug(f"Drew {connections_drawn} assignment connections")


def validate_network_data(df_all: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required columns for network visualization.
    
    Args:
        df_all: Complete analysis DataFrame to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    
    Note:
        - Checks for presence of required network columns
        - Validates coordinate ranges and data types
        - Logs validation warnings for debugging
    """
    required_columns = [
        'topic_id', 'similarity_to_centroid', 'diversity_score', 'is_selected_representative'
    ]
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df_all.columns]
    if missing_columns:
        logger.error(f"Missing required columns for network visualization: {missing_columns}")
        return False
    
    # Check data ranges for coordinates
    coord_columns = ['similarity_to_centroid', 'diversity_score']
    for col in coord_columns:
        if df_all[col].min() < 0 or df_all[col].max() > 1:
            logger.warning(f"Coordinate column {col} has values outside expected range [0,1]")
    
    # Check boolean column
    if not df_all['is_selected_representative'].dtype == bool:
        logger.warning("is_selected_representative column should be boolean type")
    
    # Check for null values in critical columns
    critical_nulls = df_all[required_columns].isnull().sum()
    if critical_nulls.any():
        logger.warning(f"Null values found in network columns: {critical_nulls.to_dict()}")
    
    # Check if we have representative papers
    rep_count = df_all['is_selected_representative'].sum()
    if rep_count == 0:
        logger.error("No representative papers found for network visualization")
        return False
    
    logger.info("✅ Network visualization data validation passed")
    return True


def get_network_summary(df_all: pd.DataFrame) -> dict:
    """
    Generate summary information for network visualization.
    
    Args:
        df_all: Complete analysis DataFrame
        
    Returns:
        dict: Summary statistics for network analysis
    
    Example:
        >>> summary = get_network_summary(df_all)
        >>> print(f"Network density: {summary['network_density']:.2%}")
        Network density: 14.27%
    """
    try:
        # Basic counts
        total_papers = len(df_all)
        representatives = df_all[df_all['is_selected_representative']]
        non_selected = df_all[~df_all['is_selected_representative']]
        
        # Topic analysis
        topic_counts = df_all['topic_id'].value_counts()
        top_topics = topic_counts.head(5)
        
        # Assignment analysis
        has_assignments = 'representative_title' in df_all.columns
        if has_assignments:
            assigned_papers = non_selected[non_selected['representative_title'].notna()]
            assignment_coverage = len(assigned_papers) / len(non_selected) if len(non_selected) > 0 else 0
        else:
            assignment_coverage = 0
            assigned_papers = pd.DataFrame()
        
        # Network metrics
        network_density = len(representatives) / total_papers if total_papers > 0 else 0
        
        return {
            'total_papers': total_papers,
            'representative_papers': len(representatives),
            'non_selected_papers': len(non_selected),
            'total_topics': df_all['topic_id'].nunique(),
            'top_5_topics': {
                int(topic_id): int(count) for topic_id, count in top_topics.items()
            },
            'network_density': network_density,
            'has_assignment_data': has_assignments,
            'assignment_coverage': assignment_coverage,
            'avg_papers_per_topic': topic_counts.mean(),
            'largest_topic_size': topic_counts.max(),
            'smallest_topic_size': topic_counts.min(),
            'representative_ratio': len(representatives) / total_papers if total_papers > 0 else 0,
            'coordinate_ranges': {
                'centrality_range': [
                    float(df_all['similarity_to_centroid'].min()),
                    float(df_all['similarity_to_centroid'].max())
                ],
                'diversity_range': [
                    float(df_all['diversity_score'].min()),
                    float(df_all['diversity_score'].max())
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating network summary: {e}")
        return {'error': str(e)} 