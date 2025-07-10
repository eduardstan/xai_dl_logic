#!/usr/bin/env python3
"""
Metrics visualization module for Stage 3.

This module creates comprehensive visualizations showing the distribution
and relationships between analysis metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from research_analysis.utils.logging import get_logger

logger = get_logger()


def create_metrics_overview(
    df_all: pd.DataFrame,
    df_selected: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create comprehensive metrics overview visualization.
    
    Generates a 2x3 grid showing distributions and relationships of key metrics:
    centrality, diversity, and representativeness.
    
    Args:
        df_all: Complete analysis DataFrame with all papers
        df_selected: Selected representative papers DataFrame
        output_dir: Directory to save the visualization
        
    Raises:
        ValueError: If required columns are missing
        RuntimeError: If visualization generation fails
    """
    logger.info("📊 Creating comprehensive metrics overview visualization...")
    
    # Validate required columns
    required_cols = ['similarity_to_centroid', 'diversity_score', 'representativeness_score']
    _validate_metrics_columns(df_all, df_selected, required_cols)
    
    try:
        # Set up the figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Systematic Review: Metrics Overview', fontsize=16, fontweight='bold')
        
        # Create individual visualizations
        _plot_centrality_distribution(axes[0, 0], df_all, df_selected)
        _plot_diversity_distribution(axes[0, 1], df_all, df_selected)
        _plot_representativeness_distribution(axes[0, 2], df_all, df_selected)
        _plot_centrality_vs_diversity(axes[1, 0], df_selected)
        _plot_research_alignment(axes[1, 1], df_all, df_selected)
        _plot_selection_efficiency(axes[1, 2], df_selected)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = output_dir / "metrics_overview.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Metrics overview saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create metrics overview: {e}", exc_info=True)
        raise RuntimeError(f"Metrics visualization failed: {e}")


def _validate_metrics_columns(df_all: pd.DataFrame, df_selected: pd.DataFrame, required_cols: list) -> None:
    """Validate that required columns exist in both DataFrames."""
    missing_all = [col for col in required_cols if col not in df_all.columns]
    missing_selected = [col for col in required_cols if col not in df_selected.columns]
    
    if missing_all or missing_selected:
        raise ValueError(f"Missing required columns: {missing_all + missing_selected}")


def _plot_centrality_distribution(ax, df_all: pd.DataFrame, df_selected: pd.DataFrame) -> None:
    """Plot similarity to centroid distribution."""
    ax.hist(df_all['similarity_to_centroid'], bins=50, alpha=0.6, label='All Papers',
            color='#87ceeb', density=True, edgecolor='navy', linewidth=0.8)
    ax.hist(df_selected['similarity_to_centroid'], bins=30, alpha=0.6, label='Selected',
            color='#0066cc', density=True, edgecolor='darkblue', linewidth=1.2)
    ax.set_xlabel('Similarity to Centroid')
    ax.set_ylabel('Density')
    ax.set_title('Distribution: Similarity to Centroid')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)


def _plot_diversity_distribution(ax, df_all: pd.DataFrame, df_selected: pd.DataFrame) -> None:
    """Plot diversity score distribution."""
    ax.hist(df_all['diversity_score'], bins=50, alpha=0.6, label='All Papers',
            color='#90ee90', density=True, edgecolor='darkgreen', linewidth=0.8)
    ax.hist(df_selected['diversity_score'], bins=30, alpha=0.6, label='Selected',
            color='#006600', density=True, edgecolor='green', linewidth=1.2)
    ax.set_xlabel('Diversity Score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution: Diversity Scores')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)


def _plot_representativeness_distribution(ax, df_all: pd.DataFrame, df_selected: pd.DataFrame) -> None:
    """Plot representativeness score distribution."""
    ax.hist(df_all['representativeness_score'], bins=50, alpha=0.6, label='All Papers',
            color='#ffa5a5', density=True, edgecolor='darkred', linewidth=0.8)
    ax.hist(df_selected['representativeness_score'], bins=30, alpha=0.6, label='Selected',
            color='#cc0000', density=True, edgecolor='red', linewidth=1.2)
    ax.set_xlabel('Representativeness Score')
    ax.set_ylabel('Density')
    ax.set_title('Distribution: Representativeness Scores')
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)


def _plot_centrality_vs_diversity(ax, df_selected: pd.DataFrame) -> None:
    """Plot centrality vs diversity scatter with representativeness coloring."""
    scatter = ax.scatter(
        df_selected['similarity_to_centroid'], 
        df_selected['diversity_score'],
        c=df_selected['representativeness_score'], 
        cmap='viridis',
        s=60, alpha=0.7, edgecolors='black', linewidth=0.5
    )
    ax.set_xlabel('Similarity to Centroid (Centrality)')
    ax.set_ylabel('Diversity Score')
    ax.set_title('Selected Papers: Centrality vs Diversity\n(color = representativeness)')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Representativeness Score')


def _plot_research_alignment(ax, df_all: pd.DataFrame, df_selected: pd.DataFrame) -> None:
    """Plot research alignment comparison if alignment columns exist."""
    alignment_cols = ['xai_alignment', 'symbolic_alignment', 'subsymbolic_alignment']
    
    if all(col in df_all.columns for col in alignment_cols):
        # Calculate mean alignments
        all_means = [df_all[col].mean() for col in alignment_cols]
        selected_means = [df_selected[col].mean() for col in alignment_cols]
        
        # Create grouped bar chart
        categories = ['All Papers', 'Selected Papers']
        x = np.arange(len(categories))
        width = 0.25
        
        colors = ['#e74c3c', '#2ecc71', '#3498db']  # Red, Green, Blue
        labels = ['XAI', 'Symbolic', 'Sub-symbolic']
        
        for i, (label, color) in enumerate(zip(labels, colors)):
            values = [all_means[i], selected_means[i]]
            bars = ax.bar(x + (i - 1) * width, values, width, label=label, 
                         color=color, alpha=0.8, edgecolor='white', linewidth=1)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_ylabel('Average Alignment Score')
        ax.set_title('Research Focus Alignment Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_ylim(0, max(max(all_means), max(selected_means)) * 1.2)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Research Alignment\nData Not Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Research Focus Alignment Comparison')


def _plot_selection_efficiency(ax, df_selected: pd.DataFrame) -> None:
    """Plot selection efficiency by topic size if topic columns exist."""
    topic_cols = ['topic_id', 'cluster_size', 'papers_selected_from_cluster']
    
    if all(col in df_selected.columns for col in topic_cols):
        # Calculate topic-level metrics
        topic_data = df_selected.groupby('topic_id').agg({
            'cluster_size': 'first',
            'papers_selected_from_cluster': 'first',
            'representativeness_score': 'mean'
        }).reset_index()
        
        scatter = ax.scatter(
            topic_data['cluster_size'], 
            topic_data['representativeness_score'],
            s=topic_data['papers_selected_from_cluster'] * 10,
            alpha=0.7, 
            c=topic_data['papers_selected_from_cluster'],
            cmap='plasma', 
            edgecolors='black', 
            linewidth=0.5
        )
        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Avg Representativeness Score')
        ax.set_title('Selection Efficiency by Topic\n(size = papers selected)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Papers Selected')
    else:
        ax.text(0.5, 0.5, 'Topic Analysis\nData Not Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Selection Efficiency by Topic') 