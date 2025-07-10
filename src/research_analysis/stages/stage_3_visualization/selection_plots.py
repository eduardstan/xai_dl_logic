#!/usr/bin/env python3
"""
Selection analysis visualization module for Stage 3.

This module creates visualizations analyzing the paper selection strategy,
including cluster relationships, selection ratios, and efficiency metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from research_analysis.utils.logging import get_logger

logger = get_logger()


def create_selection_analysis(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comprehensive visualizations analyzing the paper selection strategy.
    
    Generates a 2x2 grid showing selection patterns, efficiency metrics,
    and strategy performance across different cluster sizes and topics.
    
    Args:
        df_summary: Topic-level summary DataFrame with selection statistics
        output_dir: Directory to save the visualization
        
    Raises:
        ValueError: If required columns are missing
        RuntimeError: If visualization generation fails
    """
    logger.info("Creating selection analysis visualization...")
    
    # Add missing columns if needed
    if 'topic_name' not in df_summary.columns:
        df_summary = df_summary.copy()
        df_summary['topic_name'] = df_summary['topic_id'].apply(lambda x: f"Topic {x}")
    
    if 'papers_selected' not in df_summary.columns:
        df_summary = df_summary.copy()
        df_summary['papers_selected'] = df_summary['papers_selected_from_cluster']
    
    # Required columns are checked implicitly during plotting
    
    try:
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Paper Selection Strategy Analysis', fontsize=16, fontweight='bold')
        
        # Create individual visualizations
        _plot_papers_vs_cluster_size(axes[0, 0], df_summary)
        _plot_selection_ratios(axes[0, 1], df_summary)
        _plot_centrality_vs_diversity_tradeoff(axes[1, 0], df_summary)
        _plot_topic_size_categories(axes[1, 1], df_summary)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = output_dir / "selection_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Selection analysis saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create selection analysis: {e}", exc_info=True)
        raise RuntimeError(f"Selection analysis visualization failed: {e}")





def _plot_papers_vs_cluster_size(ax, df_summary: pd.DataFrame) -> None:
    """Plot papers selected vs cluster size with centrality coloring."""
    # Use centrality coloring if available, otherwise use a default color
    if 'avg_centrality' in df_summary.columns:
        scatter = ax.scatter(
            df_summary['cluster_size'], 
            df_summary['papers_selected'],
            c=df_summary['avg_centrality'], 
            cmap='plasma', 
            s=80, 
            alpha=0.7,
            edgecolors='black', 
            linewidth=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Avg Centrality')
    else:
        ax.scatter(
            df_summary['cluster_size'], 
            df_summary['papers_selected'],
            color='#3498db',
            s=80, 
            alpha=0.7,
            edgecolors='black', 
            linewidth=0.5
        )
    
    # Add ideal selection line if we can calculate it
    if len(df_summary) > 0:
        max_cluster = df_summary['cluster_size'].max()
        ideal_ratio = df_summary['papers_selected'].sum() / df_summary['cluster_size'].sum()
        x_ideal = np.linspace(0, max_cluster, 100)
        y_ideal = x_ideal * ideal_ratio
        ax.plot(x_ideal, y_ideal, 'r--', alpha=0.7, linewidth=2, label=f'Ideal ratio ({ideal_ratio:.3f})')
        ax.legend()
    
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Papers Selected')
    ax.set_title('Papers Selected vs Cluster Size')
    ax.grid(alpha=0.3)


def _plot_selection_ratios(ax, df_summary: pd.DataFrame) -> None:
    """Plot selection ratio by topic with color-coded thresholds."""
    # Calculate selection ratios
    selection_ratios = df_summary['papers_selected'] / df_summary['cluster_size']
    
    # Color-code based on thresholds
    colors = []
    for ratio in selection_ratios:
        if ratio < 0.05:  # Less than 5%
            colors.append('#e74c3c')  # Red
        elif ratio < 0.10:  # 5-10%
            colors.append('#f39c12')  # Orange
        else:  # 10%+
            colors.append('#27ae60')  # Green
    
    # Create bar plot
    bars = ax.bar(range(len(df_summary)), selection_ratios, color=colors, alpha=0.8, edgecolor='black')
    
    # Add percentage labels on bars
    for i, (bar, ratio) in enumerate(zip(bars, selection_ratios)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{ratio:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Topic Index')
    ax.set_ylabel('Selection Ratio')
    ax.set_title('Selection Ratio by Topic\n(Red: <5%, Orange: 5-10%, Green: ≥10%)')
    ax.set_xticks(range(0, len(df_summary), max(1, len(df_summary)//10)))
    ax.grid(axis='y', alpha=0.3)


def _plot_centrality_vs_diversity_tradeoff(ax, df_summary: pd.DataFrame) -> None:
    """Plot centrality vs diversity trade-off if metrics are available."""
    if 'avg_centrality' in df_summary.columns and 'avg_diversity' in df_summary.columns:
        # Fix negative diversity scores by clamping to 0
        avg_diversity = np.maximum(df_summary['avg_diversity'], 0)
        
        # Create bubble chart
        scatter = ax.scatter(
            df_summary['avg_centrality'], 
            avg_diversity,
            s=df_summary['cluster_size']*2,  # Bubble size represents cluster size
            c=df_summary['papers_selected'], 
            cmap='viridis',
            alpha=0.7, 
            edgecolors='black', 
            linewidth=0.5
        )
        
        ax.set_xlabel('Average Centrality')
        ax.set_ylabel('Average Diversity')
        ax.set_title('Centrality vs Diversity Trade-off\n(bubble size = cluster size)')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Papers Selected')
    else:
        ax.text(0.5, 0.5, 'Centrality vs Diversity\nMetrics Not Available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Centrality vs Diversity Trade-off')


def _plot_topic_size_categories(ax, df_summary: pd.DataFrame) -> None:
    """Plot topic size categories distribution."""
    # Define size categories
    size_bins = [0, 25, 50, 100, np.inf]
    size_labels = ['Small (≤25)', 'Medium (26-50)', 'Large (51-100)', 'Very Large (>100)']
    
    # Categorize topics by size
    topic_categories = pd.cut(df_summary['cluster_size'], bins=size_bins, labels=size_labels, right=True)
    category_counts = topic_categories.value_counts()
    
    # Create pie chart
    colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
    wedges, texts, autotexts = ax.pie(
        category_counts.values, 
        labels=category_counts.index,
        autopct='%1.1f%%',
        colors=colors[:len(category_counts)],
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    
    # Enhance text appearance
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Topic Size Categories Distribution')


def get_selection_summary(df_summary: pd.DataFrame) -> dict:
    """
    Generate summary statistics for selection analysis.
    
    Args:
        df_summary: Topic-level summary DataFrame
        
    Returns:
        Dictionary with selection analysis statistics
    """
    if len(df_summary) == 0:
        return {}
    
    total_papers = df_summary['cluster_size'].sum()
    total_selected = df_summary['papers_selected'].sum()
    overall_ratio = total_selected / total_papers if total_papers > 0 else 0
    
    return {
        'total_topics': len(df_summary),
        'total_papers': total_papers,
        'total_selected': total_selected,
        'overall_selection_ratio': overall_ratio,
        'avg_cluster_size': df_summary['cluster_size'].mean(),
        'min_cluster_size': df_summary['cluster_size'].min(),
        'max_cluster_size': df_summary['cluster_size'].max(),
        'avg_papers_per_topic': df_summary['papers_selected'].mean()
    } 