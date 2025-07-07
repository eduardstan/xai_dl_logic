#!/usr/bin/env python3
"""
Selection Analysis Visualization Module for Enhanced Research Visualization Pipeline

This module provides functions to create comprehensive visualizations analyzing
the paper selection strategy, including cluster size relationships, selection
ratios, and efficiency metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from loguru import logger


def create_selection_analysis(df_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comprehensive visualizations analyzing the paper selection strategy.
    
    This function generates a 2x2 grid of visualizations showing selection
    patterns, efficiency metrics, and strategy performance across different
    cluster sizes and topic categories.
    
    Args:
        df_summary: Topic-level summary DataFrame with selection statistics
        output_dir: Output directory path for saving visualizations
    
    Raises:
        ValueError: If required columns are missing from input DataFrame
        RuntimeError: If visualization generation fails
    
    Generated visualizations:
        1. Papers selected vs cluster size (scatter plot with centrality coloring)
        2. Selection ratio by topic (bar chart with color-coded thresholds)
        3. Centrality vs diversity trade-off (bubble chart)
        4. Topic size categories distribution (pie chart)
    
    Example:
        >>> create_selection_analysis(df_summary, Path("results"))
        # Creates and saves selection_analysis.png
    
    Note:
        - Color-coded selection ratios (red<5%, orange<10%, green≥10%)
        - Includes ideal selection line based on configuration thresholds
        - Bubble sizes represent cluster sizes for multi-dimensional analysis
        - Automatically fixes negative diversity scores (clamps to 0)
        - Saves high-resolution PNG output (300 DPI)
    """
    logger.info("📈 Creating comprehensive selection analysis visualization...")
    
    # Validate required columns
    required_columns = [
        'cluster_size', 'papers_selected', 'avg_centrality', 'selection_ratio',
        'avg_diversity', 'size_category', 'topic_id'
    ]
    
    missing_columns = [col for col in required_columns if col not in df_summary.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in summary data: {missing_columns}")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Selection Strategy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Papers selected vs cluster size
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df_summary['cluster_size'], df_summary['papers_selected'], 
                             c=df_summary['avg_centrality'], cmap='coolwarm', 
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.set_xlabel('Cluster Size')
        ax1.set_ylabel('Papers Selected')
        ax1.set_title('Selection Pattern by Cluster Size\n(color = avg centrality)')
        ax1.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Avg Centrality')
        
        # Add ideal selection line based on typical thresholds
        x_ideal = np.linspace(0, df_summary['cluster_size'].max(), 100)
        y_ideal = np.minimum(3 + (x_ideal / 50), 8)  # Based on config thresholds
        ax1.plot(x_ideal, y_ideal, 'r--', alpha=0.7, label='Ideal Selection')
        ax1.legend()
        
        # 2. Selection ratio vs cluster size
        ax2 = axes[0, 1]
        colors = ['red' if x < 0.05 else 'orange' if x < 0.1 else 'green' for x in df_summary['selection_ratio']]
        bars = ax2.bar(range(len(df_summary)), df_summary['selection_ratio'], color=colors, alpha=0.7)
        ax2.set_xlabel('Topic ID')
        ax2.set_ylabel('Selection Ratio')
        ax2.set_title('Selection Ratio by Topic\n(red<5%, orange<10%, green≥10%)')
        ax2.set_xticks(range(0, len(df_summary), 5))
        ax2.set_xticklabels(df_summary['topic_id'].iloc[::5])
        ax2.grid(axis='y', alpha=0.3)
        
        # Add horizontal reference lines
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
        ax2.axhline(y=0.10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
        ax2.legend()
        
        # 3. Average centrality vs diversity trade-off
        ax3 = axes[1, 0]
        # Fix negative diversity scores (clamp to 0)
        df_summary_fixed = df_summary.copy()
        df_summary_fixed['avg_diversity'] = np.maximum(df_summary_fixed['avg_diversity'], 0)
        
        scatter = ax3.scatter(df_summary_fixed['avg_centrality'], df_summary_fixed['avg_diversity'], 
                             s=df_summary_fixed['cluster_size']*2, 
                             c=df_summary_fixed['papers_selected'], cmap='viridis', 
                             alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Average Centrality')
        ax3.set_ylabel('Average Diversity')
        ax3.set_title('Centrality vs Diversity Trade-off\n(size = cluster size, color = papers selected)')
        ax3.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Papers Selected')
        
        # Add diagonal reference line for balanced trade-off
        min_val = min(df_summary_fixed['avg_centrality'].min(), df_summary_fixed['avg_diversity'].min())
        max_val = max(df_summary_fixed['avg_centrality'].max(), df_summary_fixed['avg_diversity'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Balanced Trade-off')
        ax3.legend()
        
        # 4. Topic size categories distribution
        ax4 = axes[1, 1]
        
        if 'size_category' in df_summary.columns:
            size_counts = df_summary['size_category'].value_counts()
            colors_cat = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db']
            
            wedges, texts, autotexts = ax4.pie(size_counts.values, labels=size_counts.index, 
                                              autopct='%1.1f%%', colors=colors_cat[:len(size_counts)], 
                                              startangle=90)
            ax4.set_title('Distribution of Topic Size Categories')
            
            # Enhance text visibility
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax4.text(0.5, 0.5, 'Size Category\nData Not Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Distribution of Topic Size Categories')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = output_dir / 'selection_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"📈 Selection analysis visualization saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error creating selection analysis: {e}")
        raise RuntimeError(f"Failed to create selection analysis visualization: {e}")


def validate_selection_data(df_summary: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required columns for selection analysis.
    
    Args:
        df_summary: Summary DataFrame to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    
    Note:
        - Checks for presence of required selection columns
        - Validates data types and ranges
        - Logs validation warnings for debugging
    """
    required_columns = [
        'cluster_size', 'papers_selected', 'avg_centrality', 'selection_ratio',
        'avg_diversity', 'topic_id'
    ]
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df_summary.columns]
    if missing_columns:
        logger.error(f"Missing required columns for selection analysis: {missing_columns}")
        return False
    
    # Check data ranges
    if df_summary['selection_ratio'].min() < 0 or df_summary['selection_ratio'].max() > 1:
        logger.warning("Selection ratio values outside expected range [0,1]")
    
    if df_summary['cluster_size'].min() <= 0:
        logger.error("Cluster size must be positive")
        return False
    
    if df_summary['papers_selected'].min() < 0:
        logger.error("Papers selected cannot be negative")
        return False
    
    # Check for null values
    null_counts = df_summary[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found in selection columns: {null_counts.to_dict()}")
    
    logger.info("✅ Selection data validation passed")
    return True


def get_selection_summary(df_summary: pd.DataFrame) -> dict:
    """
    Generate summary statistics for selection analysis.
    
    Args:
        df_summary: Topic-level summary DataFrame
        
    Returns:
        dict: Summary statistics for selection analysis
    
    Example:
        >>> summary = get_selection_summary(df_summary)
        >>> print(f"Average selection ratio: {summary['avg_selection_ratio']:.2%}")
        Average selection ratio: 14.27%
    """
    try:
        # Fix negative diversity scores for calculations
        df_fixed = df_summary.copy()
        df_fixed['avg_diversity'] = np.maximum(df_fixed['avg_diversity'], 0)
        
        # Calculate selection efficiency (balance of centrality and diversity)
        efficiency = df_fixed['avg_centrality'] * df_fixed['avg_diversity']
        
        # Size category distribution
        size_distribution = {}
        if 'size_category' in df_summary.columns:
            size_counts = df_summary['size_category'].value_counts()
            size_distribution = {
                category: {
                    'count': int(count),
                    'percentage': count / len(df_summary) * 100
                }
                for category, count in size_counts.items()
            }
        
        # Selection ratio categories
        low_selection = len(df_summary[df_summary['selection_ratio'] < 0.05])
        medium_selection = len(df_summary[(df_summary['selection_ratio'] >= 0.05) & 
                                         (df_summary['selection_ratio'] < 0.10)])
        high_selection = len(df_summary[df_summary['selection_ratio'] >= 0.10])
        
        return {
            'total_topics': len(df_summary),
            'avg_cluster_size': df_summary['cluster_size'].mean(),
            'avg_papers_selected': df_summary['papers_selected'].mean(),
            'avg_selection_ratio': df_summary['selection_ratio'].mean(),
            'avg_centrality': df_summary['avg_centrality'].mean(),
            'avg_diversity': df_fixed['avg_diversity'].mean(),
            'selection_efficiency': efficiency.mean(),
            'size_distribution': size_distribution,
            'selection_ratio_categories': {
                'low_selection_topics': low_selection,
                'medium_selection_topics': medium_selection,
                'high_selection_topics': high_selection,
                'low_selection_pct': low_selection / len(df_summary) * 100,
                'medium_selection_pct': medium_selection / len(df_summary) * 100,
                'high_selection_pct': high_selection / len(df_summary) * 100
            },
            'cluster_size_stats': {
                'min_cluster_size': df_summary['cluster_size'].min(),
                'max_cluster_size': df_summary['cluster_size'].max(),
                'median_cluster_size': df_summary['cluster_size'].median(),
                'std_cluster_size': df_summary['cluster_size'].std()
            },
            'papers_selected_stats': {
                'min_papers_selected': df_summary['papers_selected'].min(),
                'max_papers_selected': df_summary['papers_selected'].max(),
                'median_papers_selected': df_summary['papers_selected'].median(),
                'std_papers_selected': df_summary['papers_selected'].std()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating selection summary: {e}")
        return {'error': str(e)}


def analyze_selection_patterns(df_summary: pd.DataFrame) -> dict:
    """
    Analyze patterns in paper selection strategy.
    
    Args:
        df_summary: Topic-level summary DataFrame
        
    Returns:
        dict: Analysis results and insights
    
    Example:
        >>> patterns = analyze_selection_patterns(df_summary)
        >>> print(f"Selection strategy: {patterns['strategy_type']}")
        Selection strategy: balanced
    """
    try:
        # Analyze relationship between cluster size and selection ratio
        correlation_size_ratio = df_summary['cluster_size'].corr(df_summary['selection_ratio'])
        
        # Analyze centrality vs diversity balance
        df_fixed = df_summary.copy()
        df_fixed['avg_diversity'] = np.maximum(df_fixed['avg_diversity'], 0)
        
        centrality_diversity_ratio = df_fixed['avg_centrality'].mean() / df_fixed['avg_diversity'].mean()
        
        # Determine strategy type
        if centrality_diversity_ratio > 1.5:
            strategy_type = "centrality_focused"
        elif centrality_diversity_ratio < 0.67:
            strategy_type = "diversity_focused"
        else:
            strategy_type = "balanced"
        
        # Analyze selection efficiency
        high_efficiency_topics = len(df_summary[
            (df_summary['avg_centrality'] > df_summary['avg_centrality'].median()) &
            (df_fixed['avg_diversity'] > df_fixed['avg_diversity'].median())
        ])
        
        return {
            'strategy_type': strategy_type,
            'centrality_diversity_ratio': centrality_diversity_ratio,
            'size_selection_correlation': correlation_size_ratio,
            'high_efficiency_topics': high_efficiency_topics,
            'high_efficiency_pct': high_efficiency_topics / len(df_summary) * 100,
            'selection_consistency': df_summary['selection_ratio'].std(),
            'centrality_consistency': df_summary['avg_centrality'].std(),
            'diversity_consistency': df_fixed['avg_diversity'].std()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing selection patterns: {e}")
        return {'error': str(e)} 