#!/usr/bin/env python3
"""
Metrics Visualization Module for Enhanced Research Visualization Pipeline

This module provides functions to create comprehensive visualizations of
analysis metrics including centrality, diversity, and representativeness scores.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from loguru import logger


def create_metrics_overview(df_all: pd.DataFrame, df_selected: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comprehensive overview visualizations of analysis metrics.
    
    This function generates a 2x3 grid of visualizations showing the distribution
    and relationships between key metrics: centrality, diversity, and representativeness.
    
    Args:
        df_all: Complete analysis DataFrame containing all papers
        df_selected: Selected representative papers DataFrame
        output_dir: Output directory path for saving visualizations
    
    Raises:
        ValueError: If required columns are missing from input DataFrames
        RuntimeError: If visualization generation fails
    
    Generated visualizations:
        1. Similarity to Centroid distribution (all vs selected)
        2. Diversity Score distribution (all vs selected)
        3. Representativeness Score distribution (all vs selected)
        4. Centrality vs Diversity scatter plot (selected papers)
        5. Research alignment comparison (XAI, Symbolic, Sub-symbolic)
        6. Selection efficiency by topic size
    
    Example:
        >>> create_metrics_overview(df_all, df_selected, Path("results"))
        # Creates and saves metrics_overview.png
    
    Note:
        - Uses balanced transparency for clear overlap visualization
        - Color-coded scatter plots for multi-dimensional analysis
        - Includes value labels and statistical information
        - Saves high-resolution PNG output (300 DPI)
        - Professional formatting suitable for academic presentation
    """
    logger.info("📊 Creating comprehensive metrics overview visualization...")
    
    # Validate required columns
    required_columns = ['similarity_to_centroid', 'diversity_score', 'representativeness_score']
    missing_all = [col for col in required_columns if col not in df_all.columns]
    missing_selected = [col for col in required_columns if col not in df_selected.columns]
    
    if missing_all or missing_selected:
        raise ValueError(f"Missing required columns: {missing_all + missing_selected}")
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Systematic Review: Metrics Overview', fontsize=16, fontweight='bold')
        
        # 1. Distribution of similarity to centroid
        ax1 = axes[0, 0]
        ax1.hist(df_all['similarity_to_centroid'], bins=50, alpha=0.6, label='All Papers', 
                 color='#87ceeb', density=True, edgecolor='navy', linewidth=0.8)
        ax1.hist(df_selected['similarity_to_centroid'], bins=30, alpha=0.6, label='Selected', 
                 color='#0066cc', density=True, edgecolor='darkblue', linewidth=1.2)
        ax1.set_xlabel('Similarity to Centroid')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution: Similarity to Centroid')
        ax1.legend(framealpha=0.9)
        ax1.grid(alpha=0.3)
        
        # 2. Distribution of diversity scores
        ax2 = axes[0, 1]
        ax2.hist(df_all['diversity_score'], bins=50, alpha=0.6, label='All Papers', 
                 color='#90ee90', density=True, edgecolor='darkgreen', linewidth=0.8)
        ax2.hist(df_selected['diversity_score'], bins=30, alpha=0.6, label='Selected', 
                 color='#006600', density=True, edgecolor='green', linewidth=1.2)
        ax2.set_xlabel('Diversity Score')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution: Diversity Scores')
        ax2.legend(framealpha=0.9)
        ax2.grid(alpha=0.3)
        
        # 3. Distribution of representativeness scores
        ax3 = axes[0, 2]
        ax3.hist(df_all['representativeness_score'], bins=50, alpha=0.6, label='All Papers', 
                 color='#ffa5a5', density=True, edgecolor='darkred', linewidth=0.8)
        ax3.hist(df_selected['representativeness_score'], bins=30, alpha=0.6, label='Selected', 
                 color='#cc0000', density=True, edgecolor='red', linewidth=1.2)
        ax3.set_xlabel('Representativeness Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution: Representativeness Scores')
        ax3.legend(framealpha=0.9)
        ax3.grid(alpha=0.3)
        
        # 4. Centrality vs Diversity scatter (selected papers)
        ax4 = axes[1, 0]
        scatter = ax4.scatter(df_selected['similarity_to_centroid'], df_selected['diversity_score'], 
                             c=df_selected['representativeness_score'], cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Similarity to Centroid (Centrality)')
        ax4.set_ylabel('Diversity Score')
        ax4.set_title('Selected Papers: Centrality vs Diversity\n(color = representativeness)')
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Representativeness Score')
        
        # 5. Research alignment comparison (all papers vs selected papers)
        ax5 = axes[1, 1]
        
        # Check if alignment columns exist
        alignment_columns = ['xai_alignment', 'symbolic_alignment', 'subsymbolic_alignment']
        if all(col in df_all.columns for col in alignment_columns):
            # Data for both all papers and selected papers
            all_alignment_data = [
                df_all['xai_alignment'].mean(),
                df_all['symbolic_alignment'].mean(),
                df_all['subsymbolic_alignment'].mean()
            ]
            selected_alignment_data = [
                df_selected['xai_alignment'].mean(),
                df_selected['symbolic_alignment'].mean(),
                df_selected['subsymbolic_alignment'].mean()
            ]
            
            # Create a cleaner grouped bar chart with categories on x-axis
            categories = ['All Papers', 'Selected Papers']
            x = np.arange(len(categories))
            width = 0.25
            
            # Define colors for each research focus
            xai_color = '#e74c3c'      # Red
            symbolic_color = '#2ecc71'  # Green  
            subsymbolic_color = '#3498db'  # Blue
            
            # Create bars for each research focus
            bars_xai = ax5.bar(x - width, [all_alignment_data[0], selected_alignment_data[0]], 
                               width, label='XAI', color=xai_color, alpha=0.8, edgecolor='white', linewidth=1)
            bars_symbolic = ax5.bar(x, [all_alignment_data[1], selected_alignment_data[1]], 
                                   width, label='Symbolic', color=symbolic_color, alpha=0.8, edgecolor='white', linewidth=1)
            bars_subsymbolic = ax5.bar(x + width, [all_alignment_data[2], selected_alignment_data[2]], 
                                      width, label='Sub-symbolic', color=subsymbolic_color, alpha=0.8, edgecolor='white', linewidth=1)
            
            ax5.set_ylabel('Average Alignment Score')
            ax5.set_title('Research Focus Alignment Comparison')
            ax5.set_xticks(x)
            ax5.set_xticklabels(categories, fontweight='bold')
            ax5.legend(loc='upper right', framealpha=0.9)
            ax5.set_ylim(0, max(max(all_alignment_data), max(selected_alignment_data)) * 1.2)
            ax5.grid(axis='y', alpha=0.3)
            
            # Add value labels on top of bars
            all_bars = [bars_xai, bars_symbolic, bars_subsymbolic]
            all_data = [
                [all_alignment_data[0], selected_alignment_data[0]],
                [all_alignment_data[1], selected_alignment_data[1]], 
                [all_alignment_data[2], selected_alignment_data[2]]
            ]
            
            for bars, data in zip(all_bars, all_data):
                for bar, value in zip(bars, data):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        else:
            # Placeholder if alignment columns don't exist
            ax5.text(0.5, 0.5, 'Research Alignment\nData Not Available', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Research Focus Alignment Comparison')
        
        # 6. Selection efficiency by topic size
        ax6 = axes[1, 2]
        
        # Check if required columns exist for topic analysis
        topic_columns = ['topic_id', 'cluster_size', 'papers_selected_from_cluster']
        if all(col in df_selected.columns for col in topic_columns):
            # Calculate selection efficiency (representativeness per paper selected)
            topic_data = df_selected.groupby('topic_id').agg({
                'cluster_size': 'first',
                'papers_selected_from_cluster': 'first',
                'representativeness_score': 'mean'
            }).reset_index()
            
            scatter = ax6.scatter(topic_data['cluster_size'], topic_data['representativeness_score'], 
                                 s=topic_data['papers_selected_from_cluster']*10, 
                                 alpha=0.7, c=topic_data['papers_selected_from_cluster'], 
                                 cmap='plasma', edgecolors='black', linewidth=0.5)
            ax6.set_xlabel('Cluster Size')
            ax6.set_ylabel('Avg Representativeness Score')
            ax6.set_title('Selection Efficiency by Topic\n(size = papers selected)')
            ax6.grid(alpha=0.3)
            plt.colorbar(scatter, ax=ax6, label='Papers Selected')
        else:
            # Placeholder if topic columns don't exist
            ax6.text(0.5, 0.5, 'Topic Analysis\nData Not Available', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Selection Efficiency by Topic')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = output_dir / 'metrics_overview.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"📊 Metrics overview visualization saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Error creating metrics overview: {e}")
        raise RuntimeError(f"Failed to create metrics overview visualization: {e}")


def validate_metrics_data(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required columns for metrics visualization.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    
    Note:
        - Checks for presence of required metric columns
        - Validates data types and ranges
        - Logs validation warnings for debugging
    """
    required_columns = ['similarity_to_centroid', 'diversity_score', 'representativeness_score']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns for metrics visualization: {missing_columns}")
        return False
    
    # Check data ranges
    for col in required_columns:
        if df[col].min() < 0 or df[col].max() > 1:
            logger.warning(f"Column {col} has values outside expected range [0,1]")
    
    # Check for null values
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found in metrics columns: {null_counts.to_dict()}")
    
    logger.info("✅ Metrics data validation passed")
    return True


def get_metrics_summary(df_all: pd.DataFrame, df_selected: pd.DataFrame) -> dict:
    """
    Generate summary statistics for metrics visualization.
    
    Args:
        df_all: Complete analysis DataFrame
        df_selected: Selected representatives DataFrame
        
    Returns:
        dict: Summary statistics for metrics
    
    Example:
        >>> summary = get_metrics_summary(df_all, df_selected)
        >>> print(f"Centrality improvement: {summary['centrality_improvement']:.1%}")
        Centrality improvement: 15.2%
    """
    try:
        metrics_cols = ['similarity_to_centroid', 'diversity_score', 'representativeness_score']
        
        summary = {
            'total_papers': len(df_all),
            'selected_papers': len(df_selected),
            'selection_ratio': len(df_selected) / len(df_all),
            'metrics_comparison': {}
        }
        
        for col in metrics_cols:
            if col in df_all.columns and col in df_selected.columns:
                all_mean = df_all[col].mean()
                selected_mean = df_selected[col].mean()
                improvement = ((selected_mean - all_mean) / all_mean) if all_mean > 0 else 0
                
                summary['metrics_comparison'][col] = {
                    'all_papers_mean': all_mean,
                    'selected_papers_mean': selected_mean,
                    'improvement_ratio': improvement,
                    'all_papers_std': df_all[col].std(),
                    'selected_papers_std': df_selected[col].std()
                }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating metrics summary: {e}")
        return {'error': str(e)} 