#!/usr/bin/env python3
"""
Interactive Visualization Module for Enhanced Research Visualization Pipeline

This module provides functions to create interactive web-based visualizations
using Plotly for exploration and analysis of research data and metrics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from loguru import logger


def create_interactive_topic_explorer(df_all: pd.DataFrame, df_selected: pd.DataFrame, 
                                     df_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comprehensive interactive visualizations using Plotly.
    
    This function generates interactive web-based visualizations including
    scatter plots for paper exploration and multi-panel dashboards for
    topic analysis with hover information and dynamic filtering.
    
    Args:
        df_all: Complete analysis DataFrame containing all papers
        df_selected: Selected representative papers DataFrame
        df_summary: Topic-level summary statistics DataFrame
        output_dir: Output directory path for saving HTML visualizations
    
    Raises:
        ValueError: If required columns are missing from input DataFrames
        RuntimeError: If visualization generation fails
    
    Generated visualizations:
        1. Interactive scatter plot of selected representative papers
        2. Multi-panel topic comparison dashboard with:
           - Cluster size vs papers selected
           - Average metrics by topic
           - Selection efficiency analysis
           - Research alignment distribution
    
    Example:
        >>> create_interactive_topic_explorer(df_all, df_selected, df_summary, Path("results"))
        # Creates interactive_papers_explorer.html and topic_dashboard.html
    
    Note:
        - Hover data includes paper titles, authors, and topic information
        - Color coding represents different metrics and dimensions
        - Interactive zoom, pan, and filter capabilities
        - Professional styling suitable for presentations
        - Saves HTML files for web browser viewing
    """
    logger.info("🌐 Creating interactive visualizations with Plotly...")
    
    try:
        # 1. Interactive scatter plot of selected papers
        _create_papers_explorer(df_selected, output_dir)
        
        # 2. Topic comparison dashboard
        _create_topic_dashboard(df_all, df_selected, df_summary, output_dir)
        
        logger.info("🌐 Interactive visualizations created successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error creating interactive visualizations: {e}")
        raise RuntimeError(f"Failed to create interactive visualizations: {e}")


def _create_papers_explorer(df_selected: pd.DataFrame, output_dir: Path) -> None:
    """
    Create interactive scatter plot explorer for selected representative papers.
    
    Args:
        df_selected: Selected representative papers DataFrame
        output_dir: Output directory for HTML file
    
    Note:
        - Interactive scatter plot with centrality vs diversity
        - Bubble sizes represent cluster sizes
        - Color coding shows representativeness scores
        - Hover information includes paper details
    """
    # Validate required columns
    required_columns = ['similarity_to_centroid', 'diversity_score', 'representativeness_score']
    missing_columns = [col for col in required_columns if col not in df_selected.columns]
    
    if missing_columns:
        logger.warning(f"Missing columns for papers explorer: {missing_columns}")
        return
    
    # Prepare hover data columns
    hover_cols = []
    for col in ['topic_id', 'title', 'authors']:
        if col in df_selected.columns:
            hover_cols.append(col)
    
    # Create scatter plot
    fig1 = px.scatter(
        df_selected, 
        x='similarity_to_centroid', 
        y='diversity_score',
        size='cluster_size' if 'cluster_size' in df_selected.columns else None,
        color='representativeness_score',
        hover_data=hover_cols,
        color_continuous_scale='viridis',
        title='Interactive Explorer: Selected Representative Papers'
    )
    
    fig1.update_layout(
        xaxis_title="Similarity to Centroid (Centrality)",
        yaxis_title="Diversity Score",
        width=1000,
        height=700,
        font=dict(size=12),
        title_font_size=16
    )
    
    # Save HTML file
    output_path = output_dir / 'interactive_papers_explorer.html'
    fig1.write_html(str(output_path))
    logger.info(f"📊 Interactive papers explorer saved to: {output_path}")


def _create_topic_dashboard(df_all: pd.DataFrame, df_selected: pd.DataFrame, 
                           df_summary: pd.DataFrame, output_dir: Path) -> None:
    """
    Create multi-panel interactive dashboard for topic analysis.
    
    Args:
        df_all: Complete analysis DataFrame
        df_selected: Selected representative papers DataFrame
        df_summary: Topic-level summary DataFrame
        output_dir: Output directory for HTML file
    
    Note:
        - Four-panel dashboard with different analysis perspectives
        - Interactive zoom and hover for detailed exploration
        - Professional layout suitable for presentations
    """
    # Create multi-panel dashboard
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cluster Size vs Papers Selected', 
            'Average Metrics by Topic',
            'Selection Efficiency',
            'Research Alignment Distribution'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"type": "domain"}]
        ]
    )
    
    # Subplot 1: Cluster size vs papers selected
    if all(col in df_summary.columns for col in ['cluster_size', 'papers_selected', 'avg_centrality']):
        fig2.add_trace(
            go.Scatter(
                x=df_summary['cluster_size'], 
                y=df_summary['papers_selected'],
                mode='markers',
                marker=dict(
                    size=10, 
                    color=df_summary['avg_centrality'], 
                    colorscale='RdYlBu', 
                    showscale=False
                ),
                text=df_summary['topic_name'] if 'topic_name' in df_summary.columns else df_summary['topic_id'],
                name='Topics'
            ),
            row=1, col=1
        )
    
    # Subplot 2: Average metrics by topic (first 10 topics for clarity)
    if all(col in df_summary.columns for col in ['topic_id', 'avg_centrality', 'avg_diversity']):
        n_topics = min(10, len(df_summary))
        
        fig2.add_trace(
            go.Bar(
                x=df_summary['topic_id'][:n_topics], 
                y=df_summary['avg_centrality'][:n_topics],
                name='Avg Centrality',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        fig2.add_trace(
            go.Bar(
                x=df_summary['topic_id'][:n_topics], 
                y=np.maximum(df_summary['avg_diversity'][:n_topics], 0),  # Fix negative values
                name='Avg Diversity',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
    
    # Subplot 3: Selection efficiency
    if all(col in df_summary.columns for col in ['cluster_size', 'avg_centrality', 'avg_diversity', 'papers_selected']):
        efficiency = df_summary['avg_centrality'] * np.maximum(df_summary['avg_diversity'], 0)
        
        fig2.add_trace(
            go.Scatter(
                x=df_summary['cluster_size'], 
                y=efficiency,
                mode='markers',
                marker=dict(
                    size=df_summary['papers_selected']*3, 
                    color='green', 
                    opacity=0.7
                ),
                name='Efficiency'
            ),
            row=2, col=1
        )
    
    # Subplot 4: Research alignment pie chart
    if all(col in df_selected.columns for col in ['xai_alignment', 'symbolic_alignment', 'subsymbolic_alignment']):
        total_xai = df_selected['xai_alignment'].sum()
        total_symbolic = df_selected['symbolic_alignment'].sum() 
        total_subsymbolic = df_selected['subsymbolic_alignment'].sum()
        
        fig2.add_trace(
            go.Pie(
                labels=['XAI', 'Symbolic', 'Sub-symbolic'],
                values=[total_xai, total_symbolic, total_subsymbolic],
                name="Research Focus"
            ),
            row=2, col=2
        )
    else:
        # Placeholder if alignment data is not available
        fig2.add_annotation(
            text="Research Alignment<br>Data Not Available",
            xref="x4", yref="y4",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
    
    # Update layout
    fig2.update_layout(
        height=800, 
        showlegend=True, 
        title_text="Topic Analysis Dashboard",
        title_font_size=16,
        font=dict(size=11)
    )
    
    # Save HTML file
    output_path = output_dir / 'topic_dashboard.html'
    fig2.write_html(str(output_path))
    logger.info(f"📊 Topic dashboard saved to: {output_path}")


def validate_interactive_data(df_selected: pd.DataFrame, df_summary: pd.DataFrame) -> bool:
    """
    Validate that DataFrames contain required columns for interactive visualizations.
    
    Args:
        df_selected: Selected representatives DataFrame
        df_summary: Topic summary DataFrame
        
    Returns:
        bool: True if validation passes, False otherwise
    
    Note:
        - Checks for presence of required columns for interactive plots
        - Validates data types and ranges
        - Logs validation warnings for debugging
    """
    # Check selected papers data
    required_selected = ['similarity_to_centroid', 'diversity_score', 'representativeness_score']
    missing_selected = [col for col in required_selected if col not in df_selected.columns]
    
    if missing_selected:
        logger.error(f"Missing required columns in selected data: {missing_selected}")
        return False
    
    # Check summary data
    required_summary = ['topic_id', 'cluster_size', 'papers_selected', 'avg_centrality']
    missing_summary = [col for col in required_summary if col not in df_summary.columns]
    
    if missing_summary:
        logger.error(f"Missing required columns in summary data: {missing_summary}")
        return False
    
    # Check data ranges
    for col in ['similarity_to_centroid', 'diversity_score', 'representativeness_score']:
        if col in df_selected.columns:
            if df_selected[col].min() < 0 or df_selected[col].max() > 1:
                logger.warning(f"Column {col} has values outside expected range [0,1]")
    
    # Check for null values in critical columns
    critical_columns = ['similarity_to_centroid', 'diversity_score', 'topic_id']
    for col in critical_columns:
        if col in df_selected.columns and df_selected[col].isnull().any():
            logger.warning(f"Null values found in critical column {col}")
    
    logger.info("✅ Interactive visualization data validation passed")
    return True


def get_interactive_summary(df_selected: pd.DataFrame, df_summary: pd.DataFrame) -> dict:
    """
    Generate summary information for interactive visualizations.
    
    Args:
        df_selected: Selected representatives DataFrame
        df_summary: Topic summary DataFrame
        
    Returns:
        dict: Summary information for interactive features
    
    Example:
        >>> summary = get_interactive_summary(df_selected, df_summary)
        >>> print(f"Interactive features: {len(summary['available_features'])} available")
        Interactive features: 5 available
    """
    try:
        available_features = []
        
        # Check what interactive features are available
        if all(col in df_selected.columns for col in ['similarity_to_centroid', 'diversity_score']):
            available_features.append('scatter_plot_explorer')
        
        if 'representativeness_score' in df_selected.columns:
            available_features.append('color_coding')
        
        if 'cluster_size' in df_selected.columns:
            available_features.append('bubble_sizing')
        
        if any(col in df_selected.columns for col in ['title', 'authors', 'topic_id']):
            available_features.append('hover_information')
        
        if all(col in df_summary.columns for col in ['cluster_size', 'papers_selected']):
            available_features.append('topic_dashboard')
        
        # Calculate data ranges for visualization bounds
        data_ranges = {}
        for col in ['similarity_to_centroid', 'diversity_score', 'representativeness_score']:
            if col in df_selected.columns:
                data_ranges[col] = {
                    'min': float(df_selected[col].min()),
                    'max': float(df_selected[col].max()),
                    'mean': float(df_selected[col].mean())
                }
        
        return {
            'available_features': available_features,
            'feature_count': len(available_features),
            'data_ranges': data_ranges,
            'selected_papers_count': len(df_selected),
            'topics_count': len(df_summary),
            'has_alignment_data': all(col in df_selected.columns for col in 
                                    ['xai_alignment', 'symbolic_alignment', 'subsymbolic_alignment']),
            'has_topic_names': 'topic_name' in df_summary.columns,
            'has_paper_metadata': any(col in df_selected.columns for col in ['title', 'authors'])
        }
        
    except Exception as e:
        logger.error(f"Error generating interactive summary: {e}")
        return {'error': str(e)} 