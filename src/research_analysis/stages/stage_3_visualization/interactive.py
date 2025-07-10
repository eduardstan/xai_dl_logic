#!/usr/bin/env python3
"""
Interactive visualization module for Stage 3.

This module creates interactive web-based visualizations using Plotly
for exploration and analysis of research data and metrics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

from research_analysis.utils.logging import get_logger

logger = get_logger()


def create_interactive_visualizations(
    df_all: pd.DataFrame,
    df_selected: pd.DataFrame,
    df_summary: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Create interactive web-based visualizations using Plotly.
    
    Generates two HTML files:
    1. Interactive scatter plot of selected representative papers
    2. Multi-panel topic comparison dashboard
    
    Args:
        df_all: Complete analysis DataFrame with all papers
        df_selected: Selected representative papers DataFrame
        df_summary: Topic-level summary statistics DataFrame
        output_dir: Directory to save HTML visualizations
        
    Raises:
        ValueError: If required columns are missing
        RuntimeError: If visualization generation fails
    """
    logger.info("🌐 Creating interactive visualizations...")
    
    try:
        # Create papers explorer
        _create_papers_explorer(df_selected, output_dir)
        
        # Create topic dashboard
        _create_topic_dashboard(df_all, df_selected, df_summary, output_dir)
        
        logger.info("✅ Interactive visualizations completed")
        
    except Exception as e:
        logger.error(f"Failed to create interactive visualizations: {e}", exc_info=True)
        raise RuntimeError(f"Interactive visualization failed: {e}")


def _create_papers_explorer(df_selected: pd.DataFrame, output_dir: Path) -> None:
    """Create interactive scatter plot of selected representative papers."""
    logger.info("Creating interactive papers explorer...")
    
    # Validate required columns
    required_cols = ['similarity_to_centroid', 'diversity_score', 'representativeness_score', 
                    'title', 'authors', 'topic_id']
    missing_cols = [col for col in required_cols if col not in df_selected.columns]
    if missing_cols:
        logger.warning(f"Missing columns for papers explorer: {missing_cols}")
        return
    
    # Create hover text with paper information
    hover_text = []
    for _, row in df_selected.iterrows():
        hover_info = (
            f"<b>{row.get('title', 'Unknown Title')}</b><br>"
            f"Authors: {row.get('authors', 'Unknown')}<br>"
            f"Topic: {row.get('topic_id', 'Unknown')}<br>"
            f"Centrality: {row.get('similarity_to_centroid', 0):.3f}<br>"
            f"Diversity: {row.get('diversity_score', 0):.3f}<br>"
            f"Representativeness: {row.get('representativeness_score', 0):.3f}"
        )
        hover_text.append(hover_info)
    
    # Create interactive scatter plot
    fig = px.scatter(
        df_selected,
        x='similarity_to_centroid',
        y='diversity_score',
        color='representativeness_score',
        size='representativeness_score',
        hover_name='title',
        color_continuous_scale='viridis',
        title='Interactive Papers Explorer: Selected Representatives',
        labels={
            'similarity_to_centroid': 'Similarity to Centroid (Centrality)',
            'diversity_score': 'Diversity Score',
            'representativeness_score': 'Representativeness Score'
        }
    )
    
    # Update traces with custom hover data
    fig.update_traces(
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text
    )
    
    # Update layout
    fig.update_layout(
        width=900,
        height=600,
        font_size=12,
        showlegend=True
    )
    
    # Save HTML file
    output_path = output_dir / "interactive_papers_explorer.html"
    pyo.plot(fig, filename=str(output_path), auto_open=False)
    logger.info(f"Papers explorer saved to: {output_path}")


def _create_topic_dashboard(
    df_all: pd.DataFrame,
    df_selected: pd.DataFrame,
    df_summary: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create multi-panel topic comparison dashboard."""
    logger.info("Creating topic dashboard...")
    
    # Add topic_name column if missing (generate from topic_id)
    if 'topic_name' not in df_summary.columns:
        df_summary = df_summary.copy()
        df_summary['topic_name'] = df_summary['topic_id'].apply(lambda x: f"Topic {x}")
    
    # Add papers_selected column if missing (alias for papers_selected_from_cluster)
    if 'papers_selected' not in df_summary.columns:
        df_summary = df_summary.copy()
        df_summary['papers_selected'] = df_summary['papers_selected_from_cluster']
    
    # Validate required columns for summary
    required_summary_cols = ['cluster_size', 'papers_selected', 'topic_name']
    missing_cols = [col for col in required_summary_cols if col not in df_summary.columns]
    if missing_cols:
        logger.warning(f"Missing columns for topic dashboard: {missing_cols}")
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cluster Size vs Papers Selected',
            'Average Metrics by Topic',
            'Selection Efficiency Analysis',
            'Research Alignment Distribution'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ]
    )
    
    # Plot 1: Cluster size vs papers selected
    fig.add_trace(
        go.Scatter(
            x=df_summary['cluster_size'],
            y=df_summary['papers_selected'],
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.7),
            text=df_summary['topic_name'],
            hovertemplate='<b>%{text}</b><br>Cluster Size: %{x}<br>Papers Selected: %{y}<extra></extra>',
            name='Topics'
        ),
        row=1, col=1
    )
    
    # Plot 2: Average metrics by topic (if available)
    if all(col in df_summary.columns for col in ['avg_centrality', 'avg_diversity']):
        fig.add_trace(
            go.Bar(
                x=df_summary['topic_name'][:10],  # Show top 10 topics
                y=df_summary['avg_centrality'][:10],
                name='Avg Centrality',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=df_summary['topic_name'][:10],
                y=np.maximum(df_summary['avg_diversity'][:10], 0),  # Clamp to 0
                name='Avg Diversity',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
    
    # Plot 3: Selection efficiency (selection ratio vs cluster size)
    selection_ratios = df_summary['papers_selected'] / df_summary['cluster_size']
    fig.add_trace(
        go.Scatter(
            x=df_summary['cluster_size'],
            y=selection_ratios,
            mode='markers',
            marker=dict(
                size=df_summary['papers_selected'],
                color=selection_ratios,
                colorscale='plasma',
                showscale=True,
                colorbar=dict(title="Selection Ratio")
            ),
            text=df_summary['topic_name'],
            hovertemplate='<b>%{text}</b><br>Cluster Size: %{x}<br>Selection Ratio: %{y:.2%}<extra></extra>',
            name='Selection Efficiency'
        ),
        row=2, col=1
    )
    
    # Plot 4: Research alignment (if available)
    alignment_cols = ['xai_alignment', 'symbolic_alignment', 'subsymbolic_alignment']
    if all(col in df_selected.columns for col in alignment_cols):
        alignment_means = [df_selected[col].mean() for col in alignment_cols]
        alignment_labels = ['XAI', 'Symbolic', 'Sub-symbolic']
        
        fig.add_trace(
            go.Bar(
                x=alignment_labels,
                y=alignment_means,
                marker_color=['red', 'green', 'blue'],
                name='Research Alignment',
                text=[f'{mean:.3f}' for mean in alignment_means],
                textposition='auto'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Topic Analysis Dashboard",
        title_x=0.5,
        height=800,
        showlegend=True,
        font_size=10
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Cluster Size", row=1, col=1)
    fig.update_yaxes(title_text="Papers Selected", row=1, col=1)
    
    fig.update_xaxes(title_text="Topic", row=1, col=2)
    fig.update_yaxes(title_text="Average Score", row=1, col=2)
    
    fig.update_xaxes(title_text="Cluster Size", row=2, col=1)
    fig.update_yaxes(title_text="Selection Ratio", row=2, col=1)
    
    fig.update_xaxes(title_text="Research Focus", row=2, col=2)
    fig.update_yaxes(title_text="Alignment Score", row=2, col=2)
    
    # Save HTML file
    output_path = output_dir / "topic_dashboard.html"
    pyo.plot(fig, filename=str(output_path), auto_open=False)
    logger.info(f"Topic dashboard saved to: {output_path}")


def create_simple_scatter_plot(df: pd.DataFrame, output_dir: Path, filename: str = "simple_scatter.html") -> None:
    """
    Create a simple scatter plot as a fallback option.
    
    Args:
        df: DataFrame with data to plot
        output_dir: Directory to save the file
        filename: Name of the output HTML file
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame provided for scatter plot")
        return
    
    try:
        # Use the first two numeric columns for x and y
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for scatter plot")
            return
        
        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}"
        )
        
        output_path = output_dir / filename
        pyo.plot(fig, filename=str(output_path), auto_open=False)
        logger.info(f"Simple scatter plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create simple scatter plot: {e}")


def validate_plotly_data(df: pd.DataFrame, required_cols: list) -> bool:
    """
    Validate DataFrame for Plotly visualization compatibility.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or len(df) == 0:
        logger.warning("DataFrame is empty or None")
        return False
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
        return False
    
    return True 