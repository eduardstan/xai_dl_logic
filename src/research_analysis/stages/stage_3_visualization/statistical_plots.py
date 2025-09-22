#!/usr/bin/env python3
"""
Statistical visualization functions for Stage 3 pipeline.

This module creates comprehensive statistical plots including distribution comparisons,
effect size visualizations, and temporal analysis charts.
"""

from typing import Dict, List, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from research_analysis.utils.logging import get_logger

logger = get_logger()

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")


def create_distribution_comparison_plots(
    selected_df: pd.DataFrame,
    non_selected_df: pd.DataFrame,
    variables: List[str],
    output_dir: Path,
    plot_config: Dict[str, Any]
) -> Path:
    """
    Create distribution comparison plots for selected vs non-selected papers.
    
    Args:
        selected_df: DataFrame with selected papers
        non_selected_df: DataFrame with non-selected papers
        variables: List of variables to plot
        output_dir: Directory to save plots
        plot_config: Plot configuration settings
        
    Returns:
        Path to saved plot file
    """
    logger.info("Creating distribution comparison plots...")
    
    n_vars = len(variables)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistical Distribution Comparison: Selected vs Non-Selected Papers', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for i, var in enumerate(variables[:6]):  # Limit to 6 plots
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get data and remove NaN values
        selected_data = selected_df[var].dropna()
        non_selected_data = non_selected_df[var].dropna()
        
        # Create histograms
        ax.hist(selected_data, bins=30, alpha=0.7, label='Selected', 
                color='skyblue', density=True, edgecolor='black', linewidth=0.5)
        ax.hist(non_selected_data, bins=30, alpha=0.7, label='Non-Selected', 
                color='lightcoral', density=True, edgecolor='black', linewidth=0.5)
        
        # Add median lines
        ax.axvline(selected_data.median(), color='blue', linestyle='--', 
                   linewidth=2, label=f'Selected Median: {selected_data.median():.3f}')
        ax.axvline(non_selected_data.median(), color='red', linestyle='--', 
                   linewidth=2, label=f'Non-Selected Median: {non_selected_data.median():.3f}')
        
        ax.set_xlabel(var.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution: {var.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(variables), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'distribution_comparison.{plot_config["plot_format"]}'
    plt.savefig(plot_path, dpi=plot_config["plot_dpi"], bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Distribution comparison plots saved to: {plot_path}")
    return plot_path


def create_effect_size_forest_plot(
    effect_size_results: Dict[str, Any],
    output_dir: Path,
    plot_config: Dict[str, Any],
    effect_size_thresholds: Dict[str, float]
) -> Path:
    """
    Create forest plot for effect sizes with confidence intervals.
    
    Args:
        effect_size_results: Dictionary with effect size results
        output_dir: Directory to save plots
        plot_config: Plot configuration settings
        
    Returns:
        Path to saved plot file
    """
    logger.info("Creating effect size forest plot...")
    
    if not effect_size_results:
        logger.warning("No effect size results to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    variables = list(effect_size_results.keys())
    effects = [result['cohens_d'] for result in effect_size_results.values()]
    cis = [result['confidence_interval'] for result in effect_size_results.values()]
    categories = [result['effect_size_category'] for result in effect_size_results.values()]
    
    # Color mapping for effect sizes
    color_map = {
        'negligible': 'gray',
        'small': 'green',
        'medium': 'orange', 
        'large': 'red'
    }
    colors = [color_map.get(cat, 'blue') for cat in categories]
    
    # Create forest plot
    y_pos = np.arange(len(variables))
    
    # Plot effect sizes with colors based on magnitude
    for i, (effect, color, category) in enumerate(zip(effects, colors, categories)):
        ax.scatter(effect, i, s=150, color=color, zorder=3, 
                   label=category.title() if category not in ax.get_legend_handles_labels()[1] else "")
    
    # Plot confidence intervals
    for i, (lower, upper) in enumerate(cis):
        ax.plot([lower, upper], [i, i], 'k-', linewidth=2, alpha=0.7)
        ax.plot([lower, lower], [i-0.1, i+0.1], 'k-', linewidth=2, alpha=0.7)
        ax.plot([upper, upper], [i-0.1, i+0.1], 'k-', linewidth=2, alpha=0.7)
    
    # Add reference lines using configurable thresholds
    small_threshold = effect_size_thresholds['small']
    medium_threshold = effect_size_thresholds['medium'] 
    large_threshold = effect_size_thresholds['large']
    
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    ax.axvline(x=small_threshold, color='green', linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(x=medium_threshold, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(x=large_threshold, color='red', linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(x=-small_threshold, color='green', linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(x=-medium_threshold, color='orange', linestyle='--', alpha=0.6, linewidth=1)
    ax.axvline(x=-large_threshold, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels([var.replace('_', ' ').title() for var in variables], fontsize=12)
    ax.set_xlabel("Cohen's d Effect Size", fontsize=14, fontweight='bold')
    ax.set_title("Effect Sizes with 95% Confidence Intervals", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add effect size reference text using configurable thresholds
    threshold_text = f'Effect Size Thresholds:\nSmall: ±{small_threshold}\nMedium: ±{medium_threshold}\nLarge: ±{large_threshold}'
    ax.text(0.02, 0.98, threshold_text, 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Create custom legend
    handles = []
    labels = []
    for category, color in color_map.items():
        if category in categories:
            handles.append(plt.scatter([], [], c=color, s=100))
            labels.append(category.title())
    
    ax.legend(handles, labels, loc='lower right', title='Effect Size Category', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'effect_size_forest_plot.{plot_config["plot_format"]}'
    plt.savefig(plot_path, dpi=plot_config["plot_dpi"], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Effect size forest plot saved to: {plot_path}")
    return plot_path


def create_temporal_analysis_plot(
    temporal_results: Dict[str, Any],
    selected_df: pd.DataFrame,
    non_selected_df: pd.DataFrame,
    output_dir: Path,
    plot_config: Dict[str, Any]
) -> Path:
    """
    Create temporal distribution analysis plot.
    
    Args:
        temporal_results: Results from temporal analysis
        selected_df: DataFrame with selected papers
        non_selected_df: DataFrame with non-selected papers
        output_dir: Directory to save plots
        plot_config: Plot configuration settings
        
    Returns:
        Path to saved plot file
    """
    logger.info("Creating temporal analysis plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Temporal Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Publication year histograms  
    selected_years = selected_df['year'].dropna()
    non_selected_years = non_selected_df['year'].dropna()
    
    # Create aesthetically pleasing bins - group by 2-year periods for better visual balance
    year_min = int(min(selected_years.min(), non_selected_years.min()))
    year_max = int(max(selected_years.max(), non_selected_years.max()))
    
    # Create bins that align with meaningful periods (every 2 years for better aesthetics)
    bin_start = year_min - (year_min % 2)  # Start from even year
    bin_end = year_max + (2 - year_max % 2) if year_max % 2 != 0 else year_max + 2
    bins = np.arange(bin_start, bin_end + 1, 2)  # 2-year bins
    
    ax1.hist(selected_years, bins=bins, alpha=0.7, label='Selected', 
             color='skyblue', edgecolor='black', linewidth=0.5)
    ax1.hist(non_selected_years, bins=bins, alpha=0.7, label='Non-Selected', 
             color='lightcoral', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Publication Year', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Publication Year Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add statistical test result
    ks_stat = temporal_results['kolmogorov_smirnov_test']['statistic']
    ks_p = temporal_results['kolmogorov_smirnov_test']['p_value']
    significant = temporal_results['kolmogorov_smirnov_test']['significant_difference']
    
    test_text = f'Kolmogorov-Smirnov Test:\nStatistic: {ks_stat:.4f}\np-value: {ks_p:.4f}\n'
    test_text += f'Significant: {"Yes" if significant else "No"}'
    ax1.text(0.02, 0.98, test_text, transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Selection rate by year
    all_df = pd.concat([selected_df, non_selected_df])
    yearly_selection = all_df.groupby('year').agg({
        'is_selected_representative': ['count', 'sum']
    })
    yearly_selection.columns = ['total_papers', 'selected_papers']
    yearly_selection['selection_rate'] = yearly_selection['selected_papers'] / yearly_selection['total_papers']
    
    # Only plot years with reasonable sample sizes
    yearly_selection_filtered = yearly_selection[yearly_selection['total_papers'] >= 5]
    
    ax2.scatter(yearly_selection_filtered.index, yearly_selection_filtered['selection_rate'], 
                s=yearly_selection_filtered['total_papers']*2, alpha=0.6, 
                c='purple', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    if len(yearly_selection_filtered) > 1:
        z = np.polyfit(yearly_selection_filtered.index, yearly_selection_filtered['selection_rate'], 1)
        p = np.poly1d(z)
        ax2.plot(yearly_selection_filtered.index, p(yearly_selection_filtered.index), 
                 "r--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Publication Year', fontsize=12)
    ax2.set_ylabel('Selection Rate', fontsize=12)
    ax2.set_title('Selection Rate by Publication Year', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add overall selection rate line
    overall_rate = len(selected_df) / (len(selected_df) + len(non_selected_df))
    ax2.axhline(y=overall_rate, color='red', linestyle='-', alpha=0.7, 
                label=f'Overall Rate: {overall_rate:.3f}')
    ax2.legend(fontsize=10)
    
    # Add size legend
    ax2.text(0.02, 0.98, 'Bubble size ∝ number of papers', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'temporal_analysis.{plot_config["plot_format"]}'
    plt.savefig(plot_path, dpi=plot_config["plot_dpi"], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Temporal analysis plot saved to: {plot_path}")
    return plot_path


def create_statistical_summary_plot(
    mann_whitney_results: Dict[str, Any],
    effect_size_results: Dict[str, Any],
    output_dir: Path,
    plot_config: Dict[str, Any],
    effect_size_thresholds: Dict[str, float]
) -> Path:
    """
    Create summary plot of statistical test results.
    
    Args:
        mann_whitney_results: Results from Mann-Whitney U tests
        effect_size_results: Results from effect size analysis
        output_dir: Directory to save plots
        plot_config: Plot configuration settings
        
    Returns:
        Path to saved plot file
    """
    logger.info("Creating statistical summary plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # fig.suptitle('Statistical Analysis Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: P-values from Mann-Whitney tests
    variables = list(mann_whitney_results.keys())
    p_values = [result['p_value'] for result in mann_whitney_results.values()]
    significant = [result['significant'] for result in mann_whitney_results.values()]
    
    colors = ['red' if sig else 'blue' for sig in significant]
    
    bars = ax1.barh(variables, p_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax1.set_xlabel('P-value', fontsize=12)
    ax1.set_title('Mann-Whitney U Test P-values', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add significance annotations
    for i, (p_val, sig) in enumerate(zip(p_values, significant)):
        ax1.text(p_val + 0.001, i, f'{p_val:.4f}{"*" if sig else ""}', 
                 va='center', fontsize=9)
    
    # Plot 2: Effect sizes
    effect_values = [result['cohens_d'] for result in effect_size_results.values()]
    categories = [result['effect_size_category'] for result in effect_size_results.values()]
    
    color_map = {
        'negligible': 'gray',
        'small': 'green',
        'medium': 'orange',
        'large': 'red'
    }
    colors = [color_map.get(cat, 'blue') for cat in categories]
    
    bars = ax2.barh(variables, effect_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Use configurable thresholds
    small_threshold = effect_size_thresholds['small']
    medium_threshold = effect_size_thresholds['medium']
    large_threshold = effect_size_thresholds['large']
    
    ax2.axvline(x=small_threshold, color='green', linestyle='--', alpha=0.6)
    ax2.axvline(x=medium_threshold, color='orange', linestyle='--', alpha=0.6)
    ax2.axvline(x=large_threshold, color='red', linestyle='--', alpha=0.6)
    ax2.axvline(x=-small_threshold, color='green', linestyle='--', alpha=0.6)
    ax2.axvline(x=-medium_threshold, color='orange', linestyle='--', alpha=0.6)
    ax2.axvline(x=-large_threshold, color='red', linestyle='--', alpha=0.6)
    
    ax2.set_xlabel("Cohen's d", fontsize=12)
    ax2.set_title('Effect Sizes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add effect size annotations
    for i, (effect, cat) in enumerate(zip(effect_values, categories)):
        ax2.text(effect + 0.02 if effect >= 0 else effect - 0.02, i, 
                 f'{effect:.3f} ({cat})', va='center', fontsize=9,
                 ha='left' if effect >= 0 else 'right')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'statistical_summary.{plot_config["plot_format"]}'
    plt.savefig(plot_path, dpi=plot_config["plot_dpi"], bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Statistical summary plot saved to: {plot_path}")
    return plot_path


def create_all_statistical_plots(
    statistical_results: Dict[str, Any],
    selected_df: pd.DataFrame,
    non_selected_df: pd.DataFrame,
    variables: List[str],
    output_dir: Path,
    config: Any
) -> Dict[str, Path]:
    """
    Create all statistical visualization plots.
    
    Args:
        statistical_results: Results from comprehensive statistical analysis
        selected_df: DataFrame with selected papers
        non_selected_df: DataFrame with non-selected papers
        variables: List of variables analyzed
        output_dir: Directory to save plots
        config: Stage 3 configuration object
        
    Returns:
        Dictionary mapping plot types to file paths
    """
    logger.info("Creating all statistical plots...")
    
    # Extract configuration values
    plot_config = config.visualizations.dict()
    effect_size_thresholds = config.statistical_analysis.tests.cohens_d.effect_size_thresholds
    
    plot_paths = {}
    
    # Distribution comparison plots
    if plot_config.get("create_statistical_plots", True):
        plot_paths['distribution_comparison'] = create_distribution_comparison_plots(
            selected_df, non_selected_df, variables, output_dir, plot_config
        )
    
    # Effect size forest plot
    if plot_config.get("create_effect_size_plots", True):
        plot_paths['effect_size_forest'] = create_effect_size_forest_plot(
            statistical_results['effect_size_analysis'], output_dir, plot_config, effect_size_thresholds
        )
    
    # Temporal analysis plot
    if plot_config.get("create_temporal_plots", True):
        plot_paths['temporal_analysis'] = create_temporal_analysis_plot(
            statistical_results['temporal_analysis'], selected_df, non_selected_df, 
            output_dir, plot_config
        )
    
    # Statistical summary plot
    plot_paths['statistical_summary'] = create_statistical_summary_plot(
        statistical_results['mann_whitney_analysis']['individual_tests'],
        statistical_results['effect_size_analysis'], output_dir, plot_config, effect_size_thresholds
    )
    
    logger.info(f"All statistical plots created: {list(plot_paths.keys())}")
    return plot_paths 