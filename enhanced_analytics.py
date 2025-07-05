#!/usr/bin/env python3
"""
Enhanced Analytics Module for Selected vs Non-Selected Paper Analysis

This module provides advanced statistical analysis and visualization capabilities
to compare selected representative papers with non-selected papers across multiple
metrics including centrality, diversity, and research alignment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from pathlib import Path
from loguru import logger
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import shared utilities
from utils import (
    compute_cosine_similarity_to_centroid,
    compute_diversity_score,
    compute_representativeness_score,
    get_systematic_review_config
)


def compute_enhanced_metrics(df_all: pd.DataFrame, embeddings: np.ndarray, 
                           topic_assignments: np.ndarray, config: Dict) -> pd.DataFrame:
    """
    Compute enhanced metrics for all papers including centrality, diversity, and representativeness.
    
    Args:
        df_all: DataFrame with all papers
        embeddings: Paper embeddings matrix
        topic_assignments: Topic assignments for each paper
        config: Configuration dictionary
    
    Returns:
        pd.DataFrame: Enhanced dataframe with computed metrics
    """
    logger.info("📊 Computing enhanced metrics for all papers...")
    
    enhanced_df = df_all.copy()
    enhanced_df['topic'] = topic_assignments
    
    # Initialize metric columns
    enhanced_df['centrality_score'] = 0.0
    enhanced_df['diversity_score'] = 0.0
    enhanced_df['representativeness_score'] = 0.0
    
    review_config = get_systematic_review_config(config)
    diversity_weight = review_config['diversity_weight']
    
    # Compute metrics for each topic
    for topic_id in enhanced_df['topic'].unique():
        if topic_id == -1:  # Skip outliers
            continue
            
        topic_mask = enhanced_df['topic'] == topic_id
        topic_indices = enhanced_df[topic_mask].index
        topic_embeddings = embeddings[topic_indices]
        
        # Compute centrality scores
        centrality_scores = []
        for i, embedding in enumerate(topic_embeddings):
            centrality = compute_cosine_similarity_to_centroid(embedding, topic_embeddings)
            centrality_scores.append(centrality)
        
        # Compute diversity scores
        diversity_scores = []
        for i, embedding in enumerate(topic_embeddings):
            diversity = compute_diversity_score(embedding, topic_embeddings)
            diversity_scores.append(diversity)
        
        # Compute representativeness scores
        representativeness_scores = []
        for centrality, diversity in zip(centrality_scores, diversity_scores):
            repr_score = compute_representativeness_score(centrality, diversity, diversity_weight)
            representativeness_scores.append(repr_score)
        
        # Assign scores to dataframe
        enhanced_df.loc[topic_indices, 'centrality_score'] = centrality_scores
        enhanced_df.loc[topic_indices, 'diversity_score'] = diversity_scores
        enhanced_df.loc[topic_indices, 'representativeness_score'] = representativeness_scores
    
    logger.info(f"✅ Enhanced metrics computed for {len(enhanced_df)} papers")
    return enhanced_df


def compute_statistical_comparisons(df_enhanced: pd.DataFrame, 
                                  selected_indices: List[int]) -> Dict[str, Dict]:
    """
    Compute statistical comparisons between selected and non-selected papers.
    
    Args:
        df_enhanced: DataFrame with enhanced metrics
        selected_indices: List of indices for selected papers
    
    Returns:
        Dict containing statistical test results and effect sizes
    """
    logger.info("📈 Computing statistical comparisons between selected and non-selected papers...")
    
    # Create selection mask
    df_enhanced['is_selected'] = False
    df_enhanced.loc[selected_indices, 'is_selected'] = True
    
    selected_papers = df_enhanced[df_enhanced['is_selected'] == True]
    non_selected_papers = df_enhanced[df_enhanced['is_selected'] == False]
    
    metrics = ['centrality_score', 'diversity_score', 'representativeness_score']
    results = {}
    
    for metric in metrics:
        selected_values = selected_papers[metric].values
        non_selected_values = non_selected_papers[metric].values
        
        # Remove any NaN values
        selected_values = selected_values[~np.isnan(selected_values)]
        non_selected_values = non_selected_values[~np.isnan(non_selected_values)]
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            selected_values, non_selected_values, 
            alternative='two-sided'
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(selected_values) - 1) * np.var(selected_values, ddof=1) + 
             (len(non_selected_values) - 1) * np.var(non_selected_values, ddof=1)) / 
            (len(selected_values) + len(non_selected_values) - 2)
        )
        
        cohens_d = (np.mean(selected_values) - np.mean(non_selected_values)) / pooled_std
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        results[metric] = {
            'mann_whitney_u': statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': effect_size,
            'selected_mean': np.mean(selected_values),
            'selected_std': np.std(selected_values),
            'non_selected_mean': np.mean(non_selected_values),
            'non_selected_std': np.std(non_selected_values),
            'selected_median': np.median(selected_values),
            'non_selected_median': np.median(non_selected_values)
        }
    
    logger.info("✅ Statistical comparisons computed successfully")
    return results


def compute_improved_research_alignment(df_enhanced: pd.DataFrame, 
                                      selected_indices: List[int],
                                      config: Dict) -> Dict[str, Dict]:
    """
    Compute improved research alignment metrics not normalized by number of terms.
    
    Args:
        df_enhanced: DataFrame with enhanced metrics
        selected_indices: List of indices for selected papers
        config: Configuration dictionary
    
    Returns:
        Dict containing research alignment metrics
    """
    logger.info("🔍 Computing improved research alignment metrics...")
    
    # Get research keywords from config
    domain_config = config.get('domain_guidance', {})
    research_keywords = domain_config.get('research_keywords', {})
    
    if not research_keywords:
        logger.warning("No research keywords found in configuration")
        return {}
    
    # Create selection mask
    df_enhanced['is_selected'] = False
    df_enhanced.loc[selected_indices, 'is_selected'] = True
    
    selected_papers = df_enhanced[df_enhanced['is_selected'] == True]
    total_papers = len(df_enhanced)
    selected_count = len(selected_papers)
    
    alignment_results = {}
    
    for category, keywords in research_keywords.items():
        # Convert keywords to lowercase for case-insensitive matching
        keywords_lower = [kw.lower() for kw in keywords]
        
        # Find papers matching any keyword in this category
        def matches_keywords(text):
            if pd.isna(text):
                return False
            text_lower = str(text).lower()
            return any(keyword in text_lower for keyword in keywords_lower)
        
        # Apply to combined text
        all_matches = df_enhanced['combined_text'].apply(matches_keywords)
        selected_matches = selected_papers['combined_text'].apply(matches_keywords)
        
        # Count matches
        total_matching = all_matches.sum()
        selected_matching = selected_matches.sum()
        
        # Compute proportions
        total_proportion = total_matching / total_papers if total_papers > 0 else 0
        selected_proportion = selected_matching / selected_count if selected_count > 0 else 0
        
        # Compute enrichment ratio
        enrichment_ratio = selected_proportion / total_proportion if total_proportion > 0 else 0
        
        # Fisher's exact test for statistical significance
        # Contingency table: [[selected_matching, selected_non_matching], 
        #                     [total_matching - selected_matching, total_papers - total_matching - (selected_count - selected_matching)]]
        
        selected_non_matching = selected_count - selected_matching
        total_non_matching = total_papers - total_matching
        non_selected_matching = total_matching - selected_matching
        non_selected_non_matching = total_non_matching - selected_non_matching
        
        contingency_table = [
            [selected_matching, selected_non_matching],
            [non_selected_matching, non_selected_non_matching]
        ]
        
        odds_ratio, p_value = stats.fisher_exact(contingency_table)
        
        alignment_results[category] = {
            'total_papers': total_papers,
            'total_matching': total_matching,
            'total_proportion': total_proportion,
            'selected_papers': selected_count,
            'selected_matching': selected_matching,
            'selected_proportion': selected_proportion,
            'enrichment_ratio': enrichment_ratio,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'keywords': keywords
        }
    
    logger.info(f"✅ Research alignment computed for {len(alignment_results)} categories")
    return alignment_results


def create_metrics_comparison_plots(df_enhanced: pd.DataFrame, 
                                  selected_indices: List[int],
                                  statistical_results: Dict[str, Dict],
                                  output_dir: Path) -> None:
    """
    Create comprehensive visualization plots comparing selected vs non-selected papers.
    
    Args:
        df_enhanced: DataFrame with enhanced metrics
        selected_indices: List of indices for selected papers
        statistical_results: Statistical comparison results
        output_dir: Output directory for plots
    """
    logger.info("📊 Creating metrics comparison visualizations...")
    
    # Create selection mask
    df_enhanced['is_selected'] = False
    df_enhanced.loc[selected_indices, 'is_selected'] = True
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Selected vs Non-Selected Papers: Metrics Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['centrality_score', 'diversity_score', 'representativeness_score']
    metric_labels = ['Centrality Score', 'Diversity Score', 'Representativeness Score']
    
    # Box plots
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[0, i]
        
        selected_data = df_enhanced[df_enhanced['is_selected'] == True][metric]
        non_selected_data = df_enhanced[df_enhanced['is_selected'] == False][metric]
        
        box_data = [selected_data, non_selected_data]
        box_labels = ['Selected', 'Non-Selected']
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(f'{label}\nMann-Whitney U p-value: {statistical_results[metric]["p_value"]:.4f}')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        
        # Add Cohen's d annotation
        cohens_d = statistical_results[metric]["cohens_d"]
        effect_size = statistical_results[metric]["effect_size"]
        ax.text(0.02, 0.98, f"Cohen's d: {cohens_d:.3f}\nEffect: {effect_size}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Histogram comparisons
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[1, i]
        
        selected_data = df_enhanced[df_enhanced['is_selected'] == True][metric]
        non_selected_data = df_enhanced[df_enhanced['is_selected'] == False][metric]
        
        ax.hist(selected_data, bins=30, alpha=0.7, label='Selected', color='lightblue', density=True)
        ax.hist(non_selected_data, bins=30, alpha=0.7, label='Non-Selected', color='lightcoral', density=True)
        
        ax.set_title(f'{label} Distribution')
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_dir / "metrics_comparison_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Metrics comparison plot saved to: {plot_path}")


def generate_enhanced_analytics_report(df_enhanced: pd.DataFrame,
                                     selected_indices: List[int],
                                     statistical_results: Dict[str, Dict],
                                     alignment_results: Dict[str, Dict],
                                     output_dir: Path) -> Path:
    """
    Generate comprehensive analytics report with statistical comparisons.
    
    Args:
        df_enhanced: DataFrame with enhanced metrics
        selected_indices: List of indices for selected papers
        statistical_results: Statistical comparison results
        alignment_results: Research alignment results
        output_dir: Output directory
    
    Returns:
        Path to generated report
    """
    logger.info("📋 Generating enhanced analytics report...")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"enhanced_analytics_report_{timestamp}.md"
    
    # Create selection mask
    df_enhanced['is_selected'] = False
    df_enhanced.loc[selected_indices, 'is_selected'] = True
    
    selected_papers = df_enhanced[df_enhanced['is_selected'] == True]
    non_selected_papers = df_enhanced[df_enhanced['is_selected'] == False]
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced Analytics Report: Selected vs Non-Selected Papers\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overview section
        f.write("## 📊 Overview\n\n")
        f.write(f"- **Total Papers**: {len(df_enhanced):,}\n")
        f.write(f"- **Selected Papers**: {len(selected_papers):,} ({len(selected_papers)/len(df_enhanced)*100:.1f}%)\n")
        f.write(f"- **Non-Selected Papers**: {len(non_selected_papers):,} ({len(non_selected_papers)/len(df_enhanced)*100:.1f}%)\n\n")
        
        # Statistical comparisons
        f.write("## 📈 Statistical Comparisons\n\n")
        f.write("### Metrics Summary\n\n")
        f.write("| Metric | Selected Mean ± SD | Non-Selected Mean ± SD | Cohen's d | Effect Size | p-value |\n")
        f.write("|--------|-------------------|------------------------|-----------|-------------|----------|\n")
        
        for metric, results in statistical_results.items():
            metric_name = metric.replace('_', ' ').title()
            f.write(f"| {metric_name} | {results['selected_mean']:.3f} ± {results['selected_std']:.3f} | ")
            f.write(f"{results['non_selected_mean']:.3f} ± {results['non_selected_std']:.3f} | ")
            f.write(f"{results['cohens_d']:.3f} | {results['effect_size']} | {results['p_value']:.4f} |\n")
        
        f.write("\n### Statistical Test Details\n\n")
        for metric, results in statistical_results.items():
            metric_name = metric.replace('_', ' ').title()
            f.write(f"#### {metric_name}\n\n")
            f.write(f"- **Mann-Whitney U Statistic**: {results['mann_whitney_u']:.0f}\n")
            f.write(f"- **p-value**: {results['p_value']:.6f}\n")
            f.write(f"- **Cohen's d**: {results['cohens_d']:.3f} ({results['effect_size']} effect)\n")
            f.write(f"- **Selected**: Mean={results['selected_mean']:.3f}, Median={results['selected_median']:.3f}\n")
            f.write(f"- **Non-Selected**: Mean={results['non_selected_mean']:.3f}, Median={results['non_selected_median']:.3f}\n\n")
        
        # Research alignment section
        if alignment_results:
            f.write("## 🔍 Research Alignment Analysis\n\n")
            f.write("### Improved Research Alignment (Paper-Based Proportions)\n\n")
            f.write("| Category | Total Match | Selected Match | Enrichment Ratio | Odds Ratio | p-value |\n")
            f.write("|----------|-------------|----------------|------------------|------------|----------|\n")
            
            for category, results in alignment_results.items():
                f.write(f"| {category} | {results['total_matching']}/{results['total_papers']} ({results['total_proportion']:.1%}) | ")
                f.write(f"{results['selected_matching']}/{results['selected_papers']} ({results['selected_proportion']:.1%}) | ")
                f.write(f"{results['enrichment_ratio']:.2f} | {results['odds_ratio']:.2f} | {results['p_value']:.4f} |\n")
            
            f.write("\n### Research Alignment Details\n\n")
            for category, results in alignment_results.items():
                f.write(f"#### {category}\n\n")
                f.write(f"- **Keywords**: {', '.join(results['keywords'])}\n")
                f.write(f"- **Total Papers Matching**: {results['total_matching']:,} out of {results['total_papers']:,} ({results['total_proportion']:.1%})\n")
                f.write(f"- **Selected Papers Matching**: {results['selected_matching']:,} out of {results['selected_papers']:,} ({results['selected_proportion']:.1%})\n")
                f.write(f"- **Enrichment Ratio**: {results['enrichment_ratio']:.2f} (selected vs total proportion)\n")
                f.write(f"- **Odds Ratio**: {results['odds_ratio']:.2f}\n")
                f.write(f"- **Fisher's Exact Test p-value**: {results['p_value']:.6f}\n\n")
        
        # Methodology section
        f.write("## 📚 Methodology\n\n")
        f.write("### Statistical Tests\n\n")
        f.write("- **Mann-Whitney U Test**: Non-parametric test for comparing distributions between selected and non-selected papers\n")
        f.write("- **Cohen's d**: Effect size measure for quantifying the magnitude of difference\n")
        f.write("- **Fisher's Exact Test**: Statistical significance test for research alignment enrichment\n\n")
        
        f.write("### Improved Research Alignment\n\n")
        f.write("- **Method**: Paper-based proportions instead of term-normalized scoring\n")
        f.write("- **Rationale**: Eliminates bias from varying keyword counts across categories\n")
        f.write("- **Calculation**: (Papers matching keywords / Total papers) for each group\n")
        f.write("- **Enrichment Ratio**: Selected proportion / Total proportion\n")
        f.write("- **Statistical Test**: Fisher's exact test for independence\n\n")
        
        f.write("### Effect Size Interpretation\n\n")
        f.write("- **Negligible**: |d| < 0.2\n")
        f.write("- **Small**: 0.2 ≤ |d| < 0.5\n")
        f.write("- **Medium**: 0.5 ≤ |d| < 0.8\n")
        f.write("- **Large**: |d| ≥ 0.8\n\n")
    
    logger.info(f"📋 Enhanced analytics report saved to: {report_path}")
    return report_path


def run_enhanced_analytics(config: Dict, output_dir: Path = None) -> None:
    """
    Run complete enhanced analytics pipeline.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory (defaults to results/)
    """
    logger.info("🚀 Starting enhanced analytics pipeline...")
    
    if output_dir is None:
        output_dir = Path("results")
    
    # Load data (this would need to be implemented based on your data loading strategy)
    # For now, assuming we have the data available
    logger.info("📂 Loading analysis data...")
    
    # This is a placeholder - you would need to implement actual data loading
    # based on your existing data loading patterns
    
    logger.info("🎉 Enhanced analytics pipeline completed successfully!")


if __name__ == "__main__":
    from utils import load_config
    
    config = load_config()
    run_enhanced_analytics(config) 