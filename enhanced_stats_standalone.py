#!/usr/bin/env python3
"""
Enhanced Research Statistics - Standalone Version
Runs advanced statistical analysis on selected vs non-selected papers
"""

from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Import shared utilities
from utils import setup_logging, load_config, get_systematic_review_config


def load_data_directly(config: Dict) -> tuple:
    """Load data directly from CSV files."""
    logger.info("📂 Loading data directly from CSV files...")
    
    # Find most recent comprehensive analysis file
    results_dir = Path("results")
    comprehensive_files = sorted(results_dir.glob("comprehensive_analysis_*.csv"))
    if not comprehensive_files:
        raise FileNotFoundError("No comprehensive analysis files found")
    
    # Load comprehensive analysis
    df_all = pd.read_csv(comprehensive_files[-1])
    logger.info(f"📊 Loaded {len(df_all)} papers from comprehensive analysis")
    
    # Find most recent selected representatives file  
    selected_files = sorted(results_dir.glob("selected_representatives_*.csv"))
    if not selected_files:
        raise FileNotFoundError("No selected representatives files found")
    
    df_selected = pd.read_csv(selected_files[-1])
    logger.info(f"📊 Loaded {len(df_selected)} selected representatives")
    
    return df_all, df_selected


def compute_statistical_comparisons(df_all: pd.DataFrame, df_selected: pd.DataFrame) -> Dict[str, Dict]:
    """Compute statistical comparisons between selected and non-selected papers."""
    logger.info("📈 Computing statistical comparisons...")
    
    # Create selection mask based on global_index
    selected_indices = set(df_selected['global_index'].tolist())
    df_all['is_selected'] = df_all.index.isin(selected_indices)
    
    selected_papers = df_all[df_all['is_selected'] == True]
    non_selected_papers = df_all[df_all['is_selected'] == False]
    
    logger.info(f"📊 Selected: {len(selected_papers)}, Non-selected: {len(non_selected_papers)}")
    
    # Metrics to compare
    metrics = {
        'centrality_score': 'similarity_to_centroid',
        'diversity_score': 'diversity_score', 
        'representativeness_score': 'representativeness_score'
    }
    
    results = {}
    
    for metric_name, column_name in metrics.items():
        if column_name not in df_all.columns:
            logger.warning(f"Column {column_name} not found, skipping {metric_name}")
            continue
            
        selected_values = selected_papers[column_name].dropna().values
        non_selected_values = non_selected_papers[column_name].dropna().values
        
        if len(selected_values) == 0 or len(non_selected_values) == 0:
            logger.warning(f"No valid values for {metric_name}")
            continue
        
        # Mann-Whitney U test
        try:
            statistic, p_value = stats.mannwhitneyu(
                selected_values, non_selected_values, 
                alternative='two-sided'
            )
        except Exception as e:
            logger.warning(f"Mann-Whitney U test failed for {metric_name}: {e}")
            continue
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(selected_values) - 1) * np.var(selected_values, ddof=1) + 
             (len(non_selected_values) - 1) * np.var(non_selected_values, ddof=1)) / 
            (len(selected_values) + len(non_selected_values) - 2)
        )
        
        if pooled_std == 0:
            cohens_d = 0
        else:
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
        
        results[metric_name] = {
            'mann_whitney_u': statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': effect_size,
            'selected_mean': np.mean(selected_values),
            'selected_std': np.std(selected_values),
            'non_selected_mean': np.mean(non_selected_values),
            'non_selected_std': np.std(non_selected_values),
            'selected_median': np.median(selected_values),
            'non_selected_median': np.median(non_selected_values),
            'selected_n': len(selected_values),
            'non_selected_n': len(non_selected_values)
        }
    
    logger.info(f"✅ Statistical comparisons computed for {len(results)} metrics")
    return results


def compute_improved_research_alignment(df_all: pd.DataFrame, df_selected: pd.DataFrame, 
                                      config: Dict) -> Dict[str, Dict]:
    """Compute improved research alignment metrics."""
    logger.info("🔍 Computing improved research alignment...")
    
    # Get research keywords from config - using systematic_review instead of domain_guidance
    systematic_config = get_systematic_review_config(config)
    research_keywords = systematic_config.get('research_keywords', {})
    
    if not research_keywords:
        logger.warning("No research keywords found in systematic_review configuration")
        return {}
    
    # Create selection mask
    selected_indices = set(df_selected['global_index'].tolist())
    df_all['is_selected'] = df_all.index.isin(selected_indices)
    
    selected_papers = df_all[df_all['is_selected'] == True]
    total_papers = len(df_all)
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
        
        # Use combined_text if available, otherwise try title + abstract
        if 'combined_text' in df_all.columns:
            text_column = 'combined_text'
        elif 'title' in df_all.columns and 'abstract' in df_all.columns:
            # Create temp_text column for both dataframes
            df_all['temp_text'] = df_all['title'].fillna('') + ' ' + df_all['abstract'].fillna('')
            selected_papers = df_all[df_all['is_selected'] == True]  # Update after column creation
            text_column = 'temp_text'
        elif 'title' in df_all.columns:
            text_column = 'title'
        else:
            logger.warning(f"No suitable text column found for keyword matching in category {category}")
            continue
        
        # Apply matching
        all_matches = df_all[text_column].apply(matches_keywords)
        selected_matches = selected_papers[text_column].apply(matches_keywords)
        
        # Count matches
        total_matching = all_matches.sum()
        selected_matching = selected_matches.sum()
        
        # Compute proportions
        total_proportion = total_matching / total_papers if total_papers > 0 else 0
        selected_proportion = selected_matching / selected_count if selected_count > 0 else 0
        
        # Compute enrichment ratio
        enrichment_ratio = selected_proportion / total_proportion if total_proportion > 0 else 0
        
        # Fisher's exact test
        selected_non_matching = selected_count - selected_matching
        non_selected_matching = total_matching - selected_matching
        non_selected_non_matching = (total_papers - total_matching) - selected_non_matching
        
        # Ensure non-negative values
        non_selected_non_matching = max(0, non_selected_non_matching)
        
        contingency_table = [
            [selected_matching, selected_non_matching],
            [non_selected_matching, non_selected_non_matching]
        ]
        
        try:
            odds_ratio, p_value = stats.fisher_exact(contingency_table)
        except Exception as e:
            logger.warning(f"Fisher's exact test failed for {category}: {e}")
            odds_ratio, p_value = 0, 1.0
        
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


def create_metrics_plots(df_all: pd.DataFrame, statistical_results: Dict, output_dir: Path) -> None:
    """Create visualization plots."""
    logger.info("📊 Creating metrics comparison plots...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig, axes = plt.subplots(2, len(statistical_results), figsize=(6*len(statistical_results), 10))
    if len(statistical_results) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Selected vs Non-Selected Papers: Metrics Comparison', fontsize=16, fontweight='bold')
    
    metric_columns = {
        'centrality_score': 'similarity_to_centroid',
        'diversity_score': 'diversity_score', 
        'representativeness_score': 'representativeness_score'
    }
    
    for i, (metric_name, results) in enumerate(statistical_results.items()):
        column_name = metric_columns.get(metric_name, metric_name)
        
        if column_name not in df_all.columns:
            continue
            
        # Box plot
        ax1 = axes[0, i]
        selected_data = df_all[df_all['is_selected'] == True][column_name].dropna()
        non_selected_data = df_all[df_all['is_selected'] == False][column_name].dropna()
        
        box_data = [selected_data, non_selected_data]
        box_labels = ['Selected', 'Non-Selected']
        
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        metric_label = metric_name.replace('_', ' ').title()
        ax1.set_title(f'{metric_label}\np-value: {results["p_value"]:.4f}')
        ax1.set_ylabel(metric_label)
        ax1.grid(True, alpha=0.3)
        
        # Add Cohen's d annotation
        cohens_d = results["cohens_d"]
        effect_size = results["effect_size"]
        ax1.text(0.02, 0.98, f"Cohen's d: {cohens_d:.3f}\nEffect: {effect_size}", 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Histogram
        ax2 = axes[1, i]
        ax2.hist(selected_data, bins=30, alpha=0.7, label='Selected', color='lightblue', density=True)
        ax2.hist(non_selected_data, bins=30, alpha=0.7, label='Non-Selected', color='lightcoral', density=True)
        ax2.set_title(f'{metric_label} Distribution')
        ax2.set_xlabel(metric_label)
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "enhanced_metrics_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Metrics comparison plot saved to: {plot_path}")


def generate_report(df_all: pd.DataFrame, statistical_results: Dict, 
                   alignment_results: Dict, output_dir: Path) -> Path:
    """Generate comprehensive report."""
    logger.info("📋 Generating enhanced analytics report...")
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"enhanced_analytics_report_{timestamp}.md"
    
    selected_count = len(df_all[df_all['is_selected'] == True])
    non_selected_count = len(df_all[df_all['is_selected'] == False])
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced Analytics Report: Selected vs Non-Selected Papers\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overview
        f.write("## 📊 Overview\n\n")
        f.write(f"- **Total Papers**: {len(df_all):,}\n")
        f.write(f"- **Selected Papers**: {selected_count:,} ({selected_count/len(df_all)*100:.1f}%)\n")
        f.write(f"- **Non-Selected Papers**: {non_selected_count:,} ({non_selected_count/len(df_all)*100:.1f}%)\n\n")
        
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
        
        # Research alignment
        if alignment_results:
            f.write("\n## 🔍 Research Alignment Analysis\n\n")
            f.write("### Improved Research Alignment (Paper-Based Proportions)\n\n")
            f.write("| Category | Total Match | Selected Match | Enrichment Ratio | Odds Ratio | p-value |\n")
            f.write("|----------|-------------|----------------|------------------|------------|----------|\n")
            
            for category, results in alignment_results.items():
                f.write(f"| {category} | {results['total_matching']}/{results['total_papers']} ({results['total_proportion']:.1%}) | ")
                f.write(f"{results['selected_matching']}/{results['selected_papers']} ({results['selected_proportion']:.1%}) | ")
                f.write(f"{results['enrichment_ratio']:.2f} | {results['odds_ratio']:.2f} | {results['p_value']:.4f} |\n")
        
        # Methodology
        f.write("\n## 📚 Methodology\n\n")
        f.write("### Improvements Made\n\n")
        f.write("1. **Paper-Based Research Alignment**: Uses proportion of papers matching keywords instead of term-normalized scoring\n")
        f.write("2. **Statistical Significance**: Mann-Whitney U test for non-parametric comparison\n")
        f.write("3. **Effect Size**: Cohen's d for quantifying practical significance\n")
        f.write("4. **Enrichment Analysis**: Fisher's exact test for keyword category enrichment\n\n")
    
    logger.info(f"📋 Report saved to: {report_path}")
    return report_path


def main():
    """Main function."""
    # Setup
    setup_logging()
    logger.info("Starting enhanced research statistics analysis (standalone)")
    
    try:
        # Load configuration
        config = load_config()
        output_dir = Path("results")
        
        # Load data
        df_all, df_selected = load_data_directly(config)
        
        # Compute statistical comparisons
        statistical_results = compute_statistical_comparisons(df_all, df_selected)
        
        # Compute research alignment
        alignment_results = compute_improved_research_alignment(df_all, df_selected, config)
        
        # Create plots
        create_metrics_plots(df_all, statistical_results, output_dir)
        
        # Generate report
        report_path = generate_report(df_all, statistical_results, alignment_results, output_dir)
        
        logger.info("🎉 Enhanced statistics analysis completed!")
        logger.info(f"📋 Report saved to: {report_path}")
        
        # Print key findings
        print(f"\n🎉 Enhanced research statistics analysis completed!")
        print(f"📊 {len(df_selected)} selected papers analyzed from {len(df_all)} total papers")
        print(f"📋 Report saved to: {report_path}")
        
        print("\n📈 Key Statistical Findings:")
        for metric, results in statistical_results.items():
            metric_name = metric.replace('_', ' ').title()
            print(f"  • {metric_name}: Cohen's d = {results['cohens_d']:.3f} ({results['effect_size']} effect)")
        
        if alignment_results:
            print("\n🔍 Research Alignment Results:")
            for category, results in alignment_results.items():
                print(f"  • {category}: {results['enrichment_ratio']:.2f}x enrichment (p={results['p_value']:.4f})")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 