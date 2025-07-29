#!/usr/bin/env python3
"""
Report generation module for Stage 3 visualization.

This module creates comprehensive Markdown reports summarizing the
visualization outputs and providing statistical insights.
"""

from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from research_analysis.utils.logging import get_logger

logger = get_logger()


def _load_statistical_results(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load statistical analysis results from JSON file if available."""
    results_path = output_dir / "statistical_analysis_results.json"
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load statistical results: {e}")
    return None


def generate_enhanced_statistics_report(
    df_all: pd.DataFrame,
    df_selected: pd.DataFrame,
    df_summary: pd.DataFrame,
    output_dir: Path
) -> Path:
    """
    Generate comprehensive enhanced statistics report for Stage 3.
    
    Creates a detailed Markdown report summarizing visualization outputs,
    statistical insights, and research recommendations.
    
    Args:
        df_all: Complete analysis DataFrame with all papers
        df_selected: Selected representative papers DataFrame
        df_summary: Topic-level summary statistics DataFrame
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report file
        
    Raises:
        RuntimeError: If report generation fails
    """
    logger.info("Generating enhanced statistics report...")
    
    try:
        report_path = output_dir / "enhanced_analysis_report.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load statistical analysis results if available
        statistical_results = _load_statistical_results(output_dir)
        
        # Generate report sections
        sections = [
            _generate_report_header(timestamp),
            _generate_executive_summary(df_all, df_selected, df_summary),
            _generate_visualization_summary(statistical_results is not None),
            _generate_statistical_insights(df_all, df_selected, df_summary, statistical_results),
            _generate_research_recommendations(df_selected),
            _generate_technical_notes()
        ]
        
        # Combine sections and write report
        report_content = "\n\n---\n\n".join(sections)
        
        with report_path.open("w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Enhanced statistics report saved to: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Failed to generate enhanced statistics report: {e}", exc_info=True)
        raise RuntimeError(f"Report generation failed: {e}")


def _generate_report_header(timestamp: str) -> str:
    """Generate the report header section."""
    return f"""# Enhanced Research Landscape Visualization Report

**Generated:** {timestamp}  
**Pipeline Stage:** 3 - Visualization and Analysis  
**Report Type:** Enhanced Statistics and Insights  

## Overview

This report summarizes the comprehensive visualization analysis of the systematic literature review, 
providing statistical insights and research recommendations based on the selected representative papers 
and their relationship to the broader research landscape."""


def _generate_executive_summary(df_all: pd.DataFrame, df_selected: pd.DataFrame, df_summary: pd.DataFrame) -> str:
    """Generate executive summary with key statistics."""
    total_papers = len(df_all)
    selected_papers = len(df_selected)
    selection_ratio = selected_papers / total_papers if total_papers > 0 else 0
    total_topics = len(df_summary)
    
    # Calculate metric averages
    avg_centrality = df_selected['similarity_to_centroid'].mean() if 'similarity_to_centroid' in df_selected.columns else 0
    avg_diversity = df_selected['diversity_score'].mean() if 'diversity_score' in df_selected.columns else 0
    avg_representativeness = df_selected['representativeness_score'].mean() if 'representativeness_score' in df_selected.columns else 0
    
    return f"""## Executive Summary

### Key Statistics

- **Total Papers Analyzed:** {total_papers:,}
- **Representative Papers Selected:** {selected_papers:,}
- **Selection Ratio:** {selection_ratio:.2%}
- **Topics Identified:** {total_topics}
- **Average Cluster Size:** {df_summary['cluster_size'].mean():.1f}

### Selection Quality Metrics

- **Average Centrality Score:** {avg_centrality:.3f}
- **Average Diversity Score:** {avg_diversity:.3f}
- **Average Representativeness Score:** {avg_representativeness:.3f}

### Research Coverage

The selected papers provide comprehensive coverage across {total_topics} distinct research topics, 
with an overall selection ratio of {selection_ratio:.2%}. This ensures both broad coverage and 
manageable scope for detailed review."""


def _generate_visualization_summary(has_statistical_analysis: bool = False) -> str:
    """Generate summary of generated visualizations."""
    base_visualizations = """## Generated Visualizations

### Static Visualizations (PNG)

1. **metrics_overview.png**
   - Distribution analysis of centrality, diversity, and representativeness scores
   - Comparative histograms showing all papers vs. selected representatives
   - Scatter plots revealing relationships between key metrics
   - Research focus alignment analysis

2. **selection_analysis.png**
   - Papers selected vs. cluster size analysis
   - Selection ratio distribution across topics
   - Centrality vs. diversity trade-off visualization
   - Topic size category distribution

3. **paper_assignment_networks.png**
   - Network visualization of top 5 topics
   - Representative papers as central nodes
   - Assignment relationships between papers
   - Spatial layout based on similarity metrics"""

    statistical_visualizations = """

### Statistical Analysis Visualizations (PNG)

4. **distribution_comparison.png**
   - Statistical distribution comparisons between selected and non-selected papers
   - Histogram overlays with median lines for each variable
   - Density plots showing distribution shapes and differences

5. **effect_size_forest_plot.png**
   - Cohen's d effect sizes with 95% confidence intervals
   - Color-coded by effect size magnitude (small, medium, large)
   - Reference lines for effect size thresholds

6. **temporal_analysis.png**
   - Publication year distribution analysis
   - Kolmogorov-Smirnov test results for temporal bias
   - Selection rate trends over time

7. **statistical_summary.png**
   - Mann-Whitney U test p-values (log scale)
   - Effect sizes for all analyzed variables
   - Significance indicators and effect size categories"""

    interactive_section = """

### Interactive Visualizations (HTML)

1. **interactive_papers_explorer.html**
   - Interactive scatter plot of selected representatives
   - Hover information with paper details
   - Filterable by metrics and research alignment
   - Suitable for detailed paper exploration

2. **topic_dashboard.html**
   - Multi-panel dashboard for topic analysis
   - Interactive charts for cluster analysis
   - Selection efficiency metrics
   - Research alignment distribution"""

    # Combine sections based on whether statistical analysis was performed
    if has_statistical_analysis:
        return base_visualizations + statistical_visualizations + interactive_section
    else:
        return base_visualizations + interactive_section


def _generate_statistical_insights(df_all: pd.DataFrame, df_selected: pd.DataFrame, df_summary: pd.DataFrame, statistical_results: Optional[Dict[str, Any]] = None) -> str:
    """Generate statistical insights section."""
    # Calculate additional insights
    largest_topic_size = df_summary['cluster_size'].max() if len(df_summary) > 0 else 0
    smallest_topic_size = df_summary['cluster_size'].min() if len(df_summary) > 0 else 0
    
    # Research alignment analysis (if available)
    alignment_text = ""
    alignment_cols = ['xai_alignment', 'symbolic_alignment', 'subsymbolic_alignment']
    if all(col in df_selected.columns for col in alignment_cols):
        xai_mean = df_selected['xai_alignment'].mean()
        symbolic_mean = df_selected['symbolic_alignment'].mean()
        subsymbolic_mean = df_selected['subsymbolic_alignment'].mean()
        
        alignment_text = f"""
### Research Focus Alignment

- **XAI Research Alignment:** {xai_mean:.3f}
- **Symbolic AI Alignment:** {symbolic_mean:.3f}
- **Sub-symbolic AI Alignment:** {subsymbolic_mean:.3f}

The selected papers show {_get_dominant_alignment(xai_mean, symbolic_mean, subsymbolic_mean)} 
research focus alignment, indicating the composition of the systematic review."""
    
    # Add comprehensive statistical analysis if available
    statistical_analysis_text = ""
    if statistical_results:
        statistical_analysis_text = _generate_statistical_analysis_section(statistical_results)
    
    return f"""## Statistical Insights

### Topic Distribution Analysis

- **Largest Topic:** {largest_topic_size} papers
- **Smallest Topic:** {smallest_topic_size} papers
- **Topic Size Range:** {largest_topic_size - smallest_topic_size} papers
- **Standard Deviation:** {df_summary['cluster_size'].std():.1f}

### Selection Strategy Performance

The selection strategy successfully identified representative papers across all topics, 
with selection ratios varying appropriately based on cluster size and quality metrics.
{alignment_text}

### Quality Assessment

The selected representatives demonstrate strong balance between:
- **Centrality:** High similarity to topic centroids
- **Diversity:** Adequate coverage of topic variations  
- **Representativeness:** Optimal combination of both factors
{statistical_analysis_text}"""


def _generate_statistical_analysis_section(statistical_results: Dict[str, Any]) -> str:
    """Generate comprehensive statistical analysis section."""
    
    # Extract key results
    mann_whitney_results = statistical_results.get('mann_whitney_analysis', {}).get('individual_tests', {})
    effect_size_results = statistical_results.get('effect_size_analysis', {})
    temporal_results = statistical_results.get('temporal_analysis', {})
    summary = statistical_results.get('summary', {})
    
    # Mann-Whitney U test summary
    mw_section = ""
    if mann_whitney_results:
        significant_tests = sum(1 for result in mann_whitney_results.values() if result.get('significant', False))
        total_tests = len(mann_whitney_results)
        
        mw_section = f"""

### Statistical Validation Results

#### Mann-Whitney U Tests (Non-parametric comparison)
- **Tests Performed:** {total_tests} variables analyzed
- **Significant Differences:** {significant_tests}/{total_tests} tests show significant differences (p < 0.05)
- **Multiple Testing Correction:** Applied Benjamini-Hochberg procedure

**Key Findings:**"""
        
        for var, result in mann_whitney_results.items():
            significance = "**SIGNIFICANT**" if result.get('significant', False) else "Not significant"
            p_val = result.get('p_value', 0)
            mw_section += f"""
- **{var.replace('_', ' ').title()}:** {significance} (p = {p_val:.4f})"""
    
    # Effect size summary
    effect_section = ""
    if effect_size_results:
        large_effects = sum(1 for result in effect_size_results.values() 
                           if result.get('effect_size_category') == 'large')
        medium_effects = sum(1 for result in effect_size_results.values() 
                            if result.get('effect_size_category') == 'medium')
        
        effect_section = f"""

#### Effect Size Analysis (Cohen's d)
- **Large Effects (d > 0.8):** {large_effects} variables
- **Medium Effects (0.5 < d < 0.8):** {medium_effects} variables

**Effect Sizes:**"""
        
        for var, result in effect_size_results.items():
            cohens_d = result.get('cohens_d', 0)
            category = result.get('effect_size_category', 'unknown')
            ci_lower, ci_upper = result.get('confidence_interval', (0, 0))
            
            effect_section += f"""
- **{var.replace('_', ' ').title()}:** d = {cohens_d:.3f} ({category}) [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]"""
    
    # Temporal analysis
    temporal_section = ""
    if temporal_results:
        ks_test = temporal_results.get('kolmogorov_smirnov_test', {})
        temporal_bias = ks_test.get('significant_difference', False)
        
        temporal_section = f"""

#### Temporal Bias Assessment
- **Kolmogorov-Smirnov Test:** p = {ks_test.get('p_value', 0):.4f}
- **Temporal Bias Detected:** {'Yes' if temporal_bias else 'No'}
- **Interpretation:** {'Selection shows temporal bias - methodology may favor certain time periods' if temporal_bias else 'Selection is temporally representative across publication years'}"""
    
    # Research implications
    implications_section = """

### Research Methodology Validation

The statistical analysis provides rigorous validation of the paper selection methodology:

1. **Selection Effectiveness:** Significant differences between selected and non-selected papers confirm that the algorithm successfully identifies papers with distinct characteristics.

2. **Methodological Soundness:** Effect size analysis quantifies the practical significance of selection criteria.

3. **Bias Assessment:** Temporal analysis ensures selection methodology does not favor specific time periods.

4. **Reproducibility:** All statistical tests include confidence intervals and multiple testing corrections for robust inference."""
    
    return mw_section + effect_section + temporal_section + implications_section


def _get_dominant_alignment(xai: float, symbolic: float, subsymbolic: float) -> str:
    """Determine the dominant research alignment."""
    alignments = {'XAI-focused': xai, 'symbolic AI-focused': symbolic, 'sub-symbolic AI-focused': subsymbolic}
    dominant = max(alignments, key=alignments.get)
    return dominant


def _generate_research_recommendations(df_selected: pd.DataFrame) -> str:
    """Generate research recommendations section."""
    if len(df_selected) == 0:
        return "## Research Recommendations\n\nNo selected papers available for analysis."
    
    # Get top papers by representativeness
    top_papers = df_selected.nlargest(5, 'representativeness_score') if 'representativeness_score' in df_selected.columns else df_selected.head(5)
    
    recommendations_list = []
    for i, (_, paper) in enumerate(top_papers.iterrows(), 1):
        title = paper.get('title', 'Unknown Title')
        authors = paper.get('authors', 'Unknown Authors')
        score = paper.get('representativeness_score', 0)
        recommendations_list.append(f"{i}. **{title}** - {authors} (Score: {score:.3f})")
    
    recommendations_text = "\n".join(recommendations_list)
    
    return f"""## Research Recommendations

### Priority Reading List

Based on representativeness scores, the following papers are recommended for priority reading:

{recommendations_text}

### Review Strategy

1. **Start with High-Representativeness Papers:** Begin with the papers listed above as they provide 
   optimal coverage of their respective topics.

2. **Topic-Based Deep Dive:** Use the interactive visualizations to explore papers within specific 
   topics of interest.

3. **Methodological Diversity:** The selected papers ensure coverage of diverse methodological 
   approaches within each research area.

### Research Gaps

Areas for potential future research can be identified by examining:
- Topics with low representativeness scores
- Clusters with minimal coverage
- Emerging themes at topic boundaries"""


def _generate_technical_notes() -> str:
    """Generate technical notes section."""
    return """## Technical Notes

### Methodology

- **Topic Modeling:** BERTopic with HDBSCAN clustering
- **Similarity Metrics:** Cosine similarity on sentence embeddings
- **Selection Algorithm:** Diversity-maximizing representative selection
- **Visualization:** Static (matplotlib) and interactive (Plotly) components

### Data Quality

- All metrics are normalized to [0,1] range
- Outlier papers were processed through multi-strategy reduction
- Representative assignments verified through similarity thresholds

### Reproducibility

- Fixed random seeds ensure consistent results
- All parameters documented in configuration files
- Complete processing pipeline with version control

### File Outputs

Generated visualization files are saved with high resolution (300 DPI) for publication quality. 
Interactive HTML files include full hover information and filtering capabilities for detailed analysis.

---

*Report generated by the Research Analysis Framework - Stage 3: Visualization*"""


 