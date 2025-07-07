#!/usr/bin/env python3
"""
Report Generation Module for Enhanced Research Visualization Pipeline

This module provides functions to generate comprehensive text-based reports
summarizing analysis results, metrics, and insights from the systematic review.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger


def generate_enhanced_statistics_report(df_all: pd.DataFrame, df_selected: pd.DataFrame, 
                                       df_summary: pd.DataFrame, output_dir: Path) -> str:
    """
    Generate comprehensive markdown report with enhanced statistics and analysis insights.
    
    This function creates a detailed markdown report summarizing the systematic
    literature review analysis, including executive summary, metrics analysis,
    top papers, and quality assessments.
    
    Args:
        df_all: Complete analysis DataFrame containing all papers
        df_selected: Selected representative papers DataFrame  
        df_summary: Topic-level summary statistics DataFrame
        output_dir: Output directory path for saving report
    
    Returns:
        str: The generated report content as markdown text
    
    Raises:
        ValueError: If required columns are missing from input DataFrames
        RuntimeError: If report generation fails
    
    Generated report sections:
        1. Executive Summary with dataset overview
        2. Metrics Overview comparing all vs selected papers
        3. Selection Strategy Performance analysis
        4. Quality Metrics Analysis with improvements
        5. Top Representative Papers listing
        6. Paper Assignment Coverage statistics
    
    Example:
        >>> report = generate_enhanced_statistics_report(df_all, df_selected, df_summary, Path("results"))
        # Creates enhanced_analysis_report.md
    
    Note:
        - Professional markdown formatting suitable for documentation
        - Statistical comparisons with improvement percentages
        - Top papers ranked by representativeness scores
        - Assignment coverage and similarity analysis
        - Comprehensive error handling and graceful degradation
    """
    logger.info("📋 Generating comprehensive enhanced statistics report...")
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start building the report
        report_sections = [
            _generate_header(timestamp),
            _generate_executive_summary(df_all, df_selected, df_summary),
            _generate_metrics_overview(df_all, df_selected),
            _generate_selection_performance(df_summary),
            _generate_quality_analysis(df_all, df_selected),
            _generate_top_papers(df_selected),
            _generate_assignment_analysis(df_all)
        ]
        
        # Combine all sections
        report = "\n\n".join(report_sections)
        
        # Save report to file
        report_path = output_dir / 'enhanced_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📋 Enhanced analysis report saved to: {report_path}")
        return report
        
    except Exception as e:
        logger.error(f"❌ Error generating enhanced statistics report: {e}")
        raise RuntimeError(f"Failed to generate enhanced statistics report: {e}")


def _generate_header(timestamp: str) -> str:
    """Generate the report header section."""
    return f"""# Enhanced Systematic Literature Review Report

**Generated on:** {timestamp}

---"""


def _generate_executive_summary(df_all: pd.DataFrame, df_selected: pd.DataFrame, 
                               df_summary: pd.DataFrame) -> str:
    """Generate the executive summary section."""
    return f"""## 📊 Executive Summary

### Dataset Overview
- **Total Papers Analyzed:** {len(df_all):,}
- **Representative Papers Selected:** {len(df_selected):,}
- **Selection Ratio:** {len(df_selected)/len(df_all)*100:.2f}%
- **Topics Covered:** {len(df_summary)}
- **Papers with Topic Assignment:** {len(df_all[df_all['topic_id'] != -1]):,}

### Metrics Overview (Selected Papers)
- **Average Centrality:** {df_selected['similarity_to_centroid'].mean():.4f} ± {df_selected['similarity_to_centroid'].std():.4f}
- **Average Diversity:** {df_selected['diversity_score'].mean():.4f} ± {df_selected['diversity_score'].std():.4f}
- **Average Representativeness:** {df_selected['representativeness_score'].mean():.4f} ± {df_selected['representativeness_score'].std():.4f}"""


def _generate_metrics_overview(df_all: pd.DataFrame, df_selected: pd.DataFrame) -> str:
    """Generate the metrics comparison section."""
    # Check if alignment columns exist
    alignment_cols = ['xai_alignment', 'symbolic_alignment', 'subsymbolic_alignment']
    has_alignment = all(col in df_selected.columns for col in alignment_cols)
    
    alignment_section = ""
    if has_alignment:
        alignment_section = f"""
### Research Alignment (Selected Papers)
- **XAI Alignment:** {df_selected['xai_alignment'].mean():.4f} ({df_selected['xai_alignment'].sum():.0f} total matches)
- **Symbolic Alignment:** {df_selected['symbolic_alignment'].mean():.4f} ({df_selected['symbolic_alignment'].sum():.0f} total matches)
- **Sub-symbolic Alignment:** {df_selected['subsymbolic_alignment'].mean():.4f} ({df_selected['subsymbolic_alignment'].sum():.0f} total matches)"""
    
    return f"""## 🎯 Metrics Analysis{alignment_section}

### Distribution Statistics Comparison
| Metric | All Papers | Selected Papers | Improvement |
|--------|------------|----------------|-------------|
| Centrality | {df_all['similarity_to_centroid'].mean():.4f} ± {df_all['similarity_to_centroid'].std():.4f} | {df_selected['similarity_to_centroid'].mean():.4f} ± {df_selected['similarity_to_centroid'].std():.4f} | {((df_selected['similarity_to_centroid'].mean() / df_all['similarity_to_centroid'].mean()) - 1) * 100:+.1f}% |
| Diversity | {df_all['diversity_score'].mean():.4f} ± {df_all['diversity_score'].std():.4f} | {df_selected['diversity_score'].mean():.4f} ± {df_selected['diversity_score'].std():.4f} | {((df_selected['diversity_score'].mean() / df_all['diversity_score'].mean()) - 1) * 100:+.1f}% |
| Representativeness | {df_all['representativeness_score'].mean():.4f} ± {df_all['representativeness_score'].std():.4f} | {df_selected['representativeness_score'].mean():.4f} ± {df_selected['representativeness_score'].std():.4f} | {((df_selected['representativeness_score'].mean() / df_all['representativeness_score'].mean()) - 1) * 100:+.1f}% |"""


def _generate_selection_performance(df_summary: pd.DataFrame) -> str:
    """Generate the selection strategy performance section."""
    # Fix negative diversity scores for calculations
    df_summary_fixed = df_summary.copy()
    df_summary_fixed['avg_diversity'] = np.maximum(df_summary_fixed['avg_diversity'], 0)
    
    # Analyze cluster size categories if available
    size_analysis = ""
    if 'size_category' in df_summary.columns:
        size_analysis = "\n### Dynamic Selection by Cluster Size\n"
        for category in ['small', 'medium', 'large', 'xlarge', 'xxlarge']:
            cat_topics = df_summary_fixed[df_summary_fixed['size_category'] == category]
            if len(cat_topics) > 0:
                size_analysis += f"- **{category.title()} clusters** ({len(cat_topics)} topics): "
                size_analysis += f"avg {cat_topics['papers_selected'].mean():.1f} papers selected, "
                size_analysis += f"{cat_topics['selection_ratio'].mean()*100:.1f}% selection ratio\n"
    
    # Top efficient selections
    efficient_analysis = "\n### Top 10 Most Efficient Selections (High Centrality)\n"
    top_efficient = df_summary_fixed.nlargest(10, 'avg_centrality')[
        ['topic_id', 'cluster_size', 'papers_selected', 'avg_centrality', 'avg_diversity']
    ]
    
    for i, (_, row) in enumerate(top_efficient.iterrows(), 1):
        topic_name = row.get('topic_name', f"Topic {row['topic_id']}")
        efficient_analysis += f"{i}. **Topic {row['topic_id']}** ({row['cluster_size']} docs → {row['papers_selected']} selected): "
        efficient_analysis += f"Centrality={row['avg_centrality']:.3f}, Diversity={row['avg_diversity']:.3f}\n"
    
    return f"""## 🎖️ Selection Strategy Performance{size_analysis}{efficient_analysis}"""


def _generate_quality_analysis(df_all: pd.DataFrame, df_selected: pd.DataFrame) -> str:
    """Generate the quality metrics analysis section."""
    # Assignment analysis
    assigned_papers = df_all[~df_all['is_selected_representative']]
    
    assignment_section = ""
    if 'representative_title' in df_all.columns:
        assigned_with_rep = assigned_papers[assigned_papers['representative_title'].notna()]
        
        if len(assigned_with_rep) > 0:
            if 'similarity_to_representative' in assigned_with_rep.columns:
                high_similarity = assigned_with_rep[assigned_with_rep['similarity_to_representative'] >= 0.8]
                avg_similarity = assigned_with_rep['similarity_to_representative'].mean()
                
                assignment_section = f"""
### Paper Assignment Coverage
- **Papers assigned to representatives:** {len(assigned_with_rep):,} ({len(assigned_with_rep)/len(assigned_papers)*100:.1f}% of non-selected)
- **High similarity assignments (≥0.8):** {len(high_similarity):,} ({len(high_similarity)/len(assigned_with_rep)*100:.1f}% of assignments)
- **Average assignment similarity:** {avg_similarity:.4f}"""
            else:
                assignment_section = f"""
### Paper Assignment Coverage
- **Papers assigned to representatives:** {len(assigned_with_rep):,} ({len(assigned_with_rep)/len(assigned_papers)*100:.1f}% of non-selected)
- **Assignment similarity data:** Not available"""
    
    return f"""## 📈 Quality Metrics Analysis{assignment_section}"""


def _generate_top_papers(df_selected: pd.DataFrame) -> str:
    """Generate the top representative papers section."""
    # Get top 10 papers by representativeness
    top_papers = df_selected.nlargest(10, 'representativeness_score')
    
    papers_list = ""
    for i, (_, paper) in enumerate(top_papers.iterrows(), 1):
        title = paper.get('title', 'Title not available')
        authors = paper.get('authors', 'Authors not available')
        topic_id = paper.get('topic_id', 'Unknown')
        
        # Truncate long titles for readability
        if len(title) > 100:
            title = title[:97] + "..."
        
        papers_list += f"""{i}. **{title}** (Topic {topic_id})
   - Authors: {authors}
   - Representativeness: {paper['representativeness_score']:.4f}
   - Centrality: {paper['similarity_to_centroid']:.4f}, Diversity: {paper['diversity_score']:.4f}

"""
    
    return f"""## 🏆 Top Representative Papers (Highest Representativeness)

{papers_list}"""


def _generate_assignment_analysis(df_all: pd.DataFrame) -> str:
    """Generate the paper assignment analysis section."""
    if 'representative_title' not in df_all.columns:
        return """## 📊 Paper Assignment Analysis

*Assignment data not available in the current dataset.*"""
    
    # Analyze assignments
    non_selected = df_all[~df_all['is_selected_representative']]
    assigned = non_selected[non_selected['representative_title'].notna()]
    
    if len(assigned) == 0:
        return """## 📊 Paper Assignment Analysis

*No paper assignments found in the current dataset.*"""
    
    # Calculate assignment statistics
    assignment_rate = len(assigned) / len(non_selected) * 100 if len(non_selected) > 0 else 0
    
    similarity_stats = ""
    if 'similarity_to_representative' in assigned.columns:
        avg_sim = assigned['similarity_to_representative'].mean()
        min_sim = assigned['similarity_to_representative'].min()
        max_sim = assigned['similarity_to_representative'].max()
        
        similarity_stats = f"""
### Assignment Quality Metrics
- **Average assignment similarity:** {avg_sim:.4f}
- **Minimum assignment similarity:** {min_sim:.4f}
- **Maximum assignment similarity:** {max_sim:.4f}
- **High-quality assignments (≥0.8):** {len(assigned[assigned['similarity_to_representative'] >= 0.8]):,} papers"""
    
    return f"""## 📊 Paper Assignment Analysis

### Assignment Coverage
- **Total non-selected papers:** {len(non_selected):,}
- **Papers with assignments:** {len(assigned):,}
- **Assignment coverage:** {assignment_rate:.1f}%{similarity_stats}"""


def validate_report_data(df_all: pd.DataFrame, df_selected: pd.DataFrame, 
                        df_summary: pd.DataFrame) -> bool:
    """
    Validate that DataFrames contain required columns for report generation.
    
    Args:
        df_all: Complete analysis DataFrame
        df_selected: Selected representatives DataFrame
        df_summary: Topic summary DataFrame
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    # Required columns for basic report
    required_all = ['similarity_to_centroid', 'diversity_score', 'representativeness_score', 'is_selected_representative']
    required_selected = ['similarity_to_centroid', 'diversity_score', 'representativeness_score']
    required_summary = ['topic_id', 'cluster_size', 'papers_selected', 'avg_centrality']
    
    # Check all dataframes
    missing_all = [col for col in required_all if col not in df_all.columns]
    missing_selected = [col for col in required_selected if col not in df_selected.columns]
    missing_summary = [col for col in required_summary if col not in df_summary.columns]
    
    if missing_all or missing_selected or missing_summary:
        logger.error(f"Missing required columns for report: all={missing_all}, selected={missing_selected}, summary={missing_summary}")
        return False
    
    # Check data consistency
    if len(df_selected) == 0:
        logger.error("No selected papers found for report generation")
        return False
    
    if len(df_summary) == 0:
        logger.error("No topic summary data found for report generation")
        return False
    
    logger.info("✅ Report data validation passed")
    return True 