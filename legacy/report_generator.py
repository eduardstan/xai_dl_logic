#!/usr/bin/env python3
"""
Report Generation Module for Systematic Literature Reviews

This module handles the generation of comprehensive analysis reports including
executive summaries, topic analyses, reading recommendations, and statistical insights.
"""

import pandas as pd
from pathlib import Path
from typing import Dict
from datetime import datetime
from loguru import logger

from utils import get_output_config, get_systematic_review_config


def generate_analysis_report(results_df: pd.DataFrame, summary_df: pd.DataFrame, 
                           selected_df: pd.DataFrame, config: Dict) -> Path:
    """
    Generate a comprehensive analysis report in Markdown format.
    
    This function creates a detailed systematic literature review report that includes:
    - Executive summary with key statistics
    - Selection configuration and methodology
    - Topic-by-topic analysis with cluster information
    - Research alignment analysis for domain relevance
    - Reading recommendations prioritized by quality metrics
    - Statistical insights and distribution analysis
    
    Args:
        results_df: Complete analysis results for all papers
        summary_df: Topic-level summary statistics
        selected_df: Selected representative papers only
        config: Configuration dictionary containing analysis parameters
    
    Returns:
        Path: Path to the generated report file
    
    Raises:
        ValueError: If input DataFrames are empty or invalid
        IOError: If report file cannot be written
        Exception: If report generation fails
    
    Example:
        >>> config = load_config('config.yaml')
        >>> results_df, summary_df, selected_df = process_topics(config)
        >>> report_path = generate_analysis_report(results_df, summary_df, selected_df, config)
        >>> print(f"Report saved to: {report_path}")
        Report saved to: results/analysis_report_20250705_185046.md
    
    Report Sections:
        1. **Executive Summary**: Total papers, selection ratios, topic coverage
        2. **Selection Configuration**: Parameters and methodology used
        3. **Topic Summary Table**: Cluster sizes, categories, selection ratios
        4. **Research Alignment Analysis**: Domain-specific keyword alignment
        5. **Reading Recommendations**: Prioritized by quality metrics
        6. **Statistical Insights**: Distribution analysis and trends
    
    Output Format:
        - Markdown format for easy reading and conversion
        - Tables for structured data presentation
        - Hierarchical organization with clear headings
        - Embedded statistics and visualizations
        - Professional academic report style
    
    File Naming:
        - Format: `analysis_report_{YYYYMMDD_HHMMSS}.md`
        - Includes timestamp for version tracking
        - Saved to configured results directory
    
    Quality Metrics Used:
        - Representativeness score (centrality + diversity balance)
        - Research alignment scores (domain relevance)
        - Topic diversity and cluster coherence
        - Selection strategy effectiveness
    
    Note:
        - Report adapts to available data columns
        - Handles missing alignment data gracefully
        - Provides actionable reading recommendations
        - Includes both summary and detailed statistics
    """
    try:
        output_config = get_output_config(config)
        review_config = get_systematic_review_config(config)
        results_dir = Path(output_config['results_dir'])
        
        # Validate input data
        if len(results_df) == 0:
            raise ValueError("Results DataFrame is empty")
        if len(summary_df) == 0:
            raise ValueError("Summary DataFrame is empty")
        if len(selected_df) == 0:
            raise ValueError("Selected DataFrame is empty")
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = results_dir / f"analysis_report_{timestamp}.md"
        
        total_papers = len(results_df)
        total_selected = len(selected_df)
        total_topics = len(summary_df)
        
        logger.info(f"📋 Generating comprehensive analysis report...")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Report Header
            f.write(f"# Advanced Systematic Literature Review Analysis Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")
            
            # Executive Summary
            f.write("## 📊 Executive Summary\n\n")
            f.write(f"- **Total Papers Analyzed:** {total_papers:,}\n")
            f.write(f"- **Representative Papers Selected:** {total_selected:,}\n")
            f.write(f"- **Topics Covered:** {total_topics}\n")
            f.write(f"- **Selection Ratio:** {total_selected/total_papers:.1%}\n")
            f.write(f"- **Selection Strategy:** {review_config.get('selection_strategy', 'iterative')}\n")
            f.write(f"- **Analysis Timestamp:** {timestamp}\n\n")
            
            # Selection Configuration
            f.write("## 🎯 Selection Configuration\n\n")
            f.write(f"- **Base Papers per Cluster:** {review_config.get('base_papers_per_cluster', 'N/A')}\n")
            f.write(f"- **Max Papers per Cluster:** {review_config.get('max_papers_per_cluster', 'N/A')}\n")
            f.write(f"- **Diversity Weight:** {review_config.get('diversity_weight', 'N/A')}\n")
            f.write(f"- **Similarity Threshold:** {review_config.get('similarity_threshold', 'N/A')}\n\n")
            
            # Statistical Overview
            f.write("## 📈 Statistical Overview\n\n")
            avg_cluster_size = summary_df['cluster_size'].mean()
            median_cluster_size = summary_df['cluster_size'].median()
            largest_cluster = summary_df['cluster_size'].max()
            smallest_cluster = summary_df['cluster_size'].min()
            
            f.write(f"- **Average Cluster Size:** {avg_cluster_size:.1f} papers\n")
            f.write(f"- **Median Cluster Size:** {median_cluster_size:.0f} papers\n")
            f.write(f"- **Largest Cluster:** {largest_cluster} papers\n")
            f.write(f"- **Smallest Cluster:** {smallest_cluster} papers\n")
            f.write(f"- **Average Selection per Topic:** {summary_df['papers_selected'].mean():.1f} papers\n\n")
            
            # Topic Summary Table
            f.write("## 📈 Topic Summary\n\n")
            f.write("| Topic ID | Topic Name | Cluster Size | Size Category | Selected | Selection Ratio | Avg Centrality | Avg Diversity |\n")
            f.write("|----------|------------|--------------|---------------|----------|-----------------|----------------|---------------|\n")
            
            for _, row in summary_df.iterrows():
                topic_name = row['topic_name'][:50] + "..." if len(str(row['topic_name'])) > 50 else row['topic_name']
                f.write(f"| {row['topic_id']} | {topic_name} | {row['cluster_size']} | {row['size_category']} | {row['papers_selected']} | {row['selection_ratio']:.1%} | {row.get('avg_centrality', 0):.3f} | {row.get('avg_diversity', 0):.3f} |\n")
            
            f.write("\n")
            
            # Research Alignment Analysis (if available)
            if 'xai_alignment' in selected_df.columns:
                f.write("## 🔍 Research Alignment Analysis\n\n")
                
                # Calculate average alignment scores
                xai_avg = selected_df['xai_alignment'].mean()
                symbolic_avg = selected_df['symbolic_alignment'].mean() if 'symbolic_alignment' in selected_df.columns else 0
                subsymbolic_avg = selected_df['subsymbolic_alignment'].mean() if 'subsymbolic_alignment' in selected_df.columns else 0
                
                f.write(f"**Average Research Alignment (Selected Papers):**\n")
                f.write(f"- **XAI Alignment:** {xai_avg:.2%}\n")
                f.write(f"- **Symbolic Alignment:** {symbolic_avg:.2%}\n")
                f.write(f"- **Sub-symbolic Alignment:** {subsymbolic_avg:.2%}\n\n")
                
                # High alignment papers
                high_xai_papers = selected_df[selected_df['xai_alignment'] > 0.5]
                f.write(f"**Papers with High XAI Alignment (>50%):** {len(high_xai_papers)} papers\n\n")
            
            # Reading Recommendations
            f.write("## 📝 Reading Recommendations\n\n")
            
            # Priority 1: High Diversity & Centrality
            f.write("### Priority 1: High Diversity & Centrality\n\n")
            high_priority = selected_df[
                (selected_df['diversity_score'] > selected_df['diversity_score'].quantile(0.75)) &
                (selected_df['similarity_to_centroid'] > selected_df['similarity_to_centroid'].quantile(0.75))
            ].sort_values('representativeness_score', ascending=False)
            
            f.write(f"**{len(high_priority)} papers** meeting both high diversity and centrality criteria:\n\n")
            for i, (_, paper) in enumerate(high_priority.head(10).iterrows(), 1):
                f.write(f"{i}. **{paper['title']}** (Topic {paper['topic_id']})\n")
                f.write(f"   - *Authors:* {paper['authors']}\n")
                f.write(f"   - *Representativeness:* {paper['representativeness_score']:.3f}\n")
                f.write(f"   - *Centrality:* {paper['similarity_to_centroid']:.3f} | *Diversity:* {paper['diversity_score']:.3f}\n\n")
            
            # Priority 2: Highest Representativeness by Topic
            f.write("### Priority 2: Highest Representativeness by Topic\n\n")
            f.write("Top representative paper from each topic:\n\n")
            
            for i, topic_id in enumerate(sorted(selected_df['topic_id'].unique()), 1):
                topic_papers = selected_df[selected_df['topic_id'] == topic_id]
                best_paper = topic_papers.loc[topic_papers['representativeness_score'].idxmax()]
                
                f.write(f"{i}. **Topic {topic_id}:** {best_paper['title']}\n")
                f.write(f"   - *Authors:* {best_paper['authors']}\n")
                f.write(f"   - *Representativeness:* {best_paper['representativeness_score']:.3f}\n")
                f.write(f"   - *Cluster Size:* {best_paper['cluster_size']} papers\n\n")
            
            # Quality Distribution Analysis
            f.write("## 📊 Quality Distribution Analysis\n\n")
            
            # Representativeness distribution
            repr_q25 = selected_df['representativeness_score'].quantile(0.25)
            repr_q50 = selected_df['representativeness_score'].quantile(0.50)
            repr_q75 = selected_df['representativeness_score'].quantile(0.75)
            
            f.write("### Representativeness Score Distribution\n\n")
            f.write(f"- **25th Percentile:** {repr_q25:.3f}\n")
            f.write(f"- **Median (50th Percentile):** {repr_q50:.3f}\n")
            f.write(f"- **75th Percentile:** {repr_q75:.3f}\n")
            f.write(f"- **Mean:** {selected_df['representativeness_score'].mean():.3f}\n")
            f.write(f"- **Standard Deviation:** {selected_df['representativeness_score'].std():.3f}\n\n")
            
            # Cluster size distribution
            f.write("### Cluster Size Distribution\n\n")
            cluster_distribution = summary_df['size_category'].value_counts()
            for category, count in cluster_distribution.items():
                f.write(f"- **{category}:** {count} topics\n")
            f.write("\n")
            
            # Research Coverage Analysis
            if 'year' in selected_df.columns:
                f.write("## 📅 Temporal Coverage Analysis\n\n")
                year_distribution = selected_df['year'].value_counts().sort_index()
                
                f.write(f"**Year Range:** {year_distribution.index.min()} - {year_distribution.index.max()}\n")
                f.write(f"**Most Productive Years:**\n")
                for year, count in year_distribution.tail(5).items():
                    f.write(f"- {year}: {count} papers\n")
                f.write("\n")
            
            # Methodology Notes
            f.write("## 🔬 Methodology Notes\n\n")
            f.write("### Selection Algorithm\n")
            f.write(f"- **Strategy:** {review_config.get('selection_strategy', 'iterative')}\n")
            f.write("- **Diversity Weight:** Balances representativeness (centrality) vs. uniqueness (diversity)\n")
            f.write("- **Similarity Threshold:** Minimum similarity for paper-to-representative assignment\n\n")
            
            f.write("### Quality Metrics\n")
            f.write("- **Similarity to Centroid:** How representative a paper is of its topic cluster\n")
            f.write("- **Diversity Score:** How unique a paper is within its cluster (1 - max similarity)\n")
            f.write("- **Representativeness Score:** Weighted combination of centrality and diversity\n\n")
            
            f.write("### Research Alignment\n")
            f.write("- **XAI Alignment:** Relevance to explainable AI terminology and concepts\n")
            f.write("- **Symbolic Alignment:** Relevance to symbolic AI and knowledge representation\n")
            f.write("- **Sub-symbolic Alignment:** Relevance to neural networks and deep learning\n\n")
            
            # Footer
            f.write("---\n\n")
            f.write(f"*Report generated by Advanced Systematic Literature Review Analyzer*\n")
            f.write(f"*Analysis completed on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")
        
        logger.info(f"📋 Analysis report saved to {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        raise


def generate_executive_summary(results_df: pd.DataFrame, summary_df: pd.DataFrame, 
                             selected_df: pd.DataFrame) -> Dict:
    """
    Generate executive summary statistics for quick overview.
    
    Args:
        results_df: Complete analysis results
        summary_df: Topic summary statistics  
        selected_df: Selected representatives
    
    Returns:
        Dict containing executive summary statistics
    """
    try:
        total_papers = len(results_df)
        total_selected = len(selected_df)
        total_topics = len(summary_df)
        
        # Basic statistics
        exec_summary = {
            'total_papers': total_papers,
            'selected_papers': total_selected,
            'total_topics': total_topics,
            'selection_ratio': total_selected / total_papers if total_papers > 0 else 0,
            'avg_cluster_size': summary_df['cluster_size'].mean() if len(summary_df) > 0 else 0,
            'avg_papers_per_topic': summary_df['papers_selected'].mean() if len(summary_df) > 0 else 0
        }
        
        # Quality metrics
        if len(selected_df) > 0:
            exec_summary.update({
                'avg_representativeness': selected_df['representativeness_score'].mean(),
                'avg_centrality': selected_df['similarity_to_centroid'].mean(),
                'avg_diversity': selected_df['diversity_score'].mean(),
                'high_quality_papers': len(selected_df[selected_df['representativeness_score'] > 0.7])
            })
        
        # Research alignment (if available)
        if len(selected_df) > 0 and 'xai_alignment' in selected_df.columns:
            exec_summary.update({
                'avg_xai_alignment': selected_df['xai_alignment'].mean(),
                'high_xai_papers': len(selected_df[selected_df['xai_alignment'] > 0.5])
            })
        
        return exec_summary
        
    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        return {'error': str(e)}


def validate_report_generation_config(config: Dict) -> bool:
    """
    Validate report generation configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Check output config
        output_config = get_output_config(config)
        
        if 'results_dir' not in output_config:
            logger.error("Missing required output config key: results_dir")
            return False
        
        # Check results directory accessibility
        results_dir = Path(output_config['results_dir'])
        if not results_dir.exists():
            try:
                results_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created results directory: {results_dir}")
            except Exception as e:
                logger.error(f"Cannot create results directory: {e}")
                return False
        
        if not results_dir.is_dir():
            logger.error(f"Results path is not a directory: {results_dir}")
            return False
        
        # Check write permissions
        test_file = results_dir / "test_write_permissions.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            logger.error(f"No write permissions for results directory: {e}")
            return False
        
        logger.info("✅ Report generation configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating report generation configuration: {e}")
        return False 