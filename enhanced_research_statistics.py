#!/usr/bin/env python3
"""
Enhanced Research Statistics Pipeline
Runs advanced statistical analysis on selected vs non-selected papers
"""

from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np

# Import shared utilities
from utils import setup_logging, load_config

# Import data loading
from data_loader import load_analysis_results

# Import enhanced analytics
from enhanced_analytics import (
    compute_enhanced_metrics,
    compute_statistical_comparisons,
    compute_improved_research_alignment,
    create_metrics_comparison_plots,
    generate_enhanced_analytics_report
)


def main():
    """
    Main execution function for enhanced research statistics analysis.
    
    This function orchestrates advanced statistical analysis comparing selected
    representative papers with non-selected papers across multiple metrics:
    - Centrality, diversity, and representativeness scores
    - Statistical significance tests (Mann-Whitney U)
    - Effect size measurements (Cohen's d)
    - Improved research alignment analysis
    - Comprehensive visualization generation
    
    The analysis provides insights into the quality and characteristics of the
    paper selection algorithm and validates the representativeness of selected papers.
    """
    # Setup
    setup_logging()
    logger.info("Starting enhanced research statistics analysis")
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Create output directory
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Starting enhanced research statistics analysis...")
        
        # Load analysis results
        logger.info("📂 Loading analysis results...")
        topic_model, topic_assignments, embeddings, documents = load_analysis_results(config)
        
        # Create a proper DataFrame from the loaded data
        if isinstance(documents, list):
            # Convert documents list to DataFrame
            df_with_topics = pd.DataFrame(documents)
        else:
            df_with_topics = documents
        
        # Add topic assignments
        df_with_topics['topic'] = topic_assignments
        
        # Load selected representatives
        logger.info("📂 Loading selected representatives...")
        selected_files = sorted(output_dir.glob("selected_representatives_*.csv"))
        if not selected_files:
            raise FileNotFoundError("No selected representatives files found")
        
        selected_df = pd.read_csv(selected_files[-1])  # Use most recent
        
        # Get the actual indices from the selected dataframe using global_index
        if 'global_index' in selected_df.columns:
            selected_indices = selected_df['global_index'].tolist()
        else:
            # Fallback to row indices if global_index not available
            selected_indices = selected_df.index.tolist()
        
        logger.info(f"📊 Loaded {len(selected_df)} selected representatives from {len(df_with_topics)} total papers")
        
        # Compute enhanced metrics
        logger.info("⚡ Computing enhanced metrics...")
        df_enhanced = compute_enhanced_metrics(
            df_with_topics, embeddings, topic_assignments, config
        )
        
        # Ensure selected indices are valid for the enhanced dataframe
        valid_selected_indices = [idx for idx in selected_indices if 0 <= idx < len(df_enhanced)]
        if len(valid_selected_indices) != len(selected_indices):
            logger.warning(f"Some selected indices were invalid. Using {len(valid_selected_indices)} out of {len(selected_indices)} indices")
            selected_indices = valid_selected_indices
        
        # Compute statistical comparisons
        logger.info("📈 Computing statistical comparisons...")
        statistical_results = compute_statistical_comparisons(df_enhanced, selected_indices)
        
        # Compute improved research alignment
        logger.info("🔍 Computing improved research alignment...")
        alignment_results = compute_improved_research_alignment(
            df_enhanced, selected_indices, config
        )
        
        # Create visualization plots
        logger.info("📊 Creating metrics comparison plots...")
        create_metrics_comparison_plots(
            df_enhanced, selected_indices, statistical_results, output_dir
        )
        
        # Generate comprehensive report
        logger.info("📋 Generating enhanced analytics report...")
        report_path = generate_enhanced_analytics_report(
            df_enhanced, selected_indices, statistical_results, alignment_results, output_dir
        )
        
        logger.info("🎉 Enhanced statistics analysis completed successfully!")
        logger.info(f"📊 Analyzed {len(df_enhanced)} papers ({len(selected_indices)} selected)")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info(f"📋 Report saved to: {report_path}")
        
        # Print summary results
        print(f"\n🎉 Enhanced research statistics analysis completed!")
        print(f"📊 {len(selected_indices)} selected papers analyzed from {len(df_enhanced)} total papers")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📋 Report saved to: {report_path}")
        
        # Print key findings
        print("\n📈 Key Statistical Findings:")
        for metric, results in statistical_results.items():
            metric_name = metric.replace('_', ' ').title()
            print(f"  • {metric_name}: Cohen's d = {results['cohens_d']:.3f} ({results['effect_size']} effect)")
        
        if alignment_results:
            print("\n🔍 Research Alignment Results:")
            for category, results in alignment_results.items():
                print(f"  • {category}: {results['enrichment_ratio']:.2f}x enrichment (p={results['p_value']:.4f})")
        
    except Exception as e:
        logger.error(f"❌ Enhanced statistics analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 