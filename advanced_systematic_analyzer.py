#!/usr/bin/env python3
"""
Advanced Systematic Review Analyzer with Diverse Representative Paper Selection

This module implements a sophisticated paper selection strategy that:
1. Selects diverse representative papers within each cluster
2. Computes comprehensive similarity metrics
3. Maps non-selected papers to their most similar representatives
4. Uses configurable parameters from config.yaml
"""

from pathlib import Path
from loguru import logger

# Import shared utilities
from utils import setup_logging, load_config

# Import extracted functionality modules
from topic_processor import process_topics
from report_generator import generate_analysis_report


def main():
    """
    Main execution function orchestrating the complete systematic literature review analysis.
    
    This function coordinates all stages of systematic literature review analysis:
    - Configuration loading and validation
    - Topic processing and representative paper selection
    - Comprehensive analysis report generation
    - Results saving with scientific integrity
    
    The pipeline is designed for reproducibility and scientific rigor with:
    - Comprehensive error handling and logging
    - Configurable parameters for different analysis strategies
    - Detailed progress tracking with emoji indicators
    - Professional output suitable for academic publication
    
    Raises:
        FileNotFoundError: If configuration file is missing
        RuntimeError: If any pipeline stage fails critically
        ValueError: If configuration parameters are invalid
    
    Note:
        - Creates output directories as specified in configuration
        - All operations are logged for transparency and debugging
        - Results include selected representatives and comprehensive reports
        - Follows same pattern as bib_analyzer.py for consistency
    """
    # Setup
    setup_logging()
    logger.info("Starting advanced systematic literature review analysis")
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Ensure output directories exist
        results_dir = Path(config['output']['results_dir'])
        results_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Starting systematic literature review analysis...")
        
        # Process topics and select representatives
        results_df, summary_df, selected_df = process_topics(config)
        
        # Generate comprehensive analysis report
        report_path = generate_analysis_report(results_df, summary_df, selected_df, config)
        
        logger.info("🎉 Analysis completed successfully!")
        logger.info(f"📊 {len(selected_df)} representatives selected from {len(results_df)} total papers")
        logger.info(f"📁 Results saved to: {results_dir}")
        logger.info(f"📋 Report saved to: {report_path}")
        
        # Print summary results
        print(f"\n🎉 Advanced systematic review analysis completed!")
        print(f"📊 {len(selected_df)} representative papers selected from {len(results_df)} total papers")
        print(f"📁 Results saved to: {results_dir}")
        print(f"📋 Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 