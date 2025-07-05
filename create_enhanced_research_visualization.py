#!/usr/bin/env python3
"""
Enhanced Research Landscape Visualization
Create comprehensive visualizations incorporating diversity, centrality, and representativeness metrics
"""

from pathlib import Path
from loguru import logger

# Import shared utilities
from utils import setup_logging

# Import data loading functionality
from data_loader_viz import load_analysis_data
from metrics_visualizations import create_metrics_overview
from selection_visualizations import create_selection_analysis
from interactive_visualizations import create_interactive_topic_explorer
from network_visualizations import create_paper_assignment_network
from report_generation import generate_enhanced_statistics_report


def main():
    """
    Main execution function orchestrating the complete enhanced research visualization pipeline.
    
    This function coordinates all stages of visualization generation:
    - Configuration and logging setup
    - Data loading with validation from analysis results
    - Comprehensive metrics overview visualization creation
    - Selection strategy analysis visualization generation
    - Interactive Plotly-based visualization development
    - Paper assignment network visualization creation  
    - Enhanced statistics report generation
    - Results saving with scientific integrity
    
    The pipeline is designed for reproducibility and scientific rigor with:
    - Comprehensive error handling and logging
    - Modular visualization components for maintainability
    - Professional output suitable for academic publication
    - Interactive elements for enhanced data exploration
    - Detailed progress tracking with emoji indicators
    
    Raises:
        FileNotFoundError: If analysis result files are missing
        RuntimeError: If any visualization stage fails critically
        ValueError: If loaded data has invalid structure
    
    Note:
        - Expects analysis results in 'results/' directory from previous pipeline runs
        - Creates multiple output formats: PNG, HTML, and Markdown
        - All operations are logged for transparency and debugging
        - Follows same pattern as bib_analyzer.py and advanced_systematic_analyzer.py
    """
    # Setup
    setup_logging()
    logger.info("Starting enhanced research landscape visualization")
    
    try:
        # Create output directory
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Starting enhanced research visualization...")
        
        # Load data
        df_all, df_summary, df_selected = load_analysis_data()
        
        # Create visualizations
        logger.info("📊 Creating metrics overview...")
        create_metrics_overview(df_all, df_selected, output_dir)
        
        logger.info("📈 Creating selection analysis...")
        create_selection_analysis(df_summary, output_dir)
        
        logger.info("🌐 Creating interactive visualizations...")
        create_interactive_topic_explorer(df_all, df_selected, df_summary, output_dir)
        
        logger.info("🕸️ Creating paper assignment networks...")
        create_paper_assignment_network(df_all, output_dir)
        
        logger.info("📋 Generating enhanced statistics report...")
        generate_enhanced_statistics_report(df_all, df_selected, df_summary, output_dir)
        
        logger.info("✅ Enhanced visualizations completed!")
        logger.info(f"📁 All outputs saved to: {output_dir.absolute()}")
        logger.info("📊 Generated files:")
        logger.info("  - metrics_overview.png")
        logger.info("  - selection_analysis.png") 
        logger.info("  - interactive_papers_explorer.html")
        logger.info("  - topic_dashboard.html")
        logger.info("  - paper_assignment_networks.png")
        logger.info("  - enhanced_analysis_report.md")
        
        # Print summary for user visibility
        print(f"\n✅ Enhanced visualizations completed!")
        print(f"📁 All outputs saved to: {output_dir.absolute()}")
        print("📊 Generated files:")
        print("  - metrics_overview.png")
        print("  - selection_analysis.png") 
        print("  - interactive_papers_explorer.html")
        print("  - topic_dashboard.html")
        print("  - paper_assignment_networks.png")
        print("  - enhanced_analysis_report.md")
        
    except Exception as e:
        logger.error(f"❌ Visualization generation failed: {e}")
        raise


if __name__ == "__main__":
    main() 