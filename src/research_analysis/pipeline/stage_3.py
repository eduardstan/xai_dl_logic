#!/usr/bin/env python3
"""
Orchestrator for Stage 3 of the research analysis pipeline: Visualization.

This module loads the artifacts from Stage 2 (paper selection) and executes
the visualization generation, interactive dashboards, and reporting workflow.
"""

from pathlib import Path

from research_analysis.config.models import AppConfig
from research_analysis.stages.stage_3_visualization.data_loader import load_stage_2_artifacts
from research_analysis.stages.stage_3_visualization.metrics import create_metrics_overview
from research_analysis.stages.stage_3_visualization.selection_plots import create_selection_analysis
from research_analysis.stages.stage_3_visualization.interactive import create_interactive_visualizations
from research_analysis.stages.stage_3_visualization.network import create_network_visualizations
from research_analysis.stages.stage_3_visualization.report import generate_enhanced_statistics_report
from research_analysis.utils.logging import get_logger

logger = get_logger()


def run_stage_3(
    config: AppConfig,
    stage_2_input_dir: Path,
    output_dir: Path,
) -> None:
    """
    Executes the full Stage 3 visualization and reporting pipeline.

    Args:
        config: The application configuration.
        stage_2_input_dir: The path to the Stage 2 output directory.
        output_dir: The directory to save Stage 3 artifacts.
        
    Raises:
        FileNotFoundError: If Stage 2 artifacts are missing
        RuntimeError: If any visualization step fails critically
    """
    logger.info("🚀 Starting Stage 3: Visualization and Reporting")
    output_dir.mkdir(exist_ok=True, parents=True)

    try:
        # 1. Load artifacts from Stage 2
        logger.info(f"Loading artifacts from {stage_2_input_dir}...")
        df_all, df_summary, df_selected = load_stage_2_artifacts(stage_2_input_dir)

        # 2. Create static visualizations
        logger.info("📊 Creating metrics overview...")
        create_metrics_overview(df_all, df_selected, output_dir)

        logger.info("📈 Creating selection analysis...")
        create_selection_analysis(df_summary, output_dir)

        logger.info("🕸️ Creating network visualizations...")
        create_network_visualizations(df_all, output_dir)

        # 3. Create interactive visualizations
        logger.info("🌐 Creating interactive visualizations...")
        create_interactive_visualizations(df_all, df_selected, df_summary, output_dir)

        # 4. Generate comprehensive report
        logger.info("📋 Generating enhanced statistics report...")
        report_path = generate_enhanced_statistics_report(df_all, df_selected, df_summary, output_dir)

        # 5. Log completion summary
        _log_completion_summary(df_all, df_selected, df_summary, output_dir)

        logger.info("🎉 Stage 3 completed successfully!")

    except Exception as e:
        logger.error(f"❌ Stage 3 failed: {e}", exc_info=True)
        raise RuntimeError(f"Stage 3 visualization pipeline failed: {e}")


def _log_completion_summary(df_all, df_selected, df_summary, output_dir: Path) -> None:
    """Log a summary of completed Stage 3 operations."""
    logger.info("✅ Enhanced visualizations completed!")
    logger.info(f"📁 All outputs saved to: {output_dir.absolute()}")
    logger.info("📊 Generated files:")
    logger.info("  - metrics_overview.png")
    logger.info("  - selection_analysis.png")
    logger.info("  - paper_assignment_networks.png")
    logger.info("  - interactive_papers_explorer.html")
    logger.info("  - topic_dashboard.html")
    logger.info("  - enhanced_analysis_report.md")

    # Statistics for the user
    total_papers = len(df_all)
    selected_papers = len(df_selected)
    total_topics = len(df_summary)
    
    logger.info(f"📈 Analysis summary:")
    logger.info(f"  - {total_papers:,} total papers analyzed")
    logger.info(f"  - {selected_papers:,} representative papers selected")
    logger.info(f"  - {total_topics} topics with visualizations")
    logger.info(f"  - {selected_papers/total_papers:.2%} selection ratio")

    # Print final summary for user visibility
    print(f"\n✅ Enhanced visualizations completed!")
    print(f"📁 All outputs saved to: {output_dir.absolute()}")
    print("📊 Generated files:")
    print("  - metrics_overview.png")
    print("  - selection_analysis.png")
    print("  - paper_assignment_networks.png")
    print("  - interactive_papers_explorer.html")
    print("  - topic_dashboard.html")
    print("  - enhanced_analysis_report.md")
    print(f"\n📈 Analyzed {total_papers:,} papers across {total_topics} topics")
    print(f"🎯 Selected {selected_papers:,} representatives ({selected_papers/total_papers:.2%} ratio)")


def validate_stage_2_artifacts(stage_2_dir: Path) -> bool:
    """
    Validate that all required Stage 2 artifacts exist.
    
    Args:
        stage_2_dir: Directory containing Stage 2 outputs
        
    Returns:
        True if all required files exist, False otherwise
    """
    required_files = [
        "comprehensive_analysis.csv",
        "selection_summary.csv", 
        "selected_representatives.csv"
    ]
    
    for filename in required_files:
        if not (stage_2_dir / filename).exists():
            logger.error(f"Missing required Stage 2 file: {filename}")
            return False
    
    logger.info("✅ Stage 2 artifacts validation passed")
    return True


def create_visualization_summary(output_dir: Path) -> dict:
    """
    Create a summary of all generated visualization files.
    
    Args:
        output_dir: Directory containing Stage 3 outputs
        
    Returns:
        Dictionary with file information and statistics
    """
    visualization_files = {
        'static_images': [],
        'interactive_html': [],
        'reports': [],
        'total_size': 0
    }
    
    # Expected output files
    static_files = ['metrics_overview.png', 'selection_analysis.png', 'paper_assignment_networks.png']
    interactive_files = ['interactive_papers_explorer.html', 'topic_dashboard.html']
    report_files = ['enhanced_analysis_report.md']
    
    for filename in static_files:
        file_path = output_dir / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            visualization_files['static_images'].append({
                'name': filename,
                'size': file_size,
                'path': str(file_path)
            })
            visualization_files['total_size'] += file_size
    
    for filename in interactive_files:
        file_path = output_dir / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            visualization_files['interactive_html'].append({
                'name': filename,
                'size': file_size,
                'path': str(file_path)
            })
            visualization_files['total_size'] += file_size
    
    for filename in report_files:
        file_path = output_dir / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            visualization_files['reports'].append({
                'name': filename,
                'size': file_size,
                'path': str(file_path)
            })
            visualization_files['total_size'] += file_size
    
    return visualization_files 