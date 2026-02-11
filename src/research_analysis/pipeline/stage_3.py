#!/usr/bin/env python3
"""
Orchestrator for Stage 3 of the research analysis pipeline: Visualization.

This module loads the artifacts from Stage 2 (paper selection) and executes
the visualization generation, interactive dashboards, and reporting workflow.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import yaml
import pandas as pd # Added for data manipulation in statistical analysis preparation

from research_analysis.config.models import AppConfig
from research_analysis.stages.stage_3_visualization.data_loader import load_stage_2_artifacts
from research_analysis.stages.stage_3_visualization.metrics import create_metrics_overview
from research_analysis.stages.stage_3_visualization.selection_plots import create_selection_analysis
from research_analysis.stages.stage_3_visualization.dashboards import create_interactive_visualizations
from research_analysis.stages.stage_3_visualization.network import create_network_visualizations
from research_analysis.stages.stage_3_visualization.report import generate_enhanced_statistics_report
from research_analysis.stages.stage_3_visualization.statistical_analysis import comprehensive_statistical_analysis
from research_analysis.stages.stage_3_visualization.statistical_plots import create_all_statistical_plots
from research_analysis.utils.logging import get_logger

logger = get_logger()


def run_stage_3(config: AppConfig, output_dir: Path) -> None:
    """
    Executes the full Stage 3 visualization and reporting pipeline.

    Args:
        config: The application configuration.
        output_dir: The *root* output directory for the entire pipeline run.
        
    Raises:
        FileNotFoundError: If Stage 2 artifacts are missing
        RuntimeError: If any visualization step fails critically
    """
    logger.info("Starting Stage 3: Visualization and Reporting")
    
    # Construct input and output paths from config
    stage_2_input_dir = output_dir / config.pipeline.paths.stage_2_path
    stage_3_output_dir = output_dir / config.pipeline.paths.stage_3_path
    stage_3_output_dir.mkdir(exist_ok=True, parents=True)

    try:
        # 1. Load artifacts from Stage 2
        logger.info(f"Loading artifacts from {stage_2_input_dir}...")
        df_all, df_summary, df_selected = load_stage_2_artifacts(stage_2_input_dir)

        # 2. Load model from Stage 1 for hierarchy visualization
        stage_1_input_dir = output_dir / config.pipeline.paths.stage_1_path
        model_path = stage_1_input_dir / "bertopic_model"
        if not model_path.exists():
            logger.warning(f"BERTopic model not found at {model_path}. Skipping hierarchy visualization.")
            topic_model = None
        else:
            from bertopic import BERTopic
            topic_model = BERTopic.load(model_path)
            
        # 3. HDBSCAN Hierarchy Visualization
        if topic_model:
            from research_analysis.stages.stage_3_visualization.hdbscan_hierarchy import visualize_hdbscan_structure
            # Compute hierarchy using the ORIGINAL pool only (3749 papers)
            # This avoids length mismatch issues with the trained BERTopic model
            df_original = df_all[df_all['source'] != 'r1_reviewer'].copy()
            docs = df_original['title'].fillna('') + " " + df_original['abstract'].fillna('')
            
            logger.info(f"Visualizing hierarchy using {len(df_original)} original papers...")
            visualize_hdbscan_structure(topic_model, stage_3_output_dir, docs=docs.tolist())

        # 4. Load Medoid Comparison Results from Stage 2
        medoid_results = None
        medoid_path = stage_2_input_dir / "medoid_comparison.json"
        if medoid_path.exists():
            import json
            try:
                with open(medoid_path, 'r') as f:
                    medoid_results = json.load(f)
                logger.info("Loaded medoid comparison results.")
            except Exception as e:
                logger.warning(f"Could not load medoid comparison results: {e}")

        # 5. Create static visualizations
        logger.info("Creating metrics overview...")
        create_metrics_overview(df_all, df_selected, stage_3_output_dir)

        logger.info("Creating selection analysis...")
        create_selection_analysis(df_summary, stage_3_output_dir)

        logger.info("Creating network visualizations...")
        create_network_visualizations(df_all, stage_3_output_dir)

        # 6. Create interactive visualizations
        logger.info("Creating interactive visualizations...")
        create_interactive_visualizations(df_all, df_selected, df_summary, stage_3_output_dir)

        # 7. Perform statistical analysis (if enabled)
        statistical_results = None
        statistical_plots = {}
        
        if config.stage_3.statistical_analysis.enabled:
            # Perform statistical analysis
            logger.info("Performing comprehensive statistical analysis...")
            
            # Filter for original pool to ensure selection statistics reflect the standard workflow
            # (R1 papers were not part of the selection pool and should not shift the background distribution)
            is_r1 = df_all.get('source', pd.Series([False] * len(df_all))) == 'r1_reviewer'
            original_pool_df = df_all.iloc[~is_r1.values]
            
            selected_df = original_pool_df[original_pool_df['is_selected_representative'] == True]
            non_selected_df = original_pool_df[original_pool_df['is_selected_representative'] == False]
            
            # Get variables to analyze
            variables = (config.stage_3.statistical_analysis.primary_metrics + 
                        config.stage_3.statistical_analysis.research_alignment)
            
            # Perform statistical analysis
            statistical_results = comprehensive_statistical_analysis(
                selected_df=selected_df,
                non_selected_df=non_selected_df,
                variables=variables,
                config=config.stage_3
            )
            
            # Create statistical plots
            if any(config.stage_3.visualizations.dict().values()):
                logger.info("Creating statistical visualizations...")
                statistical_plots = create_all_statistical_plots(
                    statistical_results=statistical_results,
                    selected_df=selected_df,
                    non_selected_df=non_selected_df,
                    variables=variables,
                    output_dir=stage_3_output_dir,
                    config=config.stage_3
                )
            
            # Save statistical results
            if config.stage_3.output.save_statistical_results:
                _save_statistical_results(statistical_results, stage_3_output_dir)

        # Save configuration used for this stage
        config_path = stage_3_output_dir / "stage_3_config_used.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config.dict(), f, default_flow_style=False)
        logger.info(f"Stage 3 configuration saved to: {config_path}")

        # 8. Generate comprehensive report
        logger.info("Generating enhanced statistics report...")
        generate_enhanced_statistics_report(
            df_all, df_selected, df_summary, stage_3_output_dir, 
            medoid_results=medoid_results
        )

        # 6. Log completion summary
        _log_completion_summary(df_all, df_selected, df_summary, stage_3_output_dir, statistical_plots)

        logger.info("Stage 3 completed successfully!")

    except Exception as e:
        logger.error(f"Stage 3 failed: {e}", exc_info=True)
        raise RuntimeError(f"Stage 3 visualization pipeline failed: {e}")


def _save_statistical_results(statistical_results: Dict[str, Any], output_dir: Path) -> None:
    """Save statistical analysis results to JSON file."""
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {str(key) if isinstance(key, tuple) else key: convert_to_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(statistical_results)
    
    results_path = output_dir / "statistical_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Statistical results saved to: {results_path}")


def _log_completion_summary(df_all, df_selected, df_summary, output_dir: Path, statistical_plots: Dict[str, Path] = None) -> None:
    """Log a summary of completed Stage 3 operations."""
    logger.info("Enhanced visualizations completed!")
    logger.info(f"All outputs saved to: {output_dir.absolute()}")
    logger.info("Generated files:")
    logger.info("  - metrics_overview.png")
    logger.info("  - selection_analysis.png")
    logger.info("  - paper_assignment_networks.png")
    logger.info("  - interactive_papers_explorer.html")
    logger.info("  - topic_dashboard.html")
    logger.info("  - enhanced_analysis_report.md")
    
    # Log statistical plots if they were created
    if statistical_plots:
        logger.info("  Statistical Analysis Files:")
        for plot_name, plot_path in statistical_plots.items():
            if plot_path:
                logger.info(f"    - {plot_path.name}")
        logger.info("    - statistical_analysis_results.json")
    logger.info("  - stage_3_config_used.yaml")

    # Statistics for the user
    total_papers = len(df_all)
    selected_papers = len(df_selected)
    total_topics = len(df_summary)
    
    logger.info(f"Analysis summary:")
    logger.info(f"  - {total_papers:,} total papers analyzed")
    logger.info(f"  - {selected_papers:,} representative papers selected")
    logger.info(f"  - {total_topics} topics with visualizations")
    logger.info(f"  - {selected_papers/total_papers:.2%} selection ratio")

    # Print final summary for user visibility
    print(f"\nEnhanced visualizations completed!")
    print(f"All outputs saved to: {output_dir.absolute()}")
    print("Generated files:")
    print("  - metrics_overview.png")
    print("  - selection_analysis.png")
    print("  - paper_assignment_networks.png")
    print("  - interactive_papers_explorer.html")
    print("  - topic_dashboard.html")
    print("  - enhanced_analysis_report.md")
    
    # Print statistical files if they were created
    if statistical_plots:
        print("  Statistical Analysis Files:")
        for plot_name, plot_path in statistical_plots.items():
            if plot_path:
                print(f"    - {plot_path.name}")
        print("    - statistical_analysis_results.json")
    print("  - stage_3_config_used.yaml")
    print(f"\nAnalyzed {total_papers:,} papers across {total_topics} topics")
    print(f"Selected {selected_papers:,} representatives ({selected_papers/total_papers:.2%} ratio)")


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
    
    logger.info("Stage 2 artifacts validation passed")
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