#!/usr/bin/env python3
"""
Main CLI entry point for the research_analysis framework.

This module provides a command-line interface using Typer for running
the complete pipeline or individual stages.
"""

from pathlib import Path
from typing import Optional
import typer
from typing_extensions import Annotated

from research_analysis.config.loader import load_config
from research_analysis.pipeline.orchestrator import run_full_pipeline
from research_analysis.pipeline.stage_1 import run_stage_1
from research_analysis.pipeline.stage_2 import run_stage_2
from research_analysis.pipeline.stage_3 import run_stage_3
from research_analysis.utils.logging import setup_logging

app = typer.Typer(
    name="research-analysis",
    help="A unified framework for systematic bibliography analysis.",
    add_completion=False,
)


@app.command()
def run_pipeline(
    config_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--config-dir",
            "-c",
            help="Directory containing configuration files",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help="Override output directory (default: from config)",
        ),
    ] = None,
) -> None:
    """
    Run the complete research analysis pipeline across all stages.
    
    This command executes Stage 1 (Topic Modeling), Stage 2 (Paper Selection),
    and Stage 3 (Visualization) in sequence.
    """
    if config_dir is None:
        config_dir = Path("configs")
    
    try:
        # Load configuration
        config = load_config(
            pipeline_config_path=config_dir / "pipeline.yaml",
            stage_1_config_path=config_dir / "stage_1_topic_model.yaml",
            stage_2_config_path=config_dir / "stage_2_selection.yaml",
        )
        
        # Override output directory if specified
        if output_dir:
            config.pipeline.paths.outputs = str(output_dir)
        
        # Setup logging
        setup_logging(config.pipeline.logging)
        
        # Run the full pipeline
        result_dir = run_full_pipeline(config)
        
        typer.echo(f"✅ Pipeline completed successfully!")
        typer.echo(f"📁 Results saved to: {result_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Pipeline failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def run_stage(
    stage: Annotated[
        int,
        typer.Option(
            "--stage",
            "-s",
            help="Stage number to run (1, 2, or 3)",
            min=1,
            max=3,
        ),
    ],
    config_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--config-dir",
            "-c",
            help="Directory containing configuration files",
        ),
    ] = None,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory (required for individual stages)",
        ),
    ] = None,
) -> None:
    """
    Run a specific stage of the research analysis pipeline.
    
    STAGE 1: Topic Modeling - Analyzes bibliography and discovers topics
    STAGE 2: Paper Selection - Selects representative papers from each topic  
    STAGE 3: Visualization - Creates visualizations and reports
    
    Note: Stages 2 and 3 require artifacts from previous stages.
    """
    if config_dir is None:
        config_dir = Path("configs")
    
    if output_dir is None:
        typer.echo("❌ --output-dir is required when running individual stages", err=True)
        raise typer.Exit(1)
    
    try:
        # Load configuration
        config = load_config(
            pipeline_config_path=config_dir / "pipeline.yaml",
            stage_1_config_path=config_dir / "stage_1_topic_model.yaml",
            stage_2_config_path=config_dir / "stage_2_selection.yaml",
        )
        
        # Override output directory
        config.pipeline.paths.outputs = str(output_dir.parent)
        
        # Setup logging
        setup_logging(config.pipeline.logging)
        
        # Run the specified stage
        if stage == 1:
            typer.echo("🚀 Running Stage 1: Topic Modeling")
            run_stage_1(config, output_dir)
            typer.echo("✅ Stage 1 completed successfully!")
            
        elif stage == 2:
            typer.echo("🚀 Running Stage 2: Paper Selection")
            run_stage_2(config, output_dir)
            typer.echo("✅ Stage 2 completed successfully!")
            
        elif stage == 3:
            typer.echo("🚀 Running Stage 3: Visualization")
            run_stage_3(config, output_dir)
            typer.echo("✅ Stage 3 completed successfully!")
        
        typer.echo(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        typer.echo(f"❌ Stage {stage} failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def validate_config(
    config_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--config-dir",
            "-c",
            help="Directory containing configuration files",
        ),
    ] = None,
) -> None:
    """
    Validate configuration files for syntax and completeness.
    
    This command checks that all required configuration files exist
    and can be parsed correctly.
    """
    if config_dir is None:
        config_dir = Path("configs")
    
    try:
        typer.echo("🔍 Validating configuration files...")
        
        # Check if config files exist
        required_files = [
            "pipeline.yaml",
            "stage_1_topic_model.yaml", 
            "stage_2_selection.yaml"
        ]
        
        for filename in required_files:
            config_path = config_dir / filename
            if not config_path.exists():
                typer.echo(f"❌ Missing config file: {config_path}", err=True)
                raise typer.Exit(1)
            typer.echo(f"✓ Found: {config_path}")
        
        # Try to load the configuration
        config = load_config(
            pipeline_config_path=config_dir / "pipeline.yaml",
            stage_1_config_path=config_dir / "stage_1_topic_model.yaml",
            stage_2_config_path=config_dir / "stage_2_selection.yaml",
        )
        
        typer.echo("✅ All configuration files are valid!")
        typer.echo(f"📊 Pipeline will output to: {config.pipeline.paths.outputs}")
        
    except Exception as e:
        typer.echo(f"❌ Configuration validation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo("research-analysis v0.1.0")
    typer.echo("A unified framework for systematic bibliography analysis.")


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main() 