#!/usr/bin/env python3
"""
BERTopic Analysis for Academic Bibliography - Main Orchestrator
Streamlined main script that coordinates the analysis pipeline.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

# Import shared utilities
from utils import setup_logging, load_config

# Import specialized modules
from bibliography import parse_bib_file
from embeddings import prepare_embeddings
from bertopic_setup import setup_bertopic_model
from topic_training import train_topic_model
from visualization import create_visualizations
from results_saver import save_results, generate_summary_report

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    """
    Main execution function orchestrating the complete BERTopic analysis pipeline.
    
    This function coordinates all stages of academic bibliography analysis:
    - Configuration loading and validation
    - Bibliography parsing from BIB file with caching
    - Text embedding generation with GPU optimization
    - BERTopic model setup and training
    - Outlier reduction for improved topic coverage
    - Comprehensive visualization generation
    - Results saving with scientific integrity
    - Summary report generation
    
    The pipeline is designed for reproducibility and scientific rigor with:
    - Comprehensive error handling and logging
    - Intelligent caching for faster iterations
    - Configurable parameters for different use cases
    - Detailed progress tracking with emoji indicators
    - Professional output suitable for academic publication
    
    Raises:
        FileNotFoundError: If configuration or BIB file is missing
        RuntimeError: If any pipeline stage fails critically
        ValueError: If configuration parameters are invalid
    
    Note:
        - Creates 'bertopic_analysis' output directory with structured results
        - All intermediate results are cached for faster subsequent runs
        - Supports both standard and guided topic modeling workflows
        - Generates publication-ready visualizations and reports
        - Logs all operations for transparency and debugging
    """
    # Setup
    setup_logging()
    logger.info("Starting BERTopic analysis for academic bibliography")
    
    # Load configuration
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Set random seed for reproducibility
    np.random.seed(config['random_seed'])
    
    # Create output directory
    output_dir = "bertopic_analysis"
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Parse BIB file (cached)
        df = parse_bib_file("merged.bib", config)
        docs = df['combined_text'].tolist()
        
        # Generate embeddings (cached)
        embeddings = prepare_embeddings(docs, config)
        
        # Setup and train BERTopic model
        topic_model = setup_bertopic_model(config)
        topics, probs = train_topic_model(topic_model, docs, embeddings, config)
        
        # Create visualizations
        create_visualizations(topic_model, docs, embeddings, topics, config, output_dir)
        
        # Save results
        save_results(topic_model, df, topics, probs, embeddings, config, output_dir)
        
        # Generate summary report
        generate_summary_report(topic_model, df, topics, config, output_dir)
        
        logger.info("Analysis completed successfully!")
        print(f"\n✅ Analysis complete! Check the '{output_dir}' directory for results.")
        print(f"📊 Interactive visualizations are available in '{output_dir}/plots/'")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 