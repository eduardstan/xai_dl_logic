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
from outlier_reduction import apply_outlier_reduction
from visualization import create_visualizations
from results_saver import save_results, generate_summary_report

# BERTopic import
from bertopic import BERTopic

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def train_topic_model(topic_model: BERTopic, docs: List[str], embeddings: np.ndarray,
                     config: Dict, logger) -> Tuple[List[int], np.ndarray]:
    """
    Train the BERTopic model with pre-computed embeddings.
    Includes post-training outlier reduction following BERTopic best practices.
    
    Args:
        topic_model: Configured BERTopic model
        docs: List of documents
        embeddings: Pre-computed document embeddings
        config: Configuration dictionary
        logger: Configured logger instance
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger.info("🚀 Training BERTopic model with pre-computed embeddings")
    
    # Use pre-computed embeddings to avoid recomputation
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    # Log initial analysis results
    n_topics_initial = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers_initial = sum(1 for t in topics if t == -1)
    
    logger.info(f"Initial analysis complete:")
    logger.info(f"  - Found {n_topics_initial} topics")
    logger.info(f"  - {n_outliers_initial} outlier documents")
    logger.info(f"  - Coverage: {((len(docs) - n_outliers_initial) / len(docs) * 100):.1f}%")
    
    # Apply outlier reduction if configured
    topics = apply_outlier_reduction(topic_model, docs, topics, probs, embeddings, config, logger)
    
    # Log final analysis results
    n_topics_final = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers_final = sum(1 for t in topics if t == -1)
    
    logger.info(f"✅ Final analysis complete:")
    logger.info(f"  - Final topic count: {n_topics_final}")
    logger.info(f"  - {n_outliers_final} outlier documents") 
    logger.info(f"  - Coverage: {((len(docs) - n_outliers_final) / len(docs) * 100):.1f}%")
    
    if n_outliers_initial > n_outliers_final:
        outliers_reduced = n_outliers_initial - n_outliers_final
        logger.info(f"🎯 Outlier reduction: {outliers_reduced} documents reassigned to topics")
    
    # Cluster size distribution analysis
    logger.info("📊 Cluster size distribution:")
    cluster_sizes = []
    for topic_id in sorted(set(topics)):
        if topic_id != -1:  # Skip outliers
            count = topics.count(topic_id)
            cluster_sizes.append(count)
    
    if cluster_sizes:
        logger.info(f"  📈 Largest cluster: {max(cluster_sizes)} documents")
        logger.info(f"  📉 Smallest cluster: {min(cluster_sizes)} documents")
        logger.info(f"  📊 Average cluster size: {sum(cluster_sizes) / len(cluster_sizes):.1f}")
        logger.info(f"  🎯 Median cluster size: {sorted(cluster_sizes)[len(cluster_sizes)//2]}")
    
    return topics, probs


def main():
    """Main execution function."""
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
        topics, probs = train_topic_model(topic_model, docs, embeddings, config, logger)
        
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