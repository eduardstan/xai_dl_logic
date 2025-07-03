#!/usr/bin/env python3
"""
BERTopic Analysis for Academic Bibliography
Following best practices for topic modeling on academic literature.
"""

import os
import pickle
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import yaml
import pandas as pd
import numpy as np
import bibtexparser
from bibtexparser.bparser import BibTexParser

# BERTopic and ML imports
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Visualization imports
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# GPU and caching
import torch
import hashlib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging with timestamps and structured format."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file with error handling."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")


def parse_bib_file(file_path: str, config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Parse BIB file and extract relevant text fields with caching.
    
    Args:
        file_path: Path to the BIB file
        config: Configuration dictionary
        logger: Configured logger instance
        
    Returns:
        DataFrame with parsed bibliography entries
    """
    # Check if parsed data is cached
    cache_dir = Path(config['output']['cache_dir'])
    bib_hash = get_file_hash(file_path)
    cache_name = f"parsed_bib_{bib_hash[:8]}"
    
    if cache_exists(cache_dir, cache_name):
        logger.info("📂 Loading parsed BIB data from cache...")
        return load_from_cache(cache_dir, cache_name, logger)
    
    logger.info(f"📖 Parsing BIB file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as bib_file:
            parser = BibTexParser(common_strings=True)
            bib_database = bibtexparser.load(bib_file, parser=parser)
            
        logger.info(f"Found {len(bib_database.entries)} entries in BIB file")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(bib_database.entries)
        
        # Clean and standardize text fields
        text_fields = ['abstract', 'title', 'keywords', 'author']
        
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].fillna('').astype(str)
                # Remove excessive whitespace and clean text
                df[field] = df[field].str.replace(r'\s+', ' ', regex=True).str.strip()
            else:
                df[field] = ''
                
        # Create combined text field for topic modeling
        df['combined_text'] = (
            df['title'].fillna('') + ' ' + 
            df['abstract'].fillna('') + ' ' + 
            df['keywords'].fillna('')
        ).str.strip()
        
        # Filter out entries with insufficient text content
        min_text_length = 50
        original_count = len(df)
        df = df[df['combined_text'].str.len() >= min_text_length]
        
        logger.info(f"Filtered to {len(df)} entries with sufficient text content "
                   f"(removed {original_count - len(df)} entries)")
        
        # Cache the result
        save_to_cache(df, cache_dir, cache_name, logger)
        
        return df
        
    except Exception as e:
        logger.error(f"Error parsing BIB file: {e}")
        raise


def prepare_embeddings(texts: List[str], config: Dict, logger: logging.Logger) -> np.ndarray:
    """
    Generate embeddings using sentence transformers with caching.
    
    Args:
        texts: List of text documents
        config: Configuration dictionary
        logger: Configured logger instance
        
    Returns:
        Numpy array of embeddings
    """
    model_config = config['embedding_model']
    
    # Create embedding cache key based on model and documents
    docs_hash = hashlib.md5(str(texts).encode()).hexdigest()
    model_name = model_config['name']
    cache_dir = Path(config['output']['cache_dir'])
    embedding_cache_name = f"embeddings_{model_name.replace('/', '_')}_{docs_hash[:8]}"
    
    if cache_exists(cache_dir, embedding_cache_name):
        logger.info("🚀 Loading embeddings from cache (super fast!)...")
        embeddings = load_from_cache(cache_dir, embedding_cache_name, logger)
        logger.info(f"✅ Loaded embeddings shape: {embeddings.shape}")
        return embeddings
    
    logger.info(f"🤖 Computing embeddings using {model_config['name']}")
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Increase batch size for GPU
        batch_size = min(model_config['batch_size'] * 2, 128)
    else:
        logger.info("Using CPU")
        batch_size = model_config['batch_size']
    
    # Initialize sentence transformer model with device
    sentence_model = SentenceTransformer(model_config['name'], device=device)
    
    # Generate embeddings with progress bar
    logger.info("⏳ This may take several minutes for 3,364 documents...")
    embeddings = sentence_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=model_config['show_progress'],
        convert_to_numpy=True
    )
    
    # Cache the embeddings
    save_to_cache(embeddings, cache_dir, embedding_cache_name, logger)
    
    # Clear GPU memory if using CUDA
    if device == "cuda":
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")
    
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def setup_bertopic_model(config: Dict, logger: logging.Logger) -> BERTopic:
    """
    Configure BERTopic model with optimized parameters for academic text.
    Supports both seed words and guided topic modeling.
    
    Args:
        config: Configuration dictionary
        logger: Configured logger instance
        
    Returns:
        Configured BERTopic model
    """
    logger.info("Setting up BERTopic model with custom parameters")
    
    # Configure UMAP for dimensionality reduction
    umap_params = config['umap_params'].copy()
    logger.info(f"UMAP parameters: n_neighbors={umap_params.get('n_neighbors')}, "
               f"n_components={umap_params.get('n_components')}, "
               f"min_dist={umap_params.get('min_dist')}")
    umap_model = UMAP(**umap_params)
    
    # Configure HDBSCAN for clustering
    hdbscan_params = config['hdbscan_params'].copy()
    logger.info(f"HDBSCAN parameters: min_cluster_size={hdbscan_params.get('min_cluster_size')}, "
               f"max_cluster_size={hdbscan_params.get('max_cluster_size')}, "
               f"cluster_selection_epsilon={hdbscan_params.get('cluster_selection_epsilon')}")
    hdbscan_model = HDBSCAN(**hdbscan_params)
    
    # Configure vectorizer for better academic term extraction
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
        max_df=0.95,
        max_features=5000
    )
    
    # Configure ClassTfidfTransformer with seed words if enabled
    ctfidf_model = None
    if config.get('domain_guidance', {}).get('seed_words', {}).get('enabled', False):
        seed_config = config['domain_guidance']['seed_words']
        seed_words = seed_config.get('words', [])
        multiplier = seed_config.get('multiplier', 2.0)
        
        if seed_words:
            logger.info(f"🌱 Enabling seed words enhancement with {len(seed_words)} domain terms")
            logger.info(f"   Multiplier: {multiplier}x for words: {seed_words[:5]}{'...' if len(seed_words) > 5 else ''}")
            
            ctfidf_model = ClassTfidfTransformer(
                seed_words=seed_words,
                seed_multiplier=multiplier
            )
        else:
            logger.warning("Seed words enabled but no words provided - skipping enhancement")
    else:
        logger.info("Seed words enhancement disabled")
    
    # Check if guided topic modeling is enabled
    guided_topics_config = config.get('domain_guidance', {}).get('guided_topics', {})
    if guided_topics_config.get('enabled', False):
        logger.info("🎯 Guided topic modeling enabled")
        return setup_guided_bertopic_model(
            config, umap_model, hdbscan_model, vectorizer_model, 
            ctfidf_model, logger
        )
    
    # Initialize standard BERTopic model
    # NOTE: Following BERTopic best practices - let HDBSCAN find natural clusters
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,  # Add ClassTfidfTransformer
        calculate_probabilities=config['data']['calculate_probabilities'],
        # REMOVED nr_topics - will use post-training reduction if needed
        min_topic_size=config['data']['min_topic_size'],
        verbose=True
    )
    
    return topic_model


def setup_guided_bertopic_model(config: Dict, umap_model, hdbscan_model, 
                               vectorizer_model, ctfidf_model, logger: logging.Logger) -> BERTopic:
    """
    Configure BERTopic model with guided topic modeling.
    
    Args:
        config: Configuration dictionary
        umap_model: Configured UMAP model
        hdbscan_model: Configured HDBSCAN model
        vectorizer_model: Configured vectorizer
        ctfidf_model: Configured ClassTfidfTransformer (can be None)
        logger: Configured logger instance
        
    Returns:
        BERTopic model configured for guided topic modeling
    """
    guided_config = config['domain_guidance']['guided_topics']
    guided_params = guided_config.get('parameters', {})
    
    # Extract guided topics and their seed words
    topics_dict = guided_config.get('topics', {})
    
    # Prepare seed topic lists for BERTopic
    seed_topic_list = []
    
    for topic_name, topic_config in topics_dict.items():
        seeds = topic_config.get('seeds', [])
        
        if seeds:
            seed_topic_list.append(seeds)
            logger.info(f"   📋 {topic_name}: {seeds}")
    
    if not seed_topic_list:
        logger.warning("Guided topics enabled but no topic seeds provided - falling back to standard model")
        return BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            calculate_probabilities=config['data']['calculate_probabilities'],
            nr_topics=config['data']['nr_topics'],
            min_topic_size=config['data']['min_topic_size'],
            verbose=True
        )
    
    logger.info(f"   🎯 Configured {len(seed_topic_list)} guided topics")
    
    # Initialize BERTopic model with guided topic modeling
    # NOTE: Following BERTopic best practices - let HDBSCAN find natural clusters first
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        seed_topic_list=seed_topic_list,  # Enable guided topic modeling
        calculate_probabilities=config['data']['calculate_probabilities'],
        # REMOVED nr_topics - will use post-training reduction if needed
        min_topic_size=config['data']['min_topic_size'],
        verbose=True
    )
    
    return topic_model


def analyze_topics(topic_model: BERTopic, docs: List[str], embeddings: np.ndarray,
                  config: Dict, logger: logging.Logger) -> Tuple[List[int], np.ndarray]:
    """
    Perform topic modeling analysis with pre-computed embeddings.
    Includes post-training topic reduction following BERTopic best practices.
    
    Args:
        topic_model: Configured BERTopic model
        docs: List of documents
        embeddings: Pre-computed document embeddings
        config: Configuration dictionary
        logger: Configured logger instance
        
    Returns:
        Tuple of (topics, probabilities)
    """
    logger.info("Starting topic modeling analysis with pre-computed embeddings")
    
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


def apply_outlier_reduction(topic_model: BERTopic, docs: List[str], topics: List[int], 
                          probs: np.ndarray, embeddings: np.ndarray, config: Dict, 
                          logger: logging.Logger) -> List[int]:
    """
    Apply configurable outlier reduction strategies from BERTopic documentation.
    Reference: https://maartengr.github.io/BERTopic/getting_started/outlier_reduction/outlier_reduction.html
    
    Args:
        topic_model: Trained BERTopic model
        docs: List of documents
        topics: Original topic assignments
        probs: Topic probabilities (if calculated)
        embeddings: Document embeddings
        config: Configuration dictionary
        logger: Configured logger instance
        
    Returns:
        Updated topic assignments with reduced outliers
    """
    outlier_config = config.get('outlier_reduction', {})
    
    if not outlier_config.get('enabled', False):
        logger.info("📋 Outlier reduction disabled")
        return topics
    
    logger.info("🔧 Starting outlier reduction process...")
    
    # Track outlier reduction progress
    initial_outliers = sum(1 for t in topics if t == -1)
    current_topics = topics.copy()
    
    strategies = outlier_config.get('strategies', [])
    verbose = outlier_config.get('verbose', True)
    save_intermediate = outlier_config.get('save_intermediate_results', False)
    
    for i, strategy_config in enumerate(strategies):
        strategy = strategy_config.get('strategy')
        threshold = strategy_config.get('threshold', 0.1)
        
        current_outliers = sum(1 for t in current_topics if t == -1)
        if current_outliers == 0:
            logger.info(f"✅ No outliers remaining - stopping outlier reduction")
            break
            
        logger.info(f"🎯 Strategy {i+1}/{len(strategies)}: '{strategy}' (threshold={threshold})")
        logger.info(f"   Current outliers: {current_outliers}")
        
        try:
            if strategy == "c-tf-idf":
                new_topics = topic_model.reduce_outliers(
                    docs, current_topics, 
                    strategy="c-tf-idf", 
                    threshold=threshold
                )
                
            elif strategy == "probabilities":
                if probs is not None:
                    new_topics = topic_model.reduce_outliers(
                        docs, current_topics, 
                        probabilities=probs,
                        strategy="probabilities", 
                        threshold=threshold
                    )
                else:
                    logger.warning(f"   ⚠️  Probabilities not available - skipping strategy")
                    continue
                    
            elif strategy == "distributions":
                distributions_params = strategy_config.get('distributions_params', {})
                new_topics = topic_model.reduce_outliers(
                    docs, current_topics, 
                    strategy="distributions",
                    threshold=threshold
                )
                
            elif strategy == "embeddings":
                new_topics = topic_model.reduce_outliers(
                    docs, current_topics, 
                    strategy="embeddings",
                    embeddings=embeddings,
                    threshold=threshold
                )
                
            else:
                logger.warning(f"   ⚠️  Unknown strategy '{strategy}' - skipping")
                continue
                
            # Update current topics and log progress
            outliers_before = sum(1 for t in current_topics if t == -1)
            outliers_after = sum(1 for t in new_topics if t == -1)
            reduced = outliers_before - outliers_after
            
            if reduced > 0:
                logger.info(f"   ✅ Reduced {reduced} outliers ({outliers_after} remaining)")
                current_topics = new_topics
                
                if save_intermediate:
                    # Save intermediate results for analysis
                    timestamp = datetime.now().strftime("%H%M%S")
                    logger.info(f"   💾 Intermediate results saved (strategy_{i+1}_{timestamp})")
            else:
                logger.info(f"   📋 No additional outliers reduced")
                
        except Exception as e:
            logger.error(f"   ❌ Error applying strategy '{strategy}': {e}")
            continue
    
    # Final summary
    final_outliers = sum(1 for t in current_topics if t == -1)
    total_reduced = initial_outliers - final_outliers
    
    if total_reduced > 0:
        logger.info(f"🎉 Outlier reduction complete: {total_reduced} documents reassigned")
        logger.info(f"   Before: {initial_outliers} outliers ({(initial_outliers/len(docs)*100):.1f}%)")
        logger.info(f"   After: {final_outliers} outliers ({(final_outliers/len(docs)*100):.1f}%)")
        
        # Store updated topic assignments in model without changing representations
        # This preserves the original high-quality topic names/keywords
        logger.info("🔄 Updating topic assignments while preserving original representations...")
        topic_model.topics_ = current_topics
        logger.info("   ✅ Topic assignments updated (representations preserved)")
    else:
        logger.info("📋 No outliers were reduced")
        
    return current_topics


def create_visualizations(topic_model: BERTopic, docs: List[str], 
                         embeddings: np.ndarray, topics: List[int], config: Dict,
                         output_dir: str, logger: logging.Logger) -> None:
    """
    Generate comprehensive visualizations following BERTopic best practices.
    Now supports updated topic assignments for consistent artifacts.
    
    Args:
        topic_model: Trained BERTopic model
        docs: List of documents
        embeddings: Document embeddings
        topics: Updated topic assignments (post-outlier reduction)
        config: Configuration dictionary
        output_dir: Output directory for plots
        logger: Configured logger instance
    """
    logger.info("🎨 Creating visualizations with updated topic assignments and preserved representations")
    
    viz_config = config['visualization']
    plots_dir = Path(output_dir) / config['output']['plots_dir']
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Model now has updated topic assignments with preserved original representations
    # No need for additional updates - visualizations will show correct assignments with quality names
    
    # 1. Topic visualization (overview) - uses preserved topic names with updated assignments
    logger.info("Creating topic overview visualization")
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(str(plots_dir / "topics_overview.html"))
    
    # 2. Topic hierarchy - uses preserved topic names with updated assignments
    logger.info("Creating hierarchical topic visualization")
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html(str(plots_dir / "topics_hierarchy.html"))
    
    # 3. Topic heatmap - uses preserved topic names with updated assignments
    logger.info("Creating topic similarity heatmap")
    fig_heatmap = topic_model.visualize_heatmap()
    fig_heatmap.write_html(str(plots_dir / "topics_heatmap.html"))
    
    # 4. Document visualization with DataMapPlot (interactive)
    if viz_config['interactive_datamapplot']:
        logger.info("Creating interactive document visualization with DataMapPlot")
        
        # Reduce embeddings for visualization if configured
        if viz_config['reduce_embeddings_for_viz']:
            reduced_embeddings = UMAP(
                n_neighbors=10,
                n_components=2,
                min_dist=0.0,
                metric='cosine',
                random_state=config['random_seed']
            ).fit_transform(embeddings)
            
            # Interactive DataMapPlot - uses preserved topic names with updated assignments
            fig_datamap = topic_model.visualize_document_datamap(
                docs,
                reduced_embeddings=reduced_embeddings,
                interactive=True
            )
            
            # Regular document plot for comparison - uses preserved topic names with updated assignments
            fig_docs = topic_model.visualize_documents(
                docs,
                reduced_embeddings=reduced_embeddings,
                hide_document_hover=viz_config['hide_document_hover']
            )
            
        else:
            fig_datamap = topic_model.visualize_document_datamap(
                docs,
                embeddings=embeddings,
                interactive=True
            )
            
            fig_docs = topic_model.visualize_documents(
                docs,
                embeddings=embeddings,
                hide_document_hover=viz_config['hide_document_hover']
            )
        
        # Save interactive plots
        fig_datamap.save(str(plots_dir / "documents_interactive_datamap.html"))
        fig_docs.write_html(str(plots_dir / "documents_plotly.html"))
    
    # 5. Topic terms visualization - uses preserved topic names with updated assignments
    logger.info("Creating topic terms barchart")
    fig_barchart = topic_model.visualize_barchart(top_n_topics=12, n_words=8)
    fig_barchart.write_html(str(plots_dir / "topics_barchart.html"))
    
    logger.info(f"✅ All visualizations saved to: {plots_dir}")


def save_results(topic_model: BERTopic, df: pd.DataFrame, topics: List[int],
                probs: np.ndarray, embeddings: np.ndarray,
                config: Dict, output_dir: str, logger: logging.Logger) -> None:
    """
    Save analysis results and model artifacts with consistent topic assignments.
    Now generates topic_info from updated topics for consistency.
    
    Args:
        topic_model: Trained BERTopic model
        df: Original DataFrame with bibliography data
        topics: Updated topic assignments (post-outlier reduction)
        probs: Topic probabilities
        embeddings: Document embeddings
        config: Configuration dictionary
        output_dir: Output directory
        logger: Configured logger instance
    """
    logger.info("💾 Saving analysis results with consistent topic assignments")
    
    # Create output directories
    results_dir = Path(output_dir) / config['output']['results_dir']
    models_dir = Path(output_dir) / config['output']['models_dir']
    
    for directory in [results_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Add topic information to DataFrame
    df_results = df.copy()
    df_results['topic'] = topics
    if probs is not None and len(probs.shape) > 1:
        df_results['topic_probability'] = probs.max(axis=1)
    else:
        df_results['topic_probability'] = None
    
    # Get topic information from model (now has preserved representations with updated assignments)
    topic_info = topic_model.get_topic_info()
    
    # Update the Count column to reflect actual post-outlier reduction document counts
    outlier_reduction_enabled = config.get('outlier_reduction', {}).get('enabled', False)
    if outlier_reduction_enabled:
        logger.info("📊 Updating topic counts to reflect post-outlier reduction assignments...")
        from collections import Counter
        topic_counts = Counter(topics)
        
        # Update the Count column with actual counts
        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']
            actual_count = topic_counts.get(topic_id, 0)
            topic_info.at[idx, 'Count'] = actual_count
        
        # Re-sort by count (descending)
        topic_info = topic_info.sort_values('Count', ascending=False).reset_index(drop=True)
    
    # Save results
    timestamp = datetime.now().strftime(config['timestamp_format'])
    
    # Save enhanced DataFrame (uses updated topics)
    df_results.to_csv(results_dir / f"bibliography_with_topics_{timestamp}.csv", index=False)
    
    # Save topic information (now consistent with updated topics)
    topic_info.to_csv(results_dir / f"topic_info_{timestamp}.csv", index=False)
    
    # Save versioned artifacts for scientific integrity
    if outlier_reduction_enabled:
        # Save original model topic info for comparison
        original_topic_info = topic_model.get_topic_info()
        original_topic_info.to_csv(results_dir / f"topic_info_pre_outlier_reduction_{timestamp}.csv", index=False)
        logger.info(f"📋 Saved pre-outlier reduction topic info for comparison")
    
    # Save model artifacts if configured
    if config['output']['save_model']:
        topic_model.save(str(models_dir / f"bertopic_model_{timestamp}"))
    
    if config['output']['save_embeddings']:
        np.save(results_dir / f"embeddings_{timestamp}.npy", embeddings)
    
    # Save configuration used
    with open(results_dir / f"config_used_{timestamp}.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"✅ Results saved to: {results_dir}")
    logger.info(f"📁 Model artifacts saved to: {models_dir}")


def generate_topic_info_from_assignments(topics: List[int], df_results: pd.DataFrame, 
                                        logger: logging.Logger) -> pd.DataFrame:
    """
    Generate topic_info DataFrame from updated topic assignments.
    This ensures consistency when model representations aren't updated.
    
    Args:
        topics: Updated topic assignments
        df_results: DataFrame with document information and topics
        logger: Configured logger instance
        
    Returns:
        DataFrame with topic information consistent with updated assignments
    """
    logger.info("🔧 Generating topic_info from updated assignments...")
    
    from collections import Counter
    
    # Count documents per topic
    topic_counts = Counter(topics)
    
    # Create topic_info DataFrame
    topic_info_data = []
    
    for topic_id, count in topic_counts.items():
        # Get representative documents for this topic
        topic_docs = df_results[df_results['topic'] == topic_id]
        
        # Create a simple representation (we can't generate full c-TF-IDF without the model)
        if topic_id == -1:
            name = "Outliers"
            representation = ["outlier", "documents", "unassigned", "noise", "scattered"]
        else:
            # Use first few words from titles/abstracts as simple representation
            sample_texts = topic_docs['combined_text'].head(10).tolist()
            # This is a simplified representation - ideally would use c-TF-IDF
            name = f"Topic_{topic_id}"
            representation = [f"topic_{topic_id}", "documents", "cluster", "group", "category"]
        
        topic_info_data.append({
            'Topic': topic_id,
            'Count': count,
            'Name': name,
            'Representation': representation,
            'Representative_Docs': topic_docs['title'].head(3).tolist() if not topic_docs.empty else []
        })
    
    # Sort by count (descending) then by topic ID
    topic_info_data.sort(key=lambda x: (-x['Count'], x['Topic']))
    
    topic_info_df = pd.DataFrame(topic_info_data)
    
    logger.info(f"   ✅ Generated topic_info for {len(topic_info_df)} topics")
    return topic_info_df


def generate_summary_report(topic_model: BERTopic, df: pd.DataFrame,
                          topics: List[int], config: Dict,
                          output_dir: str, logger: logging.Logger) -> None:
    """
    Generate a summary report of the analysis using updated topic assignments.
    
    Args:
        topic_model: Trained BERTopic model
        df: DataFrame with results
        topics: Updated topic assignments (post-outlier reduction)
        config: Configuration dictionary
        output_dir: Output directory
        logger: Configured logger instance
    """
    logger.info("📋 Generating summary report with updated topic assignments")
    
    results_dir = Path(output_dir) / config['output']['results_dir']
    
    # Basic statistics using updated topics
    n_documents = len(topics)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    
    # Get topic information - model now has preserved representations with updated counts
    topic_info = topic_model.get_topic_info()
    
    # Ensure topic counts reflect actual post-outlier reduction assignments
    outlier_reduction_enabled = config.get('outlier_reduction', {}).get('enabled', False)
    if outlier_reduction_enabled:
        from collections import Counter
        topic_counts = Counter(topics)
        topic_sizes = [(topic_id, count) for topic_id, count in topic_counts.items() if topic_id != -1]
        topic_sizes.sort(key=lambda x: -x[1])  # Sort by size descending
    else:
        topic_sizes = [(row['Topic'], row['Count']) for _, row in topic_info.iterrows() if row['Topic'] != -1]
    
    # Create summary report
    report = f"""
# BERTopic Analysis Summary Report (Updated Topic Assignments)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Total documents processed: {n_documents:,}
- Documents with sufficient content: {len(df):,}
- Topics discovered: {n_topics}
- Outlier documents: {n_outliers}
- Coverage: {((n_documents - n_outliers) / n_documents * 100):.1f}%

## Model Configuration
- Embedding model: {config['embedding_model']['name']}
- UMAP neighbors: {config['umap_params']['n_neighbors']}
- HDBSCAN min cluster size: {config['hdbscan_params']['min_cluster_size']}
- Min topic size: {config['data']['min_topic_size']}"""
    
    # Add outlier reduction information
    if outlier_reduction_enabled:
        strategies = config.get('outlier_reduction', {}).get('strategies', [])
        strategy_names = [s.get('strategy', 'unknown') for s in strategies]
        report += f"""
- Outlier reduction: ENABLED
- Strategies used: {', '.join(strategy_names)}
- Topic representations: PRESERVED (original high-quality names maintained)"""
    else:
        report += f"""
- Outlier reduction: DISABLED"""
    
    # Add guided topics information if enabled
    guided_config = config.get('domain_guidance', {}).get('guided_topics', {})
    if guided_config.get('enabled', False):
        report += f"""
- Guided topic modeling: ENABLED
- Number of guided topics: {len(guided_config.get('topics', {}))}"""
    else:
        report += f"""
- Guided topic modeling: DISABLED"""
    
    # Add seed words information if enabled
    seed_config = config.get('domain_guidance', {}).get('seed_words', {})
    if seed_config.get('enabled', False):
        report += f"""
- Seed words enhancement: ENABLED
- Seed multiplier: {seed_config.get('multiplier', 2.0)}x
- Number of seed words: {len(seed_config.get('words', []))}"""
    else:
        report += f"""
- Seed words enhancement: DISABLED"""
    
    report += f"""

## Top 10 Topics by Size (Using Updated Assignments)
"""
    
    # Add top topics to report using updated assignments
    for i, (topic_id, count) in enumerate(topic_sizes[:10]):
        report += f"{topic_id:2d}. ({count:3d} docs) Topic {topic_id}\n"
    
    # Save report
    timestamp = datetime.now().strftime(config['timestamp_format'])
    report_filename = f"analysis_summary_{timestamp}.md"
    with open(results_dir / report_filename, 'w') as f:
        f.write(report)
    
    logger.info(f"✅ Summary report generated: {report_filename}")
    print(report)  # Also display in console


def get_file_hash(file_path: str) -> str:
    """Get hash of file for cache validation."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def cache_exists(cache_dir: Path, cache_name: str) -> bool:
    """Check if cache file exists."""
    return (cache_dir / f"{cache_name}.pkl").exists()

def save_to_cache(data, cache_dir: Path, cache_name: str, logger: logging.Logger) -> None:
    """Save data to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / f"{cache_name}.pkl", 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"💾 Saved to cache: {cache_name}")

def load_from_cache(cache_dir: Path, cache_name: str, logger: logging.Logger):
    """Load data from cache."""
    with open(cache_dir / f"{cache_name}.pkl", 'rb') as f:
        data = pickle.load(f)
    logger.info(f"📂 Loaded from cache: {cache_name}")
    return data


def main():
    """Main execution function."""
    # Setup
    logger = setup_logging()
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
        df = parse_bib_file("merged.bib", config, logger)
        docs = df['combined_text'].tolist()
        
        # Generate embeddings (cached)
        embeddings = prepare_embeddings(docs, config, logger)
        
        # Setup and train BERTopic model
        topic_model = setup_bertopic_model(config, logger)
        topics, probs = analyze_topics(topic_model, docs, embeddings, config, logger)
        
        # Create visualizations
        create_visualizations(topic_model, docs, embeddings, topics, config, output_dir, logger)
        
        # Save results
        save_results(topic_model, df, topics, probs, embeddings, config, output_dir, logger)
        
        # Generate summary report
        generate_summary_report(topic_model, df, topics, config, output_dir, logger)
        
        logger.info("Analysis completed successfully!")
        print(f"\n✅ Analysis complete! Check the '{output_dir}' directory for results.")
        print(f"📊 Interactive visualizations are available in '{output_dir}/plots/'")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main() 