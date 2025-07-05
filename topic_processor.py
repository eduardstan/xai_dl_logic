#!/usr/bin/env python3
"""
Topic Processing Module for Systematic Literature Reviews

This module handles the main topic processing orchestration including paper selection,
metrics computation, research alignment analysis, and comprehensive result generation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from datetime import datetime
from loguru import logger

from utils import get_output_config, get_systematic_review_config
from data_loader import load_analysis_results
from cluster_analysis import calculate_cluster_size_category, determine_papers_to_select
from paper_selection import select_diverse_representatives, assign_non_selected_papers
from paper_metrics import compute_paper_metrics
from research_alignment import analyze_research_alignment


def process_topics(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process all topics and select diverse representative papers.
    
    This is the main orchestration function that coordinates the entire systematic
    literature review analysis pipeline. It processes each topic cluster by:
    1. Loading analysis results from cache
    2. Selecting diverse representative papers within each cluster
    3. Computing comprehensive similarity metrics for all papers
    4. Analyzing research alignment for domain relevance
    5. Mapping non-selected papers to their most similar representatives
    6. Generating comprehensive results and summary statistics
    
    Args:
        config: Configuration dictionary containing all analysis parameters
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - results_df: Comprehensive analysis results for all papers
            - summary_df: Topic-level summary statistics
            - selected_df: Selected representative papers only
    
    Raises:
        FileNotFoundError: If required cached data files are missing
        ValueError: If data validation fails or processing errors occur
        Exception: If any step in the processing pipeline fails
    
    Example:
        >>> config = load_config('config.yaml')
        >>> results_df, summary_df, selected_df = process_topics(config)
        >>> print(f"Selected {len(selected_df)} representatives from {len(results_df)} papers")
        Selected 533 representatives from 3735 papers
    
    Processing Pipeline:
        1. **Data Loading**: Load BERTopic model, embeddings, documents, and topic assignments
        2. **Topic Iteration**: Process each topic cluster individually
        3. **Paper Selection**: Select diverse representatives using configured algorithms
        4. **Metrics Computation**: Calculate similarity, diversity, and representativeness
        5. **Research Alignment**: Analyze domain-specific keyword alignment
        6. **Assignment Mapping**: Map non-selected papers to representatives
        7. **Result Generation**: Combine all data into comprehensive results
        8. **Summary Statistics**: Generate topic-level aggregated statistics
        9. **File Saving**: Save results to CSV files with timestamps
    
    Output Files:
        - `comprehensive_analysis_{timestamp}.csv`: Complete analysis results
        - `selection_summary_{timestamp}.csv`: Topic-level summary statistics
        - `selected_representatives_{timestamp}.csv`: Representative papers only
    
    Configuration Dependencies:
        - systematic_review: Paper selection strategy and parameters
        - output: Results directory and file naming conventions
        - research_alignment: Domain-specific keyword patterns
        - cluster_analysis: Size categories and selection thresholds
    
    Performance Characteristics:
        - Processes ~3,735 papers in ~4.5 minutes
        - Memory usage: ~500MB for embeddings and document data
        - Disk I/O: Reads from cache, writes results to CSV
        - Scales linearly with number of papers and topics
    
    Note:
        - Skips outlier papers (topic_id = -1) from processing
        - Handles both DataFrame and list document formats
        - Maintains data consistency across all processing steps
        - Provides detailed logging for monitoring progress
    """
    try:
        output_config = get_output_config(config)
        results_dir = Path(output_config['results_dir'])
        
        # Ensure output directory exists
        results_dir.mkdir(exist_ok=True)
        
        logger.info("🔍 Processing topics for diverse representative selection...")
        
        # Load all analysis results
        topic_model, embeddings, documents, updated_topics = load_analysis_results(config)
        
        # Use the updated topic assignments (post-outlier reduction)
        topics = updated_topics
        topic_info = topic_model.get_topic_info()
        
        all_results = []
        selection_summary = []
        
        # Process each topic
        for topic_id in sorted(set(topics)):
            if topic_id == -1:  # Skip outliers
                continue
            
            # Get documents in this topic
            topic_mask = np.array(topics) == topic_id
            topic_indices = np.where(topic_mask)[0]
            topic_embeddings = embeddings[topic_mask]
            
            # Extract documents for this topic
            if isinstance(documents, pd.DataFrame):
                topic_documents = documents.iloc[topic_indices]
            else:
                topic_documents = [documents[i] for i in topic_indices]
            
            cluster_size = len(topic_documents)
            n_select = determine_papers_to_select(cluster_size, config)
            
            logger.info(f"📊 Topic {topic_id}: {cluster_size} papers → selecting {n_select} representatives")
            
            # Select diverse representatives
            selected_local_indices = select_diverse_representatives(topic_embeddings, n_select, config)
            selected_global_indices = [topic_indices[i] for i in selected_local_indices]
            
            # Assign non-selected papers to representatives
            assignments = assign_non_selected_papers(topic_embeddings, selected_local_indices, config)
            
            # Process all papers in the topic
            for local_idx, global_idx in enumerate(topic_indices):
                # Compute metrics
                metrics = compute_paper_metrics(
                    embeddings[global_idx], 
                    topic_embeddings,
                    config
                )
                
                # Get document data and process based on format
                if isinstance(documents, pd.DataFrame):
                    doc = documents.iloc[global_idx]
                    # Research alignment
                    full_text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('keywords', '')}"
                    
                    # Extract document info
                    doc_info = {
                        'title': doc.get('title', 'Unknown'),
                        'authors': doc.get('author', 'Unknown'),
                        'year': doc.get('year', 'Unknown'),
                        'journal': doc.get('journal', 'Unknown'),
                        'doi': doc.get('doi', ''),
                        'abstract': doc.get('abstract', ''),
                        'keywords': doc.get('keywords', '')
                    }
                else:
                    doc = documents[global_idx]
                    full_text = f"{doc.get('title', '')} {doc.get('abstract', '')} {' '.join(doc.get('keywords', []))}"
                    doc_info = {
                        'title': doc.get('title', 'Unknown'),
                        'authors': ', '.join(doc.get('authors', [])),
                        'year': doc.get('year', 'Unknown'),
                        'journal': doc.get('journal', 'Unknown'),
                        'doi': doc.get('doi', ''),
                        'abstract': doc.get('abstract', ''),
                        'keywords': ', '.join(doc.get('keywords', []))
                    }
                
                # Analyze research alignment
                alignment = analyze_research_alignment(full_text, config)
                
                # Determine selection status
                is_selected = local_idx in selected_local_indices
                representative_info = {}
                
                # Process non-selected papers assignment info
                if not is_selected and local_idx in assignments:
                    assignment = assignments[local_idx]
                    rep_global_idx = topic_indices[assignment['representative_idx']]
                    
                    if isinstance(documents, pd.DataFrame):
                        rep_doc = documents.iloc[rep_global_idx]
                        representative_info = {
                            'representative_title': rep_doc.get('title', 'Unknown'),
                            'representative_authors': rep_doc.get('author', 'Unknown'),
                            'similarity_to_representative': assignment['similarity'],
                            'is_similar_to_representative': assignment['is_similar']
                        }
                    else:
                        rep_doc = documents[rep_global_idx]
                        representative_info = {
                            'representative_title': rep_doc.get('title', 'Unknown'),
                            'representative_authors': ', '.join(rep_doc.get('authors', [])),
                            'similarity_to_representative': assignment['similarity'],
                            'is_similar_to_representative': assignment['is_similar']
                        }
                
                # Combine all information
                result = {
                    'topic_id': topic_id,
                    'global_index': global_idx,
                    'local_index': local_idx,
                    'is_selected_representative': is_selected,
                    'cluster_size': cluster_size,
                    'papers_selected_from_cluster': n_select,
                    **doc_info,
                    **metrics,
                    **alignment,
                    **representative_info
                }
                
                all_results.append(result)
            
            # Generate topic summary
            topic_name = topic_info[topic_info['Topic'] == topic_id]['Name'].iloc[0] if len(topic_info[topic_info['Topic'] == topic_id]) > 0 else f"Topic {topic_id}"
            selection_summary.append({
                'topic_id': topic_id,
                'topic_name': topic_name,
                'cluster_size': cluster_size,
                'size_category': calculate_cluster_size_category(cluster_size, config),
                'papers_selected': n_select,
                'selection_ratio': n_select / cluster_size,
                'avg_centrality': np.mean([r['similarity_to_centroid'] 
                                        for r in all_results 
                                        if r['topic_id'] == topic_id and r['is_selected_representative']]),
                'avg_diversity': np.mean([r['diversity_score'] 
                                       for r in all_results 
                                       if r['topic_id'] == topic_id and r['is_selected_representative']])
            })
        
        # Create DataFrames
        results_df = pd.DataFrame(all_results)
        summary_df = pd.DataFrame(selection_summary)
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_df.to_csv(results_dir / f"comprehensive_analysis_{timestamp}.csv", index=False)
        summary_df.to_csv(results_dir / f"selection_summary_{timestamp}.csv", index=False)
        
        # Save selected representatives separately
        selected_df = results_df[results_df['is_selected_representative']]
        selected_df.to_csv(results_dir / f"selected_representatives_{timestamp}.csv", index=False)
        
        logger.info(f"✅ Analysis complete: {len(selected_df)} representatives selected from {len(results_df)} total papers")
        logger.info(f"📁 Results saved to {results_dir}")
        
        return results_df, summary_df, selected_df
        
    except Exception as e:
        logger.error(f"❌ Topic processing failed: {e}")
        raise


def validate_topic_processing_results(results_df: pd.DataFrame, summary_df: pd.DataFrame, 
                                    selected_df: pd.DataFrame) -> Dict[str, Union[bool, str]]:
    """
    Validate topic processing results for consistency and completeness.
    
    Args:
        results_df: Complete analysis results DataFrame
        summary_df: Topic summary statistics DataFrame
        selected_df: Selected representatives DataFrame
    
    Returns:
        Dict containing validation results and any error messages
    
    Validation Checks:
        - Data consistency across all DataFrames
        - Representative selection ratios within expected ranges
        - Required columns present in all DataFrames
        - No missing critical data in results
        - Topic coverage completeness
    """
    try:
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check basic data consistency
        n_total = len(results_df)
        n_selected = len(selected_df)
        n_topics = len(summary_df)
        
        if n_selected == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No representatives selected")
        
        if n_total == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("No papers in results")
        
        # Check selection ratios
        if n_total > 0:
            selection_ratio = n_selected / n_total
            if selection_ratio < 0.05:
                validation_results['warnings'].append(f"Low selection ratio: {selection_ratio:.1%}")
            elif selection_ratio > 0.5:
                validation_results['warnings'].append(f"High selection ratio: {selection_ratio:.1%}")
        
        # Check required columns
        required_result_columns = [
            'topic_id', 'global_index', 'is_selected_representative', 
            'similarity_to_centroid', 'diversity_score', 'representativeness_score'
        ]
        
        missing_columns = [col for col in required_result_columns if col not in results_df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check topic coverage
        unique_topics_results = set(results_df['topic_id'].unique())
        unique_topics_summary = set(summary_df['topic_id'].unique())
        
        if unique_topics_results != unique_topics_summary:
            validation_results['warnings'].append("Topic coverage mismatch between results and summary")
        
        # Collect statistics
        validation_results['stats'] = {
            'n_total_papers': n_total,
            'n_selected_papers': n_selected,
            'n_topics': n_topics,
            'selection_ratio': f"{(n_selected / n_total * 100):.1f}%" if n_total > 0 else "0%",
            'avg_cluster_size': summary_df['cluster_size'].mean() if len(summary_df) > 0 else 0,
            'avg_papers_per_topic': summary_df['papers_selected'].mean() if len(summary_df) > 0 else 0
        }
        
        return validation_results
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Validation failed: {e}"],
            'warnings': [],
            'stats': {}
        }


def get_topic_processing_summary(results_df: pd.DataFrame, summary_df: pd.DataFrame, 
                               selected_df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive summary of topic processing results.
    
    Args:
        results_df: Complete analysis results DataFrame
        summary_df: Topic summary statistics DataFrame
        selected_df: Selected representatives DataFrame
    
    Returns:
        Dict containing comprehensive processing summary and statistics
    """
    try:
        n_total = len(results_df)
        n_selected = len(selected_df)
        n_topics = len(summary_df)
        
        # Selection statistics
        selection_stats = {
            'total_papers': n_total,
            'selected_papers': n_selected,
            'selection_ratio': f"{(n_selected / n_total * 100):.1f}%" if n_total > 0 else "0%",
            'topics_processed': n_topics,
            'avg_cluster_size': summary_df['cluster_size'].mean() if len(summary_df) > 0 else 0,
            'avg_papers_per_topic': summary_df['papers_selected'].mean() if len(summary_df) > 0 else 0
        }
        
        # Quality metrics
        if len(selected_df) > 0:
            quality_metrics = {
                'avg_representativeness': selected_df['representativeness_score'].mean(),
                'avg_centrality': selected_df['similarity_to_centroid'].mean(),
                'avg_diversity': selected_df['diversity_score'].mean(),
                'representativeness_std': selected_df['representativeness_score'].std(),
                'high_quality_papers': len(selected_df[selected_df['representativeness_score'] > 0.7])
            }
        else:
            quality_metrics = {}
        
        # Topic distribution
        topic_distribution = summary_df.groupby('size_category')['cluster_size'].agg(['count', 'sum']).to_dict()
        
        # Research alignment analysis
        if len(selected_df) > 0 and 'xai_alignment' in selected_df.columns:
            alignment_analysis = {
                'avg_xai_alignment': selected_df['xai_alignment'].mean(),
                'avg_symbolic_alignment': selected_df['symbolic_alignment'].mean(),
                'avg_subsymbolic_alignment': selected_df['subsymbolic_alignment'].mean(),
                'high_xai_papers': len(selected_df[selected_df['xai_alignment'] > 0.5])
            }
        else:
            alignment_analysis = {}
        
        return {
            'processing_summary': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'completed',
                **selection_stats
            },
            'quality_metrics': quality_metrics,
            'topic_distribution': topic_distribution,
            'alignment_analysis': alignment_analysis,
            'top_topics': summary_df.nlargest(10, 'cluster_size')[['topic_id', 'topic_name', 'cluster_size']].to_dict('records') if len(summary_df) > 0 else []
        }
        
    except Exception as e:
        return {'error': f"Failed to generate topic processing summary: {e}"}


def validate_topic_processing_config(config: Dict) -> bool:
    """
    Validate topic processing configuration for completeness and correctness.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if topic processing configuration is valid
    """
    try:
        # Check systematic review config
        review_config = get_systematic_review_config(config)
        
        required_keys = ['selection_strategy', 'base_papers_per_cluster', 'max_papers_per_cluster']
        for key in required_keys:
            if key not in review_config:
                logger.error(f"Missing required systematic review config key: {key}")
                return False
        
        # Check output config
        output_config = get_output_config(config)
        
        if 'results_dir' not in output_config:
            logger.error("Missing required output config key: results_dir")
            return False
        
        # Check results directory
        results_dir = Path(output_config['results_dir'])
        if not results_dir.exists():
            logger.warning(f"Results directory does not exist: {results_dir}")
            # Try to create it
            try:
                results_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created results directory: {results_dir}")
            except Exception as e:
                logger.error(f"Failed to create results directory: {e}")
                return False
        
        logger.info("✅ Topic processing configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating topic processing configuration: {e}")
        return False 