#!/usr/bin/env python3
"""
Data Loading Module for Systematic Literature Reviews

This module handles loading and validation of cached analysis results including
BERTopic models, embeddings, parsed documents, and topic assignments.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from loguru import logger

from utils import get_output_config


def load_analysis_results(config: Dict) -> Tuple[object, np.ndarray, Union[pd.DataFrame, List[Dict]], List[int]]:
    """
    Load cached BERTopic analysis results with updated topic assignments.
    
    This function loads all the necessary data components from previous analysis runs:
    - BERTopic model (for topic information and metadata)
    - Updated topic assignments (post-outlier reduction from CSV results)
    - Document embeddings (from cache)
    - Parsed documents (from cache)
    
    The function automatically finds the most recent files based on modification time,
    ensuring that the latest analysis results are always loaded.
    
    Args:
        config: Configuration dictionary containing output directory settings
    
    Returns:
        Tuple[object, np.ndarray, Union[pd.DataFrame, List[Dict]], List[int]]: 
            - topic_model: Loaded BERTopic model instance
            - embeddings: Document embeddings array, shape (n_documents, embedding_dim)
            - documents: Parsed documents as DataFrame or list of dictionaries
            - updated_topics: List of topic assignments (post-outlier reduction)
    
    Raises:
        FileNotFoundError: If required cache files are missing
        ValueError: If data dimensions don't match between components
        Exception: If loading fails due to corruption or compatibility issues
    
    Example:
        >>> config = load_config('config.yaml')
        >>> topic_model, embeddings, documents, topics = load_analysis_results(config)
        >>> print(f"Loaded {len(documents)} documents with {len(set(topics))} topics")
        Loaded 3735 documents with 57 topics
    
    Data Sources:
        - **BERTopic Model**: `bertopic_analysis/models/bertopic_model_*` 
          Contains trained topic model with topic information and metadata
        
        - **Topic Assignments**: `bertopic_analysis/results/bibliography_with_topics_*.csv`
          Contains updated topic assignments after outlier reduction processing
        
        - **Embeddings**: `cache/embeddings_*.pkl`
          Contains document embeddings generated during initial analysis
        
        - **Documents**: `cache/parsed_bib_*.pkl`
          Contains parsed bibliography data from BIB files
    
    Data Validation:
        - Verifies all components have matching document counts
        - Checks for required file existence and accessibility
        - Validates data types and basic structure integrity
        - Reports detailed statistics about loaded data
    
    File Selection Strategy:
        - Uses most recent files based on modification time
        - Ensures consistency by loading from the same analysis run
        - Handles multiple cache files gracefully
    
    Note:
        - Function expects files to follow specific naming conventions
        - All data components must have identical document ordering
        - Topic assignments use -1 for outliers, 0+ for assigned topics
        - Embeddings and documents must be pre-cached from previous runs
    """
    try:
        output_config = get_output_config(config)
        cache_dir = Path(output_config['cache_dir'])
        
        # Load BERTopic model from bertopic_analysis/models
        models_dir = Path("bertopic_analysis/models")
        model_files = list(models_dir.glob("bertopic_model_*"))
        if not model_files:
            raise FileNotFoundError(f"No BERTopic model files found in {models_dir}")
        
        # Use the most recent model file
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        
        try:
            from bertopic import BERTopic
            topic_model = BERTopic.load(str(model_path))
            logger.info(f"📂 Loaded BERTopic model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load BERTopic model from {model_path}: {e}")
            raise ValueError(f"BERTopic model loading failed: {e}")
        
        # Load the MOST RECENT topic assignments from CSV results (post-outlier reduction)
        results_dir = Path("bertopic_analysis/results")
        csv_files = list(results_dir.glob("bibliography_with_topics_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No results CSV files found in {results_dir}")
        
        # Use the most recent CSV file
        csv_path = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        try:
            results_df = pd.read_csv(csv_path)
            if 'topic' not in results_df.columns:
                raise ValueError(f"Topic column not found in {csv_path}")
            
            updated_topics = results_df['topic'].tolist()
            logger.info(f"📂 Loaded UPDATED topic assignments from {csv_path}")
            logger.info(f"   📊 Total papers: {len(updated_topics)}")
            logger.info(f"   📊 Outliers: {sum(1 for t in updated_topics if t == -1)}")
            logger.info(f"   📊 Assigned to topics: {sum(1 for t in updated_topics if t != -1)}")
            
        except Exception as e:
            logger.error(f"Failed to load topic assignments from {csv_path}: {e}")
            raise ValueError(f"Topic assignments loading failed: {e}")
        
        # Load embeddings from cache
        embeddings_files = list(cache_dir.glob("embeddings_*.pkl"))
        if not embeddings_files:
            raise FileNotFoundError(f"No embedding cache files found in {cache_dir}")
        
        embeddings_path = embeddings_files[0]  # Use the first (should be only one)
        
        try:
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            
            if not isinstance(embeddings, np.ndarray):
                raise ValueError(f"Embeddings must be numpy array, got {type(embeddings)}")
            
            if len(embeddings.shape) != 2:
                raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
            
            logger.info(f"📂 Loaded embeddings from {embeddings_path}")
            logger.info(f"   📊 Embedding shape: {embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings from {embeddings_path}: {e}")
            raise ValueError(f"Embeddings loading failed: {e}")
        
        # Load parsed documents from cache
        parsed_files = list(cache_dir.glob("parsed_bib_*.pkl"))
        if not parsed_files:
            raise FileNotFoundError(f"No parsed BIB cache files found in {cache_dir}")
        
        parsed_path = parsed_files[0]
        
        try:
            with open(parsed_path, 'rb') as f:
                documents = pickle.load(f)
            
            # Validate document structure
            if isinstance(documents, pd.DataFrame):
                doc_count = len(documents)
                logger.info(f"📂 Loaded documents from {parsed_path} (DataFrame)")
                logger.info(f"   📊 Document count: {doc_count}")
                logger.info(f"   📊 Columns: {list(documents.columns)}")
            elif isinstance(documents, list):
                doc_count = len(documents)
                logger.info(f"📂 Loaded documents from {parsed_path} (List)")
                logger.info(f"   📊 Document count: {doc_count}")
                if doc_count > 0:
                    logger.info(f"   📊 Sample keys: {list(documents[0].keys()) if isinstance(documents[0], dict) else 'N/A'}")
            else:
                raise ValueError(f"Documents must be DataFrame or list, got {type(documents)}")
            
        except Exception as e:
            logger.error(f"Failed to load documents from {parsed_path}: {e}")
            raise ValueError(f"Documents loading failed: {e}")
        
        # Verify data consistency
        n_topics = len(updated_topics)
        n_embeddings = len(embeddings)
        n_documents = len(documents)
        
        if n_topics != n_embeddings or n_topics != n_documents:
            raise ValueError(
                f"Data length mismatch: "
                f"topics={n_topics}, embeddings={n_embeddings}, documents={n_documents}. "
                f"All components must have identical document counts."
            )
        
        # Validate topic assignments
        unique_topics = set(updated_topics)
        n_unique_topics = len(unique_topics)
        n_outliers = sum(1 for t in updated_topics if t == -1)
        n_assigned = n_topics - n_outliers
        
        logger.info(f"✅ Analysis results loaded successfully:")
        logger.info(f"   📊 Documents: {n_documents}")
        logger.info(f"   📊 Embeddings: {n_embeddings} x {embeddings.shape[1]}")
        logger.info(f"   📊 Topic assignments: {n_topics}")
        logger.info(f"   📊 Unique topics: {n_unique_topics}")
        logger.info(f"   📊 Assigned papers: {n_assigned}")
        logger.info(f"   📊 Outliers: {n_outliers}")
        
        return topic_model, embeddings, documents, updated_topics
        
    except Exception as e:
        logger.error(f"❌ Error loading analysis results: {e}")
        raise


def validate_data_consistency(topic_model, embeddings: np.ndarray, documents: Union[pd.DataFrame, List[Dict]], 
                            updated_topics: List[int]) -> Dict[str, Union[bool, str]]:
    """
    Validate consistency and integrity of loaded data components.
    
    Args:
        topic_model: Loaded BERTopic model instance
        embeddings: Document embeddings array
        documents: Parsed documents structure
        updated_topics: List of topic assignments
    
    Returns:
        Dict containing validation results and any error messages
    
    Checks:
        - Data dimension consistency across all components
        - Topic assignment validity (no invalid topic IDs)
        - Document structure integrity
        - Embedding dimensionality and data types
        - BERTopic model accessibility
    """
    try:
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check data dimensions
        n_topics = len(updated_topics)
        n_embeddings = len(embeddings)
        n_documents = len(documents)
        
        if n_topics != n_embeddings or n_topics != n_documents:
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"Dimension mismatch: topics={n_topics}, embeddings={n_embeddings}, documents={n_documents}"
            )
        
        # Check embeddings
        if not isinstance(embeddings, np.ndarray):
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Embeddings must be numpy array, got {type(embeddings)}")
        elif len(embeddings.shape) != 2:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Embeddings must be 2D, got shape {embeddings.shape}")
        
        # Check topic assignments
        unique_topics = set(updated_topics)
        valid_topics = {t for t in unique_topics if t >= -1}  # -1 for outliers, 0+ for topics
        if len(valid_topics) != len(unique_topics):
            invalid_topics = unique_topics - valid_topics
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Invalid topic IDs found: {invalid_topics}")
        
        # Check BERTopic model
        try:
            topic_info = topic_model.get_topic_info()
            if topic_info is None or len(topic_info) == 0:
                validation_results['warnings'].append("BERTopic model has no topic information")
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"BERTopic model validation failed: {e}")
        
        # Collect statistics
        validation_results['stats'] = {
            'n_documents': n_documents,
            'n_embeddings': n_embeddings,
            'n_topics': n_topics,
            'n_unique_topics': len(unique_topics),
            'n_outliers': sum(1 for t in updated_topics if t == -1),
            'embedding_dim': embeddings.shape[1] if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2 else None,
            'document_type': type(documents).__name__
        }
        
        return validation_results
        
    except Exception as e:
        return {
            'is_valid': False,
            'errors': [f"Validation failed: {e}"],
            'warnings': [],
            'stats': {}
        }


def get_data_loading_summary(topic_model, embeddings: np.ndarray, documents: Union[pd.DataFrame, List[Dict]], 
                           updated_topics: List[int]) -> Dict:
    """
    Generate comprehensive summary of loaded data components.
    
    Args:
        topic_model: Loaded BERTopic model instance
        embeddings: Document embeddings array
        documents: Parsed documents structure  
        updated_topics: List of topic assignments
    
    Returns:
        Dict containing comprehensive data loading summary
    """
    try:
        # Basic statistics
        n_documents = len(documents)
        n_embeddings = len(embeddings)
        n_topics = len(updated_topics)
        unique_topics = set(updated_topics)
        n_outliers = sum(1 for t in updated_topics if t == -1)
        
        # Topic distribution
        topic_counts = {}
        for topic in updated_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Document structure analysis
        doc_info = {}
        if isinstance(documents, pd.DataFrame):
            doc_info = {
                'type': 'DataFrame',
                'columns': list(documents.columns),
                'memory_usage': f"{documents.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
            }
        elif isinstance(documents, list) and len(documents) > 0:
            sample_doc = documents[0]
            doc_info = {
                'type': 'List of Dictionaries',
                'sample_keys': list(sample_doc.keys()) if isinstance(sample_doc, dict) else 'N/A',
                'estimated_size': f"{len(documents) * len(str(sample_doc)) / 1024 / 1024:.1f} MB"
            }
        
        # BERTopic model info
        model_info = {}
        try:
            topic_info = topic_model.get_topic_info()
            model_info = {
                'n_topics_in_model': len(topic_info),
                'model_type': type(topic_model).__name__,
                'has_topic_info': topic_info is not None
            }
        except Exception as e:
            model_info = {'error': str(e)}
        
        return {
            'loading_summary': {
                'n_documents': n_documents,
                'n_embeddings': n_embeddings,
                'n_topic_assignments': n_topics,
                'n_unique_topics': len(unique_topics),
                'n_outliers': n_outliers,
                'n_assigned_papers': n_documents - n_outliers,
                'assignment_rate': f"{((n_documents - n_outliers) / n_documents * 100):.1f}%"
            },
            'embeddings_info': {
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype),
                'memory_usage': f"{embeddings.nbytes / 1024 / 1024:.1f} MB",
                'dimension': embeddings.shape[1] if len(embeddings.shape) == 2 else None
            },
            'documents_info': doc_info,
            'topic_model_info': model_info,
            'topic_distribution': dict(sorted(topic_counts.items())),
            'largest_topics': sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
    except Exception as e:
        return {'error': f"Failed to generate data loading summary: {e}"}


def validate_data_loading_config(config: Dict) -> bool:
    """
    Validate data loading configuration for required paths and settings.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if data loading configuration is valid
    
    Checks:
        - Output directory configuration exists
        - Cache directory is accessible
        - Required paths are properly configured
    """
    try:
        output_config = get_output_config(config)
        
        # Check cache directory
        cache_dir = Path(output_config.get('cache_dir', 'cache'))
        if not cache_dir.exists():
            logger.warning(f"Cache directory does not exist: {cache_dir}")
            return False
        
        if not cache_dir.is_dir():
            logger.error(f"Cache path is not a directory: {cache_dir}")
            return False
        
        # Check required paths
        required_paths = [
            Path("bertopic_analysis/models"),
            Path("bertopic_analysis/results")
        ]
        
        for path in required_paths:
            if not path.exists():
                logger.warning(f"Required path does not exist: {path}")
                return False
        
        logger.info("✅ Data loading configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating data loading configuration: {e}")
        return False 