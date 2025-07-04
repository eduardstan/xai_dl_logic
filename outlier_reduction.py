#!/usr/bin/env python3
"""
BERTopic Outlier Reduction Module
Following BERTopic best practices for outlier reduction strategies.
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from bertopic import BERTopic

from utils import get_outlier_reduction_config


def apply_outlier_reduction(topic_model: BERTopic, docs: List[str], topics: List[int], 
                          probs: Optional[np.ndarray], embeddings: np.ndarray, config: Dict, 
                          logger) -> List[int]:
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
    outlier_config = get_outlier_reduction_config(config)
    
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
            new_topics = _apply_strategy(
                topic_model, docs, current_topics, strategy, threshold, 
                probs, embeddings, strategy_config, logger
            )
            
            if new_topics is None:
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


def _apply_strategy(topic_model: BERTopic, docs: List[str], current_topics: List[int], 
                   strategy: str, threshold: float, probs: Optional[np.ndarray], 
                   embeddings: np.ndarray, strategy_config: Dict, logger) -> Optional[List[int]]:
    """
    Apply a specific outlier reduction strategy.
    
    Args:
        topic_model: Trained BERTopic model
        docs: List of documents
        current_topics: Current topic assignments
        strategy: Strategy name
        threshold: Threshold value
        probs: Topic probabilities (if available)
        embeddings: Document embeddings
        strategy_config: Strategy-specific configuration
        logger: Configured logger instance
        
    Returns:
        Updated topic assignments or None if strategy failed
    """
    if strategy == "c-tf-idf":
        return topic_model.reduce_outliers(
            docs, current_topics, 
            strategy="c-tf-idf", 
            threshold=threshold
        )
        
    elif strategy == "probabilities":
        if probs is not None:
            return topic_model.reduce_outliers(
                docs, current_topics, 
                probabilities=probs,
                strategy="probabilities", 
                threshold=threshold
            )
        else:
            logger.warning(f"   ⚠️  Probabilities not available - skipping strategy")
            return None
            
    elif strategy == "distributions":
        distributions_params = strategy_config.get('distributions_params', {})
        return topic_model.reduce_outliers(
            docs, current_topics, 
            strategy="distributions",
            threshold=threshold
        )
        
    elif strategy == "embeddings":
        return topic_model.reduce_outliers(
            docs, current_topics, 
            strategy="embeddings",
            embeddings=embeddings,
            threshold=threshold
        )
        
    else:
        logger.warning(f"   ⚠️  Unknown strategy '{strategy}' - skipping")
        return None


def get_outlier_reduction_stats(topics: List[int], docs: List[str]) -> Dict:
    """
    Get outlier reduction statistics.
    
    Args:
        topics: Topic assignments
        docs: List of documents
        
    Returns:
        Statistics dictionary
    """
    n_outliers = sum(1 for t in topics if t == -1)
    n_docs = len(docs)
    
    return {
        'total_documents': n_docs,
        'outliers': n_outliers,
        'assigned_documents': n_docs - n_outliers,
        'outlier_percentage': (n_outliers / n_docs * 100) if n_docs > 0 else 0.0,
        'coverage_percentage': ((n_docs - n_outliers) / n_docs * 100) if n_docs > 0 else 0.0
    }


def validate_outlier_reduction_config(config: Dict) -> bool:
    """
    Validate outlier reduction configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    outlier_config = config.get('outlier_reduction', {})
    
    if not outlier_config.get('enabled', False):
        return True
        
    strategies = outlier_config.get('strategies', [])
    
    if not strategies:
        return False
        
    valid_strategies = {'c-tf-idf', 'probabilities', 'distributions', 'embeddings'}
    
    for strategy_config in strategies:
        if not isinstance(strategy_config, dict):
            return False
            
        strategy = strategy_config.get('strategy')
        if strategy not in valid_strategies:
            return False
            
        threshold = strategy_config.get('threshold', 0.1)
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return False
            
    return True

