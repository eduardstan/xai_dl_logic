#!/usr/bin/env python3
"""
Cluster Analysis Module for XAI Deep Learning Logic
Handles cluster size categorization and paper selection strategies for systematic literature reviews.
"""

from typing import Dict
from loguru import logger

from utils import get_systematic_review_config


def calculate_cluster_size_category(cluster_size: int, config: Dict) -> str:
    """
    Determine cluster size category based on configured thresholds.
    
    This function categorizes clusters into predefined size categories to enable
    differentiated processing strategies based on cluster characteristics.
    
    Args:
        cluster_size: Number of papers in the cluster
        config: Configuration dictionary containing systematic review settings
                with cluster_thresholds section defining size boundaries
        
    Returns:
        str: Cluster size category ('small', 'medium', 'large', 'xlarge', 
             'xxlarge', 'xxxlarge')
    
    Raises:
        KeyError: If config lacks required cluster_thresholds section
        ValueError: If cluster_size is invalid or thresholds are malformed
    
    Example:
        >>> config = {'systematic_review': {'cluster_thresholds': {
        ...     'small': 10, 'medium': 25, 'large': 50, 'xlarge': 100, 'xxlarge': 200
        ... }}}
        >>> calculate_cluster_size_category(35, config)
        'medium'
    
    Note:
        - Categories are ordered from smallest to largest
        - Thresholds are inclusive (cluster_size <= threshold)
        - 'xxxlarge' is used for clusters exceeding all thresholds
        - Used to determine appropriate selection strategies per cluster size
    """
    if cluster_size <= 0:
        raise ValueError(f"Invalid cluster_size: {cluster_size}. Must be positive integer.")
    
    try:
        review_config = get_systematic_review_config(config)
        thresholds = review_config['cluster_thresholds']
        
        # Validate thresholds structure
        required_keys = {'small', 'medium', 'large', 'xlarge', 'xxlarge'}
        missing_keys = required_keys - set(thresholds.keys())
        if missing_keys:
            raise KeyError(f"Missing required threshold keys: {missing_keys}")
        
        # Determine category based on thresholds
        if cluster_size <= thresholds['small']:
            return 'small'
        elif cluster_size <= thresholds['medium']:
            return 'medium'
        elif cluster_size <= thresholds['large']:
            return 'large'
        elif cluster_size <= thresholds['xlarge']:
            return 'xlarge'
        elif cluster_size <= thresholds['xxlarge']:
            return 'xxlarge'
        else:
            return 'xxxlarge'
            
    except KeyError as e:
        logger.error(f"Missing required configuration key: {e}")
        raise KeyError(f"Configuration must include 'systematic_review.cluster_thresholds': {e}")
    except Exception as e:
        logger.error(f"Error in cluster size categorization: {e}")
        raise ValueError(f"Failed to categorize cluster size: {e}")


def determine_papers_to_select(cluster_size: int, config: Dict) -> int:
    """
    Determine how many papers to select based on cluster size and configured selection strategy.
    
    This function serves as the main dispatcher for paper selection strategies,
    choosing between threshold-based and ratio-based approaches based on configuration.
    
    Args:
        cluster_size: Number of papers in the cluster
        config: Configuration dictionary containing systematic review settings
                with selection strategy and parameters
        
    Returns:
        int: Number of papers to select from the cluster (1 to cluster_size)
    
    Raises:
        ValueError: If cluster_size is invalid or selection parameters are malformed
        KeyError: If required configuration sections are missing
    
    Example:
        >>> config = {'systematic_review': {
        ...     'selection_strategy_type': 'ratio_based',
        ...     'target_selection_ratio': 0.15,
        ...     'min_papers_per_cluster': 2,
        ...     'max_papers_per_cluster': 10
        ... }}
        >>> determine_papers_to_select(50, config)
        7  # 50 * 0.15 = 7.5 → 7
    
    Note:
        - Supports both 'threshold_based' and 'ratio_based' selection strategies
        - Defaults to 'threshold_based' if strategy not specified
        - Always returns at least 1 paper and at most cluster_size papers
        - Selection strategy affects representative diversity and analysis scope
    """
    if cluster_size <= 0:
        raise ValueError(f"Invalid cluster_size: {cluster_size}. Must be positive integer.")
    
    try:
        review_config = get_systematic_review_config(config)
        selection_strategy = review_config.get('selection_strategy_type', 'threshold_based')
        
        if selection_strategy == 'ratio_based':
            return _determine_papers_ratio_based(cluster_size, config)
        elif selection_strategy == 'threshold_based':
            return _determine_papers_threshold_based(cluster_size, config)
        else:
            logger.warning(f"Unknown selection strategy '{selection_strategy}', defaulting to threshold_based")
            return _determine_papers_threshold_based(cluster_size, config)
            
    except Exception as e:
        logger.error(f"Error determining papers to select: {e}")
        raise ValueError(f"Failed to determine paper selection count: {e}")


def _determine_papers_threshold_based(cluster_size: int, config: Dict) -> int:
    """
    Determine paper selection count using threshold-based strategy.
    
    This strategy assigns papers based on discrete cluster size categories,
    with each category having a predetermined number of papers to select.
    
    Args:
        cluster_size: Number of papers in the cluster
        config: Configuration dictionary with systematic review settings
        
    Returns:
        int: Number of papers to select using threshold-based strategy
    
    Note:
        - Uses cluster size categories to determine selection count
        - Provides stepped increases in selection based on cluster size
        - Good for ensuring minimum representation across all cluster sizes
        - Selection count mapping: small=base, medium=base+1, ..., xxxlarge=max
    """
    try:
        review_config = get_systematic_review_config(config)
        base = review_config['base_papers_per_cluster']
        max_papers = review_config['max_papers_per_cluster']
        category = calculate_cluster_size_category(cluster_size, config)
        
        # Map cluster size categories to selection counts
        size_mapping = {
            'small': base,
            'medium': base + 1,
            'large': base + 2,
            'xlarge': base + 3,
            'xxlarge': base + 4,
            'xxxlarge': max_papers
        }
        
        selected_count = size_mapping[category]
        
        # Ensure we don't select more papers than exist in the cluster
        return min(selected_count, cluster_size)
        
    except Exception as e:
        logger.error(f"Error in threshold-based paper selection: {e}")
        raise ValueError(f"Failed threshold-based selection: {e}")


def _determine_papers_ratio_based(cluster_size: int, config: Dict) -> int:
    """
    Determine paper selection count using ratio-based strategy.
    
    This strategy selects a fixed percentage of papers from each cluster,
    providing more uniform sampling across clusters of different sizes.
    
    Args:
        cluster_size: Number of papers in the cluster
        config: Configuration dictionary with systematic review settings
        
    Returns:
        int: Number of papers to select using ratio-based strategy
    
    Note:
        - Applies target_selection_ratio to cluster_size
        - Enforces min_papers_per_cluster and max_papers_per_cluster constraints
        - Provides proportional representation across cluster sizes
        - Good for maintaining consistent sampling density
        - Always selects at least 1 paper and at most cluster_size papers
    """
    try:
        review_config = get_systematic_review_config(config)
        target_ratio = review_config.get('target_selection_ratio', 0.15)  # Default 15%
        min_papers = review_config.get('min_papers_per_cluster', 2)
        max_papers = review_config.get('max_papers_per_cluster', 10)
        
        # Validate parameters
        if not (0.0 < target_ratio <= 1.0):
            raise ValueError(f"Invalid target_selection_ratio: {target_ratio}. Must be between 0 and 1.")
        if min_papers < 1:
            raise ValueError(f"Invalid min_papers_per_cluster: {min_papers}. Must be >= 1.")
        if max_papers < min_papers:
            raise ValueError(f"max_papers_per_cluster ({max_papers}) must be >= min_papers_per_cluster ({min_papers})")
        
        # Calculate papers based on ratio
        ratio_papers = max(1, int(cluster_size * target_ratio))
        
        # Apply min/max constraints
        selected_papers = min(max(ratio_papers, min_papers), max_papers)
        
        # Ensure we don't select more papers than exist in the cluster
        return min(selected_papers, cluster_size)
        
    except Exception as e:
        logger.error(f"Error in ratio-based paper selection: {e}")
        raise ValueError(f"Failed ratio-based selection: {e}")


def validate_cluster_analysis_config(config: Dict) -> bool:
    """
    Validate cluster analysis configuration for completeness and correctness.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if cluster analysis configuration is valid
        
    Note:
        - Checks for required systematic_review section
        - Validates cluster_thresholds structure and values
        - Validates selection strategy parameters
        - Logs specific validation issues for debugging
    """
    try:
        review_config = get_systematic_review_config(config)
        
        # Check cluster thresholds
        if 'cluster_thresholds' not in review_config:
            logger.error("Missing 'cluster_thresholds' in systematic_review configuration")
            return False
        
        thresholds = review_config['cluster_thresholds']
        required_keys = {'small', 'medium', 'large', 'xlarge', 'xxlarge'}
        missing_keys = required_keys - set(thresholds.keys())
        if missing_keys:
            logger.error(f"Missing required threshold keys: {missing_keys}")
            return False
        
        # Validate threshold values are positive integers in ascending order
        prev_threshold = 0
        for key in ['small', 'medium', 'large', 'xlarge', 'xxlarge']:
            threshold = thresholds[key]
            if not isinstance(threshold, int) or threshold <= prev_threshold:
                logger.error(f"Invalid threshold '{key}': {threshold}. Must be positive integer > {prev_threshold}")
                return False
            prev_threshold = threshold
        
        # Check selection strategy parameters
        if 'base_papers_per_cluster' not in review_config:
            logger.error("Missing 'base_papers_per_cluster' in configuration")
            return False
        
        if 'max_papers_per_cluster' not in review_config:
            logger.error("Missing 'max_papers_per_cluster' in configuration")
            return False
        
        base_papers = review_config['base_papers_per_cluster']
        max_papers = review_config['max_papers_per_cluster']
        
        if not isinstance(base_papers, int) or base_papers < 1:
            logger.error(f"Invalid base_papers_per_cluster: {base_papers}. Must be positive integer")
            return False
        
        if not isinstance(max_papers, int) or max_papers < base_papers:
            logger.error(f"Invalid max_papers_per_cluster: {max_papers}. Must be >= base_papers_per_cluster")
            return False
        
        # Check ratio-based parameters if strategy is ratio_based
        strategy = review_config.get('selection_strategy_type', 'threshold_based')
        if strategy == 'ratio_based':
            target_ratio = review_config.get('target_selection_ratio', 0.15)
            if not isinstance(target_ratio, (int, float)) or not (0.0 < target_ratio <= 1.0):
                logger.error(f"Invalid target_selection_ratio: {target_ratio}. Must be between 0 and 1")
                return False
        
        logger.info("✅ Cluster analysis configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating cluster analysis configuration: {e}")
        return False


def get_cluster_analysis_summary(cluster_size: int, selected_count: int, config: Dict) -> Dict:
    """
    Generate summary information for cluster analysis decisions.
    
    Args:
        cluster_size: Number of papers in the cluster
        selected_count: Number of papers selected
        config: Configuration dictionary
        
    Returns:
        Dict containing cluster analysis summary and rationale
    """
    try:
        review_config = get_systematic_review_config(config)
        category = calculate_cluster_size_category(cluster_size, config)
        strategy = review_config.get('selection_strategy_type', 'threshold_based')
        
        return {
            'cluster_size': cluster_size,
            'selected_count': selected_count,
            'selection_ratio': selected_count / cluster_size,
            'size_category': category,
            'selection_strategy': strategy,
            'efficiency_score': selected_count / cluster_size,  # Higher is more selective
            'coverage_score': min(selected_count / 5, 1.0)  # Normalize to reasonable coverage
        }
        
    except Exception as e:
        logger.error(f"Error generating cluster analysis summary: {e}")
        return {'error': str(e)} 