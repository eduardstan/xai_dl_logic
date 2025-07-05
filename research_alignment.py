#!/usr/bin/env python3
"""
Research Alignment Analysis Module for XAI Deep Learning Logic
Analyzes text alignment with XAI, symbolic, and sub-symbolic research using wildcard pattern matching.
"""

import re
from typing import Dict
from loguru import logger

from utils import get_systematic_review_config


def analyze_research_alignment(text: str, config: Dict) -> Dict[str, float]:
    """
    Analyze alignment with XAI, symbolic, and sub-symbolic research using wildcard matching.
    
    This function performs sophisticated pattern matching to determine how well a text
    aligns with different research categories defined in the configuration. It supports
    both exact word matching and flexible wildcard patterns for comprehensive analysis.
    
    Args:
        text: Input text to analyze (typically combined title + abstract + keywords)
        config: Configuration dictionary containing systematic review settings with
               research_keywords section defining category-specific terms
        
    Returns:
        Dict[str, float]: Alignment scores for each research category where:
                         - Keys follow pattern '{category}_alignment' (e.g., 'xai_alignment')
                         - Values are ratios (0.0-1.0) of matched terms to total terms
                         - Higher scores indicate stronger alignment with that research area
    
    Raises:
        KeyError: If config lacks required 'systematic_review.research_keywords' section
        ValueError: If text is empty or research keywords are malformed
    
    Example:
        >>> config = {'systematic_review': {'research_keywords': {
        ...     'xai': ['explainable', 'interpretable', 'transparent'],
        ...     'symbolic': ['logic*', 'formal', 'rule*']
        ... }}}
        >>> text = "Explainable AI using logical rules for transparent reasoning"
        >>> scores = analyze_research_alignment(text, config)
        >>> print(scores)
        {'xai_alignment': 0.67, 'symbolic_alignment': 0.67}
    
    Note:
        - Supports wildcard patterns: 'neural*' matches 'neural', 'neurons', 'neurally'
        - Uses word boundary matching to avoid false positives
        - Case-insensitive matching for robustness
        - Returns 0.0 for categories with no matching terms
        - Designed for academic literature analysis with technical terminology
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for alignment analysis")
        return {}
    
    try:
        # Get research keywords from systematic review configuration
        review_config = get_systematic_review_config(config)
        keywords = review_config['research_keywords']
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        alignment_scores = {}
        for category, terms in keywords.items():
            if not terms:
                logger.warning(f"No terms defined for category '{category}'")
                alignment_scores[f'{category}_alignment'] = 0.0
                continue
                
            matches = 0
            for term in terms:
                term_lower = term.lower()
                
                # Handle wildcard patterns
                if '*' in term_lower:
                    pattern = _build_wildcard_pattern(term_lower)
                    if re.search(pattern, text_lower):
                        matches += 1
                else:
                    # Exact word matching for terms without wildcards
                    if re.search(r'\b' + re.escape(term_lower) + r'\b', text_lower):
                        matches += 1
            
            alignment_scores[f'{category}_alignment'] = matches / len(terms)
        
        return alignment_scores
        
    except KeyError as e:
        logger.error(f"Missing required configuration key: {e}")
        raise KeyError(f"Configuration must include 'systematic_review.research_keywords': {e}")
    except Exception as e:
        logger.error(f"Error in research alignment analysis: {e}")
        raise ValueError(f"Failed to analyze research alignment: {e}")


def _build_wildcard_pattern(term_lower: str) -> str:
    """
    Build regex pattern for wildcard matching with proper word boundaries.
    
    Args:
        term_lower: Lowercase term with wildcard characters (*)
        
    Returns:
        str: Compiled regex pattern for word boundary matching
        
    Note:
        - Handles various wildcard positions (prefix, suffix, infix)
        - Ensures word boundary matching to avoid false positives
        - Optimized for academic terminology patterns
    """
    if term_lower.endswith('*') and not term_lower.startswith('*'):
        # Pattern like "neural*" -> match words starting with "neural"
        base_term = term_lower[:-1]  # Remove the '*'
        pattern = r'\b' + re.escape(base_term) + r'\w*\b'
    elif term_lower.startswith('*') and not term_lower.endswith('*'):
        # Pattern like "*symbolic" -> match words ending with "symbolic"
        base_term = term_lower[1:]  # Remove the '*'
        pattern = r'\b\w*' + re.escape(base_term) + r'\b'
    elif '*' in term_lower:
        # Pattern like "neuro*symbolic" -> match words with that pattern
        parts = term_lower.split('*')
        if len(parts) == 2:
            pattern = r'\b' + re.escape(parts[0]) + r'\w*' + re.escape(parts[1]) + r'\b'
        else:
            # Multiple wildcards - more complex pattern
            pattern_parts = []
            for i, part in enumerate(parts):
                if i == 0:
                    pattern_parts.append(r'\b' + re.escape(part))
                elif i == len(parts) - 1:
                    pattern_parts.append(re.escape(part) + r'\b')
                else:
                    pattern_parts.append(re.escape(part))
            pattern = r'\w*'.join(pattern_parts)
    else:
        # Fallback for malformed patterns
        pattern = r'\b' + re.escape(term_lower) + r'\b'
    
    return pattern


def validate_research_keywords(config: Dict) -> bool:
    """
    Validate research keywords configuration for completeness and correctness.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if research keywords configuration is valid
        
    Note:
        - Checks for required systematic_review.research_keywords section
        - Validates that each category has non-empty term lists
        - Logs specific validation issues for debugging
    """
    try:
        review_config = get_systematic_review_config(config)
        
        if 'research_keywords' not in review_config:
            logger.error("Missing 'research_keywords' in systematic_review configuration")
            return False
        
        keywords = review_config['research_keywords']
        
        if not keywords:
            logger.error("Empty research_keywords configuration")
            return False
        
        for category, terms in keywords.items():
            if not isinstance(terms, list):
                logger.error(f"Research keywords for '{category}' must be a list")
                return False
                
            if not terms:
                logger.warning(f"Empty term list for category '{category}'")
            
            # Check for valid string terms
            for term in terms:
                if not isinstance(term, str) or not term.strip():
                    logger.error(f"Invalid term in category '{category}': {term}")
                    return False
        
        logger.info(f"✅ Research keywords validation passed for {len(keywords)} categories")
        return True
        
    except Exception as e:
        logger.error(f"Error validating research keywords: {e}")
        return False


def get_alignment_summary(alignment_scores: Dict[str, float]) -> Dict[str, any]:
    """
    Generate summary statistics for research alignment scores.
    
    Args:
        alignment_scores: Dictionary of alignment scores from analyze_research_alignment()
        
    Returns:
        Dict containing summary statistics and insights
        
    Note:
        - Calculates overall alignment metrics
        - Identifies strongest and weakest research areas
        - Provides interpretation guidelines
    """
    if not alignment_scores:
        return {'total_categories': 0, 'strongest_alignment': None, 'weakest_alignment': None}
    
    # Remove '_alignment' suffix for cleaner category names
    clean_scores = {k.replace('_alignment', ''): v for k, v in alignment_scores.items()}
    
    strongest = max(clean_scores.items(), key=lambda x: x[1])
    weakest = min(clean_scores.items(), key=lambda x: x[1])
    
    return {
        'total_categories': len(clean_scores),
        'average_alignment': sum(clean_scores.values()) / len(clean_scores),
        'strongest_alignment': {'category': strongest[0], 'score': strongest[1]},
        'weakest_alignment': {'category': weakest[0], 'score': weakest[1]},
        'high_alignment_categories': [k for k, v in clean_scores.items() if v > 0.5],
        'alignment_scores': clean_scores
    } 