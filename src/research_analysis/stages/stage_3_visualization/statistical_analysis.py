#!/usr/bin/env python3
"""
Statistical analysis functions for Stage 3 visualization pipeline.

This module provides comprehensive statistical validation of paper selection
methodology using Mann-Whitney U tests, Cohen's d effect sizes, and temporal
distribution analysis.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from research_analysis.utils.logging import get_logger

logger = get_logger()


def mann_whitney_u_test(
    selected_values: np.ndarray, 
    non_selected_values: np.ndarray,
    variable_name: str,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test for comparing selected vs non-selected papers.
    
    Args:
        selected_values: Values for selected papers
        non_selected_values: Values for non-selected papers
        variable_name: Name of the variable being tested
        alpha: Significance level
        
    Returns:
        Dictionary with test results including U statistic, p-value, and interpretation
    """
    # Remove NaN values
    selected_clean = selected_values[~np.isnan(selected_values)]
    non_selected_clean = non_selected_values[~np.isnan(non_selected_values)]
    
    # Perform two-tailed test
    statistic, p_value = stats.mannwhitneyu(
        selected_clean, 
        non_selected_clean, 
        alternative=alternative
    )
    
    # Calculate effect size (rank-biserial correlation)
    n1, n2 = len(selected_clean), len(non_selected_clean)
    rank_biserial = (2 * statistic) / (n1 * n2) - 1
    
    # Descriptive statistics
    selected_median = np.median(selected_clean)
    non_selected_median = np.median(non_selected_clean)
    
    return {
        'variable': variable_name,
        'u_statistic': float(statistic),
        'p_value': float(p_value),
        'n_selected': n1,
        'n_non_selected': n2,
        'rank_biserial_correlation': float(rank_biserial),
        'significant': p_value < alpha,
        'selected_median': float(selected_median),
        'non_selected_median': float(non_selected_median),
        'interpretation': _interpret_mann_whitney_result(p_value, rank_biserial, alpha)
    }


def cohens_d_effect_size(
    selected_values: np.ndarray,
    non_selected_values: np.ndarray,
    variable_name: str,
    confidence_level: float = 0.95,
    effect_size_thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Calculate Cohen's d effect size with confidence intervals.
    
    Args:
        selected_values: Values for selected papers
        non_selected_values: Values for non-selected papers
        variable_name: Name of the variable
        confidence_level: Confidence level for CI
        
    Returns:
        Dictionary with effect size results
    """
    # Remove NaN values
    selected_clean = selected_values[~np.isnan(selected_values)]
    non_selected_clean = non_selected_values[~np.isnan(non_selected_values)]
    
    n1, n2 = len(selected_clean), len(non_selected_clean)
    mean1, mean2 = np.mean(selected_clean), np.mean(non_selected_clean)
    std1, std2 = np.std(selected_clean, ddof=1), np.std(non_selected_clean, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std
    
    # Confidence interval
    alpha = 1 - confidence_level
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
    t_critical = stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
    ci_lower = cohens_d - t_critical * se_d
    ci_upper = cohens_d + t_critical * se_d
    
    effect_size_category = _categorize_effect_size(abs(cohens_d), effect_size_thresholds)
    
    return {
        'variable': variable_name,
        'cohens_d': float(cohens_d),
        'confidence_interval': (float(ci_lower), float(ci_upper)),
        'effect_size_category': effect_size_category,
        'practical_significance': abs(cohens_d) >= 0.2,
        'sample_sizes': {'selected': n1, 'non_selected': n2},
        'means': {'selected': float(mean1), 'non_selected': float(mean2)},
        'interpretation': _interpret_cohens_d(cohens_d, effect_size_category)
    }


def temporal_distribution_analysis(
    selected_df: pd.DataFrame,
    non_selected_df: pd.DataFrame,
    year_column: str = 'year',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Analyze temporal distribution patterns in paper selection.
    
    Args:
        selected_df: DataFrame with selected papers
        non_selected_df: DataFrame with non-selected papers
        year_column: Column name for publication year
        alpha: Significance level
        
    Returns:
        Dictionary with temporal analysis results
    """
    selected_years = selected_df[year_column].dropna().values
    non_selected_years = non_selected_df[year_column].dropna().values
    
    # Kolmogorov-Smirnov test for distribution similarity
    ks_stat, ks_p_value = stats.ks_2samp(selected_years, non_selected_years)
    
    # Year-wise selection rates
    all_df = pd.concat([selected_df, non_selected_df])
    yearly_stats = all_df.groupby(year_column).agg({
        'is_selected_representative': ['count', 'sum', 'mean']
    }).round(3)
    
    # Descriptive statistics
    selected_year_range = (int(selected_years.min()), int(selected_years.max()))
    non_selected_year_range = (int(non_selected_years.min()), int(non_selected_years.max()))
    
    return {
        'kolmogorov_smirnov_test': {
            'statistic': float(ks_stat),
            'p_value': float(ks_p_value),
            'significant_difference': ks_p_value < alpha
        },
        'descriptive_stats': {
            'selected_year_range': selected_year_range,
            'non_selected_year_range': non_selected_year_range,
            'selected_median_year': float(np.median(selected_years)),
            'non_selected_median_year': float(np.median(non_selected_years)),
            'selected_mean_year': float(np.mean(selected_years)),
            'non_selected_mean_year': float(np.mean(non_selected_years))
        },
        'yearly_selection_rates': yearly_stats.to_dict(),
        'temporal_bias_assessment': _assess_temporal_bias(ks_p_value, alpha)
    }


def comprehensive_statistical_analysis(
    selected_df: pd.DataFrame,
    non_selected_df: pd.DataFrame,
    variables: List[str],
    config: Any
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis for all specified variables.
    
    Args:
        selected_df: DataFrame with selected papers
        non_selected_df: DataFrame with non-selected papers
        variables: List of variables to analyze
        config: Stage 3 configuration object
        
    Returns:
        Comprehensive analysis results
    """
    logger.info(f"Starting comprehensive statistical analysis for {len(variables)} variables")
    
    # Extract configuration values
    mann_whitney_alpha = config.statistical_analysis.tests.mann_whitney_u.alpha
    mann_whitney_alternative = config.statistical_analysis.tests.mann_whitney_u.alternative
    confidence_level = config.statistical_analysis.tests.cohens_d.confidence_level
    temporal_alpha = config.statistical_analysis.tests.temporal_analysis.alpha
    multiple_testing_method = config.statistical_analysis.multiple_testing_correction.method
    
    mann_whitney_results = {}
    effect_size_results = {}
    p_values = []
    
    # Analyze each variable
    for var in variables:
        if var in selected_df.columns and var in non_selected_df.columns:
            logger.info(f"Analyzing variable: {var}")
            
            selected_vals = selected_df[var].values
            non_selected_vals = non_selected_df[var].values
            
            # Skip if all values are NaN
            if np.all(np.isnan(selected_vals)) or np.all(np.isnan(non_selected_vals)):
                logger.warning(f"Skipping {var} - contains only NaN values")
                continue
            
            # Mann-Whitney U test
            mw_result = mann_whitney_u_test(selected_vals, non_selected_vals, var, mann_whitney_alpha, mann_whitney_alternative)
            mann_whitney_results[var] = mw_result
            p_values.append(mw_result['p_value'])
            
            # Cohen's d effect size
            effect_result = cohens_d_effect_size(selected_vals, non_selected_vals, var, confidence_level, config.statistical_analysis.tests.cohens_d.effect_size_thresholds)
            effect_size_results[var] = effect_result
    
    # Apply multiple testing correction
    if p_values:
        corrected_results = _apply_multiple_testing_correction(
            mann_whitney_results, p_values, multiple_testing_method, mann_whitney_alpha
        )
    else:
        corrected_results = {'method': multiple_testing_method, 'no_tests_performed': True}
    
    # Temporal analysis
    temporal_results = temporal_distribution_analysis(selected_df, non_selected_df, 'year', temporal_alpha)
    
    logger.info("Statistical analysis completed successfully")
    
    return {
        'mann_whitney_analysis': {
            'individual_tests': mann_whitney_results,
            'multiple_testing_correction': corrected_results
        },
        'effect_size_analysis': effect_size_results,
        'temporal_analysis': temporal_results,
        'summary': _generate_analysis_summary(mann_whitney_results, effect_size_results, temporal_results)
    }


# Helper functions

def _interpret_mann_whitney_result(p_value: float, rank_biserial: float, alpha: float) -> str:
    """Interpret Mann-Whitney U test results."""
    significance = "significant" if p_value < alpha else "not significant"
    
    if abs(rank_biserial) < 0.1:
        effect_magnitude = "negligible"
    elif abs(rank_biserial) < 0.3:
        effect_magnitude = "small"
    elif abs(rank_biserial) < 0.5:
        effect_magnitude = "medium"
    else:
        effect_magnitude = "large"
    
    direction = "higher" if rank_biserial > 0 else "lower"
    
    return f"Difference is {significance} (p={p_value:.4f}) with {effect_magnitude} effect size. Selected papers have {direction} values."


def _categorize_effect_size(cohens_d: float, effect_size_thresholds: Dict[str, float] = None) -> str:
    """Categorize Cohen's d effect size using configurable thresholds."""
    abs_d = abs(cohens_d)
    
    # Use default thresholds if not provided
    if effect_size_thresholds is None:
        effect_size_thresholds = {'small': 0.2, 'medium': 0.5, 'large': 0.8}
    
    small_threshold = effect_size_thresholds['small']
    medium_threshold = effect_size_thresholds['medium']
    large_threshold = effect_size_thresholds['large']
    
    if abs_d < small_threshold:
        return "negligible"
    elif abs_d < medium_threshold:
        return "small"
    elif abs_d < large_threshold:
        return "medium"
    else:
        return "large"


def _interpret_cohens_d(cohens_d: float, category: str) -> str:
    """Interpret Cohen's d effect size."""
    direction = "higher" if cohens_d > 0 else "lower"
    return f"Selected papers have {direction} values with {category} effect size (d={cohens_d:.3f})"


def _assess_temporal_bias(ks_p_value: float, alpha: float) -> str:
    """Assess temporal bias based on Kolmogorov-Smirnov test."""
    if ks_p_value < alpha:
        return "Significant temporal bias detected - selection methodology may favor certain time periods"
    else:
        return "No significant temporal bias detected - selection is temporally representative"


def _apply_multiple_testing_correction(
    test_results: Dict[str, Any],
    p_values: List[float],
    method: str,
    alpha: float
) -> Dict[str, Any]:
    """Apply multiple testing correction."""
    rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=alpha, method=method
    )
    
    return {
        'method': method,
        'original_alpha': alpha,
        'original_p_values': p_values,
        'adjusted_p_values': p_adjusted.tolist(),
        'rejected_hypotheses': rejected.tolist(),
        'significant_tests_after_correction': int(sum(rejected)),
        'total_tests': len(p_values)
    }


def _generate_analysis_summary(
    mann_whitney_results: Dict[str, Any],
    effect_size_results: Dict[str, Any],
    temporal_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate summary of statistical analysis."""
    return {
        'total_variables_analyzed': len(mann_whitney_results),
        'significant_mann_whitney_tests': sum(1 for r in mann_whitney_results.values() if r['significant']),
        'large_effect_sizes': sum(1 for r in effect_size_results.values() if r['effect_size_category'] == 'large'),
        'medium_effect_sizes': sum(1 for r in effect_size_results.values() if r['effect_size_category'] == 'medium'),
        'temporal_bias_detected': temporal_results['kolmogorov_smirnov_test']['significant_difference']
    } 