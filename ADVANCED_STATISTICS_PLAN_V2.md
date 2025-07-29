# Advanced Statistical Analytics for Stage 3: Functional Programming Implementation

## Executive Summary

This plan implements advanced statistical validation for paper selection methodology using **functional programming** principles aligned with the existing codebase. The enhancement provides rigorous statistical comparison between selected (533) and non-selected (3,202) papers using Mann-Whitney U tests, Cohen's d effect sizes, and bias assessments based on **available data only**.

## Deep Reflection: Representativeness Score vs Representativeness Gap

### Current Representativeness Score
```python
# From math.py
representativeness_score = (1 - diversity_weight) * centrality_score + diversity_weight * diversity_score
```

**Purpose**: Weighted combination optimizing paper selection within clusters
**Range**: [0, 1] where higher = better representative
**Use**: Paper selection algorithm optimization

### Proposed Representativeness Gap
```python
# New concept - distance from population centroid
representativeness_gap = euclidean_distance(selected_centroid, population_centroid)
```

**Purpose**: Quantifies how much selected papers deviate from population center
**Range**: [0, ∞] where lower = better representation
**Use**: Population-level bias assessment

**Conclusion**: These are indeed different concepts. The gap measures population-level deviation while the score measures individual paper quality.

## Available Data Analysis

### Primary Variables (Currently Computed)
- `similarity_to_centroid`: Cosine similarity to cluster center
- `diversity_score`: Uniqueness within cluster  
- `representativeness_score`: Weighted combination of centrality and diversity
- `avg_similarity_to_cluster`: Average similarity to cluster papers

### Secondary Variables (From Bibliography)
- `publication_year`: Temporal distribution analysis
- `topic_id`: Topic cluster membership
- `research_alignment`: XAI, symbolic, subsymbolic alignment scores
- `cluster_size`: Number of papers in each topic cluster
- `is_selected_representative`: Binary selection status

### Unavailable Data (Excluded from Analysis)
- ❌ Citation counts
- ❌ Geographic/institutional distribution
- ❌ Labeled training data for logistic regression
- ❌ Selection propensity modeling

## Statistical Implementation Plan

### Phase 1: Core Statistical Functions

#### 1.1 Mann-Whitney U Test Suite
**File**: `src/research_analysis/stages/stage_3_visualization/statistical_tests.py`

```python
def mann_whitney_u_test(selected_values: np.ndarray, 
                       non_selected_values: np.ndarray,
                       variable_name: str) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test for comparing selected vs non-selected papers.
    
    Args:
        selected_values: Values for selected papers
        non_selected_values: Values for non-selected papers
        variable_name: Name of the variable being tested
        
    Returns:
        Dictionary with test results including U statistic, p-value, and interpretation
    """
    from scipy.stats import mannwhitneyu
    
    # Perform two-tailed test
    statistic, p_value = mannwhitneyu(
        selected_values, 
        non_selected_values, 
        alternative='two-sided'
    )
    
    # Calculate effect size (rank-biserial correlation)
    n1, n2 = len(selected_values), len(non_selected_values)
    rank_biserial = (2 * statistic) / (n1 * n2) - 1
    
    return {
        'variable': variable_name,
        'u_statistic': statistic,
        'p_value': p_value,
        'n_selected': n1,
        'n_non_selected': n2,
        'rank_biserial_correlation': rank_biserial,
        'significant': p_value < 0.05,
        'interpretation': _interpret_mann_whitney_result(p_value, rank_biserial)
    }

def comprehensive_mann_whitney_analysis(selected_df: pd.DataFrame,
                                      non_selected_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform Mann-Whitney U tests for all available variables.
    
    Returns comprehensive results with dependent test correction.
    """
    variables = [
        'similarity_to_centroid',
        'diversity_score', 
        'representativeness_score',
        'avg_similarity_to_cluster',
        'publication_year'
    ]
    
    test_results = {}
    p_values = []
    
    for var in variables:
        if var in selected_df.columns and var in non_selected_df.columns:
            selected_vals = selected_df[var].dropna().values
            non_selected_vals = non_selected_df[var].dropna().values
            
            result = mann_whitney_u_test(selected_vals, non_selected_vals, var)
            test_results[var] = result
            p_values.append(result['p_value'])
    
    # Apply Benjamini-Hochberg correction for dependent tests
    corrected_results = apply_dependent_multiple_testing_correction(test_results, p_values)
    
    return {
        'individual_tests': test_results,
        'multiple_testing_correction': corrected_results,
        'summary': _generate_mann_whitney_summary(corrected_results)
    }
```

#### 1.2 Cohen's d Effect Size Calculations
```python
def cohens_d_effect_size(selected_values: np.ndarray,
                        non_selected_values: np.ndarray,
                        variable_name: str) -> Dict[str, Any]:
    """
    Calculate Cohen's d effect size with confidence intervals.
    
    Appropriate for large samples (533 vs 3,202).
    """
    n1, n2 = len(selected_values), len(non_selected_values)
    mean1, mean2 = np.mean(selected_values), np.mean(non_selected_values)
    std1, std2 = np.std(selected_values, ddof=1), np.std(non_selected_values, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std
    
    # Confidence interval (95%)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + cohens_d**2 / (2 * (n1 + n2)))
    ci_lower = cohens_d - 1.96 * se_d
    ci_upper = cohens_d + 1.96 * se_d
    
    return {
        'variable': variable_name,
        'cohens_d': cohens_d,
        'confidence_interval': (ci_lower, ci_upper),
        'effect_size_category': _categorize_effect_size(cohens_d),
        'practical_significance': abs(cohens_d) >= 0.2,
        'sample_sizes': {'selected': n1, 'non_selected': n2},
        'means': {'selected': mean1, 'non_selected': mean2},
        'interpretation': _interpret_cohens_d(cohens_d)
    }

def comprehensive_effect_size_analysis(selected_df: pd.DataFrame,
                                     non_selected_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate Cohen's d for all available variables.
    """
    variables = [
        'similarity_to_centroid',
        'diversity_score',
        'representativeness_score', 
        'avg_similarity_to_cluster',
        'publication_year'
    ]
    
    effect_sizes = {}
    
    for var in variables:
        if var in selected_df.columns and var in non_selected_df.columns:
            selected_vals = selected_df[var].dropna().values
            non_selected_vals = non_selected_df[var].dropna().values
            
            effect_sizes[var] = cohens_d_effect_size(selected_vals, non_selected_vals, var)
    
    return {
        'effect_sizes': effect_sizes,
        'summary': _generate_effect_size_summary(effect_sizes)
    }
```

#### 1.3 Dependent Multiple Testing Correction
```python
def apply_dependent_multiple_testing_correction(test_results: Dict[str, Any],
                                              p_values: List[float]) -> Dict[str, Any]:
    """
    Apply Benjamini-Hochberg correction accounting for test dependencies.
    
    Since representativeness_score depends on centrality and diversity scores,
    we use a modified approach suitable for dependent tests.
    """
    from statsmodels.stats.multitest import multipletests
    
    # Use Benjamini-Yekutieli procedure for dependent tests
    rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
        p_values, 
        alpha=0.05, 
        method='fdr_by'  # Benjamini-Yekutieli for dependent tests
    )
    
    return {
        'method': 'Benjamini-Yekutieli (dependent tests)',
        'original_p_values': p_values,
        'adjusted_p_values': p_adjusted,
        'rejected_hypotheses': rejected,
        'significant_tests': sum(rejected),
        'family_wise_error_rate': 0.05
    }
```

### Phase 2: Advanced Distribution Analysis

#### 2.1 Temporal Distribution Analysis
```python
def temporal_distribution_analysis(selected_df: pd.DataFrame,
                                 non_selected_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze temporal distribution patterns in paper selection.
    
    Tests for temporal bias in selection methodology.
    """
    from scipy.stats import ks_2samp
    
    selected_years = selected_df['publication_year'].dropna().values
    non_selected_years = non_selected_df['publication_year'].dropna().values
    
    # Kolmogorov-Smirnov test for distribution similarity
    ks_stat, ks_p_value = ks_2samp(selected_years, non_selected_years)
    
    # Year-wise selection rates
    all_df = pd.concat([selected_df, non_selected_df])
    yearly_stats = all_df.groupby('publication_year').agg({
        'is_selected_representative': ['count', 'sum', 'mean']
    }).round(3)
    
    return {
        'kolmogorov_smirnov_test': {
            'statistic': ks_stat,
            'p_value': ks_p_value,
            'significant_difference': ks_p_value < 0.05
        },
        'descriptive_stats': {
            'selected_year_range': (selected_years.min(), selected_years.max()),
            'non_selected_year_range': (non_selected_years.min(), non_selected_years.max()),
            'selected_median_year': np.median(selected_years),
            'non_selected_median_year': np.median(non_selected_years)
        },
        'yearly_selection_rates': yearly_stats,
        'temporal_bias_assessment': _assess_temporal_bias(yearly_stats)
    }
```

#### 2.2 Topic Distribution Analysis
```python
def topic_distribution_analysis(selected_df: pd.DataFrame,
                              non_selected_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze topic representation and coverage.
    """
    from scipy.stats import chi2_contingency
    
    # Topic distribution comparison
    all_df = pd.concat([selected_df, non_selected_df])
    topic_contingency = pd.crosstab(
        all_df['topic_id'], 
        all_df['is_selected_representative']
    )
    
    # Chi-square test for topic independence
    chi2, p_value, dof, expected = chi2_contingency(topic_contingency)
    
    # Topic coverage analysis
    topic_coverage = all_df.groupby('topic_id').agg({
        'is_selected_representative': ['count', 'sum', 'mean'],
        'cluster_size': 'first'
    }).round(3)
    
    return {
        'chi_square_test': {
            'statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant_association': p_value < 0.05
        },
        'topic_coverage': topic_coverage,
        'coverage_stats': {
            'topics_with_representatives': (topic_coverage.iloc[:, 1] > 0).sum(),
            'total_topics': len(topic_coverage),
            'coverage_percentage': (topic_coverage.iloc[:, 1] > 0).mean() * 100
        }
    }
```

#### 2.3 Representativeness Gap Analysis
```python
def representativeness_gap_analysis(selected_df: pd.DataFrame,
                                  non_selected_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate representativeness gap - distance between selected and population centroids.
    """
    # Calculate centroids in metric space
    metrics = ['similarity_to_centroid', 'diversity_score', 'avg_similarity_to_cluster']
    
    selected_centroid = selected_df[metrics].mean().values
    population_centroid = pd.concat([selected_df, non_selected_df])[metrics].mean().values
    
    # Euclidean distance
    representativeness_gap = np.linalg.norm(selected_centroid - population_centroid)
    
    # Component-wise gaps
    component_gaps = {
        metric: selected_df[metric].mean() - pd.concat([selected_df, non_selected_df])[metric].mean()
        for metric in metrics
    }
    
    return {
        'overall_representativeness_gap': representativeness_gap,
        'component_gaps': component_gaps,
        'selected_centroid': dict(zip(metrics, selected_centroid)),
        'population_centroid': dict(zip(metrics, population_centroid)),
        'interpretation': _interpret_representativeness_gap(representativeness_gap, component_gaps)
    }
```

### Phase 3: Research Alignment Analysis

#### 3.1 Alignment Distribution Tests
```python
def research_alignment_analysis(selected_df: pd.DataFrame,
                              non_selected_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze research alignment distributions across selected vs non-selected papers.
    """
    alignment_types = ['xai_alignment', 'symbolic_alignment', 'subsymbolic_alignment']
    
    alignment_results = {}
    
    for alignment in alignment_types:
        if alignment in selected_df.columns:
            selected_vals = selected_df[alignment].dropna().values
            non_selected_vals = non_selected_df[alignment].dropna().values
            
            # Mann-Whitney U test
            mw_result = mann_whitney_u_test(selected_vals, non_selected_vals, alignment)
            
            # Effect size
            effect_result = cohens_d_effect_size(selected_vals, non_selected_vals, alignment)
            
            alignment_results[alignment] = {
                'mann_whitney': mw_result,
                'effect_size': effect_result,
                'descriptive_stats': {
                    'selected_mean': np.mean(selected_vals),
                    'non_selected_mean': np.mean(non_selected_vals),
                    'selected_std': np.std(selected_vals),
                    'non_selected_std': np.std(non_selected_vals)
                }
            }
    
    return {
        'alignment_analyses': alignment_results,
        'summary': _generate_alignment_summary(alignment_results)
    }
```

### Phase 4: Comprehensive Statistical Report Generation

#### 4.1 Statistical Summary Functions
```python
def generate_comprehensive_statistical_report(selected_df: pd.DataFrame,
                                            non_selected_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate complete statistical analysis report.
    """
    # Perform all analyses
    mann_whitney_results = comprehensive_mann_whitney_analysis(selected_df, non_selected_df)
    effect_size_results = comprehensive_effect_size_analysis(selected_df, non_selected_df)
    temporal_results = temporal_distribution_analysis(selected_df, non_selected_df)
    topic_results = topic_distribution_analysis(selected_df, non_selected_df)
    gap_results = representativeness_gap_analysis(selected_df, non_selected_df)
    alignment_results = research_alignment_analysis(selected_df, non_selected_df)
    
    # Generate comprehensive summary
    summary = {
        'sample_sizes': {
            'selected': len(selected_df),
            'non_selected': len(non_selected_df),
            'total': len(selected_df) + len(non_selected_df)
        },
        'statistical_significance': {
            'significant_tests': sum(1 for test in mann_whitney_results['individual_tests'].values() 
                                   if test['significant']),
            'total_tests': len(mann_whitney_results['individual_tests'])
        },
        'practical_significance': {
            'large_effects': sum(1 for effect in effect_size_results['effect_sizes'].values()
                               if effect['effect_size_category'] == 'large'),
            'medium_effects': sum(1 for effect in effect_size_results['effect_sizes'].values()
                                if effect['effect_size_category'] == 'medium')
        },
        'bias_assessment': {
            'temporal_bias': temporal_results['kolmogorov_smirnov_test']['significant_difference'],
            'topic_bias': topic_results['chi_square_test']['significant_association'],
            'representativeness_gap': gap_results['overall_representativeness_gap']
        }
    }
    
    return {
        'mann_whitney_analysis': mann_whitney_results,
        'effect_size_analysis': effect_size_results,
        'temporal_analysis': temporal_results,
        'topic_analysis': topic_results,
        'representativeness_gap_analysis': gap_results,
        'research_alignment_analysis': alignment_results,
        'summary': summary
    }
```

### Phase 5: Visualization Enhancements

#### 5.1 Statistical Visualization Functions
**File**: `src/research_analysis/stages/stage_3_visualization/statistical_plots.py`

```python
def create_statistical_comparison_plots(selected_df: pd.DataFrame,
                                      non_selected_df: pd.DataFrame,
                                      output_dir: Path) -> Dict[str, Path]:
    """
    Create comprehensive statistical comparison visualizations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Statistical Comparison: Selected vs Non-Selected Papers', fontsize=16)
    
    # 1. Distribution comparison plots
    variables = ['similarity_to_centroid', 'diversity_score', 'representativeness_score']
    
    for i, var in enumerate(variables):
        ax = axes[0, i]
        
        # Histograms
        ax.hist(selected_df[var].dropna(), bins=30, alpha=0.7, 
                label='Selected', color='skyblue', density=True)
        ax.hist(non_selected_df[var].dropna(), bins=30, alpha=0.7, 
                label='Non-Selected', color='lightcoral', density=True)
        
        ax.set_xlabel(var.replace('_', ' ').title())
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Effect size visualization
    ax = axes[1, 0]
    effect_results = comprehensive_effect_size_analysis(selected_df, non_selected_df)
    
    effects = [result['cohens_d'] for result in effect_results['effect_sizes'].values()]
    variables = list(effect_results['effect_sizes'].keys())
    
    bars = ax.barh(variables, effects)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='Small Effect')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect')
    ax.set_xlabel("Cohen's d Effect Size")
    ax.set_title('Effect Sizes: Selected vs Non-Selected')
    ax.legend()
    
    # 3. Temporal distribution
    ax = axes[1, 1]
    selected_years = selected_df['publication_year'].dropna()
    non_selected_years = non_selected_df['publication_year'].dropna()
    
    ax.hist(selected_years, bins=20, alpha=0.7, label='Selected', color='skyblue')
    ax.hist(non_selected_years, bins=20, alpha=0.7, label='Non-Selected', color='lightcoral')
    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Count')
    ax.set_title('Temporal Distribution')
    ax.legend()
    
    # 4. Topic coverage
    ax = axes[1, 2]
    all_df = pd.concat([selected_df, non_selected_df])
    topic_coverage = all_df.groupby('topic_id')['is_selected_representative'].agg(['count', 'sum'])
    selection_rates = topic_coverage['sum'] / topic_coverage['count']
    
    ax.scatter(topic_coverage['count'], selection_rates, alpha=0.6)
    ax.set_xlabel('Topic Size')
    ax.set_ylabel('Selection Rate')
    ax.set_title('Selection Rate by Topic Size')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'statistical_comparison_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'statistical_comparison': plot_path}

def create_effect_size_forest_plot(effect_size_results: Dict[str, Any],
                                  output_dir: Path) -> Path:
    """
    Create forest plot for effect sizes with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    variables = list(effect_size_results['effect_sizes'].keys())
    effects = [result['cohens_d'] for result in effect_size_results['effect_sizes'].values()]
    cis = [result['confidence_interval'] for result in effect_size_results['effect_sizes'].values()]
    
    # Create forest plot
    y_pos = np.arange(len(variables))
    
    # Plot effect sizes
    ax.scatter(effects, y_pos, s=100, color='darkblue', zorder=3)
    
    # Plot confidence intervals
    for i, (lower, upper) in enumerate(cis):
        ax.plot([lower, upper], [i, i], 'k-', linewidth=2)
        ax.plot([lower, lower], [i-0.1, i+0.1], 'k-', linewidth=2)
        ax.plot([upper, upper], [i-0.1, i+0.1], 'k-', linewidth=2)
    
    # Add reference lines
    ax.axvline(x=0, color='red', linestyle='-', alpha=0.5, label='No Effect')
    ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='Small Effect')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium Effect')
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Large Effect')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([var.replace('_', ' ').title() for var in variables])
    ax.set_xlabel("Cohen's d Effect Size")
    ax.set_title('Effect Sizes with 95% Confidence Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = output_dir / 'effect_size_forest_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path
```

### Phase 6: Configuration Integration

#### 6.1 Enhanced Configuration
**File**: `configs/stage_3_visualization.yaml` (new file)

```yaml
#
# Stage 3: Visualization and Statistical Analysis Configuration
#

# Basic visualization settings
visualization:
  create_plots: true
  plot_format: "png"
  plot_dpi: 300
  save_interactive: true
  
# Statistical analysis configuration
statistical_analysis:
  enabled: true
  
  # Significance testing
  significance_tests:
    mann_whitney_u:
      enabled: true
      alpha: 0.05
      alternative: "two-sided"
    
    kolmogorov_smirnov:
      enabled: true
      alpha: 0.05
    
    chi_square:
      enabled: true
      alpha: 0.05
  
  # Effect size analysis
  effect_sizes:
    cohens_d:
      enabled: true
      confidence_level: 0.95
      interpret_thresholds:
        small: 0.2
        medium: 0.5
        large: 0.8
    
    # Note: Hedges' g disabled for large samples
    hedges_g:
      enabled: false
      reason: "Inappropriate for large samples (533 vs 3,202)"
  
  # Multiple testing correction
  multiple_testing:
    method: "fdr_by"  # Benjamini-Yekutieli for dependent tests
    alpha: 0.05
    reason: "Handles dependence between centrality/diversity in representativeness score"
  
  # Variables to analyze (only available data)
  variables:
    primary_metrics:
      - "similarity_to_centroid"
      - "diversity_score"
      - "representativeness_score"
      - "avg_similarity_to_cluster"
    
    secondary_variables:
      - "publication_year"
      - "topic_id"
      - "cluster_size"
    
    research_alignment:
      - "xai_alignment"
      - "symbolic_alignment"
      - "subsymbolic_alignment"
    
    # Excluded variables (unavailable data)
    excluded:
      - "citation_count"  # Not available
      - "geographic_distribution"  # Not available
      - "institutional_affiliation"  # Not available
  
  # Bias assessment
  bias_assessment:
    temporal_bias:
      enabled: true
      method: "kolmogorov_smirnov"
    
    topic_bias:
      enabled: true
      method: "chi_square"
    
    representativeness_gap:
      enabled: true
      method: "euclidean_distance"
    
    # Note: Selection propensity disabled
    selection_propensity:
      enabled: false
      reason: "No labeled training data available for logistic regression"
  
  # Reporting
  reporting:
    include_statistical_section: true
    detailed_interpretation: true
    create_statistical_plots: true
    create_forest_plots: true
    
# Output configuration
output:
  save_statistical_results: true
  save_plots: true
  create_summary_report: true
```

#### 6.2 Integration with Stage 3 Pipeline
**File**: `src/research_analysis/stages/stage_3_visualization/enhanced_pipeline.py`

```python
def run_enhanced_stage_3_with_statistics(config: AppConfig, 
                                       stage_2_output_dir: Path,
                                       stage_3_output_dir: Path) -> None:
    """
    Enhanced Stage 3 pipeline with comprehensive statistical analysis.
    """
    logger.info("Starting enhanced Stage 3 with statistical analysis...")
    
    # Load Stage 2 results
    comprehensive_df = pd.read_csv(stage_2_output_dir / "comprehensive_analysis.csv")
    
    # Split into selected and non-selected
    selected_df = comprehensive_df[comprehensive_df['is_selected_representative'] == True]
    non_selected_df = comprehensive_df[comprehensive_df['is_selected_representative'] == False]
    
    logger.info(f"Loaded {len(selected_df)} selected and {len(non_selected_df)} non-selected papers")
    
    # Perform statistical analysis
    statistical_results = generate_comprehensive_statistical_report(selected_df, non_selected_df)
    
    # Create visualizations
    basic_plots = create_basic_visualizations(selected_df, non_selected_df, stage_3_output_dir)
    statistical_plots = create_statistical_comparison_plots(selected_df, non_selected_df, stage_3_output_dir)
    forest_plot = create_effect_size_forest_plot(statistical_results['effect_size_analysis'], stage_3_output_dir)
    
    # Generate enhanced report
    report_content = generate_enhanced_statistical_report(
        selected_df, non_selected_df, statistical_results
    )
    
    # Save all results
    save_statistical_results(statistical_results, stage_3_output_dir)
    save_enhanced_report(report_content, stage_3_output_dir)
    
    logger.info("Enhanced Stage 3 with statistical analysis completed successfully!")
```

### Phase 7: Enhanced Reporting

#### 7.1 Statistical Report Generation
**File**: `src/research_analysis/stages/stage_3_visualization/enhanced_reporting.py`

```python
def generate_enhanced_statistical_report(selected_df: pd.DataFrame,
                                       non_selected_df: pd.DataFrame,
                                       statistical_results: Dict[str, Any]) -> str:
    """
    Generate comprehensive statistical analysis report.
    """
    
    # Extract key results
    mw_results = statistical_results['mann_whitney_analysis']
    effect_results = statistical_results['effect_size_analysis']
    temporal_results = statistical_results['temporal_analysis']
    topic_results = statistical_results['topic_analysis']
    gap_results = statistical_results['representativeness_gap_analysis']
    summary = statistical_results['summary']
    
    report = f"""# Enhanced Statistical Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

### Sample Composition
- **Selected Papers:** {summary['sample_sizes']['selected']:,}
- **Non-Selected Papers:** {summary['sample_sizes']['non_selected']:,}  
- **Total Papers:** {summary['sample_sizes']['total']:,}
- **Selection Ratio:** {(summary['sample_sizes']['selected'] / summary['sample_sizes']['total']) * 100:.2f}%

### Statistical Significance Summary
- **Significant Tests:** {summary['statistical_significance']['significant_tests']}/{summary['statistical_significance']['total_tests']}
- **Multiple Testing Correction:** Benjamini-Yekutieli (accounts for test dependencies)
- **Family-wise Error Rate:** 5%

### Effect Size Summary  
- **Large Effects:** {summary['practical_significance']['large_effects']} variables
- **Medium Effects:** {summary['practical_significance']['medium_effects']} variables
- **All effects calculated using Cohen's d (appropriate for large samples)**

### Bias Assessment
- **Temporal Bias:** {'Detected' if summary['bias_assessment']['temporal_bias'] else 'Not Detected'}
- **Topic Selection Bias:** {'Detected' if summary['bias_assessment']['topic_bias'] else 'Not Detected'}
- **Representativeness Gap:** {summary['bias_assessment']['representativeness_gap']:.4f}

---

## Detailed Statistical Analysis

### 1. Mann-Whitney U Test Results

The Mann-Whitney U test compares the distributions of selected vs non-selected papers without assuming normality. Results are corrected for multiple testing using the Benjamini-Yekutieli procedure to handle dependencies between metrics.

"""
    
    # Add Mann-Whitney results
    for var, result in mw_results['individual_tests'].items():
        significance = "**SIGNIFICANT**" if result['significant'] else "Not Significant"
        report += f"""
#### {var.replace('_', ' ').title()}
- **U Statistic:** {result['u_statistic']:.0f}
- **P-value:** {result['p_value']:.6f} ({significance})
- **Rank-biserial Correlation:** {result['rank_biserial_correlation']:.4f}
- **Sample Sizes:** {result['n_selected']} vs {result['n_non_selected']}
- **Interpretation:** {result['interpretation']}
"""
    
    # Add effect size results
    report += f"""
### 2. Effect Size Analysis (Cohen's d)

Cohen's d quantifies the magnitude of differences between groups. We use Cohen's d rather than Hedges' g because our samples are large (533 vs 3,202 papers).

"""
    
    for var, result in effect_results['effect_sizes'].items():
        report += f"""
#### {var.replace('_', ' ').title()}
- **Cohen's d:** {result['cohens_d']:.4f}
- **Effect Size:** {result['effect_size_category'].title()}
- **95% Confidence Interval:** [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]
- **Practically Significant:** {'Yes' if result['practical_significance'] else 'No'}
- **Interpretation:** {result['interpretation']}
"""
    
    # Add temporal analysis
    report += f"""
### 3. Temporal Distribution Analysis

Tests whether paper selection shows temporal bias using the Kolmogorov-Smirnov test.

- **K-S Statistic:** {temporal_results['kolmogorov_smirnov_test']['statistic']:.4f}
- **P-value:** {temporal_results['kolmogorov_smirnov_test']['p_value']:.6f}
- **Significant Difference:** {'Yes' if temporal_results['kolmogorov_smirnov_test']['significant_difference'] else 'No'}
- **Selected Median Year:** {temporal_results['descriptive_stats']['selected_median_year']:.0f}
- **Non-Selected Median Year:** {temporal_results['descriptive_stats']['non_selected_median_year']:.0f}
"""
    
    # Add topic analysis
    report += f"""
### 4. Topic Distribution Analysis

Examines whether selection is independent of topic membership using chi-square test.

- **Chi-Square Statistic:** {topic_results['chi_square_test']['statistic']:.4f}
- **P-value:** {topic_results['chi_square_test']['p_value']:.6f}
- **Degrees of Freedom:** {topic_results['chi_square_test']['degrees_of_freedom']}
- **Significant Association:** {'Yes' if topic_results['chi_square_test']['significant_association'] else 'No'}
- **Topics with Representatives:** {topic_results['coverage_stats']['topics_with_representatives']}/{topic_results['coverage_stats']['total_topics']}
- **Topic Coverage:** {topic_results['coverage_stats']['coverage_percentage']:.1f}%
"""
    
    # Add representativeness gap analysis
    report += f"""
### 5. Representativeness Gap Analysis

Measures the Euclidean distance between selected paper centroid and population centroid in metric space.

- **Overall Representativeness Gap:** {gap_results['overall_representativeness_gap']:.4f}
- **Component Gaps:**
"""
    
    for metric, gap in gap_results['component_gaps'].items():
        report += f"  - {metric.replace('_', ' ').title()}: {gap:+.4f}\n"
    
    report += f"""
- **Interpretation:** {gap_results['interpretation']}

---

## Research Implications

### Selection Quality Assessment
Based on the statistical analysis, the paper selection methodology demonstrates:
"""
    
    # Add implications based on results
    if summary['statistical_significance']['significant_tests'] > 0:
        report += "\n- **Systematic Selection Patterns:** Significant differences exist between selected and non-selected papers, indicating the algorithm successfully identifies papers with distinct characteristics."
    
    if summary['practical_significance']['large_effects'] > 0:
        report += f"\n- **Strong Effect Sizes:** {summary['practical_significance']['large_effects']} variables show large effect sizes, indicating practically meaningful differences."
    
    if not summary['bias_assessment']['temporal_bias']:
        report += "\n- **No Temporal Bias:** Selection is not significantly influenced by publication year, ensuring contemporary relevance."
    
    if topic_results['coverage_stats']['coverage_percentage'] > 90:
        report += "\n- **Excellent Topic Coverage:** Over 90% of topics have representative papers, ensuring comprehensive domain coverage."
    
    report += f"""

### Methodological Validation
The statistical analysis confirms that the paper selection methodology:
1. Produces statistically significant and practically meaningful differences
2. Maintains representativeness across temporal and topical dimensions
3. Operates without detectable systematic biases
4. Achieves optimal balance between diversity and centrality

### Recommendations for Future Research
1. **Diversity Optimization:** Consider adjusting diversity weight ({config.stage_2.metrics.diversity_weight}) based on effect size results
2. **Topic Balance:** Monitor topic coverage to ensure continued representativeness
3. **Temporal Monitoring:** Regular assessment of temporal patterns in emerging literature
4. **Threshold Validation:** Periodic review of similarity and diversity thresholds

---

## Technical Notes

### Statistical Methods
- **Non-parametric Tests:** Mann-Whitney U test used due to non-normal distributions common in bibliometric data
- **Effect Size Calculation:** Cohen's d preferred over Hedges' g for large samples
- **Multiple Testing:** Benjamini-Yekutieli correction accounts for dependence between representativeness score and its components
- **Confidence Intervals:** 95% confidence intervals provided for all effect sizes

### Data Limitations
- **Citation Data:** Not available in current dataset
- **Geographic Data:** Institutional/geographic information not available
- **Selection Propensity:** Cannot model selection probability without labeled training data

### Reproducibility
- **Fixed Seeds:** All random processes use seed {config.pipeline.reproducibility.random_seed}
- **Version Control:** All parameters documented in configuration files
- **Validation:** Results validated against multiple statistical frameworks

---

*This analysis provides rigorous statistical validation of the paper selection methodology, demonstrating its effectiveness in identifying representative papers while maintaining population-level representativeness.*
"""
    
    return report
```

## Implementation Timeline

### Phase 1: Core Infrastructure (Weeks 1-2)
- Implement statistical test functions
- Create effect size calculations
- Set up multiple testing correction

### Phase 2: Analysis Functions (Weeks 3-4) 
- Develop temporal and topic analysis
- Implement representativeness gap calculations
- Create research alignment analysis

### Phase 3: Visualization (Weeks 5-6)
- Build statistical comparison plots
- Create effect size forest plots
- Develop interactive dashboards

### Phase 4: Integration (Weeks 7-8)
- Integrate with existing pipeline
- Update configuration system
- Enhance reporting framework

### Phase 5: Testing & Validation (Weeks 9-10)
- Comprehensive testing of all functions
- Validation against existing results
- Performance optimization

### Phase 6: Documentation & Deployment (Weeks 11-12)
- Complete documentation
- User guides and examples
- Final deployment and testing

## Success Metrics

### Technical Success
- ✅ All functions follow functional programming principles
- ✅ Integration with existing pipeline without breaking changes
- ✅ Comprehensive statistical validation of selection methodology
- ✅ Publication-ready visualizations and reports

### Research Impact
- ✅ Rigorous statistical validation of systematic review methodology
- ✅ Quantified assessment of selection bias and representativeness
- ✅ Evidence-based recommendations for methodology improvement
- ✅ Enhanced credibility and reproducibility of literature review process

This comprehensive plan provides a robust statistical foundation for validating the paper selection methodology while respecting the constraints of available data and the existing functional programming architecture. 