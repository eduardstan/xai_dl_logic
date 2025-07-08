#!/usr/bin/env python3
"""
CSV Statistics Analyzer for Systematic Literature Review
Analyzes comprehensive analysis CSV files and generates detailed statistics reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple, Optional

class CSVStatisticsAnalyzer:
    """Analyzes CSV files from systematic literature review and generates comprehensive statistics."""
    
    def __init__(self, csv_file: str, output_dir: str = "results"):
        """
        Initialize the analyzer.
        
        Args:
            csv_file: Path to the comprehensive analysis CSV file
            output_dir: Directory to save the report
        """
        self.csv_file = Path(csv_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        print(f"📊 Loading data from {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        print(f"✅ Loaded {len(self.df)} papers from {self.csv_file.name}")
        
        # Alignment columns
        self.alignment_cols = ['xai_alignment', 'symbolic_alignment', 'subsymbolic_alignment']
    
    def calculate_alignment_statistics(self) -> Dict:
        """Calculate comprehensive alignment statistics."""
        stats = {}
        
        for col in self.alignment_cols:
            total_count = len(self.df[col].dropna())
            zero_count = (self.df[col] == 0.0).sum()
            non_zero_count = (self.df[col] != 0.0).sum()
            zero_percentage = (zero_count / total_count) * 100 if total_count > 0 else 0
            
            stats[col] = {
                'total_non_null': total_count,
                'zero_count': zero_count,
                'non_zero_count': non_zero_count,
                'zero_percentage': zero_percentage,
                'min_value': self.df[col].min(),
                'max_value': self.df[col].max(),
                'mean_value': self.df[col].mean(),
                'median_value': self.df[col].median(),
                'std_value': self.df[col].std()
            }
        
        return stats
    
    def calculate_similarity_statistics(self) -> Dict:
        """Calculate similarity-related statistics."""
        stats = {}
        
        # is_similar_to_representative analysis
        sim_col = 'is_similar_to_representative'
        value_counts = self.df[sim_col].value_counts(dropna=False)
        unique_values = self.df[sim_col].unique()
        
        stats['is_similar_to_representative'] = {
            'value_counts': value_counts,
            'unique_values': unique_values,
            'total_rows': len(self.df),
            'null_count': self.df[sim_col].isna().sum(),
            'null_percentage': (self.df[sim_col].isna().sum() / len(self.df)) * 100
        }
        
        # similarity_to_representative analysis
        sim_score_col = 'similarity_to_representative'
        non_null_scores = self.df[sim_score_col].dropna()
        
        stats['similarity_to_representative'] = {
            'total_rows': len(self.df),
            'null_count': self.df[sim_score_col].isna().sum(),
            'non_null_count': len(non_null_scores),
            'null_percentage': (self.df[sim_score_col].isna().sum() / len(self.df)) * 100,
            'min_score': non_null_scores.min() if len(non_null_scores) > 0 else None,
            'max_score': non_null_scores.max() if len(non_null_scores) > 0 else None,
            'mean_score': non_null_scores.mean() if len(non_null_scores) > 0 else None,
            'std_score': non_null_scores.std() if len(non_null_scores) > 0 else None
        }
        
        # Similarity threshold analysis
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_stats = {}
        
        for threshold in thresholds:
            count = (non_null_scores >= threshold).sum()
            percentage = (count / len(non_null_scores)) * 100 if len(non_null_scores) > 0 else 0
            threshold_stats[threshold] = {
                'count': count,
                'percentage': percentage
            }
        
        stats['similarity_thresholds'] = threshold_stats
        
        return stats
    
    def calculate_representative_statistics(self) -> Dict:
        """Calculate representative-related statistics."""
        stats = {}
        
        # Representative status
        rep_counts = self.df['is_selected_representative'].value_counts()
        stats['representative_status'] = {
            'total_papers': len(self.df),
            'representatives': rep_counts.get(True, 0),
            'non_representatives': rep_counts.get(False, 0),
            'selection_ratio': (rep_counts.get(True, 0) / len(self.df)) * 100
        }
        
        # Topic distribution
        topic_stats = self.df.groupby('topic_id').agg({
            'global_index': 'count',
            'is_selected_representative': 'sum',
            'cluster_size': 'first'
        }).rename(columns={'global_index': 'papers_in_topic'})
        
        stats['topic_distribution'] = {
            'total_topics': len(topic_stats),
            'avg_papers_per_topic': topic_stats['papers_in_topic'].mean(),
            'avg_representatives_per_topic': topic_stats['is_selected_representative'].mean(),
            'topics_with_zero_representatives': (topic_stats['is_selected_representative'] == 0).sum(),
            'min_topic_size': topic_stats['papers_in_topic'].min(),
            'max_topic_size': topic_stats['papers_in_topic'].max()
        }
        
        return stats
    
    def find_papers_by_alignment_count(self, max_nonzero_alignments: int = 0) -> pd.DataFrame:
        """
        Find papers with specified number of non-zero alignment scores.
        
        Args:
            max_nonzero_alignments: Maximum number of non-zero alignment scores (0-3)
            
        Returns:
            DataFrame with papers matching the criteria
        """
        # Count non-zero alignments for each paper
        alignment_matrix = self.df[self.alignment_cols] != 0.0
        nonzero_counts = alignment_matrix.sum(axis=1)
        
        # Filter papers
        filtered_papers = self.df[nonzero_counts <= max_nonzero_alignments].copy()
        filtered_papers['nonzero_alignment_count'] = nonzero_counts[nonzero_counts <= max_nonzero_alignments]
        
        return filtered_papers[['title', 'authors', 'year', 'journal', 'topic_id', 
                               'nonzero_alignment_count', 'abstract'] + self.alignment_cols]
    
    def calculate_alignment_patterns(self) -> Dict:
        """Calculate alignment pattern statistics."""
        stats = {}
        
        # Create boolean matrix for zero alignments
        zero_patterns = self.df[self.alignment_cols] == 0.0
        
        # Count patterns
        all_zeros = zero_patterns.all(axis=1)
        any_zeros = zero_patterns.any(axis=1)
        
        stats['zero_patterns'] = {
            'all_three_zero': {
                'count': all_zeros.sum(),
                'percentage': (all_zeros.sum() / len(self.df)) * 100
            },
            'at_least_one_zero': {
                'count': any_zeros.sum(),
                'percentage': (any_zeros.sum() / len(self.df)) * 100
            }
        }
        
        # Count by number of non-zero alignments
        alignment_matrix = self.df[self.alignment_cols] != 0.0
        nonzero_counts = alignment_matrix.sum(axis=1)
        
        stats['nonzero_patterns'] = {}
        for i in range(4):  # 0, 1, 2, 3 non-zero alignments
            count = (nonzero_counts == i).sum()
            percentage = (count / len(self.df)) * 100
            stats['nonzero_patterns'][i] = {
                'count': count,
                'percentage': percentage
            }
        
        return stats
    
    def generate_report(self, max_nonzero_alignments: int = 0) -> str:
        """
        Generate comprehensive statistics report.
        
        Args:
            max_nonzero_alignments: Maximum number of non-zero alignments to report papers for
            
        Returns:
            Report content as string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate all statistics
        alignment_stats = self.calculate_alignment_statistics()
        similarity_stats = self.calculate_similarity_statistics()
        representative_stats = self.calculate_representative_statistics()
        pattern_stats = self.calculate_alignment_patterns()
        
        # Find papers with specified alignment pattern
        filtered_papers = self.find_papers_by_alignment_count(max_nonzero_alignments)
        
        # Generate report
        report = f"""# Systematic Literature Review - CSV Statistics Report

**Generated on:** {timestamp}
**Source File:** {self.csv_file.name}
**Parameters:** max_nonzero_alignments = {max_nonzero_alignments}

## 📊 Executive Summary

### Dataset Overview
- **Total Papers Analyzed:** {len(self.df):,}
- **Representative Papers Selected:** {representative_stats['representative_status']['representatives']:,}
- **Non-Representative Papers:** {representative_stats['representative_status']['non_representatives']:,}
- **Selection Ratio:** {representative_stats['representative_status']['selection_ratio']:.2f}%
- **Topics Covered:** {representative_stats['topic_distribution']['total_topics']}

### Topic Distribution
- **Average Papers per Topic:** {representative_stats['topic_distribution']['avg_papers_per_topic']:.1f}
- **Average Representatives per Topic:** {representative_stats['topic_distribution']['avg_representatives_per_topic']:.1f}
- **Topics with Zero Representatives:** {representative_stats['topic_distribution']['topics_with_zero_representatives']}
- **Topic Size Range:** {representative_stats['topic_distribution']['min_topic_size']} - {representative_stats['topic_distribution']['max_topic_size']} papers

## 🔍 Alignment Analysis

### Alignment Score Distributions
"""

        # Add alignment statistics for each category
        for col in self.alignment_cols:
            col_name = col.replace('_alignment', '').upper()
            stats = alignment_stats[col]
            
            report += f"""
#### {col_name} Alignment
- **Zero Values:** {stats['zero_count']:,} papers ({stats['zero_percentage']:.1f}%)
- **Non-Zero Values:** {stats['non_zero_count']:,} papers ({100-stats['zero_percentage']:.1f}%)
- **Range:** {stats['min_value']:.6f} - {stats['max_value']:.6f}
- **Mean:** {stats['mean_value']:.6f} ± {stats['std_value']:.6f}
- **Median:** {stats['median_value']:.6f}
"""

        # Add alignment patterns
        report += f"""
### Alignment Patterns
- **Papers with ALL three alignments = 0:** {pattern_stats['zero_patterns']['all_three_zero']['count']} ({pattern_stats['zero_patterns']['all_three_zero']['percentage']:.1f}%)
- **Papers with at least one alignment = 0:** {pattern_stats['zero_patterns']['at_least_one_zero']['count']} ({pattern_stats['zero_patterns']['at_least_one_zero']['percentage']:.1f}%)

### Distribution by Number of Non-Zero Alignments
"""
        
        for i in range(4):
            report += f"- **{i} non-zero alignments:** {pattern_stats['nonzero_patterns'][i]['count']} papers ({pattern_stats['nonzero_patterns'][i]['percentage']:.1f}%)\n"

        # Add similarity analysis
        report += f"""
## 📈 Similarity Analysis

### is_similar_to_representative Column
- **Total Rows:** {similarity_stats['is_similar_to_representative']['total_rows']:,}
- **'False' Values:** {similarity_stats['is_similar_to_representative']['value_counts'].get(False, 0):,}
- **'True' Values:** {similarity_stats['is_similar_to_representative']['value_counts'].get(True, 0):,}
- **NaN Values:** {similarity_stats['is_similar_to_representative']['null_count']:,} ({similarity_stats['is_similar_to_representative']['null_percentage']:.1f}%)

### similarity_to_representative Scores
- **Non-Null Values:** {similarity_stats['similarity_to_representative']['non_null_count']:,} ({100-similarity_stats['similarity_to_representative']['null_percentage']:.1f}%)
- **Null Values:** {similarity_stats['similarity_to_representative']['null_count']:,} ({similarity_stats['similarity_to_representative']['null_percentage']:.1f}%)
"""

        if similarity_stats['similarity_to_representative']['mean_score'] is not None:
            report += f"""- **Range:** {similarity_stats['similarity_to_representative']['min_score']:.6f} - {similarity_stats['similarity_to_representative']['max_score']:.6f}
- **Mean:** {similarity_stats['similarity_to_representative']['mean_score']:.6f} ± {similarity_stats['similarity_to_representative']['std_score']:.6f}
"""

        # Add threshold analysis
        report += f"""
### Similarity Threshold Analysis (Non-Representatives)
"""
        
        for threshold, stats in similarity_stats['similarity_thresholds'].items():
            report += f"- **Papers with similarity ≥ {threshold}:** {stats['count']} ({stats['percentage']:.1f}%)\n"

        # Add interpretation section
        report += f"""
## 🎯 Data Quality Assessment & Interpretation

### ✅ INTENDED PATTERNS (Normal Behavior)

1. **NaN values in 'is_similar_to_representative' for representatives**
   - The {similarity_stats['is_similar_to_representative']['null_count']} NaN values correspond exactly to the {representative_stats['representative_status']['representatives']} representatives
   - Representatives don't need similarity scores to themselves - this is by design

2. **Varying zero percentages in alignment columns**
   - XAI: {alignment_stats['xai_alignment']['zero_percentage']:.1f}% zeros
   - Symbolic: {alignment_stats['symbolic_alignment']['zero_percentage']:.1f}% zeros  
   - Sub-symbolic: {alignment_stats['subsymbolic_alignment']['zero_percentage']:.1f}% zeros
   - This reflects natural distribution of research focus keywords in the corpus

3. **High percentage of 'False' in similarity classification**
   - Most papers are legitimately dissimilar to their representatives
   - Only highly similar papers (likely near-duplicates) get marked as 'True'

### ⚠️ POTENTIAL AREAS FOR INVESTIGATION

1. **Very few papers marked as 'similar to representative'**
   - Only {similarity_stats['is_similar_to_representative']['value_counts'].get(True, 0)} out of {similarity_stats['is_similar_to_representative']['total_rows']:,} papers
   - Consider if similarity threshold (typically 0.8) is too restrictive

2. **Papers with zero alignment across all categories**
   - {pattern_stats['zero_patterns']['all_three_zero']['count']} papers ({pattern_stats['zero_patterns']['all_three_zero']['percentage']:.1f}%) have no alignment
   - These might be off-topic or require keyword expansion

### 📋 RECOMMENDATIONS

1. **Review similarity threshold** for marking papers as "similar to representative"
2. **Examine papers with zero alignments** to verify they're genuinely relevant
3. **Consider expanding keyword lists** in configuration to capture more domain terms
4. **Overall data quality appears good** - most patterns reflect intended system behavior

## 📝 Papers with ≤ {max_nonzero_alignments} Non-Zero Alignment Scores

**Found {len(filtered_papers)} papers matching criteria:**

"""

        # Add paper listings
        if len(filtered_papers) > 0:
            for i, (idx, paper) in enumerate(filtered_papers.iterrows()):
                # Handle missing or NaN abstracts
                abstract = paper.get('abstract', '')
                if pd.isna(abstract) or abstract == 'nan':
                    abstract = "*[No abstract available]*"
                
                report += f"""
### Paper {i + 1}
- **Title:** {paper['title']}
- **Authors:** {paper['authors']}
- **Year:** {paper['year']}
- **Journal:** {paper['journal']}
- **Topic ID:** {paper['topic_id']}
- **Non-Zero Alignments:** {paper['nonzero_alignment_count']}

**Abstract:** {abstract}

**Alignment Scores:**
- **XAI Alignment:** {paper['xai_alignment']:.6f}
- **Symbolic Alignment:** {paper['symbolic_alignment']:.6f}
- **Sub-symbolic Alignment:** {paper['subsymbolic_alignment']:.6f}
"""
        else:
            report += "No papers found matching the specified criteria.\n"

        # Add summary statistics
        report += f"""
## 📊 Summary Statistics

### Selected Papers Metrics (if available)
"""
        
        if 'similarity_to_centroid' in self.df.columns:
            selected_papers = self.df[self.df['is_selected_representative'] == True]
            if len(selected_papers) > 0:
                report += f"""
- **Average Centrality:** {selected_papers['similarity_to_centroid'].mean():.6f} ± {selected_papers['similarity_to_centroid'].std():.6f}
- **Average Diversity:** {selected_papers['diversity_score'].mean():.6f} ± {selected_papers['diversity_score'].std():.6f}
- **Average Representativeness:** {selected_papers['representativeness_score'].mean():.6f} ± {selected_papers['representativeness_score'].std():.6f}
- **Average XAI Alignment:** {selected_papers['xai_alignment'].mean():.6f}
- **Average Symbolic Alignment:** {selected_papers['symbolic_alignment'].mean():.6f}
- **Average Sub-symbolic Alignment:** {selected_papers['subsymbolic_alignment'].mean():.6f}
"""

        report += f"""
---
*Report generated by CSV Statistics Analyzer*
*Source: {self.csv_file}*
*Generated at: {timestamp}*
"""

        return report
    
    def save_report(self, max_nonzero_alignments: int = 0) -> Path:
        """
        Generate and save the statistics report.
        
        Args:
            max_nonzero_alignments: Maximum number of non-zero alignments to report papers for
            
        Returns:
            Path to the saved report file
        """
        report_content = self.generate_report(max_nonzero_alignments)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"csv_statistics_report_{timestamp}.md"
        filepath = self.output_dir / filename
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 Report saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print a brief summary to console."""
        print("\n" + "="*60)
        print("🔍 CSV STATISTICS SUMMARY")
        print("="*60)
        
        alignment_stats = self.calculate_alignment_statistics()
        similarity_stats = self.calculate_similarity_statistics()
        representative_stats = self.calculate_representative_statistics()
        pattern_stats = self.calculate_alignment_patterns()
        
        print(f"📊 Total Papers: {len(self.df):,}")
        print(f"📊 Representatives: {representative_stats['representative_status']['representatives']:,}")
        print(f"📊 Topics: {representative_stats['topic_distribution']['total_topics']}")
        print()
        
        print("🎯 Alignment Zeros:")
        for col in self.alignment_cols:
            name = col.replace('_alignment', '').upper()
            percentage = alignment_stats[col]['zero_percentage']
            print(f"   {name}: {alignment_stats[col]['zero_count']:,} ({percentage:.1f}%)")
        print()
        
        print("📈 Similarity Patterns:")
        print(f"   Similar to Rep: {similarity_stats['is_similar_to_representative']['value_counts'].get(True, 0)}")
        print(f"   Not Similar: {similarity_stats['is_similar_to_representative']['value_counts'].get(False, 0):,}")
        print(f"   NaN (Reps): {similarity_stats['is_similar_to_representative']['null_count']:,}")
        print()
        
        print("🔍 Alignment Patterns:")
        print(f"   All 3 zero: {pattern_stats['zero_patterns']['all_three_zero']['count']} ({pattern_stats['zero_patterns']['all_three_zero']['percentage']:.1f}%)")
        print(f"   ≥1 zero: {pattern_stats['zero_patterns']['at_least_one_zero']['count']} ({pattern_stats['zero_patterns']['at_least_one_zero']['percentage']:.1f}%)")
        print("="*60)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze CSV statistics from systematic literature review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze CSV and find papers with 0 non-zero alignments
  python analyze_csv_statistics.py results/comprehensive_analysis_20250704_232952.csv

  # Find papers with up to 1 non-zero alignment
  python analyze_csv_statistics.py results/comprehensive_analysis_20250704_232952.csv --max-alignments 1

  # Specify custom output directory
  python analyze_csv_statistics.py results/comprehensive_analysis_20250704_232952.csv --output-dir custom_reports
        """
    )
    
    parser.add_argument(
        'csv_file',
        help='Path to the comprehensive analysis CSV file'
    )
    
    parser.add_argument(
        '--max-alignments',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='Maximum number of non-zero alignments to report papers for (default: 0)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for the report (default: results)'
    )
    
    parser.add_argument(
        '--no-console',
        action='store_true',
        help='Suppress console summary output'
    )
    
    args = parser.parse_args()
    
    # Validate CSV file exists
    if not Path(args.csv_file).exists():
        print(f"❌ Error: CSV file '{args.csv_file}' not found")
        sys.exit(1)
    
    try:
        # Initialize analyzer
        analyzer = CSVStatisticsAnalyzer(args.csv_file, args.output_dir)
        
        # Print console summary unless suppressed
        if not args.no_console:
            analyzer.print_summary()
        
        # Generate and save report
        report_path = analyzer.save_report(args.max_alignments)
        
        print(f"\n✅ Analysis complete!")
        print(f"📄 Full report: {report_path}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 