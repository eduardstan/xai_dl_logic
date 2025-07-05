#!/usr/bin/env python3
"""
Enhanced Research Landscape Visualization
Create comprehensive visualizations incorporating diversity, centrality, and representativeness metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime

# Import data loading functionality
from data_loader_viz import load_analysis_data
from metrics_visualizations import create_metrics_overview
from selection_visualizations import create_selection_analysis
from interactive_visualizations import create_interactive_topic_explorer

def create_paper_assignment_network(df_all, output_dir):
    """Visualize the paper assignment network (representatives and their assigned papers)."""
    # Focus on a few representative topics for clarity
    selected_topics = df_all['topic_id'].value_counts().head(5).index
    
    fig, axes = plt.subplots(1, len(selected_topics), figsize=(20, 4))
    fig.suptitle('Paper Assignment Networks (Top 5 Topics)', fontsize=16, fontweight='bold')
    
    for i, topic_id in enumerate(selected_topics):
        ax = axes[i]
        topic_data = df_all[df_all['topic_id'] == topic_id].copy()
        
        representatives = topic_data[topic_data['is_selected_representative']]
        non_selected = topic_data[~topic_data['is_selected_representative']]
        
        # Plot representatives as large circles
        ax.scatter(representatives['similarity_to_centroid'], 
                  representatives['diversity_score'],
                  s=200, c='red', alpha=0.8, label='Representatives', 
                  edgecolors='black', linewidth=2)
        
        # Plot non-selected papers as smaller circles
        if len(non_selected) > 0:
            ax.scatter(non_selected['similarity_to_centroid'], 
                      non_selected['diversity_score'],
                      s=30, c='lightblue', alpha=0.6, label='Assigned Papers')
            
            # Draw lines from non-selected to their representatives
            for _, paper in non_selected.iterrows():
                if pd.notna(paper.get('representative_title', '')):
                    # Find the representative
                    rep_title = paper['representative_title']
                    rep_row = representatives[representatives['title'] == rep_title]
                    if not rep_row.empty:
                        rep = rep_row.iloc[0]
                        ax.plot([paper['similarity_to_centroid'], rep['similarity_to_centroid']], 
                               [paper['diversity_score'], rep['diversity_score']], 
                               'gray', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('Similarity to Centroid')
        ax.set_ylabel('Diversity Score')
        ax.set_title(f'Topic {topic_id}\n({len(topic_data)} papers)')
        ax.grid(alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'paper_assignment_networks.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("🕸️ Paper assignment networks visualization saved!")

def generate_enhanced_statistics_report(df_all, df_selected, df_summary, output_dir):
    """Generate comprehensive statistics report with new metrics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Fix negative diversity scores for statistics
    df_summary_fixed = df_summary.copy()
    df_summary_fixed['avg_diversity'] = np.maximum(df_summary_fixed['avg_diversity'], 0)
    
    report = f"""# Enhanced Systematic Literature Review Report

**Generated on:** {timestamp}

## 📊 Executive Summary

### Dataset Overview
- **Total Papers Analyzed:** {len(df_all):,}
- **Representative Papers Selected:** {len(df_selected):,}
- **Selection Ratio:** {len(df_selected)/len(df_all)*100:.2f}%
- **Topics Covered:** {len(df_summary)}
- **Papers with Topic Assignment:** {len(df_all[df_all['topic_id'] != -1]):,}

### Metrics Overview (Selected Papers)
- **Average Centrality:** {df_selected['similarity_to_centroid'].mean():.4f} ± {df_selected['similarity_to_centroid'].std():.4f}
- **Average Diversity:** {df_selected['diversity_score'].mean():.4f} ± {df_selected['diversity_score'].std():.4f}
- **Average Representativeness:** {df_selected['representativeness_score'].mean():.4f} ± {df_selected['representativeness_score'].std():.4f}

### Research Alignment (Selected Papers)
- **XAI Alignment:** {df_selected['xai_alignment'].mean():.4f} ({df_selected['xai_alignment'].sum():.0f} total matches)
- **Symbolic Alignment:** {df_selected['symbolic_alignment'].mean():.4f} ({df_selected['symbolic_alignment'].sum():.0f} total matches)
- **Sub-symbolic Alignment:** {df_selected['subsymbolic_alignment'].mean():.4f} ({df_selected['subsymbolic_alignment'].sum():.0f} total matches)

## 🎯 Selection Strategy Performance

### Dynamic Selection by Cluster Size
"""
    
    # Add cluster size analysis
    for category in ['small', 'medium', 'large', 'xlarge', 'xxlarge']:
        cat_topics = df_summary_fixed[df_summary_fixed['size_category'] == category]
        if len(cat_topics) > 0:
            report += f"- **{category.title()} clusters** ({len(cat_topics)} topics): "
            report += f"avg {cat_topics['papers_selected'].mean():.1f} papers selected, "
            report += f"{cat_topics['selection_ratio'].mean()*100:.1f}% selection ratio\\n"
    
    report += f"""
### Top 10 Most Efficient Selections (High Representativeness)
"""
    
    # Top efficient selections
    top_efficient = df_summary_fixed.nlargest(10, 'avg_centrality')[['topic_id', 'topic_name', 'cluster_size', 'papers_selected', 'avg_centrality', 'avg_diversity']]
    for _, row in top_efficient.iterrows():
        report += f"- **Topic {row['topic_id']}** ({row['cluster_size']} docs → {row['papers_selected']} selected): "
        report += f"Centrality={row['avg_centrality']:.3f}, Diversity={row['avg_diversity']:.3f}\\n"
    
    report += f"""
## 📈 Quality Metrics Analysis

### Distribution Statistics
| Metric | All Papers | Selected Papers | Improvement |
|--------|------------|----------------|-------------|
| Centrality | {df_all['similarity_to_centroid'].mean():.4f} ± {df_all['similarity_to_centroid'].std():.4f} | {df_selected['similarity_to_centroid'].mean():.4f} ± {df_selected['similarity_to_centroid'].std():.4f} | {((df_selected['similarity_to_centroid'].mean() / df_all['similarity_to_centroid'].mean()) - 1) * 100:+.1f}% |
| Diversity | {df_all['diversity_score'].mean():.4f} ± {df_all['diversity_score'].std():.4f} | {df_selected['diversity_score'].mean():.4f} ± {df_selected['diversity_score'].std():.4f} | {((df_selected['diversity_score'].mean() / df_all['diversity_score'].mean()) - 1) * 100:+.1f}% |
| Representativeness | {df_all['representativeness_score'].mean():.4f} ± {df_all['representativeness_score'].std():.4f} | {df_selected['representativeness_score'].mean():.4f} ± {df_selected['representativeness_score'].std():.4f} | {((df_selected['representativeness_score'].mean() / df_all['representativeness_score'].mean()) - 1) * 100:+.1f}% |

### Paper Assignment Coverage
"""
    
    # Assignment analysis
    assigned_papers = df_all[~df_all['is_selected_representative'] & df_all['representative_title'].notna()]
    high_similarity_assignments = assigned_papers[assigned_papers['similarity_to_representative'] >= 0.8]
    
    report += f"- **Papers assigned to representatives:** {len(assigned_papers):,} ({len(assigned_papers)/len(df_all)*100:.1f}%)\\n"
    report += f"- **High similarity assignments (≥0.8):** {len(high_similarity_assignments):,} ({len(high_similarity_assignments)/len(assigned_papers)*100:.1f}% of assignments)\\n"
    report += f"- **Average assignment similarity:** {assigned_papers['similarity_to_representative'].mean():.4f}\\n"
    
    report += f"""
## 🎖️ Top Representative Papers (Highest Representativeness)

"""
    
    # Top papers
    top_papers = df_selected.nlargest(10, 'representativeness_score')[['title', 'authors', 'topic_id', 'representativeness_score', 'similarity_to_centroid', 'diversity_score']]
    for i, (_, paper) in enumerate(top_papers.iterrows(), 1):
        report += f"{i}. **{paper['title']}** (Topic {paper['topic_id']})\\n"
        report += f"   - Authors: {paper['authors']}\\n"
        report += f"   - Representativeness: {paper['representativeness_score']:.4f}\\n"
        report += f"   - Centrality: {paper['similarity_to_centroid']:.4f}, Diversity: {paper['diversity_score']:.4f}\\n\\n"
    
    # Save report
    with open(output_dir / 'enhanced_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("📋 Enhanced analysis report saved!")
    return report

def main():
    """Main function to create all enhanced visualizations."""
    try:
        # Create output directory
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        print("🚀 Starting enhanced research visualization...")
        
        # Load data
        df_all, df_summary, df_selected = load_analysis_data()
        
        # Create visualizations
        print("\n📊 Creating metrics overview...")
        create_metrics_overview(df_all, df_selected, output_dir)
        
        print("\n📈 Creating selection analysis...")
        create_selection_analysis(df_summary, output_dir)
        
        print("\n🌐 Creating interactive visualizations...")
        create_interactive_topic_explorer(df_all, df_selected, df_summary, output_dir)
        
        print("\n🕸️ Creating paper assignment networks...")
        create_paper_assignment_network(df_all, output_dir)
        
        print("\n📋 Generating enhanced statistics report...")
        generate_enhanced_statistics_report(df_all, df_selected, df_summary, output_dir)
        
        print(f"\n✅ Enhanced visualizations completed!")
        print(f"📁 All outputs saved to: {output_dir.absolute()}")
        print("\n📊 Generated files:")
        print("  - metrics_overview.png")
        print("  - selection_analysis.png") 
        print("  - interactive_papers_explorer.html")
        print("  - topic_dashboard.html")
        print("  - paper_assignment_networks.png")
        print("  - enhanced_analysis_report.md")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        raise

if __name__ == "__main__":
    main() 