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

def create_selection_analysis(df_summary, output_dir):
    """Create visualizations analyzing the selection strategy."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Selection Strategy Analysis', fontsize=16, fontweight='bold')
    
    # 1. Papers selected vs cluster size
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df_summary['cluster_size'], df_summary['papers_selected'], 
                         c=df_summary['avg_centrality'], cmap='coolwarm', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Cluster Size')
    ax1.set_ylabel('Papers Selected')
    ax1.set_title('Selection Pattern by Cluster Size\n(color = avg centrality)')
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Avg Centrality')
    
    # Add ideal line
    x_ideal = np.linspace(0, df_summary['cluster_size'].max(), 100)
    y_ideal = np.minimum(3 + (x_ideal / 50), 8)  # Based on config thresholds
    ax1.plot(x_ideal, y_ideal, 'r--', alpha=0.7, label='Ideal Selection')
    ax1.legend()
    
    # 2. Selection ratio vs cluster size
    ax2 = axes[0, 1]
    colors = ['red' if x < 0.05 else 'orange' if x < 0.1 else 'green' for x in df_summary['selection_ratio']]
    bars = ax2.bar(range(len(df_summary)), df_summary['selection_ratio'], color=colors, alpha=0.7)
    ax2.set_xlabel('Topic ID')
    ax2.set_ylabel('Selection Ratio')
    ax2.set_title('Selection Ratio by Topic\n(red<5%, orange<10%, green≥10%)')
    ax2.set_xticks(range(0, len(df_summary), 5))
    ax2.set_xticklabels(df_summary['topic_id'].iloc[::5])
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Average centrality vs diversity trade-off
    ax3 = axes[1, 0]
    # Fix negative diversity scores (clamp to 0)
    df_summary_fixed = df_summary.copy()
    df_summary_fixed['avg_diversity'] = np.maximum(df_summary_fixed['avg_diversity'], 0)
    
    scatter = ax3.scatter(df_summary_fixed['avg_centrality'], df_summary_fixed['avg_diversity'], 
                         s=df_summary_fixed['cluster_size']*2, 
                         c=df_summary_fixed['papers_selected'], cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Average Centrality')
    ax3.set_ylabel('Average Diversity')
    ax3.set_title('Centrality vs Diversity Trade-off\n(size = cluster size, color = papers selected)')
    ax3.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Papers Selected')
    
    # 4. Topic size categories
    ax4 = axes[1, 1]
    size_counts = df_summary['size_category'].value_counts()
    colors_cat = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db']
    wedges, texts, autotexts = ax4.pie(size_counts.values, labels=size_counts.index, 
                                      autopct='%1.1f%%', colors=colors_cat[:len(size_counts)], 
                                      startangle=90)
    ax4.set_title('Distribution of Topic Size Categories')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'selection_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("📈 Selection analysis visualization saved!")

def create_interactive_topic_explorer(df_all, df_selected, df_summary, output_dir):
    """Create interactive visualizations using Plotly."""
    
    # 1. Interactive scatter plot of all papers
    fig1 = px.scatter(df_selected, 
                     x='similarity_to_centroid', 
                     y='diversity_score',
                     size='cluster_size',
                     color='representativeness_score',
                     hover_data=['topic_id', 'title', 'authors'],
                     color_continuous_scale='viridis',
                     title='Interactive Explorer: Selected Representative Papers')
    
    fig1.update_layout(
        xaxis_title="Similarity to Centroid (Centrality)",
        yaxis_title="Diversity Score",
        width=1000,
        height=700
    )
    
    fig1.write_html(str(output_dir / 'interactive_papers_explorer.html'))
    
    # 2. Topic comparison dashboard
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cluster Size vs Papers Selected', 
                       'Average Metrics by Topic',
                       'Selection Efficiency',
                       'Research Alignment Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "domain"}]]
    )
    
    # Subplot 1: Cluster size vs papers selected
    fig2.add_trace(
        go.Scatter(x=df_summary['cluster_size'], 
                  y=df_summary['papers_selected'],
                  mode='markers',
                  marker=dict(size=10, color=df_summary['avg_centrality'], 
                            colorscale='RdYlBu', showscale=False),
                  text=df_summary['topic_name'],
                  name='Topics'),
        row=1, col=1
    )
    
    # Subplot 2: Average metrics
    fig2.add_trace(
        go.Bar(x=df_summary['topic_id'][:10], 
               y=df_summary['avg_centrality'][:10],
               name='Avg Centrality',
               marker_color='lightblue'),
        row=1, col=2
    )
    
    fig2.add_trace(
        go.Bar(x=df_summary['topic_id'][:10], 
               y=np.maximum(df_summary['avg_diversity'][:10], 0),  # Fix negative values
               name='Avg Diversity',
               marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Subplot 3: Selection efficiency
    efficiency = df_summary['avg_centrality'] * np.maximum(df_summary['avg_diversity'], 0)
    fig2.add_trace(
        go.Scatter(x=df_summary['cluster_size'], 
                  y=efficiency,
                  mode='markers',
                  marker=dict(size=df_summary['papers_selected']*3, 
                            color='green', opacity=0.7),
                  name='Efficiency'),
        row=2, col=1
    )
    
    # Subplot 4: Research alignment pie chart
    total_xai = df_selected['xai_alignment'].sum()
    total_symbolic = df_selected['symbolic_alignment'].sum() 
    total_subsymbolic = df_selected['subsymbolic_alignment'].sum()
    
    fig2.add_trace(
        go.Pie(labels=['XAI', 'Symbolic', 'Sub-symbolic'],
               values=[total_xai, total_symbolic, total_subsymbolic],
               name="Research Focus"),
        row=2, col=2
    )
    
    fig2.update_layout(height=800, showlegend=True, 
                      title_text="Topic Analysis Dashboard")
    
    fig2.write_html(str(output_dir / 'topic_dashboard.html'))
    
    print("🌐 Interactive visualizations created!")

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