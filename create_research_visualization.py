#!/usr/bin/env python3
"""
Research Landscape Visualization
Create comprehensive visualizations for the systematic literature review
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_research_visualizations():
    """Create comprehensive research landscape visualizations."""
    # Read the data
    taxonomy_df = pd.read_csv('systematic_review/research_taxonomy.csv')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Document count by topic (top 15)
    ax1 = plt.subplot(2, 3, 1)
    top_topics = taxonomy_df.head(15)
    y_pos = np.arange(len(top_topics))
    bars = ax1.barh(y_pos, top_topics['document_count'], color='steelblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"T{row.topic_id}: {row.top_terms.split(', ')[0]}" 
                        for _, row in top_topics.iterrows()], fontsize=8)
    ax1.set_xlabel('Number of Documents')
    ax1.set_title('Top 15 Topics by Document Count', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 5, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontsize=8)
    
    # 2. Research alignment distribution
    ax2 = plt.subplot(2, 3, 2)
    scores = taxonomy_df[['xai_score', 'symbolic_score', 'subsymbolic_score']].sum()
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    wedges, texts, autotexts = ax2.pie(scores, labels=['XAI', 'Symbolic', 'Sub-symbolic'], 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Research Focus Distribution\n(Total Keyword Matches)', fontsize=12, fontweight='bold')
    
    # 3. Priority distribution
    ax3 = plt.subplot(2, 3, 3)
    priority_counts = taxonomy_df['priority'].value_counts()
    bars = ax3.bar(priority_counts.index, priority_counts.values, 
                  color=['#2ecc71', '#f39c12', '#e74c3c'], alpha=0.7)
    ax3.set_title('Topic Priority Distribution', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Topics')
    ax3.set_xlabel('Priority Level')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # 4. Topic alignment scatter plot
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(taxonomy_df['symbolic_score'], taxonomy_df['subsymbolic_score'], 
                         s=taxonomy_df['document_count']*2, alpha=0.6, 
                         c=taxonomy_df['xai_score'], cmap='Reds', 
                         edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Symbolic Score')
    ax4.set_ylabel('Sub-symbolic Score')
    ax4.set_title('Topic Alignment by Research Dimensions\n(size=docs, color=XAI)', 
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('XAI Score')
    
    # 5. Research category breakdown
    ax5 = plt.subplot(2, 3, 5)
    category_counts = taxonomy_df['category'].value_counts()
    bars = ax5.bar(range(len(category_counts)), category_counts.values, 
                  color='lightcoral', alpha=0.7)
    ax5.set_xticks(range(len(category_counts)))
    ax5.set_xticklabels(category_counts.index, rotation=45, ha='right', fontsize=8)
    ax5.set_title('Topics by Research Category', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Number of Topics')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 6. Topic size distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(taxonomy_df['document_count'], bins=15, color='skyblue', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('Number of Documents per Topic')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of Topic Sizes', fontsize=12, fontweight='bold')
    ax6.axvline(taxonomy_df['document_count'].mean(), color='red', linestyle='--', 
               label=f'Mean: {taxonomy_df["document_count"].mean():.1f}')
    ax6.legend()
    
    plt.tight_layout(pad=3.0)
    
    # Save the visualization
    output_path = Path('systematic_review')
    plt.savefig(output_path / 'research_landscape_overview.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("📊 Research landscape visualization saved!")
    
    # Create a summary statistics report
    stats_report = f"""
# Research Landscape Statistics

## Dataset Overview
- **Total Topics**: {len(taxonomy_df)}
- **Total Documents**: {taxonomy_df['document_count'].sum():,}
- **Average Documents per Topic**: {taxonomy_df['document_count'].mean():.1f}
- **Largest Topic**: {taxonomy_df['document_count'].max()} documents
- **Smallest Topic**: {taxonomy_df['document_count'].min()} documents

## Research Focus Distribution
- **XAI Keywords**: {taxonomy_df['xai_score'].sum()} total matches
- **Symbolic Keywords**: {taxonomy_df['symbolic_score'].sum()} total matches  
- **Sub-symbolic Keywords**: {taxonomy_df['subsymbolic_score'].sum()} total matches

## Priority Breakdown
- **High Priority Topics**: {len(taxonomy_df[taxonomy_df['priority'] == 'High'])} ({len(taxonomy_df[taxonomy_df['priority'] == 'High'])/len(taxonomy_df)*100:.1f}%)
- **Medium Priority Topics**: {len(taxonomy_df[taxonomy_df['priority'] == 'Medium'])} ({len(taxonomy_df[taxonomy_df['priority'] == 'Medium'])/len(taxonomy_df)*100:.1f}%)
- **Low Priority Topics**: {len(taxonomy_df[taxonomy_df['priority'] == 'Low'])} ({len(taxonomy_df[taxonomy_df['priority'] == 'Low'])/len(taxonomy_df)*100:.1f}%)

## Top 5 Largest Topics
"""
    
    top_5 = taxonomy_df.head(5)
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        stats_report += f"{i}. **Topic {row['topic_id']}** ({row['document_count']} docs): {row['top_terms']}\n"
    
    # Save statistics
    with open(output_path / 'research_statistics.md', 'w') as f:
        f.write(stats_report)
    
    print("📈 Research statistics saved!")
    return stats_report

if __name__ == "__main__":
    create_research_visualizations() 