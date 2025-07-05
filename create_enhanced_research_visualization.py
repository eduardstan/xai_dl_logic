#!/usr/bin/env python3
"""
Enhanced Research Landscape Visualization
Create comprehensive visualizations incorporating diversity, centrality, and representativeness metrics
"""

from pathlib import Path

# Import data loading functionality
from data_loader_viz import load_analysis_data
from metrics_visualizations import create_metrics_overview
from selection_visualizations import create_selection_analysis
from interactive_visualizations import create_interactive_topic_explorer
from network_visualizations import create_paper_assignment_network
from report_generation import generate_enhanced_statistics_report


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