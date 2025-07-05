#!/usr/bin/env python3
"""
Advanced Systematic Review Analyzer with Diverse Representative Paper Selection

This module implements a sophisticated paper selection strategy that:
1. Selects diverse representative papers within each cluster
2. Computes comprehensive similarity metrics
3. Maps non-selected papers to their most similar representatives
4. Uses configurable parameters from config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict
import logging

# Import extracted functionality modules
from topic_processor import process_topics
from report_generator import generate_analysis_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedSystematicAnalyzer:
    """Advanced analyzer for systematic literature reviews with diverse paper selection."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the analyzer with configuration.
        
        This class now serves as a lightweight orchestrator that coordinates
        the systematic literature review analysis using specialized modules.
        
        Args:
            config_path: Path to the YAML configuration file
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
            Exception: If initialization fails
        """
        self.config = self.load_config(config_path)
        self.results_dir = Path(self.config['output']['results_dir'])
        
        # Ensure output directories exist
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("🔬 Advanced Systematic Analyzer initialized")
        
    def load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict: Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            raise
    
    def run_analysis(self):
        """
        Run the complete systematic literature review analysis.
        
        This is the main entry point that orchestrates the entire analysis pipeline:
        1. Process topics and select representative papers
        2. Generate comprehensive analysis report
        3. Return results and report path
        
        Returns:
            Tuple containing (results_df, summary_df, selected_df, report_path)
            
        Raises:
            Exception: If any step of the analysis fails
        """
        try:
            logger.info("🚀 Starting systematic literature review analysis...")
            
            # Process topics and select representatives
            results_df, summary_df, selected_df = process_topics(self.config)
            
            # Generate comprehensive report
            report_path = generate_analysis_report(results_df, summary_df, selected_df, self.config)
            
            logger.info("🎉 Analysis completed successfully!")
            logger.info(f"📊 {len(selected_df)} representatives selected from {len(results_df)} total papers")
            logger.info(f"📁 Results saved to: {self.results_dir}")
            logger.info(f"📋 Report saved to: {report_path}")
            
            return results_df, summary_df, selected_df, report_path
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise

def main():
    """
    Main execution function.
    
    Creates an analyzer instance and runs the complete analysis pipeline.
    This function provides a simple command-line interface for the analysis.
    """
    try:
        analyzer = AdvancedSystematicAnalyzer()
        results_df, summary_df, selected_df, report_path = analyzer.run_analysis()
        
        # Print summary results
        print(f"\n🎉 Advanced systematic review analysis completed!")
        print(f"📊 {len(selected_df)} representative papers selected from {len(results_df)} total papers")
        print(f"📁 Results saved to: {analyzer.results_dir}")
        print(f"📋 Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 