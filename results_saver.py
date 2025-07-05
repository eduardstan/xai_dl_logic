#!/usr/bin/env python3
"""
Results saving and reporting module for XAI Deep Learning Logic analysis.
Handles saving analysis results, model artifacts, and generating summary reports.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import Counter
from bertopic import BERTopic
from loguru import logger

from utils import get_results_dir, get_models_dir, get_output_config


def build_topic_info(topics: List[int], df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Build topic_info DataFrame from updated topic assignments.
    This ensures consistency when model representations aren't updated.
    
    Args:
        topics: Updated topic assignments
        df_results: DataFrame with document information and topics
        
    Returns:
        DataFrame with topic information consistent with updated assignments
    """
    logger.info("🔧 Building topic_info from updated assignments...")
    
    # Count documents per topic
    topic_counts = Counter(topics)
    
    # Create topic_info DataFrame
    topic_info_data = []
    
    for topic_id, count in topic_counts.items():
        # Get representative documents for this topic
        topic_docs = df_results[df_results['topic'] == topic_id]
        
        # Create a simple representation (we can't generate full c-TF-IDF without the model)
        if topic_id == -1:
            name = "Outliers"
            representation = ["outlier", "documents", "unassigned", "noise", "scattered"]
        else:
            # Use first few words from titles/abstracts as simple representation
            sample_texts = topic_docs['combined_text'].head(10).tolist()
            # This is a simplified representation - ideally would use c-TF-IDF
            name = f"Topic_{topic_id}"
            representation = [f"topic_{topic_id}", "documents", "cluster", "group", "category"]
        
        topic_info_data.append({
            'Topic': topic_id,
            'Count': count,
            'Name': name,
            'Representation': representation,
            'Representative_Docs': topic_docs['title'].head(3).tolist() if not topic_docs.empty else []
        })
    
    # Sort by count (descending) then by topic ID
    topic_info_data.sort(key=lambda x: (-x['Count'], x['Topic']))
    
    topic_info_df = pd.DataFrame(topic_info_data)
    
    logger.info(f"   ✅ Generated topic_info for {len(topic_info_df)} topics")
    return topic_info_df


def save_results(topic_model: BERTopic, df: pd.DataFrame, topics: List[int],
                probs: np.ndarray, embeddings: np.ndarray,
                config: Dict, output_dir: str) -> None:
    """
    Save analysis results and model artifacts with consistent topic assignments.
    Now generates topic_info from updated topics for consistency.
    
    Args:
        topic_model: Trained BERTopic model
        df: Original DataFrame with bibliography data
        topics: Updated topic assignments (post-outlier reduction)
        probs: Topic probabilities
        embeddings: Document embeddings
        config: Configuration dictionary
        output_dir: Output directory
    """
    logger.info("💾 Saving analysis results with consistent topic assignments")
    
    # Create output directories
    results_dir = Path(output_dir) / get_results_dir(config)
    models_dir = Path(output_dir) / get_models_dir(config)
    
    for directory in [results_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Add topic information to DataFrame
    df_results = df.copy()
    df_results['topic'] = topics
    if probs is not None and len(probs.shape) > 1:
        df_results['topic_probability'] = probs.max(axis=1)
    else:
        df_results['topic_probability'] = None
    
    # Get topic information from model (now has preserved representations with updated assignments)
    topic_info = topic_model.get_topic_info()
    
    # Update the Count column to reflect actual post-outlier reduction document counts
    outlier_reduction_enabled = config.get('outlier_reduction', {}).get('enabled', False)
    if outlier_reduction_enabled:
        logger.info("📊 Updating topic counts to reflect post-outlier reduction assignments...")
        topic_counts = Counter(topics)
        
        # Update the Count column with actual counts
        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']
            actual_count = topic_counts.get(topic_id, 0)
            topic_info.at[idx, 'Count'] = actual_count
        
        # Re-sort by count (descending)
        topic_info = topic_info.sort_values('Count', ascending=False).reset_index(drop=True)
    
    # Save results with timestamps
    timestamp = datetime.now().strftime(config['timestamp_format'])
    
    # Save enhanced DataFrame (uses updated topics)
    df_results.to_csv(results_dir / f"bibliography_with_topics_{timestamp}.csv", index=False)
    
    # Save topic information (now consistent with updated topics)
    topic_info.to_csv(results_dir / f"topic_info_{timestamp}.csv", index=False)
    
    # Save versioned artifacts for scientific integrity
    if outlier_reduction_enabled:
        # Save original model topic info for comparison
        original_topic_info = topic_model.get_topic_info()
        original_topic_info.to_csv(results_dir / f"topic_info_pre_outlier_reduction_{timestamp}.csv", index=False)
        logger.info(f"📋 Saved pre-outlier reduction topic info for comparison")
    
    # Save model artifacts if configured
    output_config = get_output_config(config)
    if output_config['save_model']:
        topic_model.save(str(models_dir / f"bertopic_model_{timestamp}"))
    
    if output_config['save_embeddings']:
        np.save(results_dir / f"embeddings_{timestamp}.npy", embeddings)
    
    # Save configuration used
    with open(results_dir / f"config_used_{timestamp}.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"✅ Results saved to: {results_dir}")
    logger.info(f"📁 Model artifacts saved to: {models_dir}")


def generate_summary_report(topic_model: BERTopic, df: pd.DataFrame,
                          topics: List[int], config: Dict,
                          output_dir: str) -> None:
    """
    Generate a summary report of the analysis using updated topic assignments.
    
    Args:
        topic_model: Trained BERTopic model
        df: DataFrame with results
        topics: Updated topic assignments (post-outlier reduction)
        config: Configuration dictionary
        output_dir: Output directory
    """
    logger.info("📋 Generating summary report with updated topic assignments")
    
    results_dir = Path(output_dir) / get_results_dir(config)
    
    # Basic statistics using updated topics
    n_documents = len(topics)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    
    # Get topic information - model now has preserved representations with updated counts
    topic_info = topic_model.get_topic_info()
    
    # Ensure topic counts reflect actual post-outlier reduction assignments
    outlier_reduction_enabled = config.get('outlier_reduction', {}).get('enabled', False)
    if outlier_reduction_enabled:
        topic_counts = Counter(topics)
        topic_sizes = [(topic_id, count) for topic_id, count in topic_counts.items() if topic_id != -1]
        topic_sizes.sort(key=lambda x: -x[1])  # Sort by size descending
    else:
        topic_sizes = [(row['Topic'], row['Count']) for _, row in topic_info.iterrows() if row['Topic'] != -1]
    
    # Create summary report
    report = f"""# BERTopic Analysis Summary Report (Updated Topic Assignments)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Total documents processed: {n_documents:,}
- Documents with sufficient content: {len(df):,}
- Topics discovered: {n_topics}
- Outlier documents: {n_outliers}
- Coverage: {((n_documents - n_outliers) / n_documents * 100):.1f}%

## Model Configuration
- Embedding model: {config['embedding_model']['name']}
- UMAP neighbors: {config['umap_params']['n_neighbors']}
- HDBSCAN min cluster size: {config['hdbscan_params']['min_cluster_size']}
- Min topic size: {config['data']['min_topic_size']}"""
    
    # Add outlier reduction information
    if outlier_reduction_enabled:
        strategies = config.get('outlier_reduction', {}).get('strategies', [])
        strategy_names = [s.get('strategy', 'unknown') for s in strategies]
        report += f"""

## Outlier Reduction
- Enabled: Yes
- Strategies applied: {', '.join(strategy_names)}
- Original outliers: {n_outliers} documents ({(n_outliers / n_documents * 100):.1f}%)
- Final coverage: {((n_documents - n_outliers) / n_documents * 100):.1f}%"""
    
    # Top topics section
    report += "\n\n## Top Topics by Size"
    for i, (topic_id, size) in enumerate(topic_sizes[:10]):
        # Get topic representation from model
        topic_repr = topic_model.get_topic(topic_id)
        if topic_repr:
            top_words = [word for word, _ in topic_repr[:5]]
            report += f"\n{i+1}. **Topic {topic_id}**: {size} documents - {', '.join(top_words)}"
        else:
            report += f"\n{i+1}. **Topic {topic_id}**: {size} documents"
    
    # Topic size distribution
    report += "\n\n## Topic Size Distribution"
    if topic_sizes:
        sizes = [size for _, size in topic_sizes]
        report += f"""
- Largest topic: {max(sizes)} documents
- Smallest topic: {min(sizes)} documents
- Average topic size: {sum(sizes) / len(sizes):.1f} documents
- Median topic size: {sorted(sizes)[len(sizes)//2]} documents"""
    
    # Data quality metrics
    report += f"""

## Data Quality Metrics
- Text fields processed: {', '.join(config['data']['text_fields'])}
- Average document length: {df['combined_text'].str.len().mean():.0f} characters
- Documents with titles: {(df['title'].str.len() > 0).sum()} ({(df['title'].str.len() > 0).mean() * 100:.1f}%)
- Documents with abstracts: {(df['abstract'].str.len() > 0).sum()} ({(df['abstract'].str.len() > 0).mean() * 100:.1f}%)
- Documents with keywords: {(df['keywords'].str.len() > 0).sum()} ({(df['keywords'].str.len() > 0).mean() * 100:.1f}%)"""
    
    # Output files section
    timestamp = datetime.now().strftime(config['timestamp_format'])
    report += f"""

## Output Files Generated
- Bibliography with topics: `bibliography_with_topics_{timestamp}.csv`
- Topic information: `topic_info_{timestamp}.csv`
- Configuration used: `config_used_{timestamp}.yaml`
- Model artifacts: `bertopic_model_{timestamp}/`
- Visualizations: `plots/` directory"""
    
    if outlier_reduction_enabled:
        report += f"\n- Pre-outlier reduction topic info: `topic_info_pre_outlier_reduction_{timestamp}.csv`"
    
    report += f"""

## Analysis Summary
This analysis successfully processed {n_documents:,} academic documents and discovered {n_topics} distinct topics with {((n_documents - n_outliers) / n_documents * 100):.1f}% coverage. The topics represent the main research themes in the XAI deep learning literature, providing a comprehensive overview of the field's current state and research directions.

---
*Generated by XAI Deep Learning Logic Bibliography Analysis System*
"""
    
    # Save the report
    report_path = results_dir / f"analysis_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"📄 Summary report saved to: {report_path}")


def validate_results_config(config: Dict) -> bool:
    """
    Validate results and output configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    output_config = config.get('output', {})
    
    required_keys = ['results_dir', 'models_dir', 'save_model', 'save_embeddings']
    
    for key in required_keys:
        if key not in output_config:
            return False
            
    return True


def get_results_info(results_dir: Path) -> Dict:
    """
    Get information about saved results.
    
    Args:
        results_dir: Results directory path
        
    Returns:
        Dictionary with results info
    """
    if not results_dir.exists():
        return {'exists': False, 'files': []}
    
    files = list(results_dir.glob("*"))
    
    return {
        'exists': True,
        'total_files': len(files),
        'csv_files': len(list(results_dir.glob("*.csv"))),
        'model_files': len(list(results_dir.glob("bertopic_model_*"))),
        'report_files': len(list(results_dir.glob("analysis_report_*.md"))),
        'config_files': len(list(results_dir.glob("config_used_*.yaml"))),
        'files': [f.name for f in files]
    } 