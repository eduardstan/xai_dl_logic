#!/usr/bin/env python3
"""
Advanced Systematic Review Analyzer with Diverse Representative Paper Selection

This module implements a sophisticated paper selection strategy that:
1. Selects diverse representative papers within each cluster
2. Computes comprehensive similarity metrics
3. Maps non-selected papers to their most similar representatives
4. Uses configurable parameters from config.yaml
"""

import pickle
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Set
import logging
from datetime import datetime

# Import extracted research alignment functionality
from research_alignment import analyze_research_alignment

# Import extracted cluster analysis functionality
from cluster_analysis import calculate_cluster_size_category, determine_papers_to_select

# Import extracted paper selection functionality
from paper_selection import select_diverse_representatives, assign_non_selected_papers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedSystematicAnalyzer:
    """Advanced analyzer for systematic literature reviews with diverse paper selection."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the analyzer with configuration."""
        self.config = self.load_config(config_path)
        self.review_config = self.config['systematic_review']
        self.results_dir = Path(self.config['output']['results_dir'])
        self.cache_dir = Path(self.config['output']['cache_dir'])
        
        # Ensure output directories exist
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("🔬 Advanced Systematic Analyzer initialized")
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            raise
    
    def load_analysis_results(self) -> Tuple[object, np.ndarray, List[Dict], List[int]]:
        """Load cached BERTopic analysis results with updated topic assignments."""
        try:
            # Load BERTopic model from bertopic_analysis/models
            models_dir = Path("bertopic_analysis/models")
            model_files = list(models_dir.glob("bertopic_model_*"))
            if not model_files:
                raise FileNotFoundError("No BERTopic model files found in bertopic_analysis/models")
            
            # Use the most recent model file
            model_path = max(model_files, key=lambda x: x.stat().st_mtime)
            from bertopic import BERTopic
            topic_model = BERTopic.load(str(model_path))
            logger.info(f"📂 Loaded BERTopic model from {model_path}")
            
            # Load the MOST RECENT topic assignments from CSV results (post-outlier reduction)
            results_dir = Path("bertopic_analysis/results")
            csv_files = list(results_dir.glob("bibliography_with_topics_*.csv"))
            if not csv_files:
                raise FileNotFoundError("No results CSV files found in bertopic_analysis/results")
            
            # Use the most recent CSV file
            csv_path = max(csv_files, key=lambda x: x.stat().st_mtime)
            results_df = pd.read_csv(csv_path)
            updated_topics = results_df['topic'].tolist()
            logger.info(f"📂 Loaded UPDATED topic assignments from {csv_path}")
            logger.info(f"   📊 Total papers: {len(updated_topics)}")
            logger.info(f"   📊 Outliers: {sum(1 for t in updated_topics if t == -1)}")
            logger.info(f"   📊 Assigned to topics: {sum(1 for t in updated_topics if t != -1)}")
            
            # Load embeddings from cache
            embeddings_files = list(self.cache_dir.glob("embeddings_*.pkl"))
            if not embeddings_files:
                raise FileNotFoundError("No embedding cache files found")
            
            embeddings_path = embeddings_files[0]  # Use the first (should be only one)
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            logger.info(f"📂 Loaded embeddings from {embeddings_path}")
            
            # Load parsed documents from cache
            parsed_files = list(self.cache_dir.glob("parsed_bib_*.pkl"))
            if not parsed_files:
                raise FileNotFoundError("No parsed BIB cache files found")
            
            parsed_path = parsed_files[0]
            with open(parsed_path, 'rb') as f:
                documents = pickle.load(f)
            logger.info(f"📂 Loaded documents from {parsed_path}")
            
            # Verify data consistency
            if len(updated_topics) != len(embeddings) or len(updated_topics) != len(documents):
                raise ValueError(f"Data length mismatch: topics={len(updated_topics)}, embeddings={len(embeddings)}, documents={len(documents)}")
            
            logger.info(f"✅ Analysis results loaded: {len(documents)} documents, {len(embeddings)} embeddings, {len(updated_topics)} topic assignments")
            return topic_model, embeddings, documents, updated_topics
            
        except Exception as e:
            logger.error(f"❌ Error loading analysis results: {e}")
            raise
    

    
    def compute_paper_metrics(self, paper_embeddings: np.ndarray, cluster_embeddings: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics for a paper within its cluster."""
        # Similarity to cluster centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        similarity_to_centroid = cosine_similarity([paper_embeddings], [centroid])[0][0]
        
        # Average similarity to all cluster papers
        similarities = cosine_similarity([paper_embeddings], cluster_embeddings)[0]
        avg_similarity_to_cluster = np.mean(similarities)
        
        # Diversity score (1 - max similarity to other papers)
        other_similarities = similarities[similarities != 1.0]  # Exclude self-similarity
        max_similarity = np.max(other_similarities) if len(other_similarities) > 0 else 0
        # Clamp max_similarity to [0, 1] to avoid floating point precision issues
        max_similarity = np.clip(max_similarity, 0.0, 1.0)
        diversity_score = 1.0 - max_similarity
        
        # Representativeness score (combination of centrality and diversity)
        diversity_weight = self.review_config['diversity_weight']
        representativeness_score = (
            (1 - diversity_weight) * similarity_to_centroid + 
            diversity_weight * diversity_score
        )
        
        return {
            'similarity_to_centroid': similarity_to_centroid,
            'similarity_to_cluster_papers': avg_similarity_to_cluster,
            'diversity_score': diversity_score,
            'representativeness_score': representativeness_score
        }
    

    

    

    

    

    

    
    def process_topics(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process all topics and select diverse representative papers."""
        logger.info("🔍 Processing topics for diverse representative selection...")
        
        topic_model, embeddings, documents, updated_topics = self.load_analysis_results()
        
        # Use the updated topic assignments (post-outlier reduction)
        topics = updated_topics
        topic_info = topic_model.get_topic_info()
        
        all_results = []
        selection_summary = []
        
        # Process each topic
        for topic_id in sorted(set(topics)):
            if topic_id == -1:  # Skip outliers
                continue
            
            # Get documents in this topic
            topic_mask = np.array(topics) == topic_id
            topic_indices = np.where(topic_mask)[0]
            topic_embeddings = embeddings[topic_mask]
            
            # Extract documents for this topic (documents is a DataFrame)
            if isinstance(documents, pd.DataFrame):
                topic_documents = documents.iloc[topic_indices]
            else:
                topic_documents = [documents[i] for i in topic_indices]
            
            cluster_size = len(topic_documents)
            n_select = determine_papers_to_select(cluster_size, self.config)
            
            logger.info(f"📊 Topic {topic_id}: {cluster_size} papers → selecting {n_select} representatives")
            
            # Select diverse representatives
            selected_local_indices = select_diverse_representatives(topic_embeddings, n_select, self.config)
            selected_global_indices = [topic_indices[i] for i in selected_local_indices]
            
            # Assign non-selected papers to representatives
            assignments = assign_non_selected_papers(topic_embeddings, selected_local_indices, self.config)
            
            # Process all papers in the topic
            for local_idx, global_idx in enumerate(topic_indices):
                # Compute metrics
                metrics = self.compute_paper_metrics(
                    embeddings[global_idx], 
                    topic_embeddings
                )
                
                # Get document data
                if isinstance(documents, pd.DataFrame):
                    doc = documents.iloc[global_idx]
                    # Research alignment
                    full_text = f"{doc.get('title', '')} {doc.get('abstract', '')} {doc.get('keywords', '')}"
                    
                    # Extract document info
                    doc_info = {
                        'title': doc.get('title', 'Unknown'),
                        'authors': doc.get('author', 'Unknown'),
                        'year': doc.get('year', 'Unknown'),
                        'journal': doc.get('journal', 'Unknown'),
                        'doi': doc.get('doi', ''),
                        'abstract': doc.get('abstract', ''),
                        'keywords': doc.get('keywords', '')
                    }
                else:
                    doc = documents[global_idx]
                    full_text = f"{doc.get('title', '')} {doc.get('abstract', '')} {' '.join(doc.get('keywords', []))}"
                    doc_info = {
                        'title': doc.get('title', 'Unknown'),
                        'authors': ', '.join(doc.get('authors', [])),
                        'year': doc.get('year', 'Unknown'),
                        'journal': doc.get('journal', 'Unknown'),
                        'doi': doc.get('doi', ''),
                        'abstract': doc.get('abstract', ''),
                        'keywords': ', '.join(doc.get('keywords', []))
                    }
                
                alignment = analyze_research_alignment(full_text, self.config)
                
                # Selection status
                is_selected = local_idx in selected_local_indices
                representative_info = {}
                
                if not is_selected and local_idx in assignments:
                    assignment = assignments[local_idx]
                    rep_global_idx = topic_indices[assignment['representative_idx']]
                    
                    if isinstance(documents, pd.DataFrame):
                        rep_doc = documents.iloc[rep_global_idx]
                        representative_info = {
                            'representative_title': rep_doc.get('title', 'Unknown'),
                            'representative_authors': rep_doc.get('author', 'Unknown'),
                            'similarity_to_representative': assignment['similarity'],
                            'is_similar_to_representative': assignment['is_similar']
                        }
                    else:
                        rep_doc = documents[rep_global_idx]
                        representative_info = {
                            'representative_title': rep_doc.get('title', 'Unknown'),
                            'representative_authors': ', '.join(rep_doc.get('authors', [])),
                            'similarity_to_representative': assignment['similarity'],
                            'is_similar_to_representative': assignment['is_similar']
                        }
                
                # Combine all information
                result = {
                    'topic_id': topic_id,
                    'global_index': global_idx,
                    'local_index': local_idx,
                    'is_selected_representative': is_selected,
                    'cluster_size': cluster_size,
                    'papers_selected_from_cluster': n_select,
                    **doc_info,
                    **metrics,
                    **alignment,
                    **representative_info
                }
                
                all_results.append(result)
            
            # Topic summary
            topic_name = topic_info[topic_info['Topic'] == topic_id]['Name'].iloc[0] if len(topic_info[topic_info['Topic'] == topic_id]) > 0 else f"Topic {topic_id}"
            selection_summary.append({
                'topic_id': topic_id,
                'topic_name': topic_name,
                'cluster_size': cluster_size,
                'size_category': calculate_cluster_size_category(cluster_size, self.config),
                'papers_selected': n_select,
                'selection_ratio': n_select / cluster_size,
                'avg_centrality': np.mean([r['similarity_to_centroid'] 
                                        for r in all_results 
                                        if r['topic_id'] == topic_id and r['is_selected_representative']]),
                'avg_diversity': np.mean([r['diversity_score'] 
                                       for r in all_results 
                                       if r['topic_id'] == topic_id and r['is_selected_representative']])
            })
        
        # Create DataFrames
        results_df = pd.DataFrame(all_results)
        summary_df = pd.DataFrame(selection_summary)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_df.to_csv(self.results_dir / f"comprehensive_analysis_{timestamp}.csv", index=False)
        summary_df.to_csv(self.results_dir / f"selection_summary_{timestamp}.csv", index=False)
        
        # Save selected representatives separately
        selected_df = results_df[results_df['is_selected_representative']]
        selected_df.to_csv(self.results_dir / f"selected_representatives_{timestamp}.csv", index=False)
        
        logger.info(f"✅ Analysis complete: {len(selected_df)} representatives selected from {len(results_df)} total papers")
        logger.info(f"📁 Results saved to {self.results_dir}")
        
        return results_df, summary_df, selected_df
    
    def generate_analysis_report(self, results_df: pd.DataFrame, summary_df: pd.DataFrame, selected_df: pd.DataFrame):
        """Generate a comprehensive analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"analysis_report_{timestamp}.md"
        
        total_papers = len(results_df)
        total_selected = len(selected_df)
        total_topics = len(summary_df)
        
        with open(report_path, 'w') as f:
            f.write(f"# Advanced Systematic Literature Review Analysis Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📊 Executive Summary\n\n")
            f.write(f"- **Total Papers Analyzed:** {total_papers:,}\n")
            f.write(f"- **Representative Papers Selected:** {total_selected:,}\n")
            f.write(f"- **Topics Covered:** {total_topics}\n")
            f.write(f"- **Selection Ratio:** {total_selected/total_papers:.1%}\n")
            f.write(f"- **Selection Strategy:** {self.review_config['selection_strategy']}\n\n")
            
            f.write("## 🎯 Selection Configuration\n\n")
            f.write(f"- **Base Papers per Cluster:** {self.review_config['base_papers_per_cluster']}\n")
            f.write(f"- **Max Papers per Cluster:** {self.review_config['max_papers_per_cluster']}\n")
            f.write(f"- **Diversity Weight:** {self.review_config['diversity_weight']}\n")
            f.write(f"- **Similarity Threshold:** {self.review_config['similarity_threshold']}\n\n")
            
            f.write("## 📈 Topic Summary\n\n")
            f.write("| Topic ID | Topic Name | Cluster Size | Size Category | Selected | Selection Ratio |\n")
            f.write("|----------|------------|--------------|---------------|----------|----------------|\n")
            
            for _, row in summary_df.iterrows():
                f.write(f"| {row['topic_id']} | {row['topic_name'][:50]}... | {row['cluster_size']} | {row['size_category']} | {row['papers_selected']} | {row['selection_ratio']:.1%} |\n")
            
            f.write("\n## 🔍 Research Alignment Analysis\n\n")
            
            # Calculate average alignment scores
            xai_avg = selected_df['xai_alignment'].mean()
            symbolic_avg = selected_df['symbolic_alignment'].mean()
            subsymbolic_avg = selected_df['subsymbolic_alignment'].mean()
            
            f.write(f"**Average Research Alignment (Selected Papers):**\n")
            f.write(f"- XAI Alignment: {xai_avg:.2%}\n")
            f.write(f"- Symbolic Alignment: {symbolic_avg:.2%}\n")
            f.write(f"- Sub-symbolic Alignment: {subsymbolic_avg:.2%}\n\n")
            
            f.write("## 📝 Reading Recommendations\n\n")
            f.write("### Priority 1: High Diversity & Centrality\n")
            high_priority = selected_df[
                (selected_df['diversity_score'] > selected_df['diversity_score'].quantile(0.75)) &
                (selected_df['similarity_to_centroid'] > selected_df['similarity_to_centroid'].quantile(0.75))
            ].sort_values('representativeness_score', ascending=False)
            
            f.write(f"**{len(high_priority)} papers** meeting both high diversity and centrality criteria:\n\n")
            for _, paper in high_priority.head(10).iterrows():
                f.write(f"- **{paper['title']}** (Topic {paper['topic_id']})\n")
                f.write(f"  - Authors: {paper['authors']}\n")
                f.write(f"  - Representativeness: {paper['representativeness_score']:.3f}\n\n")
            
            f.write("### Priority 2: Highest Representativeness by Topic\n")
            f.write("Top representative paper from each topic:\n\n")
            
            for topic_id in sorted(selected_df['topic_id'].unique()):
                topic_papers = selected_df[selected_df['topic_id'] == topic_id]
                best_paper = topic_papers.loc[topic_papers['representativeness_score'].idxmax()]
                
                f.write(f"**Topic {topic_id}:** {best_paper['title']}\n")
                f.write(f"- Authors: {best_paper['authors']}\n")
                f.write(f"- Representativeness: {best_paper['representativeness_score']:.3f}\n")
                f.write(f"- Cluster Size: {best_paper['cluster_size']} papers\n\n")
        
        logger.info(f"📋 Analysis report saved to {report_path}")

def main():
    """Main execution function."""
    try:
        analyzer = AdvancedSystematicAnalyzer()
        results_df, summary_df, selected_df = analyzer.process_topics()
        analyzer.generate_analysis_report(results_df, summary_df, selected_df)
        
        print(f"\n🎉 Advanced systematic review analysis completed!")
        print(f"📊 {len(selected_df)} representative papers selected from {len(results_df)} total papers")
        print(f"📁 Results saved to: {analyzer.results_dir}")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 