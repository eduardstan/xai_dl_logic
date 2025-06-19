#!/usr/bin/env python3
"""
Systematic Literature Review Analyzer
Extract representative documents and create taxonomy for XAI + Symbolic + Sub-symbolic research
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_cached_analysis(cache_dir: str = "cache") -> Tuple[BERTopic, pd.DataFrame, List[int], np.ndarray]:
    """Load the cached BERTopic analysis results."""
    print("📂 Loading cached analysis results...")
    
    cache_path = Path(cache_dir)
    
    # Find the latest cached files
    bib_files = list(cache_path.glob("parsed_bib_*.pkl"))
    embedding_files = list(cache_path.glob("embeddings_*.pkl"))
    
    if not bib_files or not embedding_files:
        raise FileNotFoundError("No cached analysis found. Run bib_analyzer.py first.")
    
    # Load the most recent files
    bib_file = sorted(bib_files)[-1]
    embedding_file = sorted(embedding_files)[-1]
    
    print(f"Loading bibliography data from: {bib_file}")
    with open(bib_file, 'rb') as f:
        df = pickle.load(f)
    
    print(f"Loading embeddings from: {embedding_file}")
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load the trained model and topics
    model_dir = Path("bertopic_analysis/models")
    model_files = list(model_dir.glob("bertopic_model_*"))
    
    if model_files:
        model_file = sorted(model_files)[-1]
        print(f"Loading BERTopic model from: {model_file}")
        topic_model = BERTopic.load(str(model_file))
        
        # Get topics from the model
        docs = df['combined_text'].tolist()
        topics = topic_model.topics_
    else:
        raise FileNotFoundError("No trained BERTopic model found.")
    
    print(f"✅ Loaded: {len(df)} documents, {len(set(topics))-1} topics, embeddings shape: {embeddings.shape}")
    return topic_model, df, topics, embeddings

def get_representative_documents(topic_model: BERTopic, df: pd.DataFrame, 
                               topics: List[int], embeddings: np.ndarray,
                               n_docs_per_topic: int = 5) -> Dict[int, List[Dict]]:
    """
    Extract the most representative documents for each topic.
    
    Args:
        topic_model: Trained BERTopic model
        df: DataFrame with document data
        topics: Topic assignments
        embeddings: Document embeddings
        n_docs_per_topic: Number of representative docs per topic
        
    Returns:
        Dictionary mapping topic_id -> list of representative documents
    """
    print(f"🎯 Extracting {n_docs_per_topic} representative documents per topic...")
    
    representative_docs = {}
    topic_info = topic_model.get_topic_info()
    
    for topic_id in topic_info['Topic']:
        if topic_id == -1:  # Skip outlier topic
            continue
            
        # Get documents in this topic
        topic_mask = np.array(topics) == topic_id
        topic_indices = np.where(topic_mask)[0]
        
        if len(topic_indices) == 0:
            continue
            
        # Get embeddings for this topic
        topic_embeddings = embeddings[topic_indices]
        
        # Calculate centroid of the topic
        topic_centroid = np.mean(topic_embeddings, axis=0)
        
        # Find documents closest to centroid (most representative)
        similarities = cosine_similarity([topic_centroid], topic_embeddings)[0]
        
        # Get top N most representative documents
        top_indices = np.argsort(similarities)[-n_docs_per_topic:][::-1]
        
        topic_docs = []
        for idx in top_indices:
            doc_idx = topic_indices[idx]
            doc_info = {
                'index': int(doc_idx),
                'similarity_to_centroid': float(similarities[idx]),
                'title': df.iloc[doc_idx]['title'],
                'author': df.iloc[doc_idx]['author'],
                'year': df.iloc[doc_idx].get('year', 'Unknown'),
                'abstract': df.iloc[doc_idx].get('abstract', ''),
                'doi': df.iloc[doc_idx].get('doi', ''),
                'keywords': df.iloc[doc_idx].get('keywords', ''),
                'combined_text': df.iloc[doc_idx]['combined_text']
            }
            topic_docs.append(doc_info)
        
        representative_docs[topic_id] = topic_docs
    
    print(f"✅ Extracted representative documents for {len(representative_docs)} topics")
    return representative_docs

def analyze_research_alignment(topic_model: BERTopic, representative_docs: Dict) -> Dict[int, Dict]:
    """
    Analyze how well each topic aligns with XAI + Symbolic + Sub-symbolic research.
    """
    print("🔍 Analyzing alignment with XAI + Symbolic + Sub-symbolic research...")
    
    # Define research focus keywords
    xai_keywords = ['explainable', 'interpretable', 'accountable', 'transparent', 'xai', 'explanation', 'interpret']
    symbolic_keywords = ['symbolic', 'logic', 'formal', 'knowledge', 'ontology', 'rule', 'reasoning', 'constraint']
    subsymbolic_keywords = ['neural', 'deep', 'network', 'layer', 'tensor', 'differentiable', 'neuro']
    
    topic_alignment = {}
    topic_info = topic_model.get_topic_info()
    
    for topic_id in representative_docs.keys():
        # Get topic terms
        topic_terms = topic_model.get_topic(topic_id)
        topic_words = [term[0].lower() for term in topic_terms] if topic_terms else []
        
        # Get topic name
        topic_row = topic_info[topic_info['Topic'] == topic_id]
        topic_name = topic_row['Name'].iloc[0] if not topic_row.empty else f"Topic {topic_id}"
        
        # Analyze representative documents
        docs_text = []
        for doc in representative_docs[topic_id]:
            text = (doc['title'] + ' ' + doc['abstract'] + ' ' + doc['keywords']).lower()
            docs_text.append(text)
        
        combined_text = ' '.join(docs_text + topic_words)
        
        # Calculate alignment scores
        xai_score = sum(1 for keyword in xai_keywords if keyword in combined_text)
        symbolic_score = sum(1 for keyword in symbolic_keywords if keyword in combined_text)
        subsymbolic_score = sum(1 for keyword in subsymbolic_keywords if keyword in combined_text)
        
        # Determine research category
        total_score = xai_score + symbolic_score + subsymbolic_score
        category = "Other"
        if total_score > 0:
            if xai_score > 0 and symbolic_score > 0 and subsymbolic_score > 0:
                category = "XAI + Symbolic + Sub-symbolic"
            elif xai_score > 0 and symbolic_score > 0:
                category = "XAI + Symbolic"
            elif xai_score > 0 and subsymbolic_score > 0:
                category = "XAI + Sub-symbolic"
            elif symbolic_score > 0 and subsymbolic_score > 0:
                category = "Symbolic + Sub-symbolic"
            elif xai_score > 0:
                category = "XAI"
            elif symbolic_score > 0:
                category = "Symbolic"
            elif subsymbolic_score > 0:
                category = "Sub-symbolic"
        
        topic_alignment[topic_id] = {
            'name': topic_name,
            'xai_score': xai_score,
            'symbolic_score': symbolic_score,
            'subsymbolic_score': subsymbolic_score,
            'total_score': total_score,
            'category': category,
            'relevance_priority': 'High' if total_score >= 3 else 'Medium' if total_score >= 1 else 'Low'
        }
    
    return topic_alignment

def create_taxonomy(topic_model: BERTopic, topic_alignment: Dict) -> Dict:
    """Create a hierarchical taxonomy based on research categories."""
    print("🌳 Creating research taxonomy...")
    
    taxonomy = {
        'XAI + Symbolic + Sub-symbolic': [],
        'XAI + Symbolic': [],
        'XAI + Sub-symbolic': [],
        'Symbolic + Sub-symbolic': [],
        'XAI': [],
        'Symbolic': [],
        'Sub-symbolic': [],
        'Other': []
    }
    
    topic_info = topic_model.get_topic_info()
    
    for topic_id, alignment in topic_alignment.items():
        # Get topic details
        topic_row = topic_info[topic_info['Topic'] == topic_id]
        if topic_row.empty:
            continue
            
        topic_data = {
            'topic_id': topic_id,
            'name': alignment['name'],
            'count': int(topic_row['Count'].iloc[0]),
            'terms': [term[0] for term in topic_model.get_topic(topic_id)[:5]] if topic_model.get_topic(topic_id) else [],
            'scores': {
                'xai': alignment['xai_score'],
                'symbolic': alignment['symbolic_score'],
                'subsymbolic': alignment['subsymbolic_score']
            },
            'priority': alignment['relevance_priority']
        }
        
        taxonomy[alignment['category']].append(topic_data)
    
    # Sort each category by document count (descending)
    for category in taxonomy:
        taxonomy[category].sort(key=lambda x: x['count'], reverse=True)
    
    return taxonomy

def generate_systematic_review_report(topic_model: BERTopic, df: pd.DataFrame, 
                                    representative_docs: Dict, topic_alignment: Dict,
                                    taxonomy: Dict, output_dir: str = "systematic_review"):
    """Generate comprehensive systematic review report."""
    print("📝 Generating systematic review report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 1. Create summary statistics
    total_docs = len(df)
    total_topics = len(representative_docs)
    high_priority_topics = sum(1 for t in topic_alignment.values() if t['relevance_priority'] == 'High')
    
    # 2. Generate main report
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# Systematic Literature Review Report
## XAI + Symbolic + Sub-symbolic Analysis

**Generated:** {timestamp}
**Total Documents:** {total_docs:,}
**Topics Discovered:** {total_topics}
**High Priority Topics:** {high_priority_topics}

## Executive Summary

This systematic literature review analyzed {total_docs:,} academic papers focusing on the intersection of:
- **XAI (Explainable AI)**: explainable, interpretable, accountable, transparent approaches
- **Symbolic**: logic, formal methods, knowledge representation, reasoning
- **Sub-symbolic**: neural networks, deep learning, differentiable approaches

### Research Landscape Overview
"""
    
    # Add taxonomy overview
    for category, topics in taxonomy.items():
        if topics:
            report += f"\n### {category} ({len(topics)} topics, {sum(t['count'] for t in topics)} papers)\n"
            for topic in topics[:3]:  # Show top 3 topics per category
                terms = ', '.join(topic['terms'])
                report += f"- **Topic {topic['topic_id']}** ({topic['count']} papers): {terms}\n"
            if len(topics) > 3:
                report += f"- ... and {len(topics)-3} more topics\n"
    
    # 3. Priority reading list
    report += f"\n## Priority Reading List\n\n"
    report += "### High Priority Topics (Focus Areas)\n"
    
    high_priority_docs = []
    for topic_id, alignment in topic_alignment.items():
        if alignment['relevance_priority'] == 'High':
            topic_docs = representative_docs[topic_id]
            report += f"\n#### Topic {topic_id}: {alignment['name']}\n"
            report += f"**Research Alignment:** {alignment['category']}\n"
            report += f"**Representative Papers:**\n"
            
            for i, doc in enumerate(topic_docs[:3], 1):  # Top 3 per topic
                report += f"{i}. **{doc['title']}**\n"
                report += f"   - Authors: {doc['author']}\n"
                report += f"   - Year: {doc['year']}\n"
                if doc['doi']:
                    report += f"   - DOI: {doc['doi']}\n"
                report += f"   - Centrality: {doc['similarity_to_centroid']:.3f}\n\n"
                
                high_priority_docs.append({
                    'topic_id': topic_id,
                    'topic_name': alignment['name'],
                    'category': alignment['category'],
                    **doc
                })
    
    # 4. Save detailed data
    # Representative documents CSV
    all_repr_docs = []
    for topic_id, docs in representative_docs.items():
        for doc in docs:
            doc_data = {
                'topic_id': topic_id,
                'topic_name': topic_alignment[topic_id]['name'],
                'category': topic_alignment[topic_id]['category'],
                'priority': topic_alignment[topic_id]['relevance_priority'],
                **doc
            }
            all_repr_docs.append(doc_data)
    
    repr_df = pd.DataFrame(all_repr_docs)
    repr_df.to_csv(output_path / "representative_documents.csv", index=False)
    
    # High priority documents CSV
    priority_df = pd.DataFrame(high_priority_docs)
    if not priority_df.empty:
        priority_df.to_csv(output_path / "high_priority_documents.csv", index=False)
    
    # Topic taxonomy CSV
    taxonomy_data = []
    for category, topics in taxonomy.items():
        for topic in topics:
            taxonomy_data.append({
                'category': category,
                'topic_id': topic['topic_id'],
                'topic_name': topic['name'],
                'document_count': topic['count'],
                'top_terms': ', '.join(topic['terms']),
                'xai_score': topic['scores']['xai'],
                'symbolic_score': topic['scores']['symbolic'],
                'subsymbolic_score': topic['scores']['subsymbolic'],
                'priority': topic['priority']
            })
    
    taxonomy_df = pd.DataFrame(taxonomy_data)
    taxonomy_df.to_csv(output_path / "research_taxonomy.csv", index=False)
    
    # Save main report
    with open(output_path / "systematic_review_report.md", 'w') as f:
        f.write(report)
    
    print(f"✅ Systematic review report saved to: {output_path}")
    print(f"📊 Files generated:")
    print(f"   - systematic_review_report.md: Main report")
    print(f"   - representative_documents.csv: All representative papers")
    print(f"   - high_priority_documents.csv: Priority reading list")
    print(f"   - research_taxonomy.csv: Topic categorization")
    
    return {
        'total_docs': total_docs,
        'total_topics': total_topics,
        'high_priority_topics': high_priority_topics,
        'high_priority_docs': len(high_priority_docs),
        'taxonomy': taxonomy
    }

def main():
    """Main execution function for systematic review analysis."""
    print("🔬 Starting Systematic Literature Review Analysis")
    print("Focus: XAI + Symbolic + Sub-symbolic approaches\n")
    
    try:
        # Load cached analysis
        topic_model, df, topics, embeddings = load_cached_analysis()
        
        # Extract representative documents (top 5 per topic)
        representative_docs = get_representative_documents(
            topic_model, df, topics, embeddings, n_docs_per_topic=5
        )
        
        # Analyze research alignment
        topic_alignment = analyze_research_alignment(topic_model, representative_docs)
        
        # Create taxonomy
        taxonomy = create_taxonomy(topic_model, topic_alignment)
        
        # Generate comprehensive report
        summary = generate_systematic_review_report(
            topic_model, df, representative_docs, topic_alignment, taxonomy
        )
        
        # Print executive summary
        print(f"\n🎯 SYSTEMATIC REVIEW SUMMARY")
        print(f"{'='*50}")
        print(f"📚 Total papers analyzed: {summary['total_docs']:,}")
        print(f"🏷️  Topics discovered: {summary['total_topics']}")
        print(f"⭐ High priority topics: {summary['high_priority_topics']}")
        print(f"📖 Priority papers to read: {summary['high_priority_docs']}")
        
        print(f"\n📊 RESEARCH TAXONOMY")
        print(f"{'='*50}")
        for category, topics in summary['taxonomy'].items():
            if topics:
                total_papers = sum(t['count'] for t in topics)
                print(f"{category:.<30} {len(topics):>3} topics, {total_papers:>4} papers")
        
        print(f"\n✅ Analysis complete! Check 'systematic_review/' directory for detailed results.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you've run 'python bib_analyzer.py' first to generate the cached analysis.")

if __name__ == "__main__":
    main() 