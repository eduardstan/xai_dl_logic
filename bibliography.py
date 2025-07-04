#!/usr/bin/env python3
"""
Bibliography parsing module for XAI Deep Learning Logic analysis.
Handles BIB file parsing, text preprocessing, and data validation.
"""

import pandas as pd
import bibtexparser
from bibtexparser.bparser import BibTexParser
from pathlib import Path
from typing import Dict, List
from loguru import logger

from utils import get_file_hash, cache_exists, save_to_cache, load_from_cache


def parse_bib_file(file_path: str, config: Dict) -> pd.DataFrame:
    """
    Parse BIB file and extract relevant text fields with intelligent caching.
    
    Args:
        file_path: Path to the BIB file
        config: Configuration dictionary containing cache settings
        
    Returns:
        DataFrame with parsed and cleaned bibliography entries
        
    Raises:
        FileNotFoundError: If BIB file doesn't exist
        Exception: If parsing fails
    """
    # Validate input file
    bib_file = Path(file_path)
    if not bib_file.exists():
        raise FileNotFoundError(f"BIB file not found: {file_path}")
    
    # Check cache first
    cache_dir = Path(config['output']['cache_dir'])
    bib_hash = get_file_hash(file_path)
    cache_name = f"parsed_bib_{bib_hash[:8]}"
    
    if cache_exists(cache_dir, cache_name):
        logger.info("📂 Loading parsed BIB data from cache...")
        return load_from_cache(cache_dir, cache_name)
    
    logger.info(f"📖 Parsing BIB file: {file_path}")
    
    try:
        # Parse BIB file
        with open(file_path, 'r', encoding='utf-8') as bib_file:
            parser = BibTexParser(common_strings=True)
            bib_database = bibtexparser.load(bib_file, parser=parser)
            
        logger.info(f"Found {len(bib_database.entries)} entries in BIB file")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(bib_database.entries)
        
        # Clean and standardize data
        df = clean_bibliography_data(df)
        
        # Filter insufficient content
        df = filter_by_content_length(df, min_length=50)
        
        # Cache the processed result
        save_to_cache(df, cache_dir, cache_name)
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error parsing BIB file: {e}")
        raise


def clean_bibliography_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize bibliography text fields.
    
    Args:
        df: Raw DataFrame from BIB parser
        
    Returns:
        DataFrame with cleaned text fields and combined text column
    """
    logger.info("🧹 Cleaning bibliography data...")
    
    # Define text fields to process
    text_fields = ['abstract', 'title', 'keywords', 'author']
    
    # Clean each text field
    for field in text_fields:
        if field in df.columns:
            # Fill missing values and convert to string
            df[field] = df[field].fillna('').astype(str)
            # Remove excessive whitespace and clean text
            df[field] = df[field].str.replace(r'\s+', ' ', regex=True).str.strip()
        else:
            # Create empty column if field doesn't exist
            df[field] = ''
    
    # Create combined text field for topic modeling
    df['combined_text'] = create_combined_text_field(df)
    
    logger.info(f"✅ Cleaned {len(df)} bibliography entries")
    return df


def create_combined_text_field(df: pd.DataFrame) -> pd.Series:
    """
    Create a combined text field optimized for topic modeling.
    
    Args:
        df: DataFrame with individual text fields
        
    Returns:
        Series with combined text for each entry
    """
    # Combine title, abstract, and keywords with proper spacing
    combined = (
        df['title'].fillna('') + ' ' + 
        df['abstract'].fillna('') + ' ' + 
        df['keywords'].fillna('')
    ).str.strip()
    
    # Remove excessive whitespace
    combined = combined.str.replace(r'\s+', ' ', regex=True)
    
    return combined


def filter_by_content_length(df: pd.DataFrame, min_length: int = 50) -> pd.DataFrame:
    """
    Filter out entries with insufficient text content for meaningful analysis.
    
    Args:
        df: DataFrame with combined_text column
        min_length: Minimum character length for combined text
        
    Returns:
        Filtered DataFrame with only substantial entries
    """
    original_count = len(df)
    
    # Filter by combined text length
    df_filtered = df[df['combined_text'].str.len() >= min_length].copy()
    
    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count
    
    logger.info(f"📊 Content filtering: kept {filtered_count} entries, "
               f"removed {removed_count} with insufficient content")
    
    return df_filtered


def validate_bibliography_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate processed bibliography data and return quality metrics.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Dictionary with validation metrics and quality indicators
    """
    logger.info("✅ Validating bibliography data quality...")
    
    metrics = {
        'total_entries': len(df),
        'has_title': (df['title'].str.len() > 0).sum(),
        'has_abstract': (df['abstract'].str.len() > 0).sum(),
        'has_keywords': (df['keywords'].str.len() > 0).sum(),
        'has_author': (df['author'].str.len() > 0).sum(),
        'avg_combined_length': df['combined_text'].str.len().mean(),
        'min_combined_length': df['combined_text'].str.len().min(),
        'max_combined_length': df['combined_text'].str.len().max()
    }
    
    # Calculate completeness percentages
    for field in ['title', 'abstract', 'keywords', 'author']:
        metrics[f'{field}_completeness'] = (metrics[f'has_{field}'] / metrics['total_entries']) * 100
    
    # Log key quality metrics
    logger.info(f"📈 Data quality metrics:")
    logger.info(f"  - Total entries: {metrics['total_entries']:,}")
    logger.info(f"  - Title completeness: {metrics['title_completeness']:.1f}%")
    logger.info(f"  - Abstract completeness: {metrics['abstract_completeness']:.1f}%")
    logger.info(f"  - Keywords completeness: {metrics['keywords_completeness']:.1f}%")
    logger.info(f"  - Avg text length: {metrics['avg_combined_length']:.0f} characters")
    
    return metrics
