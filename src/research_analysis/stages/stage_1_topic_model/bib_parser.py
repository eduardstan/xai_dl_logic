#!/usr/bin/env python3
"""
Bibliography parsing module for the research analysis framework.
Handles BIB file parsing, text preprocessing, and data validation, with caching.
"""

from pathlib import Path
from typing import List

import bibtexparser
import pandas as pd
from bibtexparser.bparser import BibTexParser

from research_analysis.config.models import AppConfig
from research_analysis.utils.files import (
    cache_exists,
    get_cache_path,
    load_from_cache,
    save_to_cache,
)
from research_analysis.utils.logging import get_logger

logger = get_logger()


def parse_bib_file(config: AppConfig) -> pd.DataFrame:
    """
    Parse a BIB file and extract relevant text fields with intelligent caching.

    This function checks for a cached version of the parsed data before
    processing the raw .bib file. If a cache exists, it is loaded;
    otherwise, the file is parsed, cleaned, filtered, and then cached
    for future runs.

    Args:
        config: The application's configuration object.

    Returns:
        A DataFrame with parsed and cleaned bibliography entries.

    Raises:
        FileNotFoundError: If the specified BIB file does not exist.
        Exception: If the parsing process fails for any reason.
    """
    bib_file = Path(config.pipeline.paths.bibliography_file)
    if not bib_file.exists():
        logger.error(f"BIB file not found: {bib_file}")
        raise FileNotFoundError(f"BIB file not found: {bib_file}")

    cache_path = get_cache_path(
        cache_dir=config.pipeline.paths.cache,
        file_name="parsed_bib.pkl",
    )

    if cache_exists(cache_path):
        logger.info(f"Loading parsed bibliography data from cache: {cache_path}")
        return load_from_cache(cache_path)

    logger.info(f"Parsing BIB file: {bib_file}")
    try:
        with open(bib_file, "r", encoding="utf-8") as f:
            parser = BibTexParser(common_strings=True)
            bib_database = bibtexparser.load(f, parser=parser)

        logger.info(f"Found {len(bib_database.entries)} entries in BIB file.")
        df = pd.DataFrame(bib_database.entries)

        df = _clean_bibliography_data(df, config.stage_1.text_fields)
        df = _filter_by_content_length(df, min_length=50)

        logger.info(f"Saving parsed data to cache: {cache_path}")
        save_to_cache(df, cache_path)

        return df

    except Exception as e:
        logger.error(f"Error parsing BIB file: {e}")
        raise


def _clean_bibliography_data(
    df: pd.DataFrame, text_fields: List[str]
) -> pd.DataFrame:
    """
    Clean and standardize bibliography text fields.

    Args:
        df: Raw DataFrame from the BIB parser.
        text_fields: A list of text fields to process from the config.

    Returns:
        A DataFrame with cleaned text fields and a 'combined_text' column.
    """
    logger.info("Cleaning bibliography data...")

    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].fillna("").astype(str)
            df[field] = df[field].str.replace(r"\\s+", " ", regex=True).str.strip()
        else:
            df[field] = ""

    df["combined_text"] = (
        (df["title"].fillna("") + " " + df["abstract"].fillna(""))
        .str.strip()
        .str.replace(r"\\s+", " ", regex=True)
    )

    logger.info(f"Cleaned {len(df)} bibliography entries.")
    return df


def _filter_by_content_length(df: pd.DataFrame, min_length: int = 50) -> pd.DataFrame:
    """
    Filter out entries with insufficient text content for meaningful analysis.

    Args:
        df: DataFrame with a 'combined_text' column.
        min_length: The minimum character length required for 'combined_text'.

    Returns:
        A filtered DataFrame containing only entries with substantial content.
    """
    original_count = len(df)
    df_filtered = df[df["combined_text"].str.len() >= min_length].copy()
    removed_count = original_count - len(df_filtered)

    if removed_count > 0:
        logger.info(
            f"Filtered out {removed_count} entries with insufficient content "
            f"(<{min_length} chars)."
        )

    return df_filtered 