#!/usr/bin/env python3
"""
Bibliography parsing for pool augmentation.
"""

from pathlib import Path
import bibtexparser
import pandas as pd
from bibtexparser.bparser import BibTexParser

from research_analysis.config.models import AppConfig
from research_analysis.stages.stage_1_topic_model.bib_parser import _clean_bibliography_data
from research_analysis.utils.logging import get_logger

logger = get_logger()

def parse_augmentation_bib(config: AppConfig) -> pd.DataFrame:
    """
    Parse the R1 bib file for augmentation.
    """
    bib_file = Path(config.stage_2.augmentation.r1_bib_file)
    if not bib_file.exists():
        logger.error(f"Augmentation BIB file not found: {bib_file}")
        raise FileNotFoundError(f"Augmentation BIB file not found: {bib_file}")

    logger.info(f"Parsing augmentation BIB file: {bib_file}")
    try:
        with open(bib_file, "r", encoding="utf-8") as f:
            parser = BibTexParser(common_strings=True)
            bib_database = bibtexparser.load(f, parser=parser)

        logger.info(f"Found {len(bib_database.entries)} entries in augmentation BIB file.")
        df = pd.DataFrame(bib_database.entries)

        # Reuse the cleaning logic from Stage 1
        df = _clean_bibliography_data(df, config.stage_1.text_fields)
        
        # Add metadata
        df["source"] = "r1_reviewer"
        df["prisma_pathway"] = "other_methods"
        
        return df

    except Exception as e:
        logger.error(f"Error parsing augmentation BIB file: {e}")
        raise
