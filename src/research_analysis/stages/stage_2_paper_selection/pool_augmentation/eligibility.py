#!/usr/bin/env python3
"""
Eligibility filtering for pool augmentation.
"""

import pandas as pd
from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger

logger = get_logger()

def filter_eligible_papers(
    augmentation_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    config: AppConfig
) -> pd.DataFrame:
    """
    Include ALL papers from the augmentation source.
    User requested to avoid filtering for duplicates or short content.
    """
    logger.info(f"Including all {len(augmentation_df)} augmentation papers as requested.")
    return augmentation_df
