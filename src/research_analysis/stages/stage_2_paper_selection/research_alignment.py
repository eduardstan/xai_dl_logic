#!/usr/bin/env python3
"""
Research alignment analysis for the research analysis framework.

This module analyzes text to determine its alignment with predefined
research domains (e.g., XAI, symbolic, subsymbolic) using wildcard
pattern matching.
"""

import re
from typing import Dict

from research_analysis.config.models import AppConfig
from research_analysis.utils.logging import get_logger

logger = get_logger()


def analyze_research_alignment(text: str, config: AppConfig) -> Dict[str, float]:
    """
    Analyzes text alignment with research domains defined in the config.

    Args:
        text: The input text to analyze.
        config: The application configuration.

    Returns:
        A dictionary of alignment scores for each research category.
    """
    if not text or not text.strip():
        return {}

    try:
        keywords = config.stage_2.research_alignment.model_dump()
        text_lower = text.lower()
        alignment_scores = {}

        for category, terms in keywords.items():
            if not terms:
                logger.warning(f"No terms defined for category '{category}'")
                alignment_scores[f"{category}_alignment"] = 0.0
                continue

            matches = 0
            for term in terms:
                pattern = _build_wildcard_pattern(term.lower())
                if re.search(pattern, text_lower):
                    matches += 1

            alignment_scores[f"{category}_alignment"] = matches / len(terms) if terms else 0.0

        return alignment_scores

    except Exception as e:
        logger.error(f"Failed to analyze research alignment: {e}", exc_info=True)
        return {
            f"{cat}_alignment": 0.0
            for cat in config.stage_2.research_alignment.model_dump()
        }


def _build_wildcard_pattern(term: str) -> str:
    """
    Builds a regex pattern for wildcard matching with word boundaries.

    Args:
        term: The lowercase term, which may contain a wildcard (*).

    Returns:
        A regex pattern string.
    """
    if term.endswith("*") and not term.startswith("*"):
        base = term[:-1]
        return r"\b" + re.escape(base) + r"\w*\b"
    elif term.startswith("*") and not term.endswith("*"):
        base = term[1:]
        return r"\b\w*" + re.escape(base) + r"\b"
    elif "*" in term:
        parts = [re.escape(p) for p in term.split("*")]
        return r"\b" + r"\w*".join(parts) + r"\b"
    else:
        return r"\b" + re.escape(term) + r"\b" 