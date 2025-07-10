#!/usr/bin/env python3
"""
Tests for the stage 2 report generation module.
"""
import pandas as pd
import pytest
from pathlib import Path

from research_analysis.config.loader import load_config
from research_analysis.stages.stage_2_paper_selection.report import generate_selection_report

@pytest.fixture
def mock_data():
    """Generates mock data for testing the report generation."""
    results_data = {
        'topic_id': [0, 0, 1, 1, 1],
        'is_selected_representative': [True, False, True, False, False],
        'title': [f'Paper {i}' for i in range(5)],
        'author': [f'Author {i}' for i in range(5)],
        'similarity_to_centroid': [0.9, 0.8, 0.85, 0.75, 0.7],
        'diversity_score': [0.5, 0.6, 0.55, 0.65, 0.4],
        'representativeness_score': [0.7, 0.7, 0.7, 0.7, 0.55],
        'xai_alignment': [0.1, 0.2, 0.3, 0.4, 0.5],
        'representative_title': [None, 'Paper 0', None, 'Paper 2', 'Paper 2'],
        'similarity_to_representative': [None, 0.95, None, 0.9, 0.85]
    }
    results_df = pd.DataFrame(results_data)

    summary_data = {
        'topic_id': [0, 1],
        'cluster_size': [2, 3]
    }
    summary_df = pd.DataFrame(summary_data)

    selected_df = results_df[results_df['is_selected_representative']].copy()
    
    return results_df, summary_df, selected_df

def test_generate_selection_report(tmp_path, mock_data):
    """
    Tests the main report generation function.
    """
    config = load_config()
    results_df, summary_df, selected_df = mock_data
    
    report_path = generate_selection_report(
        config, results_df, summary_df, selected_df, tmp_path
    )

    assert report_path.exists()
    assert report_path.name == "selection_analysis_report.md"

    content = report_path.read_text()
    
    # Check for key sections
    assert "Executive Summary" in content
    assert "Selection Configuration" in content
    assert "Metrics Comparison" in content
    assert "Research Alignment" in content
    assert "Paper Assignment Analysis" in content
    assert "Reading Recommendations" in content
    
    # Check for specific data points
    expected_summary = """## 📊 Executive Summary
- **Total Papers Analyzed:** 5
- **Representative Papers Selected:** 2
- **Selection Ratio:** 40.00%
- **Topics Covered:** 2"""
    assert expected_summary in content

    # Check for assignment similarity using a more robust method
    assert "Average Assignment Similarity:" in content
    assert "0.9000" in content
    
    assert "- **Xai Alignment:** 20.00%" in content
    assert "Paper 0" in content
    assert "Paper 2" in content 