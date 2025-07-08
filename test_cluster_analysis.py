#!/usr/bin/env python3
"""
Tests for the cluster analysis and paper selection count functions.
"""

import unittest

from research_analysis.config.loader import load_config
from research_analysis.stages.stage_2_paper_selection.cluster_analysis import (
    determine_papers_to_select,
)


class TestClusterAnalysis(unittest.TestCase):
    """Unit tests for the cluster analysis functions."""

    @classmethod
    def setUpClass(cls):
        """Set up a mock configuration for testing."""
        cls.config = load_config()

    def test_ratio_based_selection(self):
        """Test the ratio-based paper selection strategy."""
        self.config.stage_2.selection_strategy.count_method = "ratio_based"
        strategy_config = self.config.stage_2.selection_strategy
        strategy_config.ratio_params.target_selection_ratio = 0.1
        strategy_config.ratio_params.min_papers_per_cluster = 3
        strategy_config.max_papers_per_cluster = 15

        test_cases = [
            (10, 3),   # Below min, selects min
            (50, 5),   # 10% of 50 = 5
            (200, 15), # 10% of 200 = 20, but capped by max
            (2, 2),    # Cannot select more than exist
        ]

        for size, expected in test_cases:
            with self.subTest(f"ratio_{size}_{expected}"):
                self.assertEqual(determine_papers_to_select(size, self.config), expected)

    def test_threshold_based_selection(self):
        """Test the threshold-based paper selection strategy."""
        self.config.stage_2.selection_strategy.count_method = "threshold_based"
        strategy_config = self.config.stage_2.selection_strategy
        strategy_config.threshold_params.base_papers_per_cluster = 5
        strategy_config.max_papers_per_cluster = 10 # Lower for testing
        strategy_config.threshold_params.cluster_thresholds = {
            "small": 10, "medium": 25, "large": 50, "xlarge": 100, "xxlarge": 200
        }

        test_cases = [
            (5, 5),    # small -> base
            (50, 7),   # large -> base + 2
            (250, 10), # xxxlarge -> capped by max_papers
            (3, 3),    # Cannot select more than exist
        ]

        for size, expected in test_cases:
            with self.subTest(f"threshold_{size}_{expected}"):
                self.assertEqual(determine_papers_to_select(size, self.config), expected)

    def test_invalid_cluster_size(self):
        """Test that invalid cluster sizes are handled gracefully."""
        self.assertEqual(determine_papers_to_select(0, self.config), 0)
        self.assertEqual(determine_papers_to_select(-10, self.config), 0)


if __name__ == "__main__":
    unittest.main() 