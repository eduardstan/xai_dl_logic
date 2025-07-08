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
        params = self.config.stage_2.selection_strategy.ratio_params
        params.target_selection_ratio = 0.1  # 10%
        params.min_papers_per_cluster = 3

        # Test cases: (cluster_size, expected_selection)
        test_cases = [
            (10, 3),  # Below min, should select min
            (50, 5),  # 10% of 50 = 5
            (100, 10), # 10% of 100 = 10
            (2, 2),   # Cannot select more than exist
        ]

        for size, expected in test_cases:
            with self.subTest(size=size, expected=expected):
                self.assertEqual(determine_papers_to_select(size, self.config), expected)

    def test_threshold_based_selection(self):
        """Test the threshold-based paper selection strategy."""
        self.config.stage_2.selection_strategy.count_method = "threshold_based"
        params = self.config.stage_2.selection_strategy.threshold_params
        params.base_papers_per_cluster = 5
        params.cluster_thresholds = {
            "small": 10,
            "medium": 25,
            "large": 50,
            "xlarge": 100,
            "xxlarge": 200,
        }

        # Test cases: (cluster_size, expected_selection)
        # small <= 10 -> base (5)
        # medium <= 25 -> base + 1 (6)
        # large <= 50 -> base + 2 (7)
        # xlarge <= 100 -> base + 3 (8)
        # xxlarge <= 200 -> base + 4 (9)
        # xxxlarge > 200 -> max_papers (from config, default 200)
        test_cases = [
            (5, 5),     # small
            (10, 5),    # small
            (11, 6),    # medium
            (25, 6),    # medium
            (50, 7),    # large
            (100, 8),   # xlarge
            (150, 9),   # xxlarge
            (250, 200), # xxxlarge (capped by max_papers_per_cluster)
            (3, 3),     # Cannot select more than exist
        ]

        for size, expected in test_cases:
            with self.subTest(size=size, expected=expected):
                self.assertEqual(determine_papers_to_select(size, self.config), expected)

    def test_invalid_cluster_size(self):
        """Test that invalid cluster sizes are handled gracefully."""
        self.assertEqual(determine_papers_to_select(0, self.config), 0)
        self.assertEqual(determine_papers_to_select(-10, self.config), 0)

    def test_unknown_strategy(self):
        """Test that an unknown strategy defaults to ratio-based."""
        self.config.stage_2.selection_strategy.count_method = "unknown_strategy"
        # It should fall back to the ratio-based calculation.
        self.assertEqual(determine_papers_to_select(50, self.config), 5)


if __name__ == "__main__":
    unittest.main() 