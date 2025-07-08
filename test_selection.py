#!/usr/bin/env python3
"""
Tests for the paper selection algorithms.
"""

import unittest

import numpy as np

from research_analysis.config.loader import load_config
from research_analysis.stages.stage_2_paper_selection.selection import (
    select_diverse_representatives,
)


class TestSelection(unittest.TestCase):
    """Unit tests for the selection algorithms."""

    @classmethod
    def setUpClass(cls):
        """Set up a mock configuration and embeddings for testing."""
        cls.config = load_config()
        # A simple, predictable set of embeddings for a cluster
        cls.cluster_embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # 0: An extreme point
                [0.8, 0.2, 0.1, 0.1],  # 1: Central-ish
                [0.7, 0.3, 0.2, 0.2],  # 2: Another central-ish
                [0.0, 1.0, 0.0, 0.0],  # 3: Another extreme point
                [0.1, 0.8, 0.1, 0.1],  # 4: Close to #3
                [0.0, 0.0, 1.0, 0.0],  # 5: A third extreme point
            ]
        )

    def test_greedy_selection_standard(self):
        """Test standard greedy selection of 3 diverse papers."""
        self.config.stage_2.metrics.use_iterative_selection = False
        n_select = 3
        selected_indices = select_diverse_representatives(
            self.cluster_embeddings, n_select, self.config
        )
        self.assertEqual(len(selected_indices), n_select)
        self.assertIsInstance(selected_indices, list)
        # The greedy algorithm should find the most distinct papers.
        self.assertCountEqual(selected_indices, [0, 3, 5])

    def test_iterative_selection_standard(self):
        """Test standard iterative selection of 3 diverse papers."""
        self.config.stage_2.metrics.use_iterative_selection = True
        n_select = 3
        selected_indices = select_diverse_representatives(
            self.cluster_embeddings, n_select, self.config
        )
        self.assertEqual(len(selected_indices), n_select)
        self.assertIsInstance(selected_indices, list)
        # The iterative algorithm should also find the most distinct papers.
        self.assertCountEqual(selected_indices, [0, 3, 5])

    def test_diversity_threshold_filtering(self):
        """Test that the diversity pre-filtering works correctly."""
        self.config.stage_2.metrics.min_diversity_threshold = 0.8
        n_select = 3
        selected_indices = select_diverse_representatives(
            self.cluster_embeddings, n_select, self.config
        )
        # Only papers 0, 3, 5 should pass this high threshold.
        self.assertCountEqual(selected_indices, [0, 3, 5])
        # Reset threshold for other tests
        self.config.stage_2.metrics.min_diversity_threshold = 0.0

    def test_select_more_papers_than_available(self):
        """Test selecting more papers than exist in the cluster."""
        n_select = 10
        selected_indices = select_diverse_representatives(
            self.cluster_embeddings, n_select, self.config
        )
        self.assertEqual(len(selected_indices), len(self.cluster_embeddings))
        self.assertCountEqual(selected_indices, list(range(len(self.cluster_embeddings))))

    def test_select_zero_papers(self):
        """Test selecting zero papers."""
        selected_indices = select_diverse_representatives(
            self.cluster_embeddings, 0, self.config
        )
        self.assertEqual(len(selected_indices), 0)

    def test_empty_cluster(self):
        """Test selection from an empty cluster."""
        empty_embeddings = np.array([])
        selected_indices = select_diverse_representatives(
            empty_embeddings, 5, self.config
        )
        self.assertEqual(len(selected_indices), 0)


if __name__ == "__main__":
    unittest.main() 