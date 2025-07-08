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

    def test_select_diverse_representatives_standard(self):
        """Test standard selection of 3 diverse papers."""
        n_select = 3
        selected_indices = select_diverse_representatives(
            self.cluster_embeddings, n_select, self.config
        )
        self.assertEqual(len(selected_indices), n_select)
        self.assertIsInstance(selected_indices, list)
        # With the mock data, the most diverse papers should be the extremes.
        # The greedy algorithm should find indices 0, 3, and 5.
        self.assertCountEqual(selected_indices, [0, 3, 5])

    def test_select_more_papers_than_available(self):
        """Test selecting more papers than exist in the cluster."""
        n_select = 10
        selected_indices = select_diverse_representatives(
            self.cluster_embeddings, n_select, self.config
        )
        # Should return all available paper indices
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