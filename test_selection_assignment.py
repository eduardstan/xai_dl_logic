#!/usr/bin/env python3
"""
Tests for the paper assignment logic in the selection module.
"""

import unittest

import numpy as np

from research_analysis.config.loader import load_config
from research_analysis.stages.stage_2_paper_selection.selection import (
    assign_non_selected_papers,
)


class TestAssignment(unittest.TestCase):
    """Unit tests for the paper assignment functions."""

    @classmethod
    def setUpClass(cls):
        """Set up a mock configuration and embeddings for testing."""
        cls.config = load_config()
        cls.config.stage_2.metrics.similarity_threshold = 0.8

        cls.cluster_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # 0 (Selected)
                [0.9, 0.1, 0.0],  # 1 (Not Selected, similar to 0)
                [0.0, 1.0, 0.0],  # 2 (Selected)
                [0.1, 0.9, 0.0],  # 3 (Not Selected, similar to 2)
                [0.0, 0.0, 1.0],  # 4 (Not Selected, equidistant)
            ]
        )
        cls.selected_indices = [0, 2]

    def test_assign_non_selected_papers(self):
        """Test the standard assignment of non-selected papers."""
        assignments = assign_non_selected_papers(
            self.cluster_embeddings, self.selected_indices, self.config
        )

        # There are 3 non-selected papers (1, 3, 4)
        self.assertEqual(len(assignments), 3)

        # Paper 1 should be assigned to representative 0
        self.assertIn(1, assignments)
        self.assertEqual(assignments[1]["representative_idx"], 0)
        self.assertAlmostEqual(assignments[1]["similarity"], 0.994, places=3)
        self.assertTrue(assignments[1]["is_similar"])

        # Paper 3 should be assigned to representative 2
        self.assertIn(3, assignments)
        self.assertEqual(assignments[3]["representative_idx"], 2)
        self.assertAlmostEqual(assignments[3]["similarity"], 0.994, places=3)
        self.assertTrue(assignments[3]["is_similar"])

        # Paper 4 is equidistant, should be assigned to the first (0)
        self.assertIn(4, assignments)
        self.assertEqual(assignments[4]["representative_idx"], 0)
        self.assertAlmostEqual(assignments[4]["similarity"], 0.0, places=3)
        self.assertFalse(assignments[4]["is_similar"])

    def test_no_non_selected_papers(self):
        """Test assignment when all papers are selected."""
        all_selected = list(range(len(self.cluster_embeddings)))
        assignments = assign_non_selected_papers(
            self.cluster_embeddings, all_selected, self.config
        )
        self.assertEqual(len(assignments), 0)

    def test_no_representatives(self):
        """Test assignment when there are no representatives."""
        assignments = assign_non_selected_papers(
            self.cluster_embeddings, [], self.config
        )
        self.assertEqual(len(assignments), 0)


if __name__ == "__main__":
    unittest.main() 