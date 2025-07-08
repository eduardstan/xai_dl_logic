#!/usr/bin/env python3
"""
Tests for the paper metrics calculation module.
"""

import unittest

import numpy as np

from research_analysis.config.loader import load_config
from research_analysis.stages.stage_2_paper_selection.metrics import (
    compute_paper_metrics,
)


class TestMetrics(unittest.TestCase):
    """Unit tests for the metrics calculation functions."""

    @classmethod
    def setUpClass(cls):
        """Set up the test class with a mock configuration and embeddings."""
        cls.config = load_config()
        # Mock embeddings for a cluster of 3 papers
        cls.cluster_embeddings = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],  # Paper 1
                [0.5, 0.6, 0.7, 0.8],  # Paper 2 (more central)
                [0.9, 0.1, 0.2, 0.3],  # Paper 3 (more diverse)
            ]
        )
        cls.paper_embedding = cls.cluster_embeddings[0]

    def test_compute_paper_metrics_structure(self):
        """Test that compute_paper_metrics returns the correct structure."""
        metrics = compute_paper_metrics(
            self.paper_embedding, self.cluster_embeddings, self.config
        )
        self.assertIsInstance(metrics, dict)
        self.assertEqual(
            set(metrics.keys()),
            {
                "similarity_to_centroid",
                "avg_similarity_to_cluster",
                "diversity_score",
                "representativeness_score",
            },
        )

    def test_compute_paper_metrics_single_paper_cluster(self):
        """Test metrics for a cluster with only one paper."""
        single_paper_cluster = self.paper_embedding.reshape(1, -1)
        metrics = compute_paper_metrics(
            self.paper_embedding, single_paper_cluster, self.config
        )
        self.assertAlmostEqual(metrics["similarity_to_centroid"], 1.0)
        self.assertAlmostEqual(metrics["avg_similarity_to_cluster"], 1.0)
        self.assertEqual(metrics["diversity_score"], 1.0)
        # Representativeness should be 1.0 with default weight of 0.5
        self.assertAlmostEqual(metrics["representativeness_score"], 1.0)

    def test_compute_paper_metrics_values(self):
        """Test the computed metric values for a simple case."""
        metrics = compute_paper_metrics(
            self.paper_embedding, self.cluster_embeddings, self.config
        )
        # Correct, updated values based on the robust implementation
        self.assertAlmostEqual(metrics["similarity_to_centroid"], 0.907, places=3)
        self.assertAlmostEqual(
            metrics["avg_similarity_to_cluster"], 0.837, places=3
        )
        self.assertAlmostEqual(metrics["diversity_score"], 0.031, places=3)
        self.assertAlmostEqual(
            metrics["representativeness_score"], 0.250, places=3
        )


if __name__ == "__main__":
    unittest.main() 