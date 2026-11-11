from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from region_recommender import (  
    RecommendationConfig,
    evaluate_recommendations,
    lognormal_intensity,
    make_matched_references,
    maximin_lhs_indices,
    recommend_regions,
    rectangle_counts,
    rectangles_overlap,
)


class RegionRecommenderTests(unittest.TestCase):
    def test_lognormal_intensity_uses_mean_and_variance(self):
        result = lognormal_intensity(
            mu=np.array([0.0, np.log(2.0)]),
            variance=np.array([0.0, 0.0]),
            cell_area_km2=1.0,
        )
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)
        with_variance = lognormal_intensity(
            mu=np.array([0.0]), variance=np.array([2.0]), cell_area_km2=2.0
        )
        self.assertAlmostEqual(with_variance[0], (np.e - 1) / 2)

    def test_rectangle_counts_matches_bruteforce_and_closed_boundaries(self):
        points = np.array([
            [0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.2, 0.8],
            [0.7, 0.3], [0.4, 0.4], [0.9, 0.1],
        ])
        boxes = np.array([
            [0.5, 0.5, 1.0, 1.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.5, 0.5, 0.4, 0.4],
        ])
        expected = []
        for cx, cy, lx, ly in boxes:
            expected.append(np.sum(
                (points[:, 0] >= cx - lx / 2) & (points[:, 0] <= cx + lx / 2)
                & (points[:, 1] >= cy - ly / 2) & (points[:, 1] <= cy + ly / 2)
            ))
        np.testing.assert_array_equal(rectangle_counts(points, boxes), expected)

    def test_matched_references_preserve_lengths_and_are_feasible(self):
        boxes = np.array([[0.2, 0.7, 0.1, 0.2], [0.8, 0.3, 0.15, 0.05]])
        references = make_matched_references(boxes, n_reference=2000, seed=20261024)
        self.assertEqual(references.shape, (2, 2000, 4))
        np.testing.assert_allclose(references[:, :, 2], np.broadcast_to(boxes[:, None, 2], (2, 2000)))
        np.testing.assert_allclose(references[:, :, 3], np.broadcast_to(boxes[:, None, 3], (2, 2000)))
        self.assertTrue(np.all(references[:, :, 0] - references[:, :, 2] / 2 >= 0))
        self.assertTrue(np.all(references[:, :, 0] + references[:, :, 2] / 2 <= 1))
        self.assertTrue(np.all(references[:, :, 1] - references[:, :, 3] / 2 >= 0))
        self.assertTrue(np.all(references[:, :, 1] + references[:, :, 3] / 2 <= 1))

    def test_evaluation_uses_raw_incident_counts_and_expected_metrics(self):
        rng = np.random.default_rng(7)
        points = rng.random((200, 2))
        boxes = np.array([[0.5, 0.5, 0.4, 0.4]])
        references = make_matched_references(boxes, 100, 20261024)

        def brute(bs):
            return np.array([
                np.sum(
                    (points[:, 0] >= b[0] - b[2] / 2) & (points[:, 0] <= b[0] + b[2] / 2)
                    & (points[:, 1] >= b[1] - b[3] / 2) & (points[:, 1] <= b[1] + b[3] / 2)
                ) for b in np.asarray(bs).reshape(-1, 4)
            ])

        result = evaluate_recommendations(
            boxes, points, domain_width_km=(30.0, 40.0), n_reference=100, seed=20261024
        )
        count = brute(boxes)[0]
        ref_counts = brute(references)
        area = 0.4 * 30 * 0.4 * 40
        expected_q3 = np.quantile(ref_counts, 0.75, method="linear")
        self.assertEqual(result["counts"][0], count)
        self.assertAlmostEqual(result["reference_q3"][0], expected_q3)
        self.assertAlmostEqual(result["region_pass"], float(count > expected_q3))
        expected_lift = (count / area) / np.mean(ref_counts / area)
        self.assertAlmostEqual(result["density_lift"], expected_lift)

    def test_gso_returns_valid_reproducible_nonoverlapping_regions(self):
        def intensity(points):
            return (
                100 * np.exp(-((points[:, 0] - 0.25) ** 2 + (points[:, 1] - 0.3) ** 2) / 0.005)
                + 80 * np.exp(-((points[:, 0] - 0.75) ** 2 + (points[:, 1] - 0.7) ** 2) / 0.005)
                + 1.0
            )

        config = RecommendationConfig(k=2, particles=40, iterations=25, n_sobol=64)
        first = recommend_regions(intensity, (35.0, 50.0), config, seed=123)
        second = recommend_regions(intensity, (35.0, 50.0), config, seed=123)
        boxes = first["boxes"]
        self.assertEqual(boxes.shape, (2, 4))
        self.assertTrue(first["success"])
        self.assertTrue(np.all(boxes[:, 2:] >= config.length_min))
        self.assertTrue(np.all(boxes[:, 2:] <= config.length_max))
        self.assertFalse(rectangles_overlap(boxes[:1], boxes[1:])[0])
        np.testing.assert_allclose(first["boxes"], second["boxes"])
        np.testing.assert_allclose(first["scores"], second["scores"])

    def test_gso_reuses_initial_candidates_when_final_particles_collapse(self):
        initial = np.array([
            [0.1, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1],
        ])
        final = np.repeat([[0.5, 0.5, 0.1, 0.1]], 4, axis=0)

        class CollapsingObjective:
            def __init__(self, intensity_fn, domain_width_km, config, box_valid_fn=None):
                self.evaluations = 0

            def valid_mask(self, particles):
                return np.ones(len(particles), dtype=bool)

            def initialize(self, rng):
                return initial.copy()

            def values(self, particles):
                self.evaluations += len(particles)
                return np.arange(len(particles), dtype=float)

            def repair(self, particles):
                return final.copy()

        config = RecommendationConfig(k=4, particles=4, iterations=1, n_sobol=64)
        with patch("region_recommender._SobolObjective", CollapsingObjective):
            result = recommend_regions(lambda points: np.ones(len(points)), (35.0, 50.0), config)
        self.assertTrue(result["success"])
        self.assertEqual(result["boxes"].shape, (4, 4))
        self.assertEqual(result["evaluations"], 12)

    def test_maximin_lhs_is_unique_and_reproducible(self):
        rng = np.random.default_rng(42)
        xy = rng.random((80, 2))
        a = maximin_lhs_indices(xy, n=15, seed=123, n_designs=4)
        b = maximin_lhs_indices(xy, n=15, seed=123, n_designs=4)
        self.assertEqual(len(np.unique(a)), 15)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
