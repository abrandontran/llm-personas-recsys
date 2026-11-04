"""Unit tests for behavioral signatures phi(u) and k-means archetype extraction.

Task 4 (Bucket 2b) of the Pre-Bouchet Pipeline.
"""

import unittest
import numpy as np
from src.data.archetypes import (
    CANONICAL_GENRES,
    compute_behavioral_signature,
    compute_all_user_signatures,
    extract_k_archetypes,
)


class TestArchetypes(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_signature_shape_and_bounds(self):
        """User watched 10 items across 18 genres, all Action, constant ratings."""
        genre_indicators = np.zeros((10, 18), dtype=float)
        genre_indicators[:, 0] = 1.0  # all Action
        ratings = np.array([4.0] * 10)
        pop_counts = np.array([100] * 10)
        phi = compute_behavioral_signature(ratings, genre_indicators, pop_counts)

        # phi = [genre_simplex (18), entropy (1), satiation_slope (1), popularity_conformity (1)] -> 21
        self.assertEqual(len(phi), 21)
        self.assertAlmostEqual(phi[0], 1.0, places=5)  # 100% action
        self.assertAlmostEqual(phi[18], 0.0, places=5)  # zero entropy
        self.assertAlmostEqual(phi[19], 0.0, places=5)  # constant ratings -> zero satiation slope
        self.assertAlmostEqual(phi[20], 0.0, places=5)  # constant pop -> zero popularity correlation

    def test_entropy_uniform_distribution(self):
        """User with uniform preference across all 18 genres should have max entropy log(18)."""
        genre_indicators = np.eye(18, dtype=float)
        ratings = np.ones(18, dtype=float) * 3.0
        pop_counts = np.arange(1, 19, dtype=float) * 50

        phi = compute_behavioral_signature(ratings, genre_indicators, pop_counts)
        self.assertEqual(len(phi), 21)
        np.testing.assert_allclose(phi[:18], 1.0 / 18.0, rtol=1e-5)
        expected_entropy = np.log(18.0)
        self.assertAlmostEqual(phi[18], expected_entropy, places=4)

    def test_satiation_decay_slope_positive_and_negative(self):
        """Test satiation decay slope with declining vs. increasing ratings within 3-step window."""
        # 4 Action movies in a row: ratings drop 4.0 -> 3.0 -> 2.0 -> 1.0 (fatigue / satiation)
        genre_indicators = np.zeros((4, 18), dtype=float)
        genre_indicators[:, 0] = 1.0  # Action
        ratings_fatigue = np.array([4.0, 3.0, 2.0, 1.0])
        pop_counts = np.array([100, 100, 100, 100])

        phi_fatigue = compute_behavioral_signature(ratings_fatigue, genre_indicators, pop_counts)
        # k = 0, 1, 2, 3 with ratings = 4, 3, 2, 1 -> OLS slope = -1.0
        self.assertAlmostEqual(phi_fatigue[19], -1.0, places=4)

        # Binge lover: ratings increase 1.0 -> 2.0 -> 3.0 -> 4.0
        ratings_binge = np.array([1.0, 2.0, 3.0, 4.0])
        phi_binge = compute_behavioral_signature(ratings_binge, genre_indicators, pop_counts)
        self.assertAlmostEqual(phi_binge[19], 1.0, places=4)

        # No repeats within 3 steps: 4 different genres -> slope defaults to 0.0
        genre_indicators_diff = np.zeros((4, 18), dtype=float)
        for i in range(4):
            genre_indicators_diff[i, i] = 1.0
        phi_diff = compute_behavioral_signature(ratings_fatigue, genre_indicators_diff, pop_counts)
        self.assertAlmostEqual(phi_diff[19], 0.0, places=4)

    def test_popularity_conformity_spearman(self):
        """Test popularity conformity with strong positive vs. negative correlation."""
        # 5 distinct items with strictly increasing ratings and strictly increasing popularity
        genre_indicators = np.zeros((5, 18), dtype=float)
        genre_indicators[:, 0] = 1.0
        ratings_pop = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pop_counts_asc = np.array([10, 50, 200, 1000, 5000])

        phi_conformist = compute_behavioral_signature(ratings_pop, genre_indicators, pop_counts_asc)
        self.assertGreater(phi_conformist[20], 0.95)

        # Reverse popularity (contrarian / indie lover)
        pop_counts_desc = np.array([5000, 1000, 200, 50, 10])
        phi_contrarian = compute_behavioral_signature(ratings_pop, genre_indicators, pop_counts_desc)
        self.assertLess(phi_contrarian[20], -0.95)

    def test_extract_k_archetypes_structure(self):
        """Extract k=5 archetypes on synthetic data and verify schema and bounds."""
        num_users = 50
        d = 32
        k = 5
        user_ids = list(range(101, 101 + num_users))
        user_embs = self.rng.normal(0, 1.0, size=(num_users, d)).astype(np.float32)

        # Generate synthetic signatures (num_users, 21)
        genre_simplex = self.rng.dirichlet(np.ones(18), size=num_users).astype(np.float32)
        entropies = -np.sum(genre_simplex * np.log(genre_simplex + 1e-12), axis=1, keepdims=True)
        satiation = self.rng.uniform(-0.5, 0.5, size=(num_users, 1)).astype(np.float32)
        conformity = self.rng.uniform(-0.8, 0.8, size=(num_users, 1)).astype(np.float32)
        user_signatures = np.hstack([genre_simplex, entropies, satiation, conformity]).astype(np.float32)

        # Synthetic item metadata and interactions
        item_metadata = {
            str(i): {
                "title": f"Movie {i}",
                "year": 1990 + (i % 20),
                "genres": [CANONICAL_GENRES[i % 18]],
                "synopsis": f"Synopsis for Movie {i}",
            }
            for i in range(1, 30)
        }

        train_interactions = {}
        for uid in user_ids:
            # 10 interactions per user
            train_interactions[uid] = [
                (self.rng.integers(1, 30), float(self.rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])), 1000 + t)
                for t in range(10)
            ]

        archetypes = extract_k_archetypes(
            user_embeddings=user_embs,
            user_signatures=user_signatures,
            user_ids=user_ids,
            train_interactions=train_interactions,
            item_metadata=item_metadata,
            k=k,
            random_state=42,
        )

        self.assertEqual(len(archetypes), k)
        for arc in archetypes:
            self.assertIn("archetype_id", arc)
            self.assertIn("user_id", arc)
            self.assertIn("u_emb", arc)
            self.assertIn("phi", arc)
            self.assertIn("tau_quit", arc)
            self.assertIn("system_prompt", arc)
            self.assertIn("top_genres", arc)
            self.assertIn("favorite_movies", arc)
            self.assertIn("genre_simplex", arc)

            self.assertEqual(len(arc["u_emb"]), 32)
            self.assertEqual(len(arc["phi"]), 21)
            self.assertEqual(len(arc["genre_simplex"]), 18)
            self.assertGreaterEqual(arc["tau_quit"], 1.0)
            self.assertLessEqual(arc["tau_quit"], 3.5)

            # Check prompt structure
            prompt = arc["system_prompt"]
            self.assertIn("Favorite Genres:", prompt)
            self.assertIn("Adored Titles:", prompt)
            self.assertIn("Satiation Tendency:", prompt)
            self.assertIn("Rating Style:", prompt)
            self.assertIn(f"{arc['tau_quit']:.1f}/5.0", prompt)

    def test_compute_all_user_signatures(self):
        """Test compute_all_user_signatures with synthetic dataset."""
        user_ids = [1, 2, 3]
        item_ids = [10, 20, 30]
        item_metadata = {
            "10": {"title": "M10", "year": 1995, "genres": ["Action"]},
            "20": {"title": "M20", "year": 1996, "genres": ["Comedy"]},
            "30": {"title": "M30", "year": 1997, "genres": ["Action", "Drama"]},
        }
        train_interactions = {
            1: [(10, 5.0, 100), (20, 4.0, 200)],
            2: [(10, 3.0, 100), (30, 4.0, 200)],
            3: [(20, 4.0, 100), (30, 2.0, 200)],
        }

        signatures, item_pop = compute_all_user_signatures(
            train_interactions=train_interactions,
            item_metadata=item_metadata,
            user_ids=user_ids,
            item_ids=item_ids,
        )

        self.assertEqual(signatures.shape, (3, 21))
        # Each simplex should sum to 1.0
        np.testing.assert_allclose(np.sum(signatures[:, :18], axis=1), 1.0, rtol=1e-5)
        # Item popularity checks
        self.assertEqual(item_pop[10], 2)
        self.assertEqual(item_pop[20], 2)
        self.assertEqual(item_pop[30], 2)


if __name__ == "__main__":
    unittest.main()
