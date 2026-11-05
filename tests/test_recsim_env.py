"""Unit tests for Gymnasium RecSys Environment and Boltzmann OPE Policy (Bucket 3)."""

import unittest
import numpy as np

from src.envs.base_env import RecEnv
from src.envs.recsim_stat_env import ParametricRecSimEnv


class TestRecSimEnv(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 50 items, 32 dims unit-normalized
        raw_embs = np.random.randn(50, 32).astype(np.float32)
        self.item_embs = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)

        # Multi-hot genres for 50 items across 18 genres
        self.item_genres = np.zeros((50, 18), dtype=np.float32)
        self.item_genres[:25, 0] = 1.0  # Items 0..24 are Action (genre 0)
        self.item_genres[25:, 1] = 1.0  # Items 25..49 are Adventure (genre 1)

        # Archetype with strong preference for Action (genre 0)
        genre_simplex = [0.0] * 18
        genre_simplex[0] = 0.8
        genre_simplex[1] = 0.2

        self.archetypes = [
            {
                "archetype_id": 0,
                "user_id": 1001,
                "u_emb": self.item_embs[0].tolist(),  # Aligned with item 0
                "genre_simplex": genre_simplex,
                "tau_quit": 2.5,
            },
            {
                "archetype_id": 1,
                "user_id": 1002,
                "u_emb": self.item_embs[25].tolist(),  # Aligned with item 25
                "genre_simplex": genre_simplex,
                "tau_quit": 4.5,  # Very high quit threshold -> quick churn
            },
        ]

    def test_step_and_observation_shape(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            max_steps=20,
            seed=42,
        )
        obs, info = env.reset(seed=42, options={"archetype_id": 0})
        self.assertEqual(obs.shape, (202,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(info["user_id"], 1001)

        # At t=0, history slots should be zero-padded
        self.assertTrue(np.all(obs[32:] == 0.0))

        # Take a step with continuous action in [-1, 1]^32
        action = np.zeros(32, dtype=np.float32)
        action[0] = 1.0
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        self.assertEqual(next_obs.shape, (202,))
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.25)
        self.assertIn("rating", step_info)
        self.assertIn("expected_rating", step_info)
        self.assertIn("recommended_item", step_info)
        self.assertEqual(step_info["step"], 1)

        # Check that first history slot in next_obs is now populated
        # Slot 1: obs[32:64] is item emb, obs[64] is rating, obs[65] is lag (0.2)
        self.assertAlmostEqual(next_obs[65], 0.2, places=4)

    def test_no_repetition_constraint(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            max_steps=10,
            seed=42,
        )
        env.reset(seed=42, options={"archetype_id": 0})

        # Provide the same action repeatedly
        fixed_action = np.ones(32, dtype=np.float32) * 0.5
        recommended = []
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(fixed_action)
            item_idx = info["recommended_item"]
            recommended.append(item_idx)
            if terminated or truncated:
                break

        # Verify all recommended items are unique
        self.assertEqual(len(recommended), len(set(recommended)))

    def test_boltzmann_ope_distribution(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            seed=42,
        )
        env.reset(seed=42, options={"archetype_id": 0})
        action = np.random.randn(32).astype(np.float32)

        # Unconsumed distribution over 50 items
        probs = env.get_action_distribution(action, tau_pi=0.1)
        self.assertEqual(probs.shape, (50,))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertTrue(np.all(probs >= 0.0))

        # Recommend an item and verify it receives 0.0 probability afterwards
        _, _, _, _, info = env.step(action)
        consumed_item = info["recommended_item"]

        probs_after = env.get_action_distribution(action, tau_pi=0.1)
        self.assertEqual(probs_after[consumed_item], 0.0)
        self.assertAlmostEqual(float(np.sum(probs_after)), 1.0, places=5)

    def test_satiation_decay_dynamics(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            beta=0.3,
            seed=42,
        )
        env.reset(seed=42, options={"archetype_id": 0})

        initial_genre_weight = float(env.w_t[0])
        # Recommend item 0 (Action, genre 0)
        # Action vector aligned with item 0
        action = self.item_embs[0].copy()
        _, _, _, _, info = env.step(action)
        rec_item = info["recommended_item"]
        self.assertEqual(self.item_genres[rec_item, 0], 1.0)

        # Genre weight for genre 0 should have decayed by exp(-0.3)
        expected_weight = initial_genre_weight * np.exp(-0.3)
        self.assertAlmostEqual(float(env.w_t[0]), float(expected_weight), places=5)

    def test_churn_termination(self):
        # Archetype 1 has tau_quit = 4.5. Expected rating will rarely exceed 4.5.
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            seed=42,
        )
        _, info = env.reset(seed=42, options={"archetype_id": 1})
        self.assertEqual(info["tau_quit"], 4.5)

        # Orthogonal action to user embedding -> expected rating ~ 3.0 < 4.5
        orthogonal_action = np.random.randn(32).astype(np.float32)
        # Step once
        _, _, terminated, _, step_info = env.step(orthogonal_action)
        self.assertTrue(terminated)
        self.assertEqual(step_info["continuation"], 0)

    def test_consecutive_low_ratings_churn(self):
        # Even if tau_quit is low (1.0), 2 consecutive ratings <= 2.0 must terminate
        custom_archetype = [{
            "archetype_id": 99,
            "user_id": 9999,
            "u_emb": [0.0] * 32,
            "genre_simplex": [1.0 / 18.0] * 18,
            "tau_quit": 1.0,  # Low threshold so rating >= tau_quit is always satisfied
        }]
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=custom_archetype,
            seed=42,
        )
        env.reset()

        from unittest.mock import MagicMock
        mock_rng = MagicMock()
        mock_rng.normal.return_value = -2.5  # Forces clamped rating 1.0 <= 2.0
        mock_rng.integers.return_value = 0
        env._np_random = mock_rng
        _, _, term1, _, info1 = env.step(np.ones(32, dtype=np.float32))
        self.assertFalse(term1)
        self.assertEqual(info1["consecutive_low_ratings"], 1)
        self.assertEqual(info1["continuation"], 1)

        # Second low rating triggers churn
        _, _, term2, _, info2 = env.step(np.ones(32, dtype=np.float32) * -1)
        self.assertTrue(term2)
        self.assertEqual(info2["consecutive_low_ratings"], 2)
        self.assertEqual(info2["continuation"], 0)

    def test_fifo_history_and_lags(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            max_steps=20,
            seed=42,
        )
        env.reset(seed=42, options={"archetype_id": 0})
        # Override tau_quit to prevent early churn
        env.tau_quit = 0.5

        actions = [np.random.randn(32).astype(np.float32) for _ in range(6)]
        items_and_ratings = []

        for t, act in enumerate(actions):
            next_obs, _, _, _, info = env.step(act)
            items_and_ratings.append((info["recommended_item"], info["rating"]))

            # Verify lags in observation
            num_slots = min(t + 1, 5)
            for k in range(num_slots):
                expected_lag = (k + 1) / 5.0
                slot_start = 32 + k * 34
                actual_lag = next_obs[slot_start + 33]
                self.assertAlmostEqual(actual_lag, expected_lag, places=5)
                # Verify rating matches the k-th most recent
                rec_idx = -(k + 1)
                self.assertAlmostEqual(next_obs[slot_start + 32], items_and_ratings[rec_idx][1], places=4)

    def test_horizon_truncation(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            max_steps=5,
            seed=42,
        )
        env.reset(seed=42, options={"archetype_id": 0})
        env.tau_quit = 0.5  # Prevent churn

        for step_i in range(5):
            act = np.random.randn(32).astype(np.float32)
            _, _, term, trunc, _ = env.step(act)
            if step_i < 4:
                self.assertFalse(trunc)
            else:
                self.assertTrue(trunc)

    def test_options_reset_archetype_lookup(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            seed=42,
        )
        # By archetype_idx
        _, info1 = env.reset(options={"archetype_idx": 1})
        self.assertEqual(info1["archetype_id"], 1)

        # By user_id
        _, info2 = env.reset(options={"user_id": 1001})
        self.assertEqual(info2["archetype_id"], 0)

    def test_boltzmann_numerical_stability(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            seed=42,
        )
        env.reset()
        action = np.ones(32, dtype=np.float32)
        # Very small temperature
        probs = env.get_action_distribution(action, tau_pi=1e-3)
        self.assertFalse(np.isnan(probs).any())
        self.assertFalse(np.isinf(probs).any())
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_array_like_consumed_items(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            seed=42,
        )
        env.reset()
        action = np.ones(32, dtype=np.float32)
        # Pass a NumPy array instead of a set
        consumed_arr = np.array([2, 5, 8], dtype=np.int64)
        probs = env.get_action_distribution(action, tau_pi=0.1, consumed_items=consumed_arr)
        self.assertEqual(probs[2], 0.0)
        self.assertEqual(probs[5], 0.0)
        self.assertEqual(probs[8], 0.0)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_non_positive_temperature_guard(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            seed=42,
        )
        env.reset()
        action = np.ones(32, dtype=np.float32)
        with self.assertRaises(ValueError):
            env.get_action_distribution(action, tau_pi=0.0)
        with self.assertRaises(ValueError):
            env.get_action_distribution(action, tau_pi=-0.1)

    def test_catalog_exhaustion_truncation(self):
        # Small catalog with only 3 items, max_steps = 10
        small_embs = self.item_embs[:3]
        small_genres = self.item_genres[:3]
        env = ParametricRecSimEnv(
            item_embeddings=small_embs,
            item_genres=small_genres,
            archetypes=self.archetypes,
            max_steps=10,
            seed=42,
        )
        env.reset(seed=42, options={"archetype_id": 0})
        env.tau_quit = 0.0  # Prevent churn

        # Step 1
        _, _, term1, trunc1, _ = env.step(np.random.randn(32).astype(np.float32))
        self.assertFalse(trunc1)
        # Step 2
        _, _, term2, trunc2, _ = env.step(np.random.randn(32).astype(np.float32))
        self.assertFalse(trunc2)
        # Step 3: All 3 items now consumed -> must set truncated = True
        _, _, term3, trunc3, _ = env.step(np.random.randn(32).astype(np.float32))
        self.assertTrue(trunc3)

    def test_nan_and_out_of_bounds_action_handling(self):
        env = ParametricRecSimEnv(
            item_embeddings=self.item_embs,
            item_genres=self.item_genres,
            archetypes=self.archetypes,
            seed=42,
        )
        env.reset(seed=42, options={"archetype_id": 0})
        # Action with NaNs and out of bounds values
        bad_action = np.array([np.nan, 99.0, -99.0] + [0.0] * 29, dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(bad_action)
        self.assertEqual(next_obs.shape, (202,))
        self.assertIn("recommended_item", info)


if __name__ == "__main__":
    unittest.main()
