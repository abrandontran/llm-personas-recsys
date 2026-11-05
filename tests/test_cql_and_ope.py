"""Unit tests for Conservative Q-Learning (CQL) and Off-Policy Evaluation (OPE) (Bucket 5)."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.cql import CQLAgent, CQLModel
from src.metrics.ope import (
    compute_snips_score,
    evaluate_doubly_robust,
    evaluate_kuairec_zeroshot,
)


class TestCQLAndOPE(unittest.TestCase):
    """Test suite for CQLModel, CQLAgent, SNIPS, and Doubly Robust OPE estimators."""

    def test_cql_model_forward_pass_and_shapes(self):
        cql = CQLModel(obs_dim=202, action_dim=32, hidden_dim=256)
        
        # Batch 2D inputs
        obs = torch.randn(8, 202)
        act = torch.randn(8, 32)
        q = cql(obs, act)
        self.assertEqual(q.shape, (8, 1))

        # 1D inputs
        obs_1d = torch.randn(202)
        act_1d = torch.randn(32)
        q_1d = cql(obs_1d, act_1d)
        self.assertEqual(q_1d.shape, (1,))

        # Concatenated single-tensor input
        x = torch.randn(8, 234)
        q_x = cql(x)
        self.assertEqual(q_x.shape, (8, 1))

    def test_cql_agent_update_and_loss_calculation(self):
        torch.manual_seed(42)
        agent = CQLAgent(
            obs_dim=202,
            action_dim=32,
            hidden_dim=256,
            lr=1e-3,
            gamma=0.99,
            alpha_cql=1.0,
        )

        batch_size = 16
        batch = {
            "obs": torch.randn(batch_size, 202),
            "act": torch.randn(batch_size, 32),
            "rew": torch.randn(batch_size, 1),
            "next_obs": torch.randn(batch_size, 202),
            "done": torch.zeros(batch_size, 1),
            "next_act": torch.randn(batch_size, 32),
        }

        metrics = agent.update(batch)
        self.assertIn("loss", metrics)
        self.assertIn("cql_loss", metrics)
        self.assertIn("td_loss", metrics)
        self.assertIn("q_data_mean", metrics)
        self.assertIn("q_rand_mean", metrics)
        self.assertFalse(np.isnan(metrics["loss"]))

    def test_cql_conservative_regularization_behavior(self):
        """Verifies CQL penalizes OOD random actions relative to logged in-distribution actions."""
        torch.manual_seed(42)
        np.random.seed(42)
        agent = CQLAgent(
            obs_dim=202,
            action_dim=32,
            hidden_dim=256,
            lr=3e-3,
            alpha_cql=5.0,  # strong conservative regularizer
        )

        batch_size = 64
        # Consistent fixed obs and fixed logged action
        fixed_obs = torch.zeros(batch_size, 202)
        fixed_act = torch.ones(batch_size, 32) * 0.5  # logged action
        rew = torch.ones(batch_size, 1) * 1.0
        next_obs = torch.zeros(batch_size, 202)
        done = torch.zeros(batch_size, 1)
        next_act = fixed_act.clone()

        batch = (fixed_obs, fixed_act, rew, next_obs, done, next_act)

        # Train for 50 steps
        for _ in range(50):
            agent.update(batch)

        # Check Q-values on logged actions vs random OOD actions
        with torch.no_grad():
            q_logged = agent.get_q_values(fixed_obs[:10], fixed_act[:10])
            rand_acts = torch.empty(10, 32).uniform_(-1.0, 1.0)
            q_rand = agent.get_q_values(fixed_obs[:10], rand_acts)

        mean_q_logged = float(q_logged.mean().item())
        mean_q_rand = float(q_rand.mean().item())
        self.assertGreater(
            mean_q_logged,
            mean_q_rand,
            f"Expected logged Q ({mean_q_logged:.3f}) > random Q ({mean_q_rand:.3f})",
        )

    def test_cql_agent_save_and_load(self):
        agent = CQLAgent(obs_dim=202, action_dim=32, hidden_dim=256)
        obs = torch.randn(4, 202)
        act = torch.randn(4, 32)
        q_orig = agent.get_q_values(obs, act).detach()

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "cql_test.pt"
            agent.save(ckpt_path)
            self.assertTrue(ckpt_path.exists())

            agent2 = CQLAgent(obs_dim=202, action_dim=32, hidden_dim=256)
            agent2.load(ckpt_path)
            q_loaded = agent2.get_q_values(obs, act).detach()

            torch.testing.assert_close(q_orig, q_loaded)

    def test_snips_estimator(self):
        # 2 trajectories, lengths 2 and 2
        rewards = [[1.0, 0.5], [0.8, -0.2]]
        weights = [[1.2, 0.9], [1.1, 1.0]]
        score = compute_snips_score(rewards, weights, gamma=0.99)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)

        # Hand calculation check:
        # num = 1.0 * 1.2 * 1.0 + 0.99 * 0.9 * 0.5 + 1.0 * 1.1 * 0.8 + 0.99 * 1.0 * (-0.2)
        #     = 1.2 + 0.4455 + 0.88 - 0.198 = 2.3275
        # den = 1.0 * 1.2 + 0.99 * 0.9 + 1.0 * 1.1 + 0.99 * 1.0
        #     = 1.2 + 0.891 + 1.1 + 0.99 = 4.181
        expected = 2.3275 / 4.181
        self.assertAlmostEqual(score, expected, places=5)

    def test_doubly_robust_estimator(self):
        # Build mock eval trajectories: 2 trajectories of 2 steps each
        # When Q-model perfectly predicts rewards (Q = r), DR should equal the model value
        class PerfectQ:
            def __call__(self, obs, act):
                return self.get_q_values(obs, act)

            def get_q_values(self, obs, act):
                if not isinstance(obs, torch.Tensor):
                    obs = torch.as_tensor(obs, dtype=torch.float32)
                if not isinstance(act, torch.Tensor):
                    act = torch.as_tensor(act, dtype=torch.float32)
                # Mock: return first element of act as predicted reward
                return act[:, :1] if act.ndim == 2 else act[:1]

        q_model = PerfectQ()

        # Policy always chooses action with first element = 1.0
        def eval_policy_fn(obs):
            act = np.zeros(32, dtype=np.float32)
            act[0] = 1.0
            return act

        # Trajectory transitions
        trajectories = [
            [
                {
                    "obs": np.zeros(202, dtype=np.float32),
                    "action": np.ones(32, dtype=np.float32) * 0.5,
                    "reward": 0.5,
                    "importance_weight": 1.0,
                },
                {
                    "obs": np.zeros(202, dtype=np.float32),
                    "action": np.ones(32, dtype=np.float32) * 0.5,
                    "reward": 0.5,
                    "importance_weight": 1.0,
                },
            ]
        ]

        v_dr = evaluate_doubly_robust(
            eval_trajectories=trajectories,
            eval_policy_fn=eval_policy_fn,
            q_model=q_model,
            gamma=0.99,
        )
        self.assertIsInstance(v_dr, float)
        # Expected: Q(o, a^pi) = 1.0 for each step; residual (r - Q(o, a)) = 0.5 - 0.5 = 0.
        # Sum = 1.0 + 0.99 * 1.0 = 1.99
        self.assertAlmostEqual(v_dr, 1.99, places=4)

    def test_kuairec_zeroshot_lookup(self):
        # 4 users, 5 items
        kuairec_matrix = np.array([
            [4.0, 1.0, 0.0, 3.0, 5.0],
            [2.0, 5.0, 4.0, 1.0, 0.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ], dtype=np.float32)

        # 5 items in 32 dimensions, orthogonal unit vectors
        kuairec_embs = np.zeros((5, 32), dtype=np.float32)
        for i in range(5):
            kuairec_embs[i, i] = 1.0

        # Policy always points toward item index 0
        def policy(obs):
            act = np.zeros(32, dtype=np.float32)
            act[0] = 1.0
            return act

        results = evaluate_kuairec_zeroshot(
            policy=policy,
            kuairec_embs=kuairec_embs,
            kuairec_matrix=kuairec_matrix,
            rating_threshold=3.0,
        )

        self.assertIn("mean_reward", results)
        self.assertIn("hit_rate", results)
        self.assertIn("mean_rating", results)
        # Item 0 ratings across 4 users: [4.0, 2.0, 5.0, 1.0] -> mean rating = 3.0
        self.assertAlmostEqual(results["mean_rating"], 3.0, places=4)
        # Hits (>= 3.0): 4.0 and 5.0 -> 2 out of 4 = 0.5
        self.assertAlmostEqual(results["hit_rate"], 0.5, places=4)

    def test_snips_clipped_weights_and_empty(self):
        # Weight clipping test
        rewards = [[1.0], [2.0]]
        weights = [[100.0], [50.0]]  # both exceed clip_weight=20.0
        score_clipped = compute_snips_score(rewards, weights, clip_weight=20.0)
        # Both weights clipped to 20.0, so weighted mean is (20*1 + 20*2)/(20+20) = 1.5
        self.assertAlmostEqual(score_clipped, 1.5, places=5)

        # Empty trajectories edge case
        score_empty = compute_snips_score([], [])
        self.assertEqual(score_empty, 0.0)

    def test_doubly_robust_empty(self):
        v_empty = evaluate_doubly_robust([], lambda o: np.zeros(32), lambda o, a: 0.0)
        self.assertEqual(v_empty, 0.0)


if __name__ == "__main__":
    unittest.main()
