"""Unit tests for Pure PyTorch Soft Actor-Critic (SAC) Agent (Bucket 4)."""

import os
import tempfile
import unittest

import numpy as np
import torch

from src.models.sac import Actor, Critic, ReplayBuffer, SACAgent


class TestSACModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.obs_dim = 202
        self.act_dim = 32
        self.hidden_dim = 256
        self.batch_size = 8

    def test_actor_shapes_bounds_and_modes(self):
        """Actor output shape (B, 32), bounds [-1, 1], log_prob (B, 1), deterministic vs stochastic."""
        actor = Actor(obs_dim=self.obs_dim, act_dim=self.act_dim, hidden_dim=self.hidden_dim)
        obs = torch.randn(self.batch_size, self.obs_dim)

        # Stochastic mode
        act_stoch, log_prob_stoch = actor(obs, deterministic=False)
        self.assertEqual(act_stoch.shape, (self.batch_size, self.act_dim))
        self.assertEqual(log_prob_stoch.shape, (self.batch_size, 1))
        self.assertTrue((act_stoch >= -1.0).all() and (act_stoch <= 1.0).all())

        # Second stochastic pass should differ due to Gaussian sampling
        act_stoch2, _ = actor(obs, deterministic=False)
        self.assertFalse(torch.allclose(act_stoch, act_stoch2))

        # Deterministic mode
        act_det1, _ = actor(obs, deterministic=True)
        act_det2, _ = actor(obs, deterministic=True)
        self.assertEqual(act_det1.shape, (self.batch_size, self.act_dim))
        self.assertTrue((act_det1 >= -1.0).all() and (act_det1 <= 1.0).all())
        # Deterministic mode must be identical across calls
        self.assertTrue(torch.allclose(act_det1, act_det2))

    def test_critic_shapes(self):
        """Twin critics take concatenated (o, a) and output shapes (B, 1)."""
        critic = Critic(obs_dim=self.obs_dim, act_dim=self.act_dim, hidden_dim=self.hidden_dim)
        obs = torch.randn(self.batch_size, self.obs_dim)
        act = torch.randn(self.batch_size, self.act_dim)

        q1, q2 = critic(obs, act)
        self.assertEqual(q1.shape, (self.batch_size, 1))
        self.assertEqual(q2.shape, (self.batch_size, 1))

    def test_replay_buffer(self):
        """ReplayBuffer stores transitions and samples correctly with circular overwrite."""
        capacity = 10
        buf = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, capacity=capacity)
        self.assertEqual(len(buf), 0)

        # Add 15 transitions to verify circular overwrite
        for i in range(15):
            o = np.full(self.obs_dim, fill_value=float(i), dtype=np.float32)
            a = np.full(self.act_dim, fill_value=float(i), dtype=np.float32)
            r = float(i)
            o_next = np.full(self.obs_dim, fill_value=float(i + 1), dtype=np.float32)
            d = bool(i % 5 == 0)
            buf.add(o, a, r, o_next, d)

        self.assertEqual(len(buf), capacity)

        batch_o, batch_a, batch_r, batch_o_next, batch_d = buf.sample(batch_size=4)
        self.assertEqual(batch_o.shape, (4, self.obs_dim))
        self.assertEqual(batch_a.shape, (4, self.act_dim))
        self.assertEqual(batch_r.shape, (4, 1))
        self.assertEqual(batch_o_next.shape, (4, self.obs_dim))
        self.assertEqual(batch_d.shape, (4, 1))
        self.assertIsInstance(batch_o, torch.Tensor)

    def test_agent_sample_action_and_q_values(self):
        """SACAgent interface: sample_action on 1D/2D inputs and get_q_values."""
        agent = SACAgent(obs_dim=self.obs_dim, action_dim=self.act_dim, hidden_dim=self.hidden_dim)

        # Batched Tensor observation
        obs_tensor = torch.randn(4, self.obs_dim)
        action, log_prob = agent.sample_action(obs_tensor)
        self.assertEqual(action.shape, (4, self.act_dim))
        self.assertEqual(log_prob.shape, (4, 1))
        self.assertTrue((action >= -1.0).all() and (action <= 1.0).all())

        # Critic evaluation via agent
        q1, q2 = agent.get_q_values(obs_tensor, action)
        self.assertEqual(q1.shape, (4, 1))
        self.assertEqual(q2.shape, (4, 1))

        # Single 1D NumPy observation (env interaction style)
        obs_np = np.random.randn(self.obs_dim).astype(np.float32)
        act_np, lp_np = agent.sample_action(obs_np)
        self.assertIsInstance(act_np, np.ndarray)
        self.assertEqual(act_np.shape, (self.act_dim,))
        self.assertTrue(np.all(act_np >= -1.0) and np.all(act_np <= 1.0))

        # Single 1D NumPy observation with return_log_prob=False
        act_np_only = agent.sample_action(obs_np, return_log_prob=False)
        self.assertIsInstance(act_np_only, np.ndarray)
        self.assertEqual(act_np_only.shape, (self.act_dim,))

    def test_agent_update_step(self):
        """Single SAC gradient update executes without divergence or NaN."""
        agent = SACAgent(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            hidden_dim=self.hidden_dim,
            lr=1e-3,
        )

        # Seed replay buffer with dummy transitions
        for _ in range(50):
            o = np.random.randn(self.obs_dim).astype(np.float32)
            a = np.random.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
            r = float(np.random.randn())
            o_prime = np.random.randn(self.obs_dim).astype(np.float32)
            d = False
            agent.replay_buffer.add(o, a, r, o_prime, d)

        losses = agent.update(batch_size=16)
        self.assertIn("critic_loss", losses)
        self.assertIn("actor_loss", losses)
        self.assertIn("alpha_loss", losses)
        self.assertIn("alpha", losses)

        self.assertFalse(np.isnan(losses["critic_loss"]))
        self.assertFalse(np.isnan(losses["actor_loss"]))
        self.assertFalse(np.isnan(losses["alpha_loss"]))
        self.assertGreater(losses["alpha"], 0.0)

    def test_save_and_load_round_trip(self):
        """SACAgent state dict save and load preserves network weights."""
        agent = SACAgent(obs_dim=self.obs_dim, action_dim=self.act_dim, hidden_dim=self.hidden_dim)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "sac_checkpoint.pt")
            agent.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            new_agent = SACAgent(obs_dim=self.obs_dim, action_dim=self.act_dim, hidden_dim=self.hidden_dim)

            # Modify weights of new agent to ensure load actually changes them
            with torch.no_grad():
                for p in new_agent.actor.parameters():
                    p.add_(1.0)

            # Before load, weights should differ
            p_old = next(iter(agent.actor.parameters()))
            p_new = next(iter(new_agent.actor.parameters()))
            self.assertFalse(torch.equal(p_old, p_new))

            # Load checkpoint
            new_agent.load(save_path)
            p_loaded = next(iter(new_agent.actor.parameters()))
            self.assertTrue(torch.equal(p_old, p_loaded))


if __name__ == "__main__":
    unittest.main()
