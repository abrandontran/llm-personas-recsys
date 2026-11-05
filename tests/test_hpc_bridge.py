"""Unit tests for HPC Bridge, Mock LLM Emulator, Slurm Suite & Milestone Visualizations (Buckets 6 & 7)."""

import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is in search path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.envs.base_env import RecEnv
from src.envs.mock_llm_env import MockLLMEnv
from scripts.evaluate_fidelity_ablation import (
    compute_ndcg_at_k,
    format_candidate_item_string,
    query_vllm_rating,
)


class TestMockLLMEnv(unittest.TestCase):
    """Test suite for MockLLMEnv emulator."""

    def setUp(self):
        np.random.seed(42)
        # 40 items with 32-dim unit embeddings
        raw_embs = np.random.randn(40, 32).astype(np.float32)
        norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
        self.item_embs = raw_embs / norms

        self.archetypes = [
            {
                "archetype_id": 0,
                "user_id": 101,
                "u_emb": self.item_embs[0].copy(),  # strongly prefers item 0
                "tau_quit": 2.5,
                "top_genres": ["Drama"],
            },
            {
                "archetype_id": 1,
                "user_id": 202,
                "u_emb": -self.item_embs[0].copy(),  # dislikes item 0
                "tau_quit": 3.5,
                "top_genres": ["Action"],
            },
        ]
        self.env = MockLLMEnv(
            item_embeddings=self.item_embs,
            archetypes=self.archetypes,
            max_steps=10,
            seed=42,
        )

    def test_subclass_and_spaces(self):
        """Verifies environment inherits from RecEnv and defines standard spaces."""
        self.assertIsInstance(self.env, RecEnv)
        self.assertEqual(self.env.observation_space.shape, (202,))
        self.assertEqual(self.env.action_space.shape, (32,))

    def test_default_archetypes_fallback(self):
        """Verifies MockLLMEnv initializes cleanly without passing archetypes."""
        fallback_env = MockLLMEnv(item_embeddings=self.item_embs)
        obs, info = fallback_env.reset()
        self.assertEqual(obs.shape, (202,))
        self.assertIn("tau_quit", info)

    def test_item_embs_alias_and_kwargs(self):
        """Verifies MockLLMEnv accepts item_embs alias and extraneous kwargs."""
        alias_env = MockLLMEnv(
            item_embs=self.item_embs,
            archetypes=self.archetypes,
            dummy_kwarg=True,
            another_flag="ignored",
        )
        obs, info = alias_env.reset()
        self.assertEqual(obs.shape, (202,))
        # Error raised when neither provided
        with self.assertRaises(ValueError):
            MockLLMEnv()

    def test_choice_probabilities_precision(self):
        """Verifies that probabilities strictly sum to 1.0 without float rounding errors."""
        env = MockLLMEnv(item_embeddings=self.item_embs, seed=123)
        env.reset()
        # Step through 50 recommendations with varying actions
        for _ in range(50):
            act = np.random.uniform(-1, 1, 32).astype(np.float32)
            _, _, term, trunc, info = env.step(act)
            logits, probs, _ = env.compute_rating_logits(self.item_embs[0])
            self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)
            if term or trunc:
                env.reset()

    def test_reset_options(self):
        """Verifies resetting with archetype options selects correct persona."""
        obs0, info0 = self.env.reset(options={"archetype_id": 0})
        self.assertEqual(info0["archetype_id"], 0)
        self.assertEqual(info0["user_id"], 101)
        # Static u_emb in observation matches archetype
        np.testing.assert_allclose(obs0[:32], self.archetypes[0]["u_emb"], atol=1e-5)

        obs1, info1 = self.env.reset(options={"archetype_id": 1})
        self.assertEqual(info1["archetype_id"], 1)
        self.assertEqual(info1["user_id"], 202)

    def test_step_contract_and_5_logits(self):
        """Verifies step outputs and 5 rating token logits in info['logits']."""
        obs, info = self.env.reset()
        action = np.ones(32, dtype=np.float32)

        next_obs, rew, term, trunc, step_info = self.env.step(action)

        self.assertEqual(next_obs.shape, (202,))
        self.assertIsInstance(rew, float)
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)
        self.assertIsInstance(step_info, dict)

        # Check logits length and rating
        self.assertIn("logits", step_info)
        logits = step_info["logits"]
        self.assertEqual(len(logits), 5)
        self.assertIn(step_info["rating"], [1, 2, 3, 4, 5])
        self.assertIn("expected_rating", step_info)
        self.assertGreaterEqual(step_info["expected_rating"], 1.0)
        self.assertLessEqual(step_info["expected_rating"], 5.0)

    def test_mips_no_repeated_items(self):
        """Verifies that items consumed are not re-recommended in subsequent steps."""
        self.env.reset(seed=123)
        consumed = []
        # Fixed action vector pointing towards item 0
        fixed_act = self.item_embs[0].copy()

        for _ in range(5):
            _, _, term, trunc, info = self.env.step(fixed_act)
            rec_item = info["recommended_item"]
            self.assertNotIn(rec_item, consumed)
            consumed.append(rec_item)
            if term or trunc:
                break

    def test_step_truncation_at_horizon(self):
        """Verifies truncation occurs when current_step reaches max_steps."""
        short_env = MockLLMEnv(
            item_embeddings=self.item_embs,
            archetypes=self.archetypes,
            max_steps=4,
            alpha_retention=0.0,
            seed=42,
        )
        short_env.reset()
        for step_idx in range(4):
            _, _, term, trunc, _ = short_env.step(np.random.randn(32).astype(np.float32))
            if step_idx < 3:
                self.assertFalse(trunc)
            else:
                self.assertTrue(trunc)

    def test_full_episode_rollout(self):
        """Verifies a full rollout runs to termination or truncation without errors."""
        obs, _ = self.env.reset(seed=999)
        done = False
        steps = 0
        while not done:
            act = np.random.uniform(-1, 1, 32).astype(np.float32)
            obs, rew, term, trunc, info = self.env.step(act)
            done = term or trunc
            steps += 1
            self.assertFalse(np.isnan(rew))
            self.assertFalse(np.any(np.isnan(obs)))

        self.assertGreater(steps, 0)
        self.assertLessEqual(steps, self.env.max_steps)


class TestSlurmSuiteAndBouchetPayload(unittest.TestCase):
    """Test suite for Slurm scripts and Bouchet HPC payload preparation."""

    def test_slurm_vllm_script(self):
        """Verifies slurm/01_launch_vllm.sh syntax, directives, and commands."""
        script_path = project_root / "slurm" / "01_launch_vllm.sh"
        self.assertTrue(script_path.exists(), f"Missing {script_path}")
        content = script_path.read_text(encoding="utf-8")

        self.assertTrue(content.startswith("#!/bin/bash"))
        self.assertIn("#SBATCH --partition=gpu", content)
        self.assertIn("#SBATCH --gpus=1", content)
        self.assertIn("#SBATCH --cpus-per-task=8", content)
        self.assertIn("#SBATCH --mem=32G", content)
        self.assertIn("#SBATCH --time=04:00:00", content)
        self.assertIn("vllm serve", content)
        self.assertIn("--enable-prefix-caching", content)
        self.assertIn("--gpu-memory-utilization 0.90", content)
        # Multi-node host recording
        self.assertIn("hostname -s > artifacts/vllm_host.txt", content)

    def test_slurm_ablation_script(self):
        """Verifies slurm/02_run_fidelity_ablation.sh syntax and directives."""
        script_path = project_root / "slurm" / "02_run_fidelity_ablation.sh"
        self.assertTrue(script_path.exists(), f"Missing {script_path}")
        content = script_path.read_text(encoding="utf-8")

        self.assertTrue(content.startswith("#!/bin/bash"))
        self.assertIn("#SBATCH --partition=gpu", content)
        self.assertIn("#SBATCH --cpus-per-task", content)
        self.assertIn("#SBATCH --mem", content)
        self.assertIn("#SBATCH --time", content)
        self.assertIn("fidelity_ablation_results.json", content)
        self.assertIn("psi_u", content)
        self.assertIn("psi_generic", content)
        # Multi-node hostname resolution and --vllm-url flag
        self.assertIn("vllm_host.txt", content)
        self.assertIn("--vllm-url", content)

    def test_prep_bouchet_payload_script(self):
        """Verifies scripts/06_prep_bouchet_payload.py executes and passes verification."""
        prep_script = project_root / "scripts" / "06_prep_bouchet_payload.py"
        self.assertTrue(prep_script.exists())

        mod = importlib.import_module("scripts.06_prep_bouchet_payload")
        artifacts_ok, _ = mod.verify_artifact_payload(project_root / "artifacts")
        slurm_ok, _ = mod.verify_slurm_scripts(project_root / "slurm")

        self.assertTrue(artifacts_ok, "Artifact payload verification failed")
        self.assertTrue(slurm_ok, "Slurm scripts verification failed")


class TestMilestoneVisualizations(unittest.TestCase):
    """Test suite for milestone figures and summary LaTeX table."""

    def test_ndcg_at_k_calculation(self):
        """Verifies compute_ndcg_at_k rank discount calculation."""
        true_r = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        pred_perfect = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        self.assertAlmostEqual(compute_ndcg_at_k(true_r, pred_perfect, k=5), 1.0, places=5)

        pred_reverse = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ndcg_rev = compute_ndcg_at_k(true_r, pred_reverse, k=5)
        self.assertLess(ndcg_rev, 1.0)
        self.assertGreater(ndcg_rev, 0.0)

    def test_vllm_query_offline_fallback(self):
        """Verifies query_vllm_rating returns None when server is unreachable."""
        res = query_vllm_rating(
            vllm_url="http://127.0.0.1:9999/v1",  # Unused port
            prompt="Test prompt",
            model_name="test-model",
            timeout=0.2,
        )
        self.assertIsNone(res)

    def test_format_candidate_item_string(self):
        """Verifies structured semantic prompt formatting for movie items."""
        meta = {"title": "Toy Story", "year": "1995", "genres": ["Animation", "Comedy"]}
        item_str = format_candidate_item_string(1, {"1": meta})
        self.assertIn("Title: Toy Story", item_str)
        self.assertIn("Year: 1995", item_str)
        self.assertIn("Genres: Animation, Comedy", item_str)

    def test_fidelity_ablation_results_json(self):
        """Verifies artifacts/fidelity_ablation_results.json structure and metrics."""
        ablation_file = project_root / "artifacts" / "fidelity_ablation_results.json"
        self.assertTrue(ablation_file.exists())

        with open(ablation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in ["full_persona", "generic_persona", "unconditioned"]:
            self.assertIn(key, data)
            m = data[key]
            self.assertIn("rmse", m)
            self.assertIn("ndcg_at_10", m)
            self.assertIn("d_js", m)
            self.assertIn("w1", m)

    def test_milestone1_figures_exist_and_non_empty(self):
        """Verifies all milestone figure files exist, are valid images, and exceed 10 KB."""
        artifacts_dir = project_root / "artifacts"
        figures = [
            "milestone1_fig1_archetypes.png",
            "milestone1_fig2_ablations.png",
            "milestone1_fig3_sac_recsim.png",
        ]
        for fig_name in figures:
            fig_path = artifacts_dir / fig_name
            self.assertTrue(fig_path.exists(), f"Figure {fig_name} missing")
            size_kb = fig_path.stat().st_size / 1024
            self.assertGreater(size_kb, 10.0, f"Figure {fig_name} too small ({size_kb:.1f} KB)")

            # Check PNG magic header bytes
            with open(fig_path, "rb") as f:
                header = f.read(8)
            self.assertEqual(header[:4], b"\x89PNG")

    def test_milestone1_table1_latex(self):
        """Verifies artifacts/milestone1_table1.tex format and content."""
        tex_path = project_root / "artifacts" / "milestone1_table1.tex"
        self.assertTrue(tex_path.exists(), f"Table file missing at {tex_path}")

        content = tex_path.read_text(encoding="utf-8")
        self.assertIn(r"\begin{table}", content)
        self.assertIn(r"\end{table}", content)
        self.assertIn("SNIPS", content)
        self.assertIn("Doubly Robust", content)
        self.assertIn("KuaiRec", content)
        self.assertIn("Random Exploration", content)
        self.assertIn("RecSim SAC", content)
        self.assertIn("Offline CQL", content)


if __name__ == "__main__":
    unittest.main()
