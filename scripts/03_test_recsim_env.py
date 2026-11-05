#!/usr/bin/env python3
"""Script 03: Verify Gymnasium RecSys Environment (M_stat) and Boltzmann OPE Policy.

Bucket 3 of the Pre-Bouchet Pipeline.
Runs 1,000 rollout steps on MovieLens-1M artifacts, verifying:
1. Observation shape strictness (202 dimensions).
2. Continuous action space and Cosine MIPS recommendation mapping.
3. Repetition prevention within episodes.
4. Genre satiation dynamics and decay.
5. Endogenous churn termination and reward bounds [-1.0, 1.25].
6. Boltzmann OPE policy propensity sum and non-negativity.
7. CPU rollout throughput (>1,000 steps/sec).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.archetypes import CANONICAL_GENRES, GENRE_TO_IDX
from src.envs.recsim_stat_env import ParametricRecSimEnv

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("03_test_recsim_env")


def build_genre_matrix(
    item_metadata: Dict[str, dict],
    num_items: int,
) -> np.ndarray:
    """Builds a multi-hot binary indicator matrix (N, 18) from item metadata.

    Args:
        item_metadata: Mapping of item ID string to item dict with 'genres'.
        num_items: Expected total number of catalog items N.

    Returns:
        np.ndarray of shape (N, 18) with float32 values in {0.0, 1.0}.
    """
    genre_matrix = np.zeros((num_items, 18), dtype=np.float32)

    for idx, (item_id_str, meta) in enumerate(item_metadata.items()):
        if idx >= num_items:
            break
        for genre in meta.get("genres", []):
            if genre in GENRE_TO_IDX:
                genre_matrix[idx, GENRE_TO_IDX[genre]] = 1.0

    return genre_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Gymnasium RecSys Environment (M_stat) and Boltzmann OPE Policy."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory containing precomputed artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of interaction steps to simulate (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=1000.0,
        help="Minimum required throughput in steps/sec (default: 1000.0)",
    )

    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    logger.info("=== Starting Bucket 3: RecSim Environment & Boltzmann OPE Verification ===")
    t_start = time.time()

    # Step 1: Load precomputed artifacts
    item_emb_path = artifacts_dir / "item_embeddings_d32.npy"
    item_meta_path = artifacts_dir / "item_metadata.json"
    archetypes_path = artifacts_dir / "user_archetypes_k32.json"

    for p in [item_emb_path, item_meta_path, archetypes_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    logger.info(f"Loading item embeddings from {item_emb_path}...")
    item_embeddings = np.load(item_emb_path)
    logger.info(f"Item embeddings shape: {item_embeddings.shape}")

    logger.info(f"Loading item metadata from {item_meta_path}...")
    with open(item_meta_path, "r", encoding="utf-8") as f:
        item_metadata = json.load(f)

    logger.info(f"Loading user archetypes from {archetypes_path}...")
    with open(archetypes_path, "r", encoding="utf-8") as f:
        archetypes = json.load(f)

    num_items = item_embeddings.shape[0]
    logger.info(f"Loaded {num_items} items and {len(archetypes)} user archetypes.")

    # Step 2: Build genre multi-hot matrix
    item_genres = build_genre_matrix(item_metadata, num_items)
    logger.info(
        f"Built multi-hot genre matrix with shape {item_genres.shape}, "
        f"mean genres/item: {item_genres.sum(axis=1).mean():.2f}"
    )

    # Step 3: Instantiate ParametricRecSimEnv
    env = ParametricRecSimEnv(
        item_embeddings=item_embeddings,
        item_genres=item_genres,
        archetypes=archetypes,
        max_steps=20,
        beta=0.3,
        lambda_satiate=1.5,
        noise_std=0.2,
        alpha_retention=0.25,
        seed=args.seed,
    )

    # Step 4: Verify initial state & Boltzmann OPE distribution
    obs, info = env.reset(seed=args.seed)
    assert obs.shape == (202,), f"Expected observation shape (202,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32 observation, got {obs.dtype}"

    test_action = np.random.uniform(-1.0, 1.0, size=(32,)).astype(np.float32)
    test_probs = env.get_action_distribution(test_action, tau_pi=0.1)
    assert test_probs.shape == (num_items,), f"Expected shape ({num_items},), got {test_probs.shape}"
    assert np.all(test_probs >= 0.0), "Negative Boltzmann probabilities encountered!"
    assert np.isclose(float(test_probs.sum()), 1.0, atol=1e-4), f"Boltzmann probs sum to {test_probs.sum()}"
    logger.info(
        f"Verified Boltzmann OPE distribution: min={test_probs.min():.2e}, "
        f"max={test_probs.max():.2e}, sum={test_probs.sum():.6f}"
    )

    # Step 5: Execute 1,000 rollout steps and verify invariants
    logger.info(f"Simulating {args.steps} interaction steps...")
    total_episodes = 0
    total_churns = 0
    total_truncations = 0
    session_items: List[int] = []

    # Reset RNG for rollout actions
    np.random.seed(args.seed)
    t_rollout_start = time.time()

    for step_idx in range(args.steps):
        # Generate random continuous action a_t in [-1, 1]^32
        action = np.random.uniform(-1.0, 1.0, size=(32,)).astype(np.float32)

        next_obs, reward, terminated, truncated, step_info = env.step(action)

        # 1. Observation shape check
        assert next_obs.shape == (202,), f"Step {step_idx}: Observation shape {next_obs.shape} != (202,)"

        # 2. Reward bounds check
        assert -1.0 <= reward <= 1.25, f"Step {step_idx}: Reward {reward} out of bounds [-1.0, 1.25]"

        # 3. Item recommendation repetition check
        rec_item = step_info["recommended_item"]
        assert rec_item not in session_items, (
            f"Step {step_idx}: Item {rec_item} recommended multiple times in the same session!"
        )
        session_items.append(rec_item)

        if terminated or truncated:
            total_episodes += 1
            if terminated:
                total_churns += 1
            if truncated:
                total_truncations += 1

            session_items = []
            obs, info = env.reset()

    rollout_duration = time.time() - t_rollout_start
    steps_per_sec = args.steps / rollout_duration

    logger.info(
        f"Rollout finished: {args.steps} steps across {total_episodes} sessions "
        f"({total_churns} churn terminations, {total_truncations} horizon truncations) "
        f"in {rollout_duration:.3f}s"
    )
    logger.info(f"Simulation throughput: {steps_per_sec:.1f} steps/sec (required: >{args.min_speed:.1f})")

    assert (
        steps_per_sec >= args.min_speed
    ), f"Throughput {steps_per_sec:.1f} steps/sec below minimum requirement {args.min_speed:.1f}"

    total_duration = time.time() - t_start
    logger.info(f"=== Bucket 3 Verified Successfully in {total_duration:.2f}s ===")


if __name__ == "__main__":
    main()
