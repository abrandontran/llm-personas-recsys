#!/usr/bin/env python3
"""Script 04: Pure PyTorch Soft Actor-Critic (SAC) Training Loop on Parametric RecSim (M_stat).

Bucket 4 of the Pre-Bouchet Pipeline.
Consumes precomputed item embeddings, metadata, and user archetypes.
Trains SAC baseline agent for 50,000 steps, serializes weights, and plots training curves.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.archetypes import GENRE_TO_IDX
from src.envs.recsim_stat_env import ParametricRecSimEnv
from src.models.sac import SACAgent

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("04_train_recsim_sac")


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


def plot_training_curves(
    episode_rewards: List[float],
    episode_lengths: List[int],
    critic_losses: List[float],
    actor_losses: List[float],
    alphas: List[float],
    output_path: Path,
) -> None:
    """Renders and saves a 3-panel training curves figure.

    Panel 1: Episode Return (raw + rolling mean).
    Panel 2: Episode Length (session churn duration).
    Panel 3: Training Diagnostics (smoothed Critic/Actor losses and temperature alpha).

    Args:
        episode_rewards: History of total episodic returns.
        episode_lengths: History of episodic trajectory lengths.
        critic_losses: Critic loss values across training updates.
        actor_losses: Actor policy loss values across training updates.
        alphas: Entropy temperature alpha values across training updates.
        output_path: File path to save output PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: Episode Return ---
    axes[0].plot(episode_rewards, alpha=0.25, color="steelblue", label="Raw Return")
    if len(episode_rewards) >= 10:
        window = max(5, min(50, len(episode_rewards) // 10))
        rolling_ret = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(
            range(window - 1, len(episode_rewards)),
            rolling_ret,
            color="navy",
            linewidth=2.0,
            label=f"Rolling Mean (w={window})",
        )
    axes[0].set_title("Episode Return (Reward Formulation)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Episode", fontsize=11)
    axes[0].set_ylabel("Cumulative Return", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right")

    # --- Panel 2: Episode Length ---
    axes[1].plot(episode_lengths, alpha=0.25, color="coral", label="Raw Length")
    if len(episode_lengths) >= 10:
        window = max(5, min(50, len(episode_lengths) // 10))
        rolling_len = np.convolve(episode_lengths, np.ones(window) / window, mode="valid")
        axes[1].plot(
            range(window - 1, len(episode_lengths)),
            rolling_len,
            color="darkred",
            linewidth=2.0,
            label=f"Rolling Mean (w={window})",
        )
    axes[1].set_title("Session Length (Endogenous Churn)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Episode", fontsize=11)
    axes[1].set_ylabel("Interaction Steps", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right")

    # --- Panel 3: Losses & Entropy Temperature ---
    ax3 = axes[2]
    if critic_losses and actor_losses:
        w_loss = max(1, min(100, len(critic_losses) // 20))
        smoothed_critic = np.convolve(critic_losses, np.ones(w_loss) / w_loss, mode="valid")
        smoothed_actor = np.convolve(actor_losses, np.ones(w_loss) / w_loss, mode="valid")

        x_loss = range(w_loss - 1, len(critic_losses))
        ax3.plot(x_loss, smoothed_critic, color="crimson", label="Critic Loss", linewidth=1.5)
        ax3.plot(x_loss, smoothed_actor, color="forestgreen", label="Actor Loss", linewidth=1.5)
        ax3.set_xlabel("Training Step", fontsize=11)
        ax3.set_ylabel("Loss", fontsize=11)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Twin axis for alpha
        if alphas:
            ax3_twin = ax3.twinx()
            ax3_twin.plot(alphas, color="purple", linestyle="--", alpha=0.85, label=r"Temp $\alpha$")
            ax3_twin.set_ylabel(r"Temperature $\alpha$", fontsize=11, color="purple")
            ax3_twin.tick_params(axis="y", labelcolor="purple")

            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_twin.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        else:
            ax3.legend(loc="upper right")
    else:
        ax3.text(0.5, 0.5, "No Loss Data", ha="center", va="center")

    ax3.set_title("Losses & Entropy Temperature", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved training curves plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Pure PyTorch SAC Agent on Parametric RecSim (M_stat)."
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
        default=50000,
        help="Total environment interaction steps to train (default: 50000)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Warm-up exploration steps with uniform random actions (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for SAC updates (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Adam learning rate for actor, critic, and alpha (default: 3e-4)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for Actor and Critic MLPs (default: 256)",
    )
    parser.add_argument(
        "--update-freq",
        type=int,
        default=1,
        help="Network update frequency in environment steps (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="artifacts/sac_recsim_baseline.pt",
        help="File path to save the trained SAC model checkpoint (default: artifacts/sac_recsim_baseline.pt)",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="artifacts/sac_recsim_curve.png",
        help="File path to save training curves figure (default: artifacts/sac_recsim_curve.png)",
    )

    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    save_path = Path(args.save_path)
    plot_path = Path(args.plot_path)

    logger.info("=== Starting Bucket 4: Soft Actor-Critic (SAC) RecSim Training ===")
    t_start = time.time()

    # Seed all random number generators
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load artifacts
    item_emb_path = artifacts_dir / "item_embeddings_d32.npy"
    item_meta_path = artifacts_dir / "item_metadata.json"
    archetypes_path = artifacts_dir / "user_archetypes_k32.json"

    for p in [item_emb_path, item_meta_path, archetypes_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    logger.info(f"Loading item embeddings from {item_emb_path}...")
    item_embeddings = np.load(item_emb_path)
    logger.info(f"Loaded item embeddings: shape={item_embeddings.shape}")

    logger.info(f"Loading item metadata from {item_meta_path}...")
    with open(item_meta_path, "r", encoding="utf-8") as f:
        item_metadata = json.load(f)

    logger.info(f"Loading user archetypes from {archetypes_path}...")
    with open(archetypes_path, "r", encoding="utf-8") as f:
        archetypes = json.load(f)

    num_items = item_embeddings.shape[0]
    item_genres = build_genre_matrix(item_metadata, num_items)
    logger.info(f"Built genre matrix: shape={item_genres.shape}")

    # 2. Initialize environment and SAC agent
    env = ParametricRecSimEnv(
        item_embeddings=item_embeddings,
        item_genres=item_genres,
        archetypes=archetypes,
        max_steps=20,
        seed=args.seed,
    )
    logger.info(f"Initialized ParametricRecSimEnv (obs_dim={env.observation_space.shape[0]}, act_dim={env.action_space.shape[0]})")

    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=0.99,
        tau=0.005,
        target_entropy=-32.0,
        buffer_capacity=100_000,
    )
    logger.info("Initialized SACAgent with LayerNorm MLPs and automatic entropy tuning.")

    # 3. Training Loop
    total_steps = args.steps
    warmup_steps = args.warmup_steps
    batch_size = args.batch_size
    update_freq = args.update_freq

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    critic_losses: List[float] = []
    actor_losses: List[float] = []
    alphas: List[float] = []

    obs, _ = env.reset(seed=args.seed)
    ep_reward = 0.0
    ep_len = 0
    ep_count = 0

    log_interval = max(500, total_steps // 20)
    t_loop_start = time.time()

    logger.info(f"Starting training loop: steps={total_steps}, warmup={warmup_steps}, update_freq={update_freq}...")

    for step in range(1, total_steps + 1):
        if step <= warmup_steps:
            # Exploration via uniform continuous action sampling
            action = env.action_space.sample().astype(np.float32)
        else:
            # Policy action sampling
            action = agent.sample_action(obs, evaluate=False, return_log_prob=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store transition in circular replay buffer
        agent.replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        ep_reward += reward
        ep_len += 1

        # SAC network updates after warm-up
        if step > warmup_steps and (step % update_freq == 0):
            losses = agent.update(batch_size=batch_size)
            if losses:
                critic_losses.append(losses["critic_loss"])
                actor_losses.append(losses["actor_loss"])
                alphas.append(losses["alpha"])

        # Episode termination / truncation
        if done:
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            ep_count += 1

            obs, _ = env.reset()
            ep_reward = 0.0
            ep_len = 0

        # Periodic logging
        if step % log_interval == 0 or step == total_steps:
            elapsed = time.time() - t_loop_start
            steps_per_sec = step / max(1e-5, elapsed)
            mean_ret = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
            mean_len = np.mean(episode_lengths[-20:]) if episode_lengths else 0.0
            cur_alpha = alphas[-1] if alphas else agent.alpha
            c_loss = critic_losses[-1] if critic_losses else 0.0
            a_loss = actor_losses[-1] if actor_losses else 0.0

            logger.info(
                f"Step {step:6d}/{total_steps} | "
                f"Episodes: {ep_count:4d} | "
                f"Mean Ret: {mean_ret:6.2f} | "
                f"Mean Len: {mean_len:4.1f} | "
                f"Critic Loss: {c_loss:6.3f} | "
                f"Actor Loss: {a_loss:6.3f} | "
                f"Alpha: {cur_alpha:5.3f} | "
                f"Speed: {steps_per_sec:5.1f} steps/s"
            )

    total_duration = time.time() - t_start
    logger.info(
        f"Completed {total_steps} interaction steps in {total_duration:.2f}s "
        f"({total_steps / total_duration:.1f} steps/s across {ep_count} episodes)."
    )

    # 4. Save model weights
    logger.info(f"Saving SAC model weights to {save_path}...")
    agent.save(save_path)
    if not save_path.exists():
        raise RuntimeError(f"Failed to create model weights checkpoint at {save_path}")
    logger.info(f"Successfully verified checkpoint at {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")

    # 5. Generate training curves plot
    logger.info(f"Rendering training curves to {plot_path}...")
    plot_training_curves(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        critic_losses=critic_losses,
        actor_losses=actor_losses,
        alphas=alphas,
        output_path=plot_path,
    )
    if not plot_path.exists():
        raise RuntimeError(f"Failed to create training curve figure at {plot_path}")
    logger.info(f"Successfully verified training curve at {plot_path} ({plot_path.stat().st_size / 1024:.1f} KB)")

    logger.info("=== Bucket 4: Soft Actor-Critic Training Completed Successfully ===")


if __name__ == "__main__":
    main()
