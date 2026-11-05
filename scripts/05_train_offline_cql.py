#!/usr/bin/env python3
"""Script 05: Offline Conservative Q-Learning (CQL) & OPE Evaluation.

Bucket 5 of the Pre-Bouchet Pipeline.
Consumes precomputed item embeddings, user embeddings, and MovieLens D_init / D_eval.
Trains CQL offline baseline agent for 20,000 steps, serializes checkpoint to
artifacts/cql_baseline.pt, computes OPE estimators (SNIPS & Doubly Robust),
and evaluates zero-shot counterfactual performance on the KuaiRec matrix.
"""

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Prevent OpenMP runtime collision between MKL and PyTorch on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add project root to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.ope import (
    compute_snips_score,
    evaluate_doubly_robust,
    evaluate_kuairec_zeroshot,
)
from src.models.cql import CQLAgent
from src.models.sac import SACAgent

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("05_train_offline_cql")


class OfflineDataset:
    """Vectorized in-memory dataset storing transitions (o_t, a_t, r_t, o_{t+1}, d_t, a_{t+1})."""

    def __init__(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
        next_act: np.ndarray,
    ) -> None:
        self.obs = torch.as_tensor(obs, dtype=torch.float32)
        self.act = torch.as_tensor(act, dtype=torch.float32)
        self.rew = torch.as_tensor(rew, dtype=torch.float32)
        self.next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
        self.done = torch.as_tensor(done, dtype=torch.float32)
        self.next_act = torch.as_tensor(next_act, dtype=torch.float32)
        self.size = len(obs)

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idxs = torch.randint(0, self.size, (batch_size,))
        return (
            self.obs[idxs],
            self.act[idxs],
            self.rew[idxs],
            self.next_obs[idxs],
            self.done[idxs],
            self.next_act[idxs],
        )

    def __len__(self) -> int:
        return self.size


def build_offline_dataset(
    train_interactions: Dict[int, List[Tuple]],
    user_ids: List[int],
    item_ids: List[int],
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
) -> OfflineDataset:
    """Constructs offline transition buffers from chronological training interactions.

    Contract:
        o_t in R^202: static u_emb + past 5 interaction slots (e_hat_i, r, Delta t_k)
        with Delta t_k = (k+1)/5.0, zero-padded at t=0.
        Action a_t = e_hat_{i_t} in R^32.
        Reward R_t = clip((r_t - 3.0) / 2.0 + 0.25 * I(c_t == 1), -1.0, 1.25).
    """
    u_map = {uid: idx for idx, uid in enumerate(user_ids)}
    i_map = {iid: idx for idx, iid in enumerate(item_ids)}

    total_transitions = sum(len(inters) for inters in train_interactions.values())
    logger.info(f"Allocating dataset buffers for {total_transitions:,} transitions...")

    obs_buf = np.zeros((total_transitions, 202), dtype=np.float32)
    act_buf = np.zeros((total_transitions, 32), dtype=np.float32)
    rew_buf = np.zeros((total_transitions, 1), dtype=np.float32)
    next_obs_buf = np.zeros((total_transitions, 202), dtype=np.float32)
    done_buf = np.zeros((total_transitions, 1), dtype=np.float32)
    next_act_buf = np.zeros((total_transitions, 32), dtype=np.float32)

    ptr = 0
    t0 = time.time()

    for uid, inters in train_interactions.items():
        if uid not in u_map or not inters:
            continue
        u_idx = u_map[uid]
        u_vec = user_embeddings[u_idx]

        # Rolling FIFO interaction history of (e_k, r_k)
        history: List[Tuple[np.ndarray, float]] = []
        n_inter = len(inters)

        for t, (iid, r_val, _) in enumerate(inters):
            if iid not in i_map:
                continue
            item_idx = i_map[iid]
            e_it = item_embeddings[item_idx]

            # 1. Current Observation o_t in R^202
            cur_obs = np.zeros(202, dtype=np.float32)
            cur_obs[:32] = u_vec
            for k in range(min(len(history), 5)):
                e_k, r_k = history[k]
                lag = float((k + 1) / 5.0)
                slot = 32 + k * 34
                cur_obs[slot : slot + 32] = e_k
                cur_obs[slot + 32] = float(r_k)
                cur_obs[slot + 33] = lag

            # 2. Action a_t in R^32
            cur_act = e_it

            # 3. Continuation flag c_t and Reward R_t
            c_t = 1 if (t < n_inter - 1) else 0
            reward = float((r_val - 3.0) / 2.0 + 0.25 * float(c_t))
            reward = float(np.clip(reward, -1.0, 1.25))
            done = 1.0 - float(c_t)

            # 4. Update history with current interaction
            history.insert(0, (e_it, float(r_val)))
            if len(history) > 5:
                history.pop()

            # 5. Next Observation o_{t+1} in R^202
            next_obs = np.zeros(202, dtype=np.float32)
            next_obs[:32] = u_vec
            for k in range(min(len(history), 5)):
                e_k, r_k = history[k]
                lag = float((k + 1) / 5.0)
                slot = 32 + k * 34
                next_obs[slot : slot + 32] = e_k
                next_obs[slot + 32] = float(r_k)
                next_obs[slot + 33] = lag

            # 6. Next Action a_{t+1}
            if t < n_inter - 1:
                next_iid = inters[t + 1][0]
                if next_iid in i_map:
                    next_act = item_embeddings[i_map[next_iid]]
                else:
                    next_act = np.zeros(32, dtype=np.float32)
            else:
                next_act = np.zeros(32, dtype=np.float32)

            obs_buf[ptr] = cur_obs
            act_buf[ptr] = cur_act
            rew_buf[ptr, 0] = reward
            next_obs_buf[ptr] = next_obs
            done_buf[ptr, 0] = done
            next_act_buf[ptr] = next_act

            ptr += 1

    # Slice buffers to actual filled count
    obs_buf = obs_buf[:ptr]
    act_buf = act_buf[:ptr]
    rew_buf = rew_buf[:ptr]
    next_obs_buf = next_obs_buf[:ptr]
    done_buf = done_buf[:ptr]
    next_act_buf = next_act_buf[:ptr]

    duration = time.time() - t0
    logger.info(f"Built offline dataset with {ptr:,} transitions in {duration:.2f}s.")
    return OfflineDataset(obs_buf, act_buf, rew_buf, next_obs_buf, done_buf, next_act_buf)


def build_eval_trajectories(
    eval_interactions: Dict[int, List[Tuple]],
    user_ids: List[int],
    item_ids: List[int],
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    max_users: int = 500,
    seed: int = 42,
) -> List[List[Dict[str, Any]]]:
    """Constructs evaluation trajectories from chronological D_eval interactions."""
    u_map = {uid: idx for idx, uid in enumerate(user_ids)}
    i_map = {iid: idx for idx, iid in enumerate(item_ids)}

    rng = np.random.default_rng(seed)
    eval_uids = [uid for uid, inters in eval_interactions.items() if inters and uid in u_map]
    if len(eval_uids) > max_users:
        eval_uids = list(rng.choice(eval_uids, size=max_users, replace=False))

    trajectories: List[List[Dict[str, Any]]] = []

    for uid in eval_uids:
        inters = eval_interactions[uid]
        u_idx = u_map[uid]
        u_vec = user_embeddings[u_idx]

        history: List[Tuple[np.ndarray, float]] = []
        traj: List[Dict[str, Any]] = []
        n_inter = len(inters)

        for t, (iid, r_val, _) in enumerate(inters):
            if iid not in i_map:
                continue
            item_idx = i_map[iid]
            e_it = item_embeddings[item_idx]

            obs = np.zeros(202, dtype=np.float32)
            obs[:32] = u_vec
            for k in range(min(len(history), 5)):
                e_k, r_k = history[k]
                lag = float((k + 1) / 5.0)
                slot = 32 + k * 34
                obs[slot : slot + 32] = e_k
                obs[slot + 32] = float(r_k)
                obs[slot + 33] = lag

            c_t = 1 if (t < n_inter - 1) else 0
            reward = float((r_val - 3.0) / 2.0 + 0.25 * float(c_t))
            reward = float(np.clip(reward, -1.0, 1.25))

            traj.append({
                "obs": obs,
                "action": e_it,
                "reward": reward,
                "item_idx": item_idx,
                "rating": float(r_val),
            })

            history.insert(0, (e_it, float(r_val)))
            if len(history) > 5:
                history.pop()

        if traj:
            trajectories.append(traj)

    return trajectories


def compute_policy_propensities(
    trajectories: List[List[Dict[str, Any]]],
    policy_fn: Callable[[np.ndarray], np.ndarray],
    item_embeddings: np.ndarray,
    tau_pi: float = 0.1,
    num_subsample_items: int = 500,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Computes importance weights rho_{m, t} = pi_e(a_{m, t} | o_{m, t}) / pi_b(a_{m, t} | o_{m, t}).

    Uses Boltzmann softmax distribution over catalog embeddings.
    """
    embs = item_embeddings
    num_items = embs.shape[0]
    pi_b_val = 1.0 / float(num_items)

    traj_rewards: List[List[float]] = []
    traj_weights: List[List[float]] = []

    for traj in trajectories:
        r_list: List[float] = []
        w_list: List[float] = []

        for step in traj:
            obs = step["obs"]
            target_idx = step["item_idx"]
            reward = step["reward"]

            # Policy continuous action vector
            action = policy_fn(obs)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(action)) + 1e-8
            a_norm = action / norm

            # Boltzmann distribution: logits = embs @ a_norm / tau_pi
            # Fast vectorized dot product
            logits = (embs @ a_norm) / tau_pi
            max_logit = float(np.max(logits))
            exp_logits = np.exp(logits - max_logit)
            sum_exp = float(np.sum(exp_logits))
            pi_e_val = float(exp_logits[target_idx] / max(1e-12, sum_exp))

            # Importance weight rho = pi_e / pi_b
            rho = float(pi_e_val / max(1e-12, pi_b_val))
            rho = min(max(rho, 0.0), 20.0)

            r_list.append(reward)
            w_list.append(rho)

        traj_rewards.append(r_list)
        traj_weights.append(w_list)

    return traj_rewards, traj_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Pure PyTorch CQL Agent on Offline MovieLens Logs (Bucket 5)."
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
        default=20000,
        help="Total CQL gradient update steps (default: 20000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for CQL updates (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="AdamW learning rate for Q-network (default: 3e-4)",
    )
    parser.add_argument(
        "--alpha-cql",
        type=float,
        default=1.0,
        help="Conservative regularizer weight (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor gamma (default: 0.99)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Polyak target update weight tau (default: 0.005)",
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
        default="artifacts/cql_baseline.pt",
        help="File path to save the trained CQL checkpoint (default: artifacts/cql_baseline.pt)",
    )

    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    save_path = Path(args.save_path)

    logger.info("=== Starting Bucket 5: Offline Conservative Q-Learning (CQL) ===")
    t_start = time.time()

    # Seed random generators
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load artifacts
    splits_path = artifacts_dir / "data_splits.pkl"
    item_emb_path = artifacts_dir / "item_embeddings_d32.npy"
    user_emb_path = artifacts_dir / "user_embeddings_d32.npy"
    kuairec_mat_path = artifacts_dir / "kuairec_matrix.npy"
    kuairec_emb_path = artifacts_dir / "kuairec_embeddings_d32.npy"

    for p in [splits_path, item_emb_path, user_emb_path, kuairec_mat_path, kuairec_emb_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    logger.info(f"Loading data splits from {splits_path}...")
    with open(splits_path, "rb") as f:
        data_splits = pickle.load(f)

    item_embeddings = np.load(item_emb_path)
    user_embeddings = np.load(user_emb_path)
    kuairec_matrix = np.load(kuairec_mat_path)
    kuairec_embeddings = np.load(kuairec_emb_path)

    logger.info(
        f"Loaded embeddings: MovieLens items={item_embeddings.shape}, "
        f"users={user_embeddings.shape}; KuaiRec matrix={kuairec_matrix.shape}, "
        f"KuaiRec items={kuairec_embeddings.shape}"
    )

    # 2. Build offline transition dataset
    dataset = build_offline_dataset(
        train_interactions=data_splits["train_interactions"],
        user_ids=data_splits["user_ids"],
        item_ids=data_splits["item_ids"],
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
    )

    # 3. Initialize CQL Agent
    agent = CQLAgent(
        obs_dim=202,
        action_dim=32,
        hidden_dim=256,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha_cql=args.alpha_cql,
        num_random_actions=10,
    )
    logger.info(
        f"Initialized CQLAgent (obs_dim=202, act_dim=32, hidden_dim=256, "
        f"lr={args.lr}, gamma={args.gamma}, alpha_cql={args.alpha_cql})"
    )

    # 4. Training Loop
    total_steps = args.steps
    batch_size = args.batch_size
    log_interval = max(500, total_steps // 20)
    t_train_start = time.time()

    logger.info(f"Starting CQL training loop: steps={total_steps:,}, batch_size={batch_size}...")

    loss_history: List[float] = []
    td_history: List[float] = []
    cql_history: List[float] = []

    for step in range(1, total_steps + 1):
        batch = dataset.sample_batch(batch_size)
        metrics = agent.update(batch)

        loss_history.append(metrics["loss"])
        td_history.append(metrics["td_loss"])
        cql_history.append(metrics["cql_loss"])

        if step % log_interval == 0 or step == total_steps:
            elapsed = time.time() - t_train_start
            steps_per_sec = step / max(1e-5, elapsed)
            logger.info(
                f"Step {step:6d}/{total_steps} | "
                f"Loss: {metrics['loss']:6.3f} | "
                f"TD Loss: {metrics['td_loss']:6.3f} | "
                f"CQL Loss: {metrics['cql_loss']:6.3f} | "
                f"Q(data): {metrics['q_data_mean']:5.2f} | "
                f"Q(rand): {metrics['q_rand_mean']:5.2f} | "
                f"Speed: {steps_per_sec:5.1f} steps/s"
            )

    train_duration = time.time() - t_train_start
    logger.info(f"Completed {total_steps:,} CQL updates in {train_duration:.2f}s ({total_steps / train_duration:.1f} steps/s).")

    # 5. Save Model Checkpoint
    logger.info(f"Saving CQL model weights to {save_path}...")
    agent.save(save_path)
    if not save_path.exists():
        raise RuntimeError(f"Failed to create model weights checkpoint at {save_path}")
    logger.info(f"Successfully verified checkpoint at {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")

    # 6. Off-Policy Evaluation (OPE) on Held-Out D_eval
    logger.info("Constructing evaluation trajectories from held-out D_eval...")
    eval_trajectories = build_eval_trajectories(
        eval_interactions=data_splits["eval_interactions"],
        user_ids=data_splits["user_ids"],
        item_ids=data_splits["item_ids"],
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        max_users=300,
        seed=args.seed,
    )
    logger.info(f"Constructed {len(eval_trajectories)} evaluation trajectories.")

    # Define candidate policies for comparison on MovieLens catalog
    def cql_policy_fn(obs: np.ndarray) -> np.ndarray:
        u_vec = obs[:32]
        sims = item_embeddings @ u_vec
        top50_idx = np.argpartition(sims, -50)[-50:]
        cands = torch.as_tensor(item_embeddings[top50_idx])
        obs_rep = torch.as_tensor(obs).unsqueeze(0).repeat(len(top50_idx), 1)
        with torch.no_grad():
            q_vals = agent.get_q_values(obs_rep, cands)
        best_cand = top50_idx[int(torch.argmax(q_vals).item())]
        return item_embeddings[best_cand]

    def random_policy_fn(obs: np.ndarray) -> np.ndarray:
        idx = int(np.random.randint(0, len(item_embeddings)))
        return item_embeddings[idx]

    # Attempt to load SAC baseline for comparison
    sac_checkpoint_path = artifacts_dir / "sac_recsim_baseline.pt"
    sac_agent: Optional[SACAgent] = None
    if sac_checkpoint_path.exists():
        try:
            sac_agent = SACAgent(obs_dim=202, action_dim=32, hidden_dim=256)
            sac_agent.load(sac_checkpoint_path)
            logger.info(f"Loaded trained SAC baseline from {sac_checkpoint_path}.")
        except Exception as e:
            logger.warning(f"Could not load SAC baseline ({e}). Skipping SAC in comparison.")

    def sac_policy_fn(obs: np.ndarray) -> np.ndarray:
        if sac_agent is not None:
            act = sac_agent.sample_action(obs, evaluate=True, return_log_prob=False)
            norm = float(np.linalg.norm(act)) + 1e-8
            best_idx = int(np.argmax(item_embeddings @ (act / norm)))
            return item_embeddings[best_idx]
        return random_policy_fn(obs)

    # Compute SNIPS & Doubly Robust estimates
    logger.info("Computing OPE metrics (SNIPS & Doubly Robust)...")
    eval_policies = {
        "Random Baseline": random_policy_fn,
        "Offline CQL": cql_policy_fn,
    }
    if sac_agent is not None:
        eval_policies["RecSim SAC"] = sac_policy_fn

    ope_results: Dict[str, Dict[str, float]] = {}

    for name, p_fn in eval_policies.items():
        r_list, w_list = compute_policy_propensities(
            eval_trajectories, p_fn, item_embeddings, tau_pi=0.1
        )
        snips_val = compute_snips_score(r_list, w_list, gamma=args.gamma)

        # Inject importance weights into trajectories for DR evaluation
        dr_trajs: List[List[Dict[str, Any]]] = []
        for traj_idx, traj in enumerate(eval_trajectories):
            new_traj = []
            for step_idx, step in enumerate(traj):
                step_copy = dict(step)
                step_copy["importance_weight"] = w_list[traj_idx][step_idx]
                new_traj.append(step_copy)
            dr_trajs.append(new_traj)

        dr_val = evaluate_doubly_robust(
            eval_trajectories=dr_trajs,
            eval_policy_fn=p_fn,
            q_model=agent,
            gamma=args.gamma,
        )

        ope_results[name] = {
            "snips": snips_val,
            "dr": dr_val,
        }
        logger.info(f"{name:16s} | SNIPS: {snips_val:6.3f} | Doubly Robust: {dr_val:6.3f}")

    # 7. Zero-Shot Counterfactual Evaluation on KuaiRec Matrix
    logger.info("Evaluating zero-shot counterfactual performance on KuaiRec ground-truth matrix...")
    kuairec_results: Dict[str, Dict[str, float]] = {}

    # Define KuaiRec-grounded policies
    def cql_kuairec_policy(obs: np.ndarray) -> np.ndarray:
        u_vec = obs[:32]
        sims = kuairec_embeddings @ u_vec
        top50_idx = np.argpartition(sims, -50)[-50:]
        cands = torch.as_tensor(kuairec_embeddings[top50_idx])
        obs_rep = torch.as_tensor(obs).unsqueeze(0).repeat(len(top50_idx), 1)
        with torch.no_grad():
            q_vals = agent.get_q_values(obs_rep, cands)
        best_cand = top50_idx[int(torch.argmax(q_vals).item())]
        return kuairec_embeddings[best_cand]

    def random_kuairec_policy(obs: np.ndarray) -> np.ndarray:
        idx = int(np.random.randint(0, len(kuairec_embeddings)))
        return kuairec_embeddings[idx]

    kuairec_policies = {
        "Random Baseline": random_kuairec_policy,
        "Offline CQL": cql_kuairec_policy,
    }
    if sac_agent is not None:
        kuairec_policies["RecSim SAC"] = lambda o: sac_agent.sample_action(o, evaluate=True, return_log_prob=False)

    for name, p_fn in kuairec_policies.items():
        res = evaluate_kuairec_zeroshot(
            policy=p_fn,
            kuairec_embs=kuairec_embeddings,
            kuairec_matrix=kuairec_matrix,
            rating_threshold=3.0,
        )
        kuairec_results[name] = res
        logger.info(
            f"{name:16s} | KuaiRec Mean Rew: {res['mean_reward']:6.3f} | "
            f"Hit Rate: {res['hit_rate']:5.1%} | Mean Rating: {res['mean_rating']:5.2f}"
        )

    # 8. Print Summary Table
    print("\n" + "=" * 90)
    print("           PRE-BOUCHET REINFORCEMENT LEARNING BASELINES COMPARISON (MILESTONE #1)")
    print("=" * 90)
    print(
        f"{'Algorithm / Policy':<22} | "
        f"{'SNIPS (D_eval)':<16} | "
        f"{'Doubly Robust':<16} | "
        f"{'KuaiRec Rew':<14} | "
        f"{'KuaiRec Hit%':<12}"
    )
    print("-" * 90)

    order = ["Random Baseline", "RecSim SAC", "Offline CQL"]
    for policy_name in order:
        if policy_name in ope_results:
            snips_s = f"{ope_results[policy_name]['snips']:+.3f}"
            dr_s = f"{ope_results[policy_name]['dr']:+.3f}"
            k_rew_s = f"{kuairec_results[policy_name]['mean_reward']:+.3f}"
            k_hit_s = f"{kuairec_results[policy_name]['hit_rate'] * 100:5.1f}%"
            print(f"{policy_name:<22} | {snips_s:<16} | {dr_s:<16} | {k_rew_s:<14} | {k_hit_s:<12}")

    print("=" * 90)
    total_time = time.time() - t_start
    print(f"Bucket 5 execution finished in {total_time:.2f} seconds.\n")


if __name__ == "__main__":
    main()
