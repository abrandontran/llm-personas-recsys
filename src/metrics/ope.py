"""Off-Policy Evaluation (OPE) estimators and counterfactual benchmarks (Bucket 5).

Includes:
- compute_snips_score: Self-Normalized Importance Sampling (SNIPS) estimator.
- evaluate_doubly_robust: Doubly Robust (DR) off-policy policy evaluator.
- evaluate_kuairec_zeroshot: Zero-shot counterfactual evaluation on KuaiRec ground-truth matrix.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch


def compute_snips_score(
    trajectories_rewards: Sequence[Sequence[float]],
    importance_weights: Sequence[Sequence[float]],
    gamma: float = 0.99,
    clip_weight: Optional[float] = 20.0,
) -> float:
    """Calculates Self-Normalized Importance Sampling (SNIPS) policy value score.

    Formula:
        V_SNIPS = (sum_m sum_t gamma^t * w_{m, t} * r_{m, t}) / (sum_m sum_t gamma^t * w_{m, t})

    Args:
        trajectories_rewards: Nested sequences of rewards [m][t] across M trajectories.
        importance_weights: Nested sequences of per-step importance weights [m][t].
        gamma: Discount factor in [0.0, 1.0] (default: 0.99).
        clip_weight: Maximum allowed importance weight to prevent high variance (default: 20.0).

    Returns:
        float scalar SNIPS estimate.
    """
    numerator = 0.0
    denominator = 0.0

    for m in range(len(trajectories_rewards)):
        r_seq = trajectories_rewards[m]
        w_seq = importance_weights[m]
        t_len = min(len(r_seq), len(w_seq))

        for t in range(t_len):
            w = float(w_seq[t])
            if clip_weight is not None and clip_weight > 0.0:
                w = min(max(w, 0.0), float(clip_weight))

            discount = float(gamma**t)
            weighted_discount = discount * w

            numerator += weighted_discount * float(r_seq[t])
            denominator += weighted_discount

    if denominator <= 0.0:
        return 0.0

    return float(numerator / denominator)


def evaluate_doubly_robust(
    eval_trajectories: Sequence[Sequence[Union[Dict[str, Any], Sequence[Any]]]],
    eval_policy_fn: Callable[[np.ndarray], np.ndarray],
    q_model: Any,
    tau_pi: float = 0.1,
    gamma: float = 0.99,
    clip_weight: Optional[float] = 20.0,
) -> float:
    """Calculates Doubly Robust (DR) off-policy value estimate across trajectories.

    Formula:
        V_DR = 1/M * sum_{m=1}^M sum_{t=0}^{H_m-1} gamma^t [ Q_hat(o_{m,t}, a_{m,t}^{pi_e})
               + rho_{m,t} * (r_{m,t} - Q_hat(o_{m,t}, a_{m,t})) ]

    Args:
        eval_trajectories: List of trajectories, where each trajectory is a sequence of
            transition dicts (or tuples) containing 'obs', 'action', 'reward', and optional
            'importance_weight' or 'rho'.
        eval_policy_fn: Callable taking obs -> evaluation action vector in [-1, 1]^32.
        q_model: Trained Q-function (CQLModel, CQLAgent, Critic, or callable) evaluating (obs, act).
        tau_pi: Boltzmann temperature hyperparameter (default: 0.1).
        gamma: Discount factor (default: 0.99).
        clip_weight: Upper bound for clipping importance weight rho (default: 20.0).

    Returns:
        float scalar Doubly Robust policy value estimate.
    """
    m_trajectories = len(eval_trajectories)
    if m_trajectories == 0:
        return 0.0

    total_dr_value = 0.0

    for trajectory in eval_trajectories:
        traj_val = 0.0
        for t, trans in enumerate(trajectory):
            if isinstance(trans, dict):
                obs = trans["obs"]
                act = trans["action"]
                rew = float(trans["reward"])
                rho = float(trans.get("importance_weight", trans.get("rho", 1.0)))
            else:
                obs = trans[0]
                act = trans[1]
                rew = float(trans[2])
                rho = float(trans[3]) if len(trans) > 3 else 1.0

            if clip_weight is not None and clip_weight > 0.0:
                rho = min(max(rho, 0.0), float(clip_weight))

            # Action selected by evaluation policy
            eval_act = eval_policy_fn(obs) if eval_policy_fn is not None else act

            # Evaluate Q-values
            if hasattr(q_model, "get_q_values"):
                q_eval = q_model.get_q_values(obs, eval_act)
                q_logged = q_model.get_q_values(obs, act)
            elif callable(q_model):
                q_eval = q_model(obs, eval_act)
                q_logged = q_model(obs, act)
            else:
                raise TypeError(f"Unsupported q_model type: {type(q_model)}")

            q_eval_val = float(q_eval.item() if isinstance(q_eval, torch.Tensor) else q_eval)
            q_logged_val = float(q_logged.item() if isinstance(q_logged, torch.Tensor) else q_logged)

            residual = rew - q_logged_val
            step_dr = q_eval_val + rho * residual

            traj_val += float(gamma**t) * step_dr

        total_dr_value += traj_val

    return float(total_dr_value / m_trajectories)


def evaluate_kuairec_zeroshot(
    policy: Any,
    kuairec_embs: np.ndarray,
    kuairec_matrix: np.ndarray,
    user_embs: Optional[np.ndarray] = None,
    rating_threshold: float = 3.0,
) -> Dict[str, float]:
    """Evaluates policy zero-shot on the fully-observed KuaiRec ground-truth matrix.

    Maps policy actions to KuaiRec items via Cosine MIPS, looks up ground-truth ratings
    directly from kuairec_matrix (1411 x 3327), and computes mean reward and hit rate.

    Args:
        policy: Evaluated policy (SACAgent, CQLAgent, or callable obs -> action).
        kuairec_embs: Matrix of shape (N_items, 32) containing unit-normalized item embeddings.
        kuairec_matrix: Dense array of shape (N_users, N_items) containing true ratings/watch ratios.
        user_embs: Optional user representations of shape (N_users, 32). If None, derived
            via rating-weighted projection onto kuairec_embs.
        rating_threshold: Numerical rating cutoff for hit rate classification (default: 3.0).

    Returns:
        dict containing 'mean_reward', 'mean_rating', 'hit_rate', 'hit_rate_observed', 'num_users'.
    """
    matrix = np.asarray(kuairec_matrix, dtype=np.float32)
    embs = np.asarray(kuairec_embs, dtype=np.float32)
    num_users, num_items = matrix.shape

    if user_embs is None:
        # Project observed user interactions onto item embedding space
        proj = matrix @ embs
        norms = np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
        user_embs = (proj / norms).astype(np.float32)
    else:
        user_embs = np.asarray(user_embs, dtype=np.float32)

    ratings: List[float] = []
    rewards: List[float] = []
    hits: List[float] = []
    observed_hits: List[float] = []

    for u in range(num_users):
        u_vec = user_embs[u]
        # Construct standard 202-dim POMDP observation at step 0
        obs = np.zeros(202, dtype=np.float32)
        obs[:32] = u_vec

        # Policy action selection
        if hasattr(policy, "sample_action"):
            # SACAgent
            act = policy.sample_action(obs, evaluate=True, return_log_prob=False)
        elif hasattr(policy, "select_action"):
            # CQLAgent with candidate action scoring
            act = policy.select_action(obs, candidate_actions=embs)
        elif callable(policy):
            act = policy(obs)
        else:
            raise TypeError(f"Unsupported policy type: {type(policy)}")

        if isinstance(act, torch.Tensor):
            act = act.cpu().numpy()

        act = np.asarray(act, dtype=np.float32).reshape(-1)

        # Cosine MIPS against KuaiRec item catalog
        norm = float(np.linalg.norm(act)) + 1e-8
        sims = embs @ (act / norm)
        chosen_item = int(np.argmax(sims))

        # Look up true rating from matrix
        r_true = float(matrix[u, chosen_item])
        ratings.append(r_true)

        # Metrics
        is_hit = 1.0 if r_true >= rating_threshold else 0.0
        hits.append(is_hit)
        observed_hits.append(1.0 if r_true > 0.0 else 0.0)

        # Composite reward formula: (r - 3.0) / 2.0 + 0.25 * I(r >= threshold)
        rew = (r_true - 3.0) / 2.0 + 0.25 * is_hit
        rewards.append(float(np.clip(rew, -1.0, 1.25)))

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_rating": float(np.mean(ratings)),
        "hit_rate": float(np.mean(hits)),
        "hit_rate_observed": float(np.mean(observed_hits)),
        "num_users": num_users,
    }
