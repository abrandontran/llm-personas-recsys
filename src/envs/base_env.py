"""Base Gymnasium environment for Sequential Recommender Systems.

Defines the POMDP observation/action spaces, FIFO interaction buffer,
Cosine MIPS candidate selection, and the Boltzmann OPE propensity policy.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class RecEnv(gym.Env):
    """Abstract Gymnasium Base Environment for RecSys POMDPs.

    State / Observation Space (R^202):
        o_t = [u_emb, e_{i_{t-1}}, r_{t-1}, Delta t_1, ..., e_{i_{t-5}}, r_{t-5}, Delta t_5]^T
        Dimension: 32 + 5 * (32 + 1 + 1) = 202.
        Delta t_k = k / 5.0 in {0.2, 0.4, 0.6, 0.8, 1.0} for k=1..5.
        Zero-padded at t=0 when past interactions are absent.

    Action Space ([-1.0, 1.0]^32):
        Continuous representation vector a_t in [-1.0, 1.0]^32, mapped to catalog items
        via Cosine Maximum Inner Product Search (MIPS) over unconsumed items.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        item_embeddings: np.ndarray,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the base RecSys environment.

        Args:
            item_embeddings: Matrix of shape (N, 32) containing float32 item vectors.
                Embeddings are unit-normalized upon initialization.
            seed: Optional integer seed for reproducible random state.
        """
        super().__init__()

        # Embeddings processing & unit-normalization
        embs = np.asarray(item_embeddings, dtype=np.float32)
        if embs.ndim != 2 or embs.shape[1] != 32:
            raise ValueError(f"item_embeddings must have shape (N, 32), got {embs.shape}")

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self.item_embeddings: np.ndarray = (embs / norms).astype(np.float32)
        self.num_items: int = self.item_embeddings.shape[0]

        # Action space: continuous 32-dimensional embedding
        self.action_space: spaces.Box = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(32,),
            dtype=np.float32,
        )

        # Observation space: 202-dimensional vector
        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(202,),
            dtype=np.float32,
        )

        # Dynamic state trackers
        self.consumed_items: Set[int] = set()
        self.history: List[Tuple[np.ndarray, float]] = []
        self.u_emb: np.ndarray = np.zeros(32, dtype=np.float32)
        self.current_step: int = 0

        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

    def mips_cosine_selection(self, action: np.ndarray) -> int:
        """Selects the best unconsumed catalog item using Cosine MIPS.

        i_t = argmax_{i in I \\ I_consumed} (a_t^T e_hat_i) / (||a_t||_2 + 1e-8)

        Args:
            action: Continuous recommendation action vector a_t in [-1, 1]^32.

        Returns:
            Integer index of the selected catalog item i_t in [0, N-1].

        Raises:
            RuntimeError: If all items in the catalog have already been consumed.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0)
        norm = float(np.linalg.norm(action)) + 1e-8
        a_norm = action / norm

        sims = self.item_embeddings @ a_norm  # (N,)

        if len(self.consumed_items) > 0:
            consumed_arr = np.fromiter(self.consumed_items, dtype=np.int64)
            valid_consumed = consumed_arr[(consumed_arr >= 0) & (consumed_arr < self.num_items)]
            if len(valid_consumed) >= self.num_items:
                raise RuntimeError("All catalog items have been consumed in this session.")
            sims[valid_consumed] = -np.inf

        selected_item = int(np.argmax(sims))
        return selected_item

    def get_action_distribution(
        self,
        action: np.ndarray,
        tau_pi: float = 0.1,
        consumed_items: Optional[Set[int]] = None,
    ) -> np.ndarray:
        """Computes Boltzmann probability distribution over all catalog items.

        pi_e(i | o_t) = exp((a_t^T e_hat_i) / (tau_pi * ||a_t||_2)) /
                        sum_{j not in I_consumed} exp((a_t^T e_hat_j) / (tau_pi * ||a_t||_2))
        for i not in I_consumed, and 0.0 for i in I_consumed.

        Args:
            action: Continuous action vector a_t in [-1, 1]^32.
            tau_pi: Boltzmann temperature hyperparameter (default: 0.1). Must be > 0.
            consumed_items: Optional set of consumed item indices. If None, uses
                self.consumed_items.

        Returns:
            np.ndarray of shape (N,) summing to 1.0, with zero probability on consumed items.
        """
        if tau_pi <= 0.0:
            raise ValueError(f"tau_pi must be strictly positive, got {tau_pi}")

        if consumed_items is None:
            consumed_items = self.consumed_items

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(np.nan_to_num(action, nan=0.0), -1.0, 1.0)
        norm = float(np.linalg.norm(action)) + 1e-8
        a_norm = action / norm

        logits = (self.item_embeddings @ a_norm) / float(tau_pi)

        # Build mask of unconsumed items
        unconsumed_mask = np.ones(self.num_items, dtype=bool)
        if consumed_items is not None and len(consumed_items) > 0:
            consumed_arr = np.asarray(
                list(consumed_items) if isinstance(consumed_items, (set, frozenset)) else consumed_items,
                dtype=np.int64,
            )
            valid_consumed = consumed_arr[(consumed_arr >= 0) & (consumed_arr < self.num_items)]
            unconsumed_mask[valid_consumed] = False

        if not np.any(unconsumed_mask):
            probs = np.full(self.num_items, 1.0 / self.num_items, dtype=np.float32)
            return probs

        # Numerically stable softmax over unconsumed items
        max_logit = float(np.max(logits[unconsumed_mask]))
        exp_logits = np.zeros(self.num_items, dtype=np.float64)
        exp_logits[unconsumed_mask] = np.exp(logits[unconsumed_mask] - max_logit)

        sum_exp = float(np.sum(exp_logits[unconsumed_mask]))
        if sum_exp > 0.0:
            probs = exp_logits / sum_exp
        else:
            probs = np.zeros(self.num_items, dtype=np.float64)
            probs[unconsumed_mask] = 1.0 / float(np.sum(unconsumed_mask))

        # Re-normalize to ensure sum is strictly 1.0 in float32
        probs = probs.astype(np.float32)
        s = float(np.sum(probs))
        if s > 0:
            probs /= s

        return probs

    def _get_observation(self) -> np.ndarray:
        """Constructs the 202-dimensional POMDP observation vector.

        Structure:
            [0:32]    : Static user embedding u_emb in R^32
            [32:66]   : Slot 1 (1 step ago): e_{i_{t-1}} (32), r_{t-1} (1), Delta t_1 = 0.2 (1)
            [66:100]  : Slot 2 (2 steps ago): e_{i_{t-2}} (32), r_{t-2} (1), Delta t_2 = 0.4 (1)
            [100:134] : Slot 3 (3 steps ago): e_{i_{t-3}} (32), r_{t-3} (1), Delta t_3 = 0.6 (1)
            [134:168] : Slot 4 (4 steps ago): e_{i_{t-4}} (32), r_{t-4} (1), Delta t_4 = 0.8 (1)
            [168:202] : Slot 5 (5 steps ago): e_{i_{t-5}} (32), r_{t-5} (1), Delta t_5 = 1.0 (1)

        Returns:
            np.ndarray of shape (202,) with dtype float32.
        """
        obs = np.zeros(202, dtype=np.float32)
        obs[:32] = self.u_emb

        for k in range(min(len(self.history), 5)):
            e_k, r_k = self.history[k]
            lag = float((k + 1) / 5.0)
            slot_start = 32 + k * 34
            obs[slot_start : slot_start + 32] = e_k
            obs[slot_start + 32] = float(r_k)
            obs[slot_start + 33] = lag

        return obs

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment for a new recommendation episode."""
        super().reset(seed=seed)
        self.consumed_items.clear()
        self.history.clear()
        self.current_step = 0
        return self._get_observation(), {}

    @abstractmethod
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single recommendation step."""
        raise NotImplementedError
