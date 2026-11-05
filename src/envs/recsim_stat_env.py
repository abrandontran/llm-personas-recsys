"""Parametric Statistical RecSim Environment (M_stat).

Implements the decoupled collaborative affinity and genre-space satiation dynamics,
endogenous session continuation (churn), and composite reward formulation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit

from src.envs.base_env import RecEnv


class ParametricRecSimEnv(RecEnv):
    """Parametric RecSim Simulation Environment (M_stat).

    State / Latent Representation:
        - Static user latent vector u_emb in R^32.
        - Dynamic genre satiation vector w_t in R^18, initialized to archetype's genre simplex.

    Decoupled Dynamics:
        1. Satiation Decay:
           w_{t+1, g} = w_{t, g} * exp(-beta * g_{i_t, g})  (beta = 0.3)
        2. Expected Rating:
           r_bar_t = 1 + 4 * sigma(u_emb^T e_hat_{i_t} + lambda_satiate * (w_t^T g_{i_t} - w_bar_0))
           where lambda_satiate = 1.5, w_bar_0 = mean(w_0), sigma is the standard logistic function.
        3. Realized Rating:
           r_t = r_bar_t + eps_t,  eps_t ~ N(0, noise_std^2), clamped to [1.0, 5.0].
        4. Endogenous Continuation & Churn:
           c_t = 1 iff r_bar_t >= tau_quit(u) AND consecutive_low_ratings (r_t <= 2.0) < 2.
           If c_t == 0, session terminates prematurely (terminated = True).
        5. Composite Reward:
           R_t = (r_t - 3.0) / 2.0 + alpha_retention * I(c_t == 1) in [-1.0, 1.25].
    """

    def __init__(
        self,
        item_embeddings: np.ndarray,
        item_genres: np.ndarray,
        archetypes: List[Dict[str, Any]],
        max_steps: int = 20,
        beta: float = 0.3,
        lambda_satiate: float = 1.5,
        noise_std: float = 0.2,
        alpha_retention: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the Parametric RecSim Environment.

        Args:
            item_embeddings: Array of shape (N, 32) containing unit-normalized item embeddings.
            item_genres: Multi-hot binary indicator matrix of shape (N, 18).
            archetypes: List of archetype dictionaries containing u_emb, genre_simplex, tau_quit.
            max_steps: Maximum interaction steps per episode (default: 20).
            beta: Genre satiation exponential decay rate (default: 0.3).
            lambda_satiate: Coefficient for genre satiation impact (default: 1.5).
            noise_std: Standard deviation of Gaussian observation noise (default: 0.2).
            alpha_retention: Continuation reward bonus (default: 0.25).
            seed: Optional integer seed for reproducible random state.
        """
        super().__init__(item_embeddings=item_embeddings, seed=seed)

        genres = np.asarray(item_genres, dtype=np.float32)
        if genres.ndim != 2 or genres.shape[1] != 18 or genres.shape[0] != self.num_items:
            raise ValueError(
                f"item_genres must have shape ({self.num_items}, 18), got {genres.shape}"
            )
        self.item_genres: np.ndarray = genres

        if not archetypes:
            raise ValueError("archetypes list must not be empty.")
        self.archetypes: List[Dict[str, Any]] = archetypes

        self.max_steps: int = max_steps
        self.beta: float = float(beta)
        self.lambda_satiate: float = float(lambda_satiate)
        self.noise_std: float = float(noise_std)
        self.alpha_retention: float = float(alpha_retention)

        # Dynamic simulation state
        self.current_archetype: Dict[str, Any] = self.archetypes[0]
        self.w_t: np.ndarray = np.full(18, 1.0 / 18.0, dtype=np.float32)
        self.w_bar_0: float = 1.0 / 18.0
        self.tau_quit: float = 3.0
        self.consecutive_low_ratings: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment for a new session with a user archetype.

        Args:
            seed: Optional integer seed.
            options: Optional dict for selecting a specific archetype via
                'archetype_id', 'archetype_idx', or 'user_id'.

        Returns:
            Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed, options=options)

        # Select user archetype
        if options and "archetype_id" in options:
            matched = [
                a for a in self.archetypes if a.get("archetype_id") == options["archetype_id"]
            ]
            self.current_archetype = matched[0] if matched else self.archetypes[0]
        elif options and "archetype_idx" in options:
            idx = int(options["archetype_idx"]) % len(self.archetypes)
            self.current_archetype = self.archetypes[idx]
        elif options and "user_id" in options:
            matched = [a for a in self.archetypes if a.get("user_id") == options["user_id"]]
            self.current_archetype = matched[0] if matched else self.archetypes[0]
        else:
            idx = int(self.np_random.integers(0, len(self.archetypes)))
            self.current_archetype = self.archetypes[idx]

        # Extract static user embedding u_emb in R^32
        u_emb_raw = self.current_archetype.get("u_emb", np.zeros(32))
        self.u_emb = np.asarray(u_emb_raw, dtype=np.float32).reshape(32)

        # Initialize dynamic genre satiation weights w_0 in R^18
        if (
            "genre_simplex" in self.current_archetype
            and len(self.current_archetype["genre_simplex"]) == 18
        ):
            w_0 = np.asarray(self.current_archetype["genre_simplex"], dtype=np.float32)
        elif "phi" in self.current_archetype and len(self.current_archetype["phi"]) >= 18:
            w_0 = np.asarray(self.current_archetype["phi"][:18], dtype=np.float32)
        else:
            w_0 = np.full(18, 1.0 / 18.0, dtype=np.float32)

        self.w_t = w_0.copy()
        self.w_bar_0 = float(np.mean(w_0))

        # Personalized churn threshold
        self.tau_quit = float(self.current_archetype.get("tau_quit", 3.0))

        # Reset episode trackers
        self.consecutive_low_ratings = 0
        self.current_step = 0

        obs = self._get_observation()
        info = {
            "user_id": self.current_archetype.get("user_id"),
            "archetype_id": self.current_archetype.get("archetype_id"),
            "step": 0,
            "tau_quit": self.tau_quit,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single recommendation interaction step.

        Args:
            action: Continuous representation vector a_t in [-1.0, 1.0]^32.

        Returns:
            Tuple of (next_observation, reward, terminated, truncated, info).
        """
        # 1. Map action to unconsumed item via Cosine MIPS
        item_idx = self.mips_cosine_selection(action)
        self.consumed_items.add(item_idx)

        e_it = self.item_embeddings[item_idx]
        g_it = self.item_genres[item_idx]

        # 2. Collaborative affinity and genre satiation
        latent_affinity = float(np.dot(self.u_emb, e_it))
        satiation_deviation = float(np.dot(self.w_t, g_it)) - self.w_bar_0
        z = latent_affinity + self.lambda_satiate * satiation_deviation

        # 3. Expected rating and realization with Gaussian noise
        r_bar_t = float(1.0 + 4.0 * expit(z))
        noise = float(self.np_random.normal(0.0, self.noise_std))
        r_t = float(np.clip(r_bar_t + noise, 1.0, 5.0))

        # 4. Exponential satiation decay for next step
        self.w_t = (self.w_t * np.exp(-self.beta * g_it)).astype(np.float32)

        # 5. Low ratings streak tracking
        if r_t <= 2.0:
            self.consecutive_low_ratings += 1
        else:
            self.consecutive_low_ratings = 0

        # 6. Endogenous session continuation check
        # Continuation requires expected rating >= tau_quit and streak < 2
        c_t = 1 if (r_bar_t >= self.tau_quit and self.consecutive_low_ratings < 2) else 0
        terminated = bool(c_t == 0)

        # 7. Composite reward in [-1.0, 1.25]
        reward = float((r_t - 3.0) / 2.0 + self.alpha_retention * float(c_t))
        reward = float(np.clip(reward, -1.0, 1.25))

        # 8. Update FIFO interaction history
        self.history.insert(0, (e_it, r_t))
        if len(self.history) > 5:
            self.history.pop()

        # 9. Step counter and horizon / item exhaustion truncation
        self.current_step += 1
        truncated = bool(
            self.current_step >= self.max_steps
            or len(self.consumed_items) >= self.num_items
        )

        next_obs = self._get_observation()
        info = {
            "recommended_item": int(item_idx),
            "expected_rating": float(r_bar_t),
            "rating": float(r_t),
            "continuation": int(c_t),
            "consecutive_low_ratings": int(self.consecutive_low_ratings),
            "step": int(self.current_step),
            "user_id": self.current_archetype.get("user_id"),
            "archetype_id": self.current_archetype.get("archetype_id"),
        }

        return next_obs, reward, terminated, truncated, info
