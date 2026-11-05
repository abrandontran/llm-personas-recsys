"""Mock LLM Gymnasium Simulation Environment (MockLLMEnv).

Bucket 6: High-throughput local emulator for vLLM rating logit scoring.
Emulates vLLM HTTP single-pass token logit extraction without GPU or network latency:
- Maintains 202-D POMDP observation space and continuous 32-D Cosine MIPS action selection.
- Evaluates expected rating via cosine similarity between archetype taste u_emb and item embedding e_i.
- Generates 5 logits for rating tokens ["1", "2", "3", "4", "5"] with calibrated Dirichlet noise.
- Samples integer ratings r_t in {1, 2, 3, 4, 5}.
- Evaluates endogenous churn continuation c_t in {0, 1} and composite reward in [-1.0, 1.25].
- Returns 5-element logit vector in info["logits"].
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.envs.base_env import RecEnv


RATING_TOKENS: List[str] = ["1", "2", "3", "4", "5"]
RATING_VALUES: np.ndarray = np.array([1, 2, 3, 4, 5], dtype=np.int64)


class MockLLMEnv(RecEnv):
    """Local CPU emulator for the vLLM rating token generation environment.

    Simulates the zero-generation logit scoring mechanism described in the proposal:
    instead of calling a remote vLLM server via HTTP, evaluates item appeal via
    latent cosine similarity, maps to token logits over {'1', '2', '3', '4', '5'},
    and samples realistic discrete ratings.
    """

    def __init__(
        self,
        item_embeddings: np.ndarray,
        archetypes: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = 20,
        alpha_retention: float = 0.25,
        noise_std: float = 0.25,
        dirichlet_alpha: float = 0.2,
        dirichlet_eps: float = 0.05,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the Mock LLM Environment.

        Args:
            item_embeddings: Array of shape (N, 32) containing float32 item vectors.
            archetypes: Optional list of user archetype dictionaries. If None or empty,
                a default neutral archetype is instantiated.
            max_steps: Maximum interaction steps per episode (default: 20).
            alpha_retention: Continuation bonus in composite reward (default: 0.25).
            noise_std: Scale parameter for rating logit Gaussian dispersion (default: 0.25).
            dirichlet_alpha: Concentration parameter for Dirichlet token noise (default: 0.2).
            dirichlet_eps: Mixing proportion for Dirichlet token noise (default: 0.05).
            temperature: Softmax temperature scaling for token logits (default: 1.0).
            seed: Optional integer seed for reproducible random state.
        """
        super().__init__(item_embeddings=item_embeddings, seed=seed)

        # Default fallback archetype if none provided
        if not archetypes:
            self.archetypes: List[Dict[str, Any]] = [
                {
                    "archetype_id": 0,
                    "user_id": 0,
                    "u_emb": np.zeros(32, dtype=np.float32),
                    "tau_quit": 3.0,
                    "top_genres": ["Drama", "Comedy"],
                }
            ]
        else:
            self.archetypes = list(archetypes)

        self.max_steps: int = int(max_steps)
        self.alpha_retention: float = float(alpha_retention)
        self.noise_std: float = float(noise_std)
        self.dirichlet_alpha: float = float(dirichlet_alpha)
        self.dirichlet_eps: float = float(dirichlet_eps)
        self.temperature: float = float(temperature)

        # Dynamic simulation state
        self.current_archetype: Dict[str, Any] = self.archetypes[0]
        self.tau_quit: float = 3.0
        self.consecutive_low_ratings: int = 0
        self.current_step: int = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment for a new recommendation episode.

        Args:
            seed: Optional integer seed.
            options: Optional dict selecting a specific user persona via
                'archetype_id', 'archetype_idx', or 'user_id'.

        Returns:
            Tuple of (initial_observation [202], info_dict).
        """
        super().reset(seed=seed, options=options)

        # Persona selection logic
        if options and "archetype_id" in options:
            matched = [a for a in self.archetypes if a.get("archetype_id") == options["archetype_id"]]
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

        # Extract static user embedding in R^32 and unit-normalize
        raw_u = self.current_archetype.get("u_emb", np.zeros(32, dtype=np.float32))
        u_arr = np.asarray(raw_u, dtype=np.float32).reshape(32)
        norm_u = float(np.linalg.norm(u_arr))
        if norm_u > 1e-8:
            self.u_emb = (u_arr / norm_u).astype(np.float32)
        else:
            self.u_emb = u_arr.copy()

        # Personalized quit threshold
        self.tau_quit = float(self.current_archetype.get("tau_quit", 3.0))

        # Reset episode counters
        self.consecutive_low_ratings = 0
        self.current_step = 0

        obs = self._get_observation()
        info: Dict[str, Any] = {
            "user_id": self.current_archetype.get("user_id"),
            "archetype_id": self.current_archetype.get("archetype_id"),
            "step": 0,
            "tau_quit": self.tau_quit,
        }
        return obs, info

    def compute_rating_logits(
        self,
        item_emb: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Evaluates taste alignment and generates 5 token logits for {'1', ..., '5'}.

        Args:
            item_emb: Unit-normalized 32-D item embedding.

        Returns:
            Tuple of:
                - logits: float32 array of shape (5,) representing raw unnormalized token logits.
                - probs: float32 array of shape (5,) representing softmax probabilities.
                - expected_rating: Continuous expected score in [1.0, 5.0].
        """
        # Cosine similarity between user taste and candidate item
        cos_sim = float(np.dot(self.u_emb, item_emb))

        # Linear mapping from similarity [-1, 1] to continuous rating [1.0, 5.0]
        # Similarity +1 -> rating 5.0; Similarity 0 -> rating 3.0; Similarity -1 -> rating 1.0
        expected_rating = float(np.clip(3.0 + 2.0 * cos_sim, 1.0, 5.0))

        # Generate base Gaussian-distance negative log-likelihoods for {1, 2, 3, 4, 5}
        ratings = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        scale = max(0.1, self.noise_std * 2.0)
        base_logits = -0.5 * ((ratings - expected_rating) / scale) ** 2

        # Convert to probabilities via numerically stable softmax
        max_l = float(np.max(base_logits))
        exp_l = np.exp(base_logits - max_l)
        base_probs = exp_l / float(np.sum(exp_l))

        # Add calibrated Dirichlet noise to mimic LLM token generation jitter
        if self.dirichlet_eps > 0.0:
            diri_noise = self.np_random.dirichlet(np.full(5, self.dirichlet_alpha))
            probs = (1.0 - self.dirichlet_eps) * base_probs + self.dirichlet_eps * diri_noise
        else:
            probs = base_probs

        probs = (probs / np.sum(probs)).astype(np.float32)

        # Scale logits with temperature
        logits = (np.log(np.clip(probs, 1e-12, 1.0)) / max(1e-5, self.temperature)).astype(np.float32)

        return logits, probs, expected_rating

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single recommendation interaction step.

        Args:
            action: Continuous action vector in [-1.0, 1.0]^32.

        Returns:
            Tuple of (next_observation [202], reward, terminated, truncated, info).
        """
        # 1. Cosine MIPS selection over unconsumed items
        item_idx = self.mips_cosine_selection(action)
        self.consumed_items.add(item_idx)
        e_it = self.item_embeddings[item_idx]

        # 2. Token logit generation and discrete rating sampling
        logits, probs, r_bar_t = self.compute_rating_logits(e_it)
        r_t = int(self.np_random.choice(RATING_VALUES, p=probs))

        # 3. Track consecutive poor recommendations
        if r_t <= 2:
            self.consecutive_low_ratings += 1
        else:
            self.consecutive_low_ratings = 0

        # 4. Endogenous churn check: continuation requires r_bar >= tau_quit and streak < 2
        c_t = 1 if (r_bar_t >= self.tau_quit and self.consecutive_low_ratings < 2) else 0
        terminated = bool(c_t == 0)

        # 5. Composite reward in [-1.0, 1.25]
        reward = float((float(r_t) - 3.0) / 2.0 + self.alpha_retention * float(c_t))
        reward = float(np.clip(reward, -1.0, 1.25))

        # 6. Update FIFO interaction history
        self.history.insert(0, (e_it, float(r_t)))
        if len(self.history) > 5:
            self.history.pop()

        # 7. Horizon / item exhaustion truncation
        self.current_step += 1
        truncated = bool(
            self.current_step >= self.max_steps
            or len(self.consumed_items) >= self.num_items
        )

        next_obs = self._get_observation()
        info: Dict[str, Any] = {
            "recommended_item": int(item_idx),
            "expected_rating": float(r_bar_t),
            "rating": int(r_t),
            "logits": logits,  # 5-element array for rating tokens "1".."5"
            "probs": probs,
            "continuation": int(c_t),
            "consecutive_low_ratings": int(self.consecutive_low_ratings),
            "step": int(self.current_step),
            "user_id": self.current_archetype.get("user_id"),
            "archetype_id": self.current_archetype.get("archetype_id"),
        }

        return next_obs, reward, terminated, truncated, info
