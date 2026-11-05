"""Pure PyTorch Conservative Q-Learning (CQL) implementation (Bucket 5).

Features:
- CQLModel: 2-layer MLP [256, 256] with LayerNorm and ReLU.
- CQLAgent: Target Q-network with Polyak update (tau=0.005), AdamW optimizer,
  and conservative regularization penalizing out-of-distribution random actions
  relative to logged dataset actions.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CQLModel(nn.Module):
    """Q-network MLP taking concatenated (obs, action) in R^234 -> R^1."""

    def __init__(
        self,
        obs_dim: int = 202,
        action_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        in_dim = obs_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Q-network.

        Args:
            obs: Observation tensor (B, obs_dim) or (obs_dim,) or concatenated input (B, in_dim).
            action: Optional action tensor (B, action_dim) or (action_dim,).

        Returns:
            Q-value tensor of shape (B, 1) or (1,).
        """
        if action is not None:
            is_1d = obs.ndim == 1
            if is_1d:
                obs = obs.unsqueeze(0)
                action = action.unsqueeze(0)
            x = torch.cat([obs, action], dim=-1)
            out = self.net(x)
            return out.squeeze(0) if is_1d else out

        # Single concatenated input
        is_1d = obs.ndim == 1
        if is_1d:
            obs = obs.unsqueeze(0)
        out = self.net(obs)
        return out.squeeze(0) if is_1d else out


class CQLAgent:
    """Conservative Q-Learning (CQL) Agent for Offline RL.

    Manages Q-network Q_theta, target Q-network Q_{theta_bar}, Polyak averaging,
    and conservative regularizer loss:
        L_CQL(theta) = alpha_CQL * (E_{o~D, a~Unif}[Q(o, a)] - E_{(o, a)~D}[Q(o, a)])
                     + 0.5 * E_{(o, a, r, o')~D}[(Q(o, a) - y)^2]
    where target y = r + gamma * (1 - d) * Q_{theta_bar}(o', a').
    """

    def __init__(
        self,
        obs_dim: int = 202,
        action_dim: int = 32,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_cql: float = 1.0,
        num_random_actions: int = 10,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha_cql = float(alpha_cql)
        self.num_random_actions = int(num_random_actions)

        # Networks
        self.q_net = CQLModel(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q_target = CQLModel(obs_dim, action_dim, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=lr)

    def get_q_values(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluates Q-network on (obs, action) pairs."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(device=self.device, dtype=torch.float32)

        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(device=self.device, dtype=torch.float32)

        return self.q_net(obs, action)

    @torch.no_grad()
    def _update_targets(self) -> None:
        """Polyak averaging update for target Q-network."""
        for target_param, param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

    def update(
        self,
        batch: Union[
            Dict[str, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ) -> Dict[str, float]:
        """Performs a single CQL gradient update step.

        Args:
            batch: Dictionary or tuple containing:
                obs: (B, obs_dim)
                act: (B, action_dim)
                rew: (B, 1)
                next_obs: (B, obs_dim)
                done: (B, 1)
                next_act: Optional (B, action_dim)

        Returns:
            Dictionary with training loss metrics.
        """
        if isinstance(batch, dict):
            obs = batch["obs"].to(device=self.device, dtype=torch.float32)
            act = batch["act"].to(device=self.device, dtype=torch.float32)
            rew = batch["rew"].to(device=self.device, dtype=torch.float32)
            next_obs = batch["next_obs"].to(device=self.device, dtype=torch.float32)
            done = batch["done"].to(device=self.device, dtype=torch.float32)
            next_act = (
                batch["next_act"].to(device=self.device, dtype=torch.float32)
                if "next_act" in batch and batch["next_act"] is not None
                else None
            )
        else:
            obs = batch[0].to(device=self.device, dtype=torch.float32)
            act = batch[1].to(device=self.device, dtype=torch.float32)
            rew = batch[2].to(device=self.device, dtype=torch.float32)
            next_obs = batch[3].to(device=self.device, dtype=torch.float32)
            done = batch[4].to(device=self.device, dtype=torch.float32)
            next_act = (
                batch[5].to(device=self.device, dtype=torch.float32)
                if len(batch) > 5 and batch[5] is not None
                else None
            )

        batch_size = obs.shape[0]

        # 1. Compute Bellman Target y
        with torch.no_grad():
            if next_act is not None:
                target_q_next = self.q_target(next_obs, next_act)
            else:
                # Max over candidate random actions for continuous offline Q-learning
                num_cands = max(1, self.num_random_actions)
                # (B * num_cands, action_dim)
                cand_acts = torch.empty(
                    batch_size * num_cands, self.action_dim, device=self.device
                ).uniform_(-1.0, 1.0)
                next_obs_rep = (
                    next_obs.unsqueeze(1)
                    .repeat(1, num_cands, 1)
                    .view(batch_size * num_cands, self.obs_dim)
                )
                cand_q = self.q_target(next_obs_rep, cand_acts).view(batch_size, num_cands)
                target_q_next = cand_q.max(dim=1, keepdim=True)[0]

            y = rew + (1.0 - done) * self.gamma * target_q_next

        # 2. Q-value on logged dataset actions
        q_data = self.q_net(obs, act)
        td_loss = 0.5 * F.mse_loss(q_data, y)

        # 3. Conservative Regularization
        # Sample random actions uniformly from [-1, 1]^action_dim
        num_rand = max(1, self.num_random_actions)
        if num_rand == 1:
            rand_acts = torch.empty(
                batch_size, self.action_dim, device=self.device
            ).uniform_(-1.0, 1.0)
            q_rand = self.q_net(obs, rand_acts)
        else:
            rand_acts = torch.empty(
                batch_size * num_rand, self.action_dim, device=self.device
            ).uniform_(-1.0, 1.0)
            obs_rep = (
                obs.unsqueeze(1)
                .repeat(1, num_rand, 1)
                .view(batch_size * num_rand, self.obs_dim)
            )
            q_rand = self.q_net(obs_rep, rand_acts).view(batch_size, num_rand)
        q_rand_mean = q_rand.mean()
        q_data_mean = q_data.mean()

        cql_loss = self.alpha_cql * (q_rand_mean - q_data_mean)
        total_loss = td_loss + cql_loss

        # 4. Gradient Step
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()

        # 5. Polyak Target Averaging
        self._update_targets()

        return {
            "loss": float(total_loss.item()),
            "td_loss": float(td_loss.item()),
            "cql_loss": float(cql_loss.item()),
            "q_data_mean": float(q_data_mean.item()),
            "q_rand_mean": float(q_rand_mean.item()),
        }

    @torch.no_grad()
    def select_action(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        candidate_actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        num_samples: int = 100,
    ) -> np.ndarray:
        """Selects greedy action maximizing Q(obs, action).

        Args:
            obs: Observation vector in R^202.
            candidate_actions: Optional pool of candidate actions (e.g. catalog embeddings N x 32).
            num_samples: Number of uniform random actions to evaluate if candidate_actions is None.

        Returns:
            Selected action vector of shape (32,) as np.ndarray.
        """
        if not isinstance(obs, torch.Tensor):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(device=self.device, dtype=torch.float32)

        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        if candidate_actions is not None:
            if not isinstance(candidate_actions, torch.Tensor):
                cand_tensor = torch.as_tensor(
                    candidate_actions, dtype=torch.float32, device=self.device
                )
            else:
                cand_tensor = candidate_actions.to(device=self.device, dtype=torch.float32)

            num_cands = cand_tensor.shape[0]
            obs_expanded = obs_tensor.repeat(num_cands, 1)
            q_vals = self.q_net(obs_expanded, cand_tensor)
            best_idx = int(torch.argmax(q_vals).item())
            return cand_tensor[best_idx].cpu().numpy()

        # Continuous action search over random samples
        rand_acts = torch.empty(num_samples, self.action_dim, device=self.device).uniform_(-1.0, 1.0)
        obs_expanded = obs_tensor.repeat(num_samples, 1)
        q_vals = self.q_net(obs_expanded, rand_acts)
        best_idx = int(torch.argmax(q_vals).item())
        return rand_acts[best_idx].cpu().numpy()

    def save(self, path: Union[str, Path]) -> None:
        """Serializes CQL agent checkpoint to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha_cql": self.alpha_cql,
            "num_random_actions": self.num_random_actions,
        }
        torch.save(state, path)

    def load(self, path: Union[str, Path]) -> None:
        """Loads serialized CQL agent checkpoint from disk."""
        state = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(state["q_net"])
        self.q_target.load_state_dict(state["q_target"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            for opt_state in self.optimizer.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(self.device)
