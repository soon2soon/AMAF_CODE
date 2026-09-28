from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from amaf.buffers.replay import DiscreteReplayBuffer
from amaf.networks.discrete import AMAFDQN, build_discrete_network


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.action_dim = int(action_dim)
        self.kind = cfg["algorithm"]
        net_cfg = cfg.get("network", {})
        self.online = build_discrete_network(self.kind, state_dim, action_dim, net_cfg).to(device)
        self.target = copy.deepcopy(self.online).to(device)
        self.target.eval()

        optim_cfg = cfg.get("optimizer", {})
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=float(optim_cfg.get("lr", 1e-3)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
        replay_cfg = cfg.get("replay", {})
        self.replay = DiscreteReplayBuffer(
            state_dim, int(replay_cfg.get("capacity", 100_000)), device
        )
        self.batch_size = int(replay_cfg.get("batch_size", 64))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.target_mode = cfg.get("target_mode", "double_dqn")
        self.learn_start = int(cfg.get("learn_start", self.batch_size))
        self.train_every = int(cfg.get("train_every", 1))
        self.target_update_interval = int(cfg.get("target_update_interval", 1000))
        self.gradient_clip = float(cfg.get("gradient_clip", 10.0))
        self.env_steps = 0
        self.updates = 0

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < float(epsilon):
            return random.randrange(self.action_dim)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.online.eval()
        with torch.no_grad():
            q = self.online(state_t)
        self.online.train()
        return int(q.argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, done) -> dict[str, float] | None:
        self.replay.add(state, action, reward, next_state, done)
        self.env_steps += 1
        if len(self.replay) < self.learn_start or self.env_steps % self.train_every != 0:
            return None
        metrics = self.learn()
        if self.env_steps % self.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return metrics

    def learn(self) -> dict[str, float]:
        state, action, reward, next_state, done = self.replay.sample(self.batch_size)
        q_expected = self.online(state).gather(1, action)
        with torch.no_grad():
            if self.target_mode == "double_dqn":
                next_action = self.online(next_state).argmax(dim=1, keepdim=True)
                q_next = self.target(next_state).gather(1, next_action)
            elif self.target_mode == "dqn":
                q_next = self.target(next_state).max(dim=1, keepdim=True).values
            else:
                raise ValueError(f"Unknown target_mode: {self.target_mode}")
            target = reward + self.gamma * (1.0 - done) * q_next

        td = target - q_expected
        loss = F.mse_loss(q_expected, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        return {
            "critic_loss": float(loss.detach().cpu()),
            "q_mean": float(q_expected.detach().mean().cpu()),
            "q_std": float(q_expected.detach().std(unbiased=False).cpu()),
            "td_error_mean": float(td.detach().abs().mean().cpu()),
            "td_error_std": float(td.detach().std(unbiased=False).cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }

    def diagnostics(self, state: np.ndarray) -> dict[str, Any] | None:
        if not isinstance(self.online, AMAFDQN):
            return None

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.online.eval()
        with torch.no_grad():
            _, diag = self.online(
                state_t,
                return_diagnostics=True,
            )
        self.online.train()

        # Effective weights actually used by final fused advantage.
        weights = diag.weights[0]
        heads = diag.heads[0]

        # Raw adaptive-router weights before anchoring.
        if diag.router_weights is not None:
            router_weights = diag.router_weights[0]
        else:
            router_weights = weights

        # State-conditioned trust tau(s).
        if diag.trust is not None:
            adaptive_trust = diag.trust[0].squeeze()
        else:
            adaptive_trust = None

        # ------------------------------------------------------
        # Effective gate diagnostics
        # ------------------------------------------------------
        entropy = -(
            weights
            * torch.log(weights.clamp_min(1e-12))
        ).sum()

        if weights.numel() > 1:
            max_entropy = torch.log(
                torch.tensor(
                    float(weights.numel()),
                    dtype=weights.dtype,
                    device=weights.device,
                )
            )
            normalized_entropy = entropy / max_entropy
        else:
            max_entropy = None
            normalized_entropy = torch.full_like(
                entropy,
                float("nan"),
            )

        effective_heads = torch.exp(entropy)
        gate_max_weight = weights.max()

        # ------------------------------------------------------
        # Raw router diagnostics
        # ------------------------------------------------------
        router_entropy = -(
            router_weights
            * torch.log(router_weights.clamp_min(1e-12))
        ).sum()

        if router_weights.numel() > 1:
            assert max_entropy is not None
            router_entropy_norm = (
                router_entropy / max_entropy
            )
        else:
            router_entropy_norm = torch.full_like(
                router_entropy,
                float("nan"),
            )

        router_effective_heads = torch.exp(
            router_entropy
        )
        router_max_weight = router_weights.max()

        # heads: [H, A]
        # Dispersion across heads for each action,
        # averaged over actions.
        head_disagreement = heads.std(
            dim=0,
            unbiased=False,
        ).mean()

        return {
            "weights": weights.detach().cpu().numpy(),
            "heads": heads.detach().cpu().numpy(),

            "gate_entropy": float(entropy.cpu()),
            "gate_entropy_norm": float(
                normalized_entropy.cpu()
            ),
            "effective_heads": float(
                effective_heads.cpu()
            ),
            "gate_max_weight": float(
                gate_max_weight.cpu()
            ),
            "head_disagreement": float(
                head_disagreement.cpu()
            ),
            "dominant_head": int(
                weights.argmax().item()
            ),

            # Anchored-gate mechanism diagnostics.
            "adaptive_trust": (
                float(adaptive_trust.cpu())
                if adaptive_trust is not None
                else float("nan")
            ),
            "router_entropy": float(
                router_entropy.cpu()
            ),
            "router_entropy_norm": float(
                router_entropy_norm.cpu()
            ),
            "router_effective_heads": float(
                router_effective_heads.cpu()
            ),
            "router_max_weight": float(
                router_max_weight.cpu()
            ),
            "router_dominant_head": int(
                router_weights.argmax().item()
            ),
            "router_weights": (
                router_weights.detach().cpu().numpy()
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env_steps": self.env_steps,
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.env_steps = int(state.get("env_steps", 0))
        self.updates = int(state.get("updates", 0))
