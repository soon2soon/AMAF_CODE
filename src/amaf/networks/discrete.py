from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AMAFDiagnostics:
    weights: torch.Tensor
    heads: torch.Tensor
    fused_advantage: torch.Tensor
    router_weights: torch.Tensor | None = None
    trust: torch.Tensor | None = None


class DQN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden: Sequence[int] = (128, 128)):
        super().__init__()
        dims = [input_dim, *hidden, action_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        value = self.value_stream(h)
        advantage = self.advantage_stream(h)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class Gate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        hidden_dim: int | None = None,
        mode: str = "adaptive",
        fixed_weights: Sequence[float] | None = None,
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.mode = mode
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, self.n_heads)
            )
        else:
            self.net = nn.Linear(input_dim, self.n_heads)

        if fixed_weights is not None:
            w = torch.tensor(list(fixed_weights), dtype=torch.float32)
            if w.numel() != self.n_heads:
                raise ValueError("fixed_weights length must equal n_heads")
            w = w / w.sum()
        else:
            w = torch.full((self.n_heads,), 1.0 / self.n_heads)
        self.register_buffer("fixed_weights", w)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "adaptive":
            return torch.softmax(self.net(h), dim=-1)
        if self.mode in {"uniform", "fixed"}:
            return self.fixed_weights.unsqueeze(0).expand(h.shape[0], -1)
        if self.mode == "random":
            # Non-learned, state-independent stochastic control. Reproducible under global seed.
            w = torch.rand((h.shape[0], self.n_heads), device=h.device, dtype=h.dtype)
            return w / w.sum(dim=-1, keepdim=True)
        if self.mode == "hard":
            logits = self.net(h)
            idx = logits.argmax(dim=-1)
            return F.one_hot(idx, num_classes=self.n_heads).to(h.dtype)
        raise ValueError(f"Unknown gate mode: {self.mode}")


class AMAFDQN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        gate_mode: str = "adaptive",
        gate_hidden_dim: int | None = None,
        fixed_weights: Sequence[float] | None = None,
        fusion_mode: str = "direct",
        trust_init: float = 0.1,
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.action_dim = int(action_dim)
        self.gate_mode = gate_mode

        # Same feature extractor as DuelingDQN.
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Same value stream as DuelingDQN.
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Extend ONLY the Dueling advantage stream from one stream to H heads.
        self.advantage_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
            for _ in range(self.n_heads)
        ])

        # Preserve the corrected parameter-free Uniform variant.
        if gate_mode == "uniform":
            self.gate = None
        else:
            self.gate = Gate(
                hidden_dim,
                self.n_heads,
                hidden_dim=gate_hidden_dim,
                mode=gate_mode,
                fixed_weights=fixed_weights,
            )

        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"direct", "uniform_anchored"}:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        self.trust_init = float(trust_init)
        self.trust_gate = None

        if self.fusion_mode == "uniform_anchored" and self.n_heads > 1:
            if not 0.0 < self.trust_init < 1.0:
                raise ValueError("trust_init must be in (0, 1)")

            trust_hidden_dim = int(
                gate_hidden_dim if gate_hidden_dim is not None else hidden_dim
            )

            self.trust_gate = nn.Sequential(
                nn.Linear(hidden_dim, trust_hidden_dim),
                nn.ReLU(),
                nn.Linear(trust_hidden_dim, 1),
            )

            # tau(s) starts exactly at trust_init for every state.
            final = self.trust_gate[-1]
            nn.init.zeros_(final.weight)
            nn.init.constant_(
                final.bias,
                math.log(self.trust_init / (1.0 - self.trust_init)),
            )

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        h = self.shared(x)
        value = self.value(h)

        heads = torch.stack(
            [head(h) for head in self.advantage_heads],
            dim=1,
        )  # [B,H,A]

        # Raw routing weights.
        if self.gate_mode == "uniform":
            router_weights = torch.full(
                (h.shape[0], self.n_heads),
                1.0 / self.n_heads,
                device=h.device,
                dtype=h.dtype,
            )
        else:
            assert self.gate is not None
            router_weights = self.gate(h)  # [B,H]

        adaptive = (
            router_weights.unsqueeze(-1) * heads
        ).sum(dim=1)

        trust = None

        if self.fusion_mode == "uniform_anchored":
            uniform = heads.mean(dim=1)

            if self.n_heads == 1:
                # Functional reduction: A_uniform == A_adaptive.
                # No unnecessary trust parameters are created for H=1.
                trust = torch.full(
                    (h.shape[0], 1),
                    self.trust_init,
                    device=h.device,
                    dtype=h.dtype,
                )
            else:
                assert self.trust_gate is not None
                trust = torch.sigmoid(self.trust_gate(h))  # [B,1]

            # A_fused = A_uniform + tau(s) * (A_adaptive - A_uniform)
            fused = uniform + trust * (adaptive - uniform)

            # Effective simplex weights actually used by the fused output:
            # w_eff = (1-tau)/H + tau*w_router
            weights = (
                (1.0 - trust) / float(self.n_heads)
                + trust * router_weights
            )
        else:
            fused = adaptive
            weights = router_weights

        q = value + fused - fused.mean(dim=1, keepdim=True)

        if return_diagnostics:
            return q, AMAFDiagnostics(
                weights=weights,
                heads=heads,
                fused_advantage=fused,
                router_weights=router_weights,
                trust=trust,
            )

        return q


def build_discrete_network(
    kind: str,
    input_dim: int,
    action_dim: int,
    cfg: dict,
) -> nn.Module:
    kind = kind.lower()

    if kind == "dqn":
        return DQN(
            input_dim,
            action_dim,
            hidden=tuple(cfg.get("hidden", [128, 128])),
        )

    if kind == "dueling_dqn":
        return DuelingDQN(
            input_dim,
            action_dim,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
        )

    if kind == "amaf_protected_dqn":
        from amaf.networks.discrete_protected import (
            ProtectedReferenceAMAFDQN,
        )

        return ProtectedReferenceAMAFDQN(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            n_heads=int(cfg.get("n_heads", 4)),
            gate_mode=cfg.get("gate_mode", "adaptive"),
            gate_hidden_dim=cfg.get("gate_hidden_dim"),
            trust_init=float(cfg.get("trust_init", 0.1)),
        )

    if kind in {"amaf_dqn", "amaf", "amaf_anchored_dqn"}:
        return AMAFDQN(
            input_dim,
            action_dim,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            n_heads=int(cfg.get("n_heads", 4)),
            gate_mode=cfg.get("gate_mode", "adaptive"),
            gate_hidden_dim=cfg.get("gate_hidden_dim"),
            fixed_weights=cfg.get("fixed_weights"),
            fusion_mode=cfg.get(
                "fusion_mode",
                "uniform_anchored"
                if kind == "amaf_anchored_dqn"
                else "direct",
            ),
            trust_init=float(cfg.get("trust_init", 0.1)),
        )

    raise ValueError(f"Unknown discrete network kind: {kind}")
