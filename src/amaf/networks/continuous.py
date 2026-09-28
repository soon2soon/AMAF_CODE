from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discrete import Gate


@dataclass
class ContinuousAMAFDiagnostics:
    weights: torch.Tensor
    heads: torch.Tensor


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden=(256, 256)):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], action_dim)
        self.max_action = float(max_action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class QCritic(nn.Module):
    """Standard TD3/SAC single Q critic."""

    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, **_) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class LegacyDuelingCritic(nn.Module):
    """Exact structural form used in the historical HalfCheetah notebook.

    Note: both the value and advantage streams are functions of a shared (s,a)
    representation; therefore this is *not* a strict V(s)+A(s,a) decomposition.
    It exists only for provenance/reproduction.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(hidden, 1)
        self.advantage_stream = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor, **_) -> torch.Tensor:
        h = self.shared(torch.cat([state, action], dim=-1))
        return self.value_stream(h) + self.advantage_stream(h)


class LegacyAMAFQFusionCritic(nn.Module):
    """Historical continuous AMAF implementation: adaptive fusion of Q candidates.

    The original notebook called these 'advantage heads', but the implementation
    has no separate V stream and directly fuses scalar Q candidates. This class is
    retained verbatim at the architectural level so legacy results can be traced.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        n_heads: int = 4,
        gate_mode: str = "adaptive",
        fixed_weights: Sequence[float] | None = None,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.gate = Gate(hidden, n_heads, mode=gate_mode, fixed_weights=fixed_weights)
        self.q_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_heads)])

    def forward(self, state: torch.Tensor, action: torch.Tensor, return_diagnostics=False, **_):
        h = self.shared(torch.cat([state, action], dim=-1))
        heads = torch.stack([head(h) for head in self.q_heads], dim=1)  # [B,H,1]
        weights = self.gate(h)
        fused = (heads * weights.unsqueeze(-1)).sum(dim=1)
        if return_diagnostics:
            return fused, ContinuousAMAFDiagnostics(weights=weights, heads=heads)
        return fused


class StateGate(nn.Module):
    """State-conditioned gate that can be shared by twin AMAF critics."""

    def __init__(
        self,
        state_dim: int,
        n_heads: int,
        hidden: int = 128,
        gate_mode: str = "adaptive",
        fixed_weights: Sequence[float] | None = None,
    ):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU())
        self.gate = Gate(hidden, n_heads, mode=gate_mode, fixed_weights=fixed_weights)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.gate(self.feature(state))


class ResidualAMAFQCritic(nn.Module):
    """Vanilla TD3 Q critic augmented with a centered AMAF residual.

    Q(s,a) = Q_base(s,a)
           + scale * sum_h w_h(s) [A_h(s,a) - A_h(s,a_ref)]

    The base Q path follows the standard TD3 critic architecture.
    The AMAF branch is zero at a == a_ref by construction.
    Each twin critic owns its own gate and residual heads.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden=(256, 256),
        n_heads: int = 4,
        gate_hidden_dim: int = 128,
        gate_mode: str = "adaptive",
        fixed_weights: Sequence[float] | None = None,
        residual_scale: float = 1.0,
        head_init_std: float = 1e-3,
    ):
        super().__init__()

        # Same action-conditioned backbone depth/width as vanilla QCritic.
        self.feature = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.base_head = nn.Linear(hidden[1], 1)

        self.state_gate = StateGate(
            state_dim,
            n_heads,
            hidden=gate_hidden_dim,
            gate_mode=gate_mode,
            fixed_weights=fixed_weights,
        )

        self.residual_heads = nn.ModuleList(
            [nn.Linear(hidden[1], 1) for _ in range(n_heads)]
        )

        # Start very close to vanilla TD3 while preserving head diversity
        # and non-zero learning signals.
        for head in self.residual_heads:
            nn.init.normal_(head.weight, mean=0.0, std=float(head_init_std))
            nn.init.zeros_(head.bias)

        self.residual_scale = float(residual_scale)

    def _feature(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.feature(torch.cat([state, action], dim=-1))

    def _heads_from_feature(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [head(h) for head in self.residual_heads],
            dim=1,
        )

    def _heads(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._heads_from_feature(self._feature(state, action))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reference_action: torch.Tensor | None = None,
        detach_reference: bool = False,
        return_diagnostics: bool = False,
        **_,
    ):
        h = self._feature(state, action)

        # Vanilla TD3 Q path.
        base_q = self.base_head(h)

        weights = self.state_gate(state)
        heads = self._heads_from_feature(h)
        fused = (weights.unsqueeze(-1) * heads).sum(dim=1)

        correction = fused

        if reference_action is not None:
            ref_heads = self._heads(state, reference_action)
            ref_fused = (weights.unsqueeze(-1) * ref_heads).sum(dim=1)

            if detach_reference:
                ref_fused = ref_fused.detach()

            correction = fused - ref_fused

        q = base_q + self.residual_scale * correction

        if return_diagnostics:
            return q, ContinuousAMAFDiagnostics(
                weights=weights,
                heads=heads,
            )

        return q


@dataclass
class SelectiveResidualAMAFDiagnostics:
    """Diagnostics for the base-preserving selective residual critic.

    `weights` contains the full H+1 simplex [NULL, residual_1, ..., residual_H].
    `heads` contains only the H residual head outputs at the current action.
    """

    weights: torch.Tensor
    heads: torch.Tensor
    null_weight: torch.Tensor
    residual_trust: torch.Tensor
    residual_weights: torch.Tensor
    conditional_weights: torch.Tensor
    correction: torch.Tensor
    base_q: torch.Tensor


class SelectiveStateGate(nn.Module):
    """State-conditioned gate with an explicit NULL / base-fallback option.

    For H residual views, the gate emits H+1 simplex weights:

        [w_null, w_1, ..., w_H]

    The residual trust is tau(s) = 1 - w_null(s).  The default initialization
    assigns 0.5 probability mass to NULL and 0.5 to the residual family, split
    uniformly across the H residual views.
    """

    def __init__(
        self,
        state_dim: int,
        n_heads: int,
        hidden: int = 128,
        gate_mode: str = "adaptive",
        fixed_weights: Sequence[float] | None = None,
        null_prior: float = 0.5,
    ):
        super().__init__()

        self.n_heads = int(n_heads)
        self.gate_mode = str(gate_mode)
        self.null_prior = float(null_prior)

        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")

        if not (0.0 < self.null_prior < 1.0):
            raise ValueError("null_prior must lie strictly between 0 and 1")

        if self.gate_mode == "adaptive":
            self.feature = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
            )
            self.logits = nn.Linear(hidden, self.n_heads + 1)

            # Principled prior:
            #   P(NULL) = null_prior
            #   P(residual family) = 1-null_prior
            # with uniform conditional mass across residual heads.
            # If residual logits are 0, the required NULL bias is
            # log(H * p_null / (1-p_null)).
            nn.init.zeros_(self.logits.weight)
            nn.init.zeros_(self.logits.bias)
            with torch.no_grad():
                null_bias = math.log(
                    self.n_heads * self.null_prior / (1.0 - self.null_prior)
                )
                self.logits.bias[0] = null_bias

            self.register_buffer("fixed", torch.empty(0), persistent=False)

        elif self.gate_mode == "uniform":
            # Uniform within the residual family, with the same explicit NULL prior.
            weights = torch.full(
                (self.n_heads + 1,),
                (1.0 - self.null_prior) / float(self.n_heads),
                dtype=torch.float32,
            )
            weights[0] = self.null_prior
            self.register_buffer("fixed", weights)
            self.feature = None
            self.logits = None

        elif self.gate_mode == "fixed":
            if fixed_weights is None:
                raise ValueError("fixed gate_mode requires fixed_weights")

            weights = torch.as_tensor(fixed_weights, dtype=torch.float32)
            if weights.numel() != self.n_heads + 1:
                raise ValueError(
                    "selective fixed_weights must contain H+1 values: "
                    "[NULL, residual_1, ..., residual_H]"
                )
            if torch.any(weights < 0):
                raise ValueError("fixed_weights must be non-negative")
            if float(weights.sum()) <= 0:
                raise ValueError("fixed_weights must have positive sum")

            weights = weights / weights.sum()
            self.register_buffer("fixed", weights)
            self.feature = None
            self.logits = None

        else:
            raise ValueError(
                f"Unknown selective gate_mode={self.gate_mode!r}; "
                "expected 'adaptive', 'uniform', or 'fixed'."
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "adaptive":
            logits = self.logits(self.feature(state))
            return torch.softmax(logits, dim=-1)

        return self.fixed.unsqueeze(0).expand(state.shape[0], -1)


class SelectiveResidualAMAFQCritic(nn.Module):
    """Base-preserving selective residual AMAF critic.

    Q(s,a) = B(s,a)
           + scale * sum_h w_h(s) [A_h(s,a) - A_h(s,a_ref)]

    where the gate is defined over H residual views plus an explicit NULL option.
    The residual weights therefore sum to tau(s)=1-w_null(s), allowing the model
    to fall back continuously to the TD3 base critic on a state-by-state basis.

    Structural invariants:
      1) The base critic and residual representation are parameter-disjoint.
      2) The centered residual is exactly zero in forward value at a == a_ref.
      3) Only the reference ACTION coordinate should be detached by the agent;
         the reference residual output remains in the critic computation graph.
      4) `detach_base=True` protects base parameters from the residual loss while
         preserving the same numeric full-Q value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden=(256, 256),
        n_heads: int = 4,
        gate_hidden_dim: int = 128,
        gate_mode: str = "adaptive",
        fixed_weights: Sequence[float] | None = None,
        residual_scale: float = 1.0,
        head_init_std: float = 1e-3,
        null_prior: float = 0.5,
        base_critic: QCritic | None = None,
    ):
        super().__init__()

        self.n_heads = int(n_heads)
        self.residual_scale = float(residual_scale)

        # A genuine TD3-compatible base critic.  TD3Agent can construct both
        # base critics before constructing residual modules, so with the same
        # random seed their initialization order can exactly match vanilla TD3.
        self.base = (
            base_critic
            if base_critic is not None
            else QCritic(state_dim, action_dim, hidden=hidden)
        )

        # Parameter-disjoint AMAF residual representation.
        self.residual_feature = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )

        self.residual_heads = nn.ModuleList(
            [nn.Linear(hidden[1], 1) for _ in range(self.n_heads)]
        )

        for head in self.residual_heads:
            nn.init.normal_(head.weight, mean=0.0, std=float(head_init_std))
            nn.init.zeros_(head.bias)

        self.state_gate = SelectiveStateGate(
            state_dim,
            self.n_heads,
            hidden=gate_hidden_dim,
            gate_mode=gate_mode,
            fixed_weights=fixed_weights,
            null_prior=null_prior,
        )

    def base_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.base(state, action)

    def _residual_feature(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.residual_feature(torch.cat([state, action], dim=-1))

    def _heads_from_feature(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [head(h) for head in self.residual_heads],
            dim=1,
        )

    def _heads(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._heads_from_feature(self._residual_feature(state, action))

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reference_action: torch.Tensor | None = None,
        detach_reference: bool = False,
        detach_base: bool = False,
        return_diagnostics: bool = False,
        **_,
    ):
        base_q = self.base_q(state, action)
        base_for_sum = base_q.detach() if detach_base else base_q

        full_weights = self.state_gate(state)           # [B, H+1]
        null_weight = full_weights[:, :1]               # [B, 1]
        residual_weights = full_weights[:, 1:]          # [B, H]
        residual_trust = 1.0 - null_weight              # [B, 1]
        conditional_weights = residual_weights / residual_trust.clamp_min(1e-12)

        heads = self._heads(state, action)               # [B, H, 1]
        fused = (residual_weights.unsqueeze(-1) * heads).sum(dim=1)
        correction = fused

        if reference_action is not None:
            ref_heads = self._heads(state, reference_action)
            ref_fused = (residual_weights.unsqueeze(-1) * ref_heads).sum(dim=1)

            if detach_reference:
                ref_fused = ref_fused.detach()

            correction = fused - ref_fused

        correction = self.residual_scale * correction
        q = base_for_sum + correction

        if return_diagnostics:
            return q, SelectiveResidualAMAFDiagnostics(
                weights=full_weights,
                heads=heads,
                null_weight=null_weight,
                residual_trust=residual_trust,
                residual_weights=residual_weights,
                conditional_weights=conditional_weights,
                correction=correction,
                base_q=base_q,
            )

        return q


class DuelingAdvantageCritic(nn.Module):
    """Strict state-value + action-conditioned advantage critic.

    `reference_action` optionally supplies an action for advantage centering:
      Q(s,a)=V(s)+A(s,a)-stopgrad(A(s,a_ref)).

    The stop-gradient option preserves an actor gradient while providing a numeric
    baseline. This is an *experimental reconciliation* of the manuscript equation,
    not the historical implementation.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reference_action: torch.Tensor | None = None,
        detach_reference: bool = True,
        **_,
    ) -> torch.Tensor:
        value = self.value(state)
        adv = self.advantage(torch.cat([state, action], dim=-1))
        if reference_action is None:
            return value + adv
        ref = self.advantage(torch.cat([state, reference_action], dim=-1))
        if detach_reference:
            ref = ref.detach()
        return value + adv - ref


class AMAFAdvantageCritic(nn.Module):
    """Paper-oriented continuous AMAF with V(s), multiple A_n(s,a), and a state gate."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gate: StateGate,
        hidden: int = 256,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.shared_gate = gate
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.adv_trunk = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.advantage_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_heads)])

    def _heads(self, state: torch.Tensor, action: torch.Tensor):
        h = self.adv_trunk(torch.cat([state, action], dim=-1))
        return torch.stack([head(h) for head in self.advantage_heads], dim=1)  # [B,H,1]

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reference_action: torch.Tensor | None = None,
        detach_reference: bool = True,
        return_diagnostics: bool = False,
    ):
        value = self.value(state)
        weights = self.shared_gate(state)
        heads = self._heads(state, action)
        fused = (weights.unsqueeze(-1) * heads).sum(dim=1)
        q = value + fused
        if reference_action is not None:
            ref_heads = self._heads(state, reference_action)
            ref = (weights.unsqueeze(-1) * ref_heads).sum(dim=1)
            if detach_reference:
                ref = ref.detach()
            q = q - ref
        if return_diagnostics:
            return q, ContinuousAMAFDiagnostics(weights=weights, heads=heads)
        return q


class GaussianActor(nn.Module):
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU()
        )
        self.mean = nn.Linear(hidden[1], action_dim)
        self.log_std = nn.Linear(hidden[1], action_dim)
        self.max_action = float(max_action)

    def forward(self, state: torch.Tensor):
        h = self.net(state)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        deterministic = torch.tanh(mean) * self.max_action
        return action, log_prob, deterministic