from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from amaf.networks.discrete import AMAFDQN, Gate


@dataclass
class ProtectedAMAFDiagnostics:
    weights: torch.Tensor
    heads: torch.Tensor
    fused_advantage: torch.Tensor
    router_weights: torch.Tensor
    trust: torch.Tensor
    reference_q: torch.Tensor
    correction: torch.Tensor


class ProtectedReferenceAMAFDQN(nn.Module):
    """
    V3.2: optimization-protected Uniform reference.

    Forward function:
        Q = Q_ref + tau(s) * center(A_adaptive - A_uniform)

    Q_ref is a canonical Uniform multi-head Dueling Q-network.

    Crucially, when detach_reference=True, the adaptive correction
    cannot backpropagate into the reference trunk/value/advantage heads.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        gate_mode: str = "adaptive",
        gate_hidden_dim: int | None = None,
        trust_init: float = 0.1,
    ):
        super().__init__()

        self.n_heads = int(n_heads)
        self.action_dim = int(action_dim)
        self.trust_init = float(trust_init)

        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")

        # ------------------------------------------------------
        # Protected reference:
        # exact canonical Uniform AMAF architecture.
        # ------------------------------------------------------
        self.reference = AMAFDQN(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_heads=self.n_heads,
            gate_mode="uniform",
            fusion_mode="direct",
        )

        self.gate = None
        self.trust_gate = None

        # H=1 must reduce exactly to Dueling.
        if self.n_heads > 1:
            if not 0.0 < self.trust_init < 1.0:
                raise ValueError("trust_init must be in (0, 1)")

            # Router must match V3.1 exactly:
            # gate_hidden_dim=None is passed through unchanged.
            self.gate = Gate(
                hidden_dim,
                self.n_heads,
                hidden_dim=gate_hidden_dim,
                mode=gate_mode,
            )

            # V3.1 trust gate independently falls back to hidden_dim.
            trust_hidden = int(
                gate_hidden_dim
                if gate_hidden_dim is not None
                else hidden_dim
            )

            self.trust_gate = nn.Sequential(
                nn.Linear(hidden_dim, trust_hidden),
                nn.ReLU(),
                nn.Linear(trust_hidden, 1),
            )

            # Same initialization as V3.1.
            final = self.trust_gate[-1]
            nn.init.zeros_(final.weight)
            nn.init.constant_(
                final.bias,
                math.log(
                    self.trust_init
                    / (1.0 - self.trust_init)
                ),
            )

    def _reference_components(
        self,
        x: torch.Tensor,
    ):
        h = self.reference.shared(x)
        value = self.reference.value(h)

        heads = torch.stack(
            [
                head(h)
                for head in self.reference.advantage_heads
            ],
            dim=1,
        )  # [B,H,A]

        uniform = heads.mean(dim=1)

        q_ref = (
            value
            + uniform
            - uniform.mean(dim=1, keepdim=True)
        )

        return q_ref, h, heads, uniform

    def forward_reference(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        q_ref, _, _, _ = self._reference_components(x)
        return q_ref

    def reference_parameter_list(self):
        return list(self.reference.parameters())

    def adaptive_parameter_list(self):
        if self.n_heads == 1:
            return []

        assert self.gate is not None
        assert self.trust_gate is not None

        return (
            list(self.gate.parameters())
            + list(self.trust_gate.parameters())
        )

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
        detach_reference: bool = False,
    ):
        q_ref, h, heads, uniform = \
            self._reference_components(x)

        # ------------------------------------------------------
        # H=1: exact Dueling reduction.
        # ------------------------------------------------------
        if self.n_heads == 1:
            router_weights = torch.ones(
                (x.shape[0], 1),
                dtype=x.dtype,
                device=x.device,
            )

            trust = torch.zeros(
                (x.shape[0], 1),
                dtype=x.dtype,
                device=x.device,
            )

            correction = torch.zeros_like(q_ref)
            q = q_ref
            weights = router_weights
            fused = uniform

        else:
            assert self.gate is not None
            assert self.trust_gate is not None

            # --------------------------------------------------
            # OPTIMIZATION PROTECTION
            # --------------------------------------------------
            if detach_reference:
                h_adapt = h.detach()
                heads_adapt = heads.detach()
                uniform_adapt = uniform.detach()
                q_base = q_ref.detach()
            else:
                h_adapt = h
                heads_adapt = heads
                uniform_adapt = uniform
                q_base = q_ref

            router_weights = self.gate(h_adapt)

            adaptive = (
                router_weights.unsqueeze(-1)
                * heads_adapt
            ).sum(dim=1)

            trust = torch.sigmoid(
                self.trust_gate(h_adapt)
            )

            delta = adaptive - uniform_adapt

            # Dueling-consistent centered correction.
            correction = trust * (
                delta
                - delta.mean(dim=1, keepdim=True)
            )

            q = q_base + correction

            # Algebraically equivalent fused advantage
            # for diagnostics.
            fused = (
                uniform_adapt
                + trust * delta
            )

            weights = (
                (1.0 - trust) / float(self.n_heads)
                + trust * router_weights
            )

        if return_diagnostics:
            return q, ProtectedAMAFDiagnostics(
                weights=weights,
                heads=heads,
                fused_advantage=fused,
                router_weights=router_weights,
                trust=trust,
                reference_q=q_ref,
                correction=correction,
            )

        return q
