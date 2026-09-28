from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from amaf.agents.dqn import DQNAgent
from amaf.networks.discrete_protected import (
    ProtectedReferenceAMAFDQN,
)


class ProtectedReferenceDQNAgent(DQNAgent):
    """
    V3.2 DQN agent.

    Reference optimizer:
        Uniform reference TD loss -> reference params only.

    Adaptive optimizer:
        Full-Q TD loss -> router/trust params only.
        Reference tensors are detached.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: dict[str, Any],
        device: torch.device,
    ):
        # build_discrete_network now knows amaf_protected_dqn.
        super().__init__(
            state_dim,
            action_dim,
            cfg,
            device,
        )

        if not isinstance(
            self.online,
            ProtectedReferenceAMAFDQN,
        ):
            raise TypeError(
                "ProtectedReferenceDQNAgent requires "
                "ProtectedReferenceAMAFDQN"
            )

        self.reference_params = \
            self.online.reference_parameter_list()

        self.adaptive_params = \
            self.online.adaptive_parameter_list()

        ref_ids = {id(p) for p in self.reference_params}
        adapt_ids = {id(p) for p in self.adaptive_params}

        overlap = ref_ids & adapt_ids
        if overlap:
            raise RuntimeError(
                "reference/adaptive parameter groups overlap"
            )

        optim_cfg = cfg.get("optimizer", {})
        lr = float(optim_cfg.get("lr", 1e-3))
        wd = float(optim_cfg.get("weight_decay", 0.0))

        # Remove the generic all-parameter optimizer created by
        # DQNAgent.__init__.
        self.optimizer = None

        self.reference_optimizer = torch.optim.Adam(
            self.reference_params,
            lr=lr,
            weight_decay=wd,
        )

        self.adaptive_optimizer = (
            torch.optim.Adam(
                self.adaptive_params,
                lr=lr,
                weight_decay=wd,
            )
            if self.adaptive_params
            else None
        )

    def _reference_target(
        self,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        with torch.no_grad():
            if self.target_mode == "double_dqn":
                next_action = (
                    self.online
                    .forward_reference(next_state)
                    .argmax(dim=1, keepdim=True)
                )

                q_next = (
                    self.target
                    .forward_reference(next_state)
                    .gather(1, next_action)
                )

            elif self.target_mode == "dqn":
                q_next = (
                    self.target
                    .forward_reference(next_state)
                    .max(dim=1, keepdim=True)
                    .values
                )
            else:
                raise ValueError(
                    f"Unknown target_mode: "
                    f"{self.target_mode}"
                )

            return (
                reward
                + self.gamma * (1.0 - done) * q_next
            )

    def _full_target(
        self,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        with torch.no_grad():
            if self.target_mode == "double_dqn":
                next_action = (
                    self.online(next_state)
                    .argmax(dim=1, keepdim=True)
                )

                q_next = (
                    self.target(next_state)
                    .gather(1, next_action)
                )

            elif self.target_mode == "dqn":
                q_next = (
                    self.target(next_state)
                    .max(dim=1, keepdim=True)
                    .values
                )
            else:
                raise ValueError(
                    f"Unknown target_mode: "
                    f"{self.target_mode}"
                )

            return (
                reward
                + self.gamma * (1.0 - done) * q_next
            )

    def learn(self) -> dict[str, float]:
        (
            state,
            action,
            reward,
            next_state,
            done,
        ) = self.replay.sample(self.batch_size)

        # ======================================================
        # Compute both losses BEFORE either optimizer steps.
        # ======================================================

        # ----- protected Uniform reference loss -----
        reference_target = self._reference_target(
            next_state,
            reward,
            done,
        )

        q_reference = (
            self.online
            .forward_reference(state)
            .gather(1, action)
        )

        reference_td = (
            reference_target - q_reference
        )

        reference_loss = F.mse_loss(
            q_reference,
            reference_target,
        )

        # ----- adaptive final-Q loss -----
        full_target = self._full_target(
            next_state,
            reward,
            done,
        )

        # detach_reference=True is the central V3.2 safeguard.
        q_full = (
            self.online(
                state,
                detach_reference=True,
            )
            .gather(1, action)
        )

        adaptive_td = full_target - q_full

        adaptive_loss = F.mse_loss(
            q_full,
            full_target,
        )

        # ======================================================
        # Reference update: reference params only.
        # ======================================================
        self.reference_optimizer.zero_grad(
            set_to_none=True
        )

        if self.adaptive_optimizer is not None:
            self.adaptive_optimizer.zero_grad(
                set_to_none=True
            )

        reference_loss.backward()

        reference_grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.reference_params,
                self.gradient_clip,
            )
        )

        self.reference_optimizer.step()

        # ======================================================
        # Adaptive update: router + trust only.
        # ======================================================
        if self.adaptive_optimizer is not None:
            adaptive_loss.backward()

            adaptive_grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    self.adaptive_params,
                    self.gradient_clip,
                )
            )

            self.adaptive_optimizer.step()
        else:
            adaptive_grad_norm = torch.tensor(
                0.0,
                device=self.device,
            )

        self.updates += 1

        ref_g = float(
            torch.as_tensor(reference_grad_norm)
            .detach().cpu()
        )

        adapt_g = float(
            torch.as_tensor(adaptive_grad_norm)
            .detach().cpu()
        )

        combined_grad = math.sqrt(
            ref_g * ref_g
            + adapt_g * adapt_g
        )

        return {
            # Generic final-Q metric retained for compatibility.
            "critic_loss":
                float(adaptive_loss.detach().cpu()),

            "reference_loss":
                float(reference_loss.detach().cpu()),

            "adaptive_loss":
                float(adaptive_loss.detach().cpu()),

            "q_mean":
                float(q_full.detach().mean().cpu()),

            "q_std":
                float(
                    q_full.detach()
                    .std(unbiased=False)
                    .cpu()
                ),

            "td_error_mean":
                float(
                    adaptive_td.detach()
                    .abs().mean().cpu()
                ),

            "td_error_std":
                float(
                    adaptive_td.detach()
                    .std(unbiased=False)
                    .cpu()
                ),

            "grad_norm": combined_grad,

            "reference_grad_norm": ref_g,
            "adaptive_grad_norm": adapt_g,
        }

    def diagnostics(
        self,
        state: np.ndarray,
    ) -> dict[str, Any] | None:

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

        weights = diag.weights[0]
        heads = diag.heads[0]
        router = diag.router_weights[0]
        trust = diag.trust[0].squeeze()

        entropy = -(
            weights
            * torch.log(weights.clamp_min(1e-12))
        ).sum()

        router_entropy = -(
            router
            * torch.log(router.clamp_min(1e-12))
        ).sum()

        if weights.numel() > 1:
            max_entropy = torch.log(
                torch.tensor(
                    float(weights.numel()),
                    dtype=weights.dtype,
                    device=weights.device,
                )
            )

            entropy_norm = entropy / max_entropy
            router_entropy_norm = (
                router_entropy / max_entropy
            )
        else:
            entropy_norm = torch.full_like(
                entropy,
                float("nan"),
            )
            router_entropy_norm = torch.full_like(
                router_entropy,
                float("nan"),
            )

        effective_heads = torch.exp(entropy)
        router_effective_heads = \
            torch.exp(router_entropy)

        head_disagreement = (
            heads.std(
                dim=0,
                unbiased=False,
            ).mean()
        )

        reference_q_abs = (
            diag.reference_q[0]
            .abs().mean()
        )

        correction_abs = (
            diag.correction[0]
            .abs().mean()
        )

        correction_ratio = (
            correction_abs
            / (reference_q_abs + 1e-8)
        )

        return {
            "weights":
                weights.detach().cpu().numpy(),

            "heads":
                heads.detach().cpu().numpy(),

            "gate_entropy":
                float(entropy.cpu()),

            "gate_entropy_norm":
                float(entropy_norm.cpu()),

            "effective_heads":
                float(effective_heads.cpu()),

            "gate_max_weight":
                float(weights.max().cpu()),

            "head_disagreement":
                float(head_disagreement.cpu()),

            "dominant_head":
                int(weights.argmax().item()),

            "adaptive_trust":
                float(trust.cpu()),

            "router_entropy":
                float(router_entropy.cpu()),

            "router_entropy_norm":
                float(router_entropy_norm.cpu()),

            "router_effective_heads":
                float(
                    router_effective_heads.cpu()
                ),

            "router_max_weight":
                float(router.max().cpu()),

            "router_dominant_head":
                int(router.argmax().item()),

            "router_weights":
                router.detach().cpu().numpy(),

            "reference_q_abs_mean":
                float(reference_q_abs.cpu()),

            "correction_abs_mean":
                float(correction_abs.cpu()),

            "correction_ratio":
                float(correction_ratio.cpu()),
        }

    def state_dict(self) -> dict[str, Any]:
        state = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),

            "reference_optimizer":
                self.reference_optimizer.state_dict(),

            "env_steps": self.env_steps,
            "updates": self.updates,
        }

        if self.adaptive_optimizer is not None:
            state["adaptive_optimizer"] = (
                self.adaptive_optimizer.state_dict()
            )

        return state

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        self.online.load_state_dict(
            state["online"]
        )

        self.target.load_state_dict(
            state["target"]
        )

        if "reference_optimizer" in state:
            self.reference_optimizer.load_state_dict(
                state["reference_optimizer"]
            )

        if (
            self.adaptive_optimizer is not None
            and "adaptive_optimizer" in state
        ):
            self.adaptive_optimizer.load_state_dict(
                state["adaptive_optimizer"]
            )

        self.env_steps = int(
            state.get("env_steps", 0)
        )

        self.updates = int(
            state.get("updates", 0)
        )
