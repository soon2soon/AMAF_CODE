from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from amaf.buffers.replay import ReplayBuffer
from amaf.networks.continuous import (
    AMAFAdvantageCritic,
    Actor,
    DuelingAdvantageCritic,
    LegacyAMAFQFusionCritic,
    LegacyDuelingCritic,
    QCritic,
    ResidualAMAFQCritic,
    SelectiveResidualAMAFQCritic,
    StateGate,
)


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        cfg: dict[str, Any],
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.critic_type = cfg.get("critic_type", "vanilla")

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)

        self.critic_1, self.critic_2 = self._build_critics(state_dim, action_dim, cfg)
        self.critic_1_target, self.critic_2_target = self._build_target_critics(cfg)

        opt_cfg = cfg.get("optimizer", {})
        actor_lr = float(opt_cfg.get("actor_lr", 3e-4))
        critic_lr = float(opt_cfg.get("critic_lr", 3e-4))

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        critic_params = self._unique_params([self.critic_1, self.critic_2])
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)

        replay_cfg = cfg.get("replay", {})
        self.replay = ReplayBuffer(
            state_dim,
            action_dim,
            int(replay_cfg.get("capacity", 1_000_000)),
            device,
        )

        self.batch_size = int(replay_cfg.get("batch_size", 256))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.policy_noise = float(cfg.get("policy_noise", 0.2))
        self.noise_clip = float(cfg.get("noise_clip", 0.5))
        self.policy_freq = int(cfg.get("policy_freq", 2))
        self.gradient_clip = cfg.get("gradient_clip")

        # Selective BP actor update:
        #   none      -> v1: unconstrained full-Q actor gradient
        #   base_safe -> v2: project residual actor gradient when it
        #                conflicts with the vanilla-TD3 base gradient
        self.actor_projection = str(
            cfg.get("actor_projection", "none")
        ).lower()

        if self.actor_projection not in {"none", "base_safe"}:
            raise ValueError(
                f"Unknown actor_projection: {self.actor_projection}"
            )

        self.total_it = 0

    @staticmethod
    def _unique_params(modules):
        seen = set()
        params = []

        for module in modules:
            for p in module.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        return params

    @staticmethod
    def _project_residual_actor_gradients(
        base_grads,
        residual_grads,
        eps: float = 1e-12,
    ):
        """Project only the conflicting component of the residual gradient.

        The gradients are loss gradients.  If

            <g_base, g_residual> < 0,

        the residual gradient is projected onto the hyperplane orthogonal
        to g_base.  Otherwise it is left unchanged.

        This preserves the vanilla-TD3 base gradient while allowing
        non-conflicting AMAF residual information to contribute.
        """

        if len(base_grads) != len(residual_grads):
            raise ValueError("Gradient list length mismatch")

        base = []
        residual = []

        for gb, gr in zip(base_grads, residual_grads):
            if gb is None and gr is None:
                base.append(None)
                residual.append(None)
                continue

            if gb is None:
                gb = torch.zeros_like(gr)

            if gr is None:
                gr = torch.zeros_like(gb)

            base.append(gb)
            residual.append(gr)

        valid = [
            (gb, gr)
            for gb, gr in zip(base, residual)
            if gb is not None
        ]

        if not valid:
            return residual, {
                "base_norm": float("nan"),
                "residual_norm": float("nan"),
                "cosine": float("nan"),
                "conflict": 0.0,
                "projection_applied": 0.0,
            }

        gb0 = valid[0][0]

        dot = torch.zeros(
            (),
            dtype=gb0.dtype,
            device=gb0.device,
        )
        base_sq = torch.zeros_like(dot)
        residual_sq = torch.zeros_like(dot)

        for gb, gr in valid:
            dot = dot + (gb * gr).sum()
            base_sq = base_sq + (gb * gb).sum()
            residual_sq = residual_sq + (gr * gr).sum()

        base_norm = torch.sqrt(base_sq)
        residual_norm = torch.sqrt(residual_sq)

        if (
            float(base_sq.detach().cpu()) > eps
            and float(residual_sq.detach().cpu()) > eps
        ):
            cosine = dot / (
                base_norm * residual_norm + eps
            )
            cosine_value = float(cosine.detach().cpu())
        else:
            cosine_value = float("nan")

        conflict = (
            float(dot.detach().cpu()) < 0.0
            and float(base_sq.detach().cpu()) > eps
        )

        if conflict:
            coeff = dot / (base_sq + eps)

            projected = [
                None if gb is None else gr - coeff * gb
                for gb, gr in zip(base, residual)
            ]
        else:
            projected = residual

        return projected, {
            "base_norm": float(base_norm.detach().cpu()),
            "residual_norm": float(residual_norm.detach().cpu()),
            "cosine": cosine_value,
            "conflict": float(conflict),
            "projection_applied": float(conflict),
        }


    def _build_critics(self, state_dim: int, action_dim: int, cfg: dict):
        ctype = self.critic_type

        if ctype == "vanilla":
            return (
                QCritic(state_dim, action_dim).to(self.device),
                QCritic(state_dim, action_dim).to(self.device),
            )

        if ctype == "legacy_dueling":
            return (
                LegacyDuelingCritic(state_dim, action_dim).to(self.device),
                LegacyDuelingCritic(state_dim, action_dim).to(self.device),
            )

        if ctype == "legacy_amaf_qfusion":
            n_heads = int(cfg.get("n_heads", 4))
            mode = cfg.get("gate_mode", "adaptive")
            fixed = cfg.get("fixed_weights")

            return (
                LegacyAMAFQFusionCritic(
                    state_dim,
                    action_dim,
                    n_heads=n_heads,
                    gate_mode=mode,
                    fixed_weights=fixed,
                ).to(self.device),
                LegacyAMAFQFusionCritic(
                    state_dim,
                    action_dim,
                    n_heads=n_heads,
                    gate_mode=mode,
                    fixed_weights=fixed,
                ).to(self.device),
            )

        if ctype == "residual_amaf":
            n_heads = int(cfg.get("n_heads", 4))

            kwargs = dict(
                n_heads=n_heads,
                gate_hidden_dim=int(cfg.get("gate_hidden_dim", 128)),
                gate_mode=cfg.get("gate_mode", "adaptive"),
                fixed_weights=cfg.get("fixed_weights"),
                residual_scale=float(cfg.get("residual_scale", 1.0)),
                head_init_std=float(cfg.get("head_init_std", 1e-3)),
            )

            return (
                ResidualAMAFQCritic(
                    state_dim,
                    action_dim,
                    **kwargs,
                ).to(self.device),
                ResidualAMAFQCritic(
                    state_dim,
                    action_dim,
                    **kwargs,
                ).to(self.device),
            )

        if ctype == "selective_residual_amaf":
            n_heads = int(cfg.get("n_heads", 4))

            # Build both TD3 base critics first.  With the same global seed this
            # preserves the vanilla TD3 critic initialization order before any
            # residual/gate parameters consume RNG state.
            base_1 = QCritic(state_dim, action_dim).to(self.device)
            base_2 = QCritic(state_dim, action_dim).to(self.device)

            kwargs = dict(
                n_heads=n_heads,
                gate_hidden_dim=int(cfg.get("gate_hidden_dim", 128)),
                gate_mode=cfg.get("gate_mode", "adaptive"),
                fixed_weights=cfg.get("fixed_weights"),
                residual_scale=float(cfg.get("residual_scale", 1.0)),
                head_init_std=float(cfg.get("head_init_std", 1e-3)),
                null_prior=float(cfg.get("null_prior", 0.5)),
            )

            return (
                SelectiveResidualAMAFQCritic(
                    state_dim,
                    action_dim,
                    base_critic=base_1,
                    **kwargs,
                ).to(self.device),
                SelectiveResidualAMAFQCritic(
                    state_dim,
                    action_dim,
                    base_critic=base_2,
                    **kwargs,
                ).to(self.device),
            )

        if ctype == "paper_dueling":
            return (
                DuelingAdvantageCritic(state_dim, action_dim).to(self.device),
                DuelingAdvantageCritic(state_dim, action_dim).to(self.device),
            )

        if ctype == "paper_amaf":
            n_heads = int(cfg.get("n_heads", 4))
            shared_gate = StateGate(
                state_dim,
                n_heads,
                hidden=int(cfg.get("gate_hidden_dim", 128)),
                gate_mode=cfg.get("gate_mode", "adaptive"),
                fixed_weights=cfg.get("fixed_weights"),
            ).to(self.device)

            return (
                AMAFAdvantageCritic(
                    state_dim,
                    action_dim,
                    shared_gate,
                    n_heads=n_heads,
                ).to(self.device),
                AMAFAdvantageCritic(
                    state_dim,
                    action_dim,
                    shared_gate,
                    n_heads=n_heads,
                ).to(self.device),
            )

        raise ValueError(f"Unknown critic_type: {ctype}")

    def _build_target_critics(self, cfg: dict):
        # paper_amaf online twin critics share one gate.
        # Target twin critics also share one target gate.
        if self.critic_type == "paper_amaf":
            n_heads = int(cfg.get("n_heads", 4))
            target_gate = copy.deepcopy(self.critic_1.shared_gate).to(self.device)

            c1 = AMAFAdvantageCritic(
                self.state_dim,
                self.action_dim,
                target_gate,
                n_heads=n_heads,
            ).to(self.device)
            c2 = AMAFAdvantageCritic(
                self.state_dim,
                self.action_dim,
                target_gate,
                n_heads=n_heads,
            ).to(self.device)

            c1.load_state_dict(self.critic_1.state_dict())
            c2.load_state_dict(self.critic_2.state_dict())
            return c1, c2

        return (
            copy.deepcopy(self.critic_1).to(self.device),
            copy.deepcopy(self.critic_2).to(self.device),
        )

    def select_action(self, state: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]

        if noise_std > 0:
            action = action + np.random.normal(0.0, noise_std, size=self.action_dim)

        return np.clip(action, -self.max_action, self.max_action)

    def add_transition(self, state, action, next_state, reward, done) -> None:
        self.replay.add(state, action, next_state, reward, done)

    def _reference_action(self, state: torch.Tensor, target: bool = False):
        if self.critic_type not in {
            "paper_dueling",
            "paper_amaf",
            "residual_amaf",
            "selective_residual_amaf",
        }:
            return None

        actor = self.actor_target if target else self.actor
        return actor(state).detach()

    def _critic_forward(
        self,
        critic,
        state,
        action,
        target=False,
        return_diagnostics=False,
    ):
        ref = self._reference_action(state, target=target)
        # The reference ACTION is already detached in _reference_action().
        # Keep the reference critic output in the graph so that critic and
        # gate gradients correspond to the centered advantage difference.
        kwargs = {"reference_action": ref, "detach_reference": False}

        if return_diagnostics:
            kwargs["return_diagnostics"] = True

        return critic(state, action, **kwargs)

    def train_step(self) -> dict[str, float] | None:
        if len(self.replay) < self.batch_size:
            return None

        self.total_it += 1
        state, action, next_state, reward, done = self.replay.sample(self.batch_size)

        if self.critic_type == "selective_residual_amaf":
            return self._train_step_selective(
                state, action, next_state, reward, done
            )

        # TD3 target
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            target_q1 = self._critic_forward(
                self.critic_1_target, next_state, next_action, target=True
            )
            target_q2 = self._critic_forward(
                self.critic_2_target, next_state, next_action, target=True
            )
            target_q = reward + (1.0 - done) * self.gamma * torch.minimum(
                target_q1, target_q2
            )

        # Critic update
        current_q1 = self._critic_forward(self.critic_1, state, action)
        current_q2 = self._critic_forward(self.critic_2, state, action)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()

        # If clipping is disabled, gradient norm is not measured.
        # NaN is preferable to a misleading 0.0 and becomes blank in CSV logging.
        grad_norm = float("nan")
        if self.gradient_clip is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._unique_params([self.critic_1, self.critic_2]),
                    float(self.gradient_clip),
                )
                .detach()
                .cpu()
            )

        self.critic_optimizer.step()

        # Delayed actor update
        actor_loss_value = float("nan")
        if self.total_it % self.policy_freq == 0:
            actor_action = self.actor(state)
            actor_q = self._critic_forward(self.critic_1, state, actor_action)
            actor_loss = -actor_q.mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_value = float(actor_loss.detach().cpu())
            self._soft_update()

        # Revision diagnostics
        # q_std: minibatch Q1(s,a) dispersion
        # td_error_std: raw TD residual dispersion
        # critic_disagreement: mean |Q1 - Q2|
        td1 = target_q - current_q1

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "q_mean": float(current_q1.detach().mean().cpu()),
            "q_std": float(current_q1.detach().std(unbiased=False).cpu()),
            "td_error_mean": float(td1.detach().abs().mean().cpu()),
            "td_error_std": float(td1.detach().std(unbiased=False).cpu()),
            "critic_disagreement": float(
                (current_q1.detach() - current_q2.detach()).abs().mean().cpu()
            ),
            "grad_norm": grad_norm,
        }

    def _train_step_selective(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> dict[str, float]:
        """Train the base-preserving selective residual critic.

        The base branch receives only a vanilla TD3 Bellman loss.  The residual
        branch receives a full selective-AMAF Bellman loss with the current base
        value detached, so residual/gate gradients cannot modify base parameters.
        The actor still uses the full Q action-gradient.
        """

        # ------------------------------------------------------------
        # Targets
        # ------------------------------------------------------------
        with torch.no_grad():
            target_actor_action = self.actor_target(next_state)

            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (target_actor_action + noise).clamp(
                -self.max_action, self.max_action
            )

            # Unperturbed, detached target actor action is the residual anchor.
            next_ref = target_actor_action.detach()

            # Vanilla TD3 base target.
            target_base_q1 = self.critic_1_target.base_q(next_state, next_action)
            target_base_q2 = self.critic_2_target.base_q(next_state, next_action)
            target_base_min = torch.minimum(target_base_q1, target_base_q2)
            target_base = reward + (1.0 - done) * self.gamma * target_base_min

            # Full selective-AMAF target; clipped double-Q is preserved.
            target_full_q1 = self.critic_1_target(
                next_state,
                next_action,
                reference_action=next_ref,
                detach_reference=False,
                detach_base=False,
            )
            target_full_q2 = self.critic_2_target(
                next_state,
                next_action,
                reference_action=next_ref,
                detach_reference=False,
                detach_base=False,
            )
            target_full_min = torch.minimum(target_full_q1, target_full_q2)
            target_full = reward + (1.0 - done) * self.gamma * target_full_min

        # ------------------------------------------------------------
        # Base critic loss: vanilla TD3 learning path only.
        # ------------------------------------------------------------
        base_q1 = self.critic_1.base_q(state, action)
        base_q2 = self.critic_2.base_q(state, action)

        base_loss = F.mse_loss(base_q1, target_base) + F.mse_loss(
            base_q2, target_base
        )

        # ------------------------------------------------------------
        # Residual/gate loss: full target, but current base is detached.
        # This numerically evaluates the same Q while protecting base params.
        # ------------------------------------------------------------
        current_ref = self.actor(state).detach()

        current_full_q1 = self.critic_1(
            state,
            action,
            reference_action=current_ref,
            detach_reference=False,
            detach_base=True,
        )
        current_full_q2 = self.critic_2(
            state,
            action,
            reference_action=current_ref,
            detach_reference=False,
            detach_base=True,
        )

        residual_loss = F.mse_loss(current_full_q1, target_full) + F.mse_loss(
            current_full_q2, target_full
        )

        critic_loss = base_loss + residual_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()

        grad_norm = float("nan")
        if self.gradient_clip is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._unique_params([self.critic_1, self.critic_2]),
                    float(self.gradient_clip),
                )
                .detach()
                .cpu()
            )

        self.critic_optimizer.step()

        # ------------------------------------------------------------
        # Delayed actor update.
        #
        # v1 ("none"):
        #   actor follows the unconstrained full-Q gradient.
        #
        # v2 ("base_safe"):
        #   decompose the actor gradient into
        #
        #       g = g_base + g_residual
        #
        #   and remove only the component of g_residual that conflicts
        #   with the vanilla-TD3 base gradient.
        #
        # The reference action is detached, so the residual forward value
        # is zero at the anchor while its action derivative survives.
        # ------------------------------------------------------------
        actor_loss_value = float("nan")

        actor_base_grad_norm = float("nan")
        actor_residual_grad_norm = float("nan")
        actor_grad_cosine = float("nan")
        actor_conflict = float("nan")
        actor_projection_applied = float("nan")

        if self.total_it % self.policy_freq == 0:
            actor_action = self.actor(state)
            actor_ref = actor_action.detach()

            if self.actor_projection == "base_safe":
                actor_params = [
                    p for p in self.actor.parameters()
                    if p.requires_grad
                ]

                # Vanilla-TD3 base objective.
                actor_base_q = self.critic_1.base_q(
                    state,
                    actor_action,
                )
                actor_base_loss = -actor_base_q.mean()

                # Residual-only action objective.
                #
                # detach_base=True prevents the base branch from
                # contributing an action gradient here.  Subtracting the
                # detached base value leaves the residual contribution
                # numerically isolated as well.
                actor_full_detached_base_q = self.critic_1(
                    state,
                    actor_action,
                    reference_action=actor_ref,
                    detach_reference=False,
                    detach_base=True,
                )

                actor_residual_q = (
                    actor_full_detached_base_q
                    - actor_base_q.detach()
                )
                actor_residual_loss = -actor_residual_q.mean()

                # Numerically this equals the full selective objective at
                # the anchor, while the two gradient components remain
                # separately observable.
                actor_loss = (
                    actor_base_loss
                    + actor_residual_loss
                )

                self.actor_optimizer.zero_grad(set_to_none=True)

                base_grads = torch.autograd.grad(
                    actor_base_loss,
                    actor_params,
                    retain_graph=True,
                    allow_unused=True,
                )

                residual_grads = torch.autograd.grad(
                    actor_residual_loss,
                    actor_params,
                    allow_unused=True,
                )

                projected_residual_grads, proj = (
                    self._project_residual_actor_gradients(
                        base_grads,
                        residual_grads,
                    )
                )

                # Explicitly install
                #
                #   g_base + g_residual_safe
                #
                # into the actor optimizer.
                for p, gb, gr in zip(
                    actor_params,
                    base_grads,
                    projected_residual_grads,
                ):
                    if gb is None and gr is None:
                        continue

                    if gb is None:
                        gb = torch.zeros_like(gr)

                    if gr is None:
                        gr = torch.zeros_like(gb)

                    p.grad = (gb + gr).detach().clone()

                self.actor_optimizer.step()

                actor_base_grad_norm = proj["base_norm"]
                actor_residual_grad_norm = proj["residual_norm"]
                actor_grad_cosine = proj["cosine"]
                actor_conflict = proj["conflict"]
                actor_projection_applied = proj[
                    "projection_applied"
                ]

            else:
                # v1: preserve the original selective-BP implementation
                # exactly for the critic-only preservation ablation.
                actor_q = self.critic_1(
                    state,
                    actor_action,
                    reference_action=actor_ref,
                    detach_reference=False,
                    detach_base=False,
                )
                actor_loss = -actor_q.mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_optimizer.step()

            actor_loss_value = float(actor_loss.detach().cpu())
            self._soft_update()

        # Full-Q diagnostics are numerically valid even though the base path was
        # detached during the residual loss.
        td1 = target_full - current_full_q1

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "base_critic_loss": float(base_loss.detach().cpu()),
            "residual_critic_loss": float(residual_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "actor_base_grad_norm": actor_base_grad_norm,
            "actor_residual_grad_norm": actor_residual_grad_norm,
            "actor_grad_cosine": actor_grad_cosine,
            "actor_conflict": actor_conflict,
            "actor_projection_applied": actor_projection_applied,
            "q_mean": float(current_full_q1.detach().mean().cpu()),
            "q_std": float(current_full_q1.detach().std(unbiased=False).cpu()),
            "td_error_mean": float(td1.detach().abs().mean().cpu()),
            "td_error_std": float(td1.detach().std(unbiased=False).cpu()),
            "critic_disagreement": float(
                (current_full_q1.detach() - current_full_q2.detach())
                .abs()
                .mean()
                .cpu()
            ),
            "grad_norm": grad_norm,
        }

    def _soft_update_module(
        self, source: torch.nn.Module, target: torch.nn.Module
    ) -> None:
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def _soft_update_named(
        self,
        source: torch.nn.Module,
        target: torch.nn.Module,
        skip_prefix: str | None = None,
    ) -> None:
        target_params = dict(target.named_parameters())

        for name, p in source.named_parameters():
            if skip_prefix and name.startswith(skip_prefix):
                continue

            tp = target_params[name]
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def _soft_update(self) -> None:
        self._soft_update_module(self.actor, self.actor_target)

        if self.critic_type == "paper_amaf":
            # Shared gate must be updated exactly once.
            self._soft_update_module(
                self.critic_1.shared_gate, self.critic_1_target.shared_gate
            )
            self._soft_update_named(
                self.critic_1,
                self.critic_1_target,
                skip_prefix="shared_gate.",
            )
            self._soft_update_named(
                self.critic_2,
                self.critic_2_target,
                skip_prefix="shared_gate.",
            )
        else:
            self._soft_update_module(self.critic_1, self.critic_1_target)
            self._soft_update_module(self.critic_2, self.critic_2_target)

    def diagnostics(
        self, state: np.ndarray, action: np.ndarray
    ) -> dict[str, Any] | None:
        if "amaf" not in self.critic_type:
            return None

        s = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        a = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        if self.critic_type == "selective_residual_amaf":
            with torch.no_grad():
                ref_action = self._reference_action(s, target=False)
                out = self.critic_1(
                    s,
                    a,
                    reference_action=ref_action,
                    detach_reference=False,
                    detach_base=False,
                    return_diagnostics=True,
                )

                if not isinstance(out, tuple):
                    return None

                _, diag = out

                full_weights = diag.weights[0]                    # [H+1]
                residual_weights = diag.residual_weights[0]      # [H]
                conditional_weights = diag.conditional_weights[0] # [H]
                null_weight = diag.null_weight[0, 0]
                residual_trust = diag.residual_trust[0, 0]

                heads = diag.heads[0].squeeze(-1)
                ref_heads = self.critic_1._heads(s, ref_action)[0].squeeze(-1)
                heads = heads - ref_heads

                # Preserve the historical AMAF diagnostic semantics for the
                # existing keys: entropy/effective-head metrics refer to the
                # conditional distribution over the H residual views.
                entropy = -(
                    conditional_weights
                    * torch.log(conditional_weights.clamp_min(1e-12))
                ).sum()

                if conditional_weights.numel() > 1:
                    max_entropy = torch.log(
                        torch.tensor(
                            float(conditional_weights.numel()),
                            dtype=conditional_weights.dtype,
                            device=conditional_weights.device,
                        )
                    )
                    normalized_entropy = entropy / max_entropy
                else:
                    normalized_entropy = torch.ones_like(entropy)

                effective_heads = torch.exp(entropy)
                gate_max_weight = conditional_weights.max()
                head_disagreement = heads.std(unbiased=False)

                # New selective-gate diagnostics include the explicit NULL option.
                full_entropy = -(
                    full_weights * torch.log(full_weights.clamp_min(1e-12))
                ).sum()
                full_max_entropy = torch.log(
                    torch.tensor(
                        float(full_weights.numel()),
                        dtype=full_weights.dtype,
                        device=full_weights.device,
                    )
                )
                full_entropy_norm = full_entropy / full_max_entropy
                full_effective_options = torch.exp(full_entropy)
                dominant_option = int(full_weights.argmax().item())

            return {
                # Backward-compatible AMAF head diagnostics (H residual views).
                "weights": conditional_weights.cpu().numpy(),
                "heads": heads.cpu().numpy(),
                "gate_entropy": float(entropy.cpu()),
                "gate_entropy_norm": float(normalized_entropy.cpu()),
                "effective_heads": float(effective_heads.cpu()),
                "gate_max_weight": float(gate_max_weight.cpu()),
                "head_disagreement": float(head_disagreement.cpu()),
                "dominant_head": int(conditional_weights.argmax().item()),

                # Selective / abstention diagnostics.
                "full_weights": full_weights.cpu().numpy(),
                "residual_weights": residual_weights.cpu().numpy(),
                "gate_null_weight": float(null_weight.cpu()),
                "residual_trust": float(residual_trust.cpu()),
                "full_gate_entropy": float(full_entropy.cpu()),
                "full_gate_entropy_norm": float(full_entropy_norm.cpu()),
                "full_effective_options": float(full_effective_options.cpu()),
                # 0 means NULL; 1..H correspond to residual heads 0..H-1.
                "dominant_option": dominant_option,
            }

        with torch.no_grad():
            out = self._critic_forward(
                self.critic_1, s, a, return_diagnostics=True
            )
            if not isinstance(out, tuple):
                return None

            _, diag = out
            weights = diag.weights[0]
            heads = diag.heads[0].squeeze(-1)

            # paper_amaf actually uses centered advantages:
            # A_h(s,a) - A_h(s,a_ref).
            if self.critic_type in {"paper_amaf", "residual_amaf"}:
                ref_action = self._reference_action(s, target=False)
                ref_heads = self.critic_1._heads(s, ref_action)[0].squeeze(-1)
                heads = heads - ref_heads

            entropy = -(
                weights * torch.log(weights.clamp_min(1e-12))
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
                normalized_entropy = torch.ones_like(entropy)

            effective_heads = torch.exp(entropy)
            gate_max_weight = weights.max()
            head_disagreement = heads.std(unbiased=False)

        return {
            "weights": weights.cpu().numpy(),
            "heads": heads.cpu().numpy(),
            "gate_entropy": float(entropy.cpu()),
            "gate_entropy_norm": float(normalized_entropy.cpu()),
            "effective_heads": float(effective_heads.cpu()),
            "gate_max_weight": float(gate_max_weight.cpu()),
            "head_disagreement": float(head_disagreement.cpu()),
            "dominant_head": int(weights.argmax().item()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
        }