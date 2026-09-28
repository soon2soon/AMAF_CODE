from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from amaf.buffers.replay import ReplayBuffer
from amaf.networks.continuous import GaussianActor, QCritic


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        cfg: dict[str, Any],
        device: torch.device,
    ):
        self.device = device
        self.max_action = float(max_action)
        self.action_dim = int(action_dim)

        self.actor = GaussianActor(
            state_dim,
            action_dim,
            max_action,
        ).to(device)

        self.q1 = QCritic(
            state_dim,
            action_dim,
        ).to(device)

        self.q2 = QCritic(
            state_dim,
            action_dim,
        ).to(device)

        self.q1_target = copy.deepcopy(
            self.q1
        ).to(device)

        self.q2_target = copy.deepcopy(
            self.q2
        ).to(device)

        opt = cfg.get("optimizer", {})
        lr = float(
            opt.get("lr", 3e-4)
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=lr,
        )

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters())
            + list(self.q2.parameters()),
            lr=lr,
        )

        self.auto_alpha = bool(
            cfg.get("auto_alpha", True)
        )

        init_alpha = float(
            cfg.get("alpha", 0.2)
        )

        self.target_entropy = float(
            cfg.get(
                "target_entropy",
                -action_dim,
            )
        )

        self.log_alpha = torch.tensor(
            np.log(init_alpha),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=lr,
        )

        replay = cfg.get(
            "replay",
            {},
        )

        self.replay = ReplayBuffer(
            state_dim,
            action_dim,
            int(
                replay.get(
                    "capacity",
                    1_000_000,
                )
            ),
            device,
        )

        self.batch_size = int(
            replay.get(
                "batch_size",
                256,
            )
        )

        self.gamma = float(
            cfg.get("gamma", 0.99)
        )

        self.tau = float(
            cfg.get("tau", 0.005)
        )

        self.updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        s = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            action, _, mean_action = (
                self.actor.sample(s)
            )

        out = (
            mean_action
            if deterministic
            else action
        )

        return out.cpu().numpy()[0]

    def add_transition(
        self,
        state,
        action,
        next_state,
        reward,
        done,
    ):
        self.replay.add(
            state,
            action,
            next_state,
            reward,
            done,
        )

    def train_step(
        self,
    ) -> dict[str, float] | None:
        if len(self.replay) < self.batch_size:
            return None

        (
            state,
            action,
            next_state,
            reward,
            done,
        ) = self.replay.sample(
            self.batch_size
        )

        # =========================================================
        # SAC target
        # =========================================================
        with torch.no_grad():
            (
                next_action,
                next_logp,
                _,
            ) = self.actor.sample(
                next_state
            )

            next_q1 = self.q1_target(
                next_state,
                next_action,
            )

            next_q2 = self.q2_target(
                next_state,
                next_action,
            )

            next_q = torch.minimum(
                next_q1,
                next_q2,
            )

            target = (
                reward
                + (1.0 - done)
                * self.gamma
                * (
                    next_q
                    - self.alpha.detach()
                    * next_logp
                )
            )

        # =========================================================
        # Critic update
        # =========================================================
        q1 = self.q1(
            state,
            action,
        )

        q2 = self.q2(
            state,
            action,
        )

        q_loss = (
            F.mse_loss(
                q1,
                target,
            )
            + F.mse_loss(
                q2,
                target,
            )
        )

        self.q_optimizer.zero_grad(
            set_to_none=True
        )

        q_loss.backward()
        self.q_optimizer.step()

        # =========================================================
        # Actor update
        # =========================================================
        (
            new_action,
            logp,
            _,
        ) = self.actor.sample(
            state
        )

        q_new = torch.minimum(
            self.q1(
                state,
                new_action,
            ),
            self.q2(
                state,
                new_action,
            ),
        )

        actor_loss = (
            self.alpha.detach()
            * logp
            - q_new
        ).mean()

        self.actor_optimizer.zero_grad(
            set_to_none=True
        )

        actor_loss.backward()
        self.actor_optimizer.step()

        # =========================================================
        # Entropy coefficient update
        # =========================================================
        alpha_loss_value = float("nan")

        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha
                * (
                    logp
                    + self.target_entropy
                ).detach()
            ).mean()

            self.alpha_optimizer.zero_grad(
                set_to_none=True
            )

            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha_loss_value = float(
                alpha_loss
                .detach()
                .cpu()
            )

        # =========================================================
        # Target update
        # =========================================================
        self._soft_update(
            self.q1,
            self.q1_target,
        )

        self._soft_update(
            self.q2,
            self.q2_target,
        )

        self.updates += 1

        # =========================================================
        # Revision diagnostics
        #
        # q_std:
        #   minibatch Q1(s,a) dispersion
        #
        # td_error_std:
        #   raw TD residual dispersion
        #
        # critic_disagreement:
        #   mean |Q1 - Q2|
        #
        # Definitions intentionally match TD3 instrumentation.
        # =========================================================
        td1 = target - q1

        return {
            "critic_loss": float(
                q_loss
                .detach()
                .cpu()
            ),
            "actor_loss": float(
                actor_loss
                .detach()
                .cpu()
            ),
            "alpha_loss": alpha_loss_value,
            "alpha": float(
                self.alpha
                .detach()
                .cpu()
            ),
            "q_mean": float(
                q1
                .detach()
                .mean()
                .cpu()
            ),
            "q_std": float(
                q1
                .detach()
                .std(unbiased=False)
                .cpu()
            ),
            "td_error_mean": float(
                td1
                .detach()
                .abs()
                .mean()
                .cpu()
            ),
            "td_error_std": float(
                td1
                .detach()
                .std(unbiased=False)
                .cpu()
            ),
            "critic_disagreement": float(
                (
                    q1.detach()
                    - q2.detach()
                )
                .abs()
                .mean()
                .cpu()
            ),
        }

    def _soft_update(
        self,
        source,
        target,
    ):
        for p, tp in zip(
            source.parameters(),
            target.parameters(),
        ):
            tp.data.mul_(
                1.0 - self.tau
            ).add_(
                self.tau * p.data
            )

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "updates": self.updates,
        }