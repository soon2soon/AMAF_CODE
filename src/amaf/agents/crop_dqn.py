from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from amaf.buffers.replay import CropReplayBuffer
from amaf.envs.crop import state_to_text
from amaf.networks.crop_text import CropAMAFDQN, build_crop_network


class CropDQNAgent:
    """DQN/Dueling/AMAF agent for the legacy gym-DSSAT text-state setup."""

    def __init__(self, action_dim: int, cfg: dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.kind = cfg["algorithm"]
        self.action_dim = int(action_dim)
        self.net_cfg = cfg.get("network", {})

        try:
            from transformers import BertTokenizerFast
        except ImportError as exc:
            raise ImportError("Install crop dependencies with `pip install -e '.[crop]'`") from exc

        enc_cfg = self.net_cfg.get("encoder", {})
        self.token_size = int(enc_cfg.get("token_size", 27))
        self.tokenizer = BertTokenizerFast.from_pretrained(
            enc_cfg.get("model_name", "distilbert-base-uncased"), use_fast=True
        )

        self.online = build_crop_network(self.kind, action_dim, self.net_cfg).to(device)
        share_target_encoder = bool(cfg.get("share_target_encoder", False))
        if share_target_encoder:
            # Matches the historical notebooks where both QNetwork objects referenced
            # the same global DistilBERT instance.
            self.target = build_crop_network(
                self.kind, action_dim, self.net_cfg, backbone=self.online.backbone
            ).to(device)
            self.target.load_state_dict(self.online.state_dict())
        else:
            self.target = copy.deepcopy(self.online).to(device)
        self.target.eval()

        opt = cfg.get("optimizer", {})
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=float(opt.get("lr", 1e-5)),
            betas=tuple(opt.get("betas", [0.9, 0.999])),
            weight_decay=float(opt.get("weight_decay", 0.0)),
        )
        replay = cfg.get("replay", {})
        self.replay = CropReplayBuffer(int(replay.get("capacity", 100_000)))
        self.batch_size = int(replay.get("batch_size", 512))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.update_every = int(cfg.get("update_every", 16))
        self.target_update_every = int(cfg.get("target_update_every", 8 * self.update_every))
        self.target_mode = cfg.get("target_mode", "dqn")
        self.gradient_clip_value = float(cfg.get("gradient_clip_value", 1.0))
        self.terminal_repeat = int(cfg.get("terminal_repeat", 1))
        self.t_step = 0
        self.updates = 0

    def _tokenize(self, states: list[np.ndarray]):
        texts = [state_to_text(s) for s in states]
        token = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.token_size,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return token["input_ids"].to(self.device), token["attention_mask"].to(self.device)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < float(epsilon):
            return random.randrange(self.action_dim)
        ids, mask = self._tokenize([state])
        self.online.eval()
        with torch.no_grad():
            q = self.online(ids, mask)
        self.online.train()
        return int(q.argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, done):
        repeats = self.terminal_repeat if done else 1
        for _ in range(repeats):
            self.replay.add(state, action, reward, next_state, done)
        # Historical behavior: duplicated terminal transitions did not increment t_step.
        self.t_step += 1
        if self.t_step % self.update_every != 0 or len(self.replay) <= self.batch_size:
            return None
        metrics = self.learn()
        if self.t_step % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return metrics

    def learn(self):
        batch = self.replay.sample(self.batch_size)
        states = [b.state for b in batch]
        next_states = [b.next_state for b in batch]
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        ids, mask = self._tokenize(states)
        next_ids, next_mask = self._tokenize(next_states)
        q_expected = self.online(ids, mask).gather(1, actions)
        with torch.no_grad():
            if self.target_mode == "double_dqn":
                next_action = self.online(next_ids, next_mask).argmax(dim=1, keepdim=True)
                q_next = self.target(next_ids, next_mask).gather(1, next_action)
            elif self.target_mode == "dqn":
                q_next = self.target(next_ids, next_mask).max(dim=1, keepdim=True).values
            else:
                raise ValueError(f"Unknown target_mode: {self.target_mode}")
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        td = q_target - q_expected
        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for p in self.online.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-self.gradient_clip_value, self.gradient_clip_value)
        self.optimizer.step()
        self.updates += 1
        return {
            "critic_loss": float(loss.detach().cpu()),
            "q_mean": float(q_expected.detach().mean().cpu()),
            "q_std": float(q_expected.detach().std(unbiased=False).cpu()),
            "td_error_mean": float(td.detach().abs().mean().cpu()),
            "td_error_std": float(td.detach().std(unbiased=False).cpu()),
        }

    def diagnostics(self, state: np.ndarray):
        if not isinstance(self.online, CropAMAFDQN):
            return None
        ids, mask = self._tokenize([state])
        self.online.eval()
        with torch.no_grad():
            _, diag = self.online(ids, mask, return_diagnostics=True)
        self.online.train()
        weights = diag.weights[0]
        heads = diag.heads[0]

        # `weights` are the effective weights actually used by the final
        # fused advantage. `router_weights` are the raw adaptive-router
        # weights before Uniform anchoring.
        if diag.router_weights is not None:
            router_weights = diag.router_weights[0]
        else:
            router_weights = weights

        if diag.trust is not None:
            adaptive_trust = diag.trust[0].squeeze()
        else:
            adaptive_trust = None

        entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum()

        max_entropy = torch.log(
            torch.tensor(
                float(weights.numel()),
                dtype=weights.dtype,
                device=weights.device,
            )
        )

        normalized_entropy = entropy / max_entropy
        effective_heads = torch.exp(entropy)
        gate_max_weight = weights.max()

        router_entropy = -(
            router_weights
            * torch.log(router_weights.clamp_min(1e-12))
        ).sum()

        router_entropy_norm = router_entropy / max_entropy
        router_effective_heads = torch.exp(router_entropy)
        router_max_weight = router_weights.max()

        # heads: [H, A]
        # Dispersion across heads for each action, averaged over actions.
        head_disagreement = heads.std(dim=0, unbiased=False).mean()

        return {
            "weights": weights.detach().cpu().numpy(),
            "heads": heads.detach().cpu().numpy(),
            "gate_entropy": float(entropy.cpu()),
            "gate_entropy_norm": float(normalized_entropy.cpu()),
            "effective_heads": float(effective_heads.cpu()),
            "gate_max_weight": float(gate_max_weight.cpu()),
            "head_disagreement": float(head_disagreement.cpu()),
            "dominant_head": int(weights.argmax().item()),

            # Anchored-gate mechanism diagnostics.
            "adaptive_trust": (
                float(adaptive_trust.cpu())
                if adaptive_trust is not None
                else float("nan")
            ),
            "router_entropy": float(router_entropy.cpu()),
            "router_entropy_norm": float(router_entropy_norm.cpu()),
            "router_effective_heads": float(router_effective_heads.cpu()),
            "router_max_weight": float(router_max_weight.cpu()),
            "router_dominant_head": int(router_weights.argmax().item()),
            "router_weights": router_weights.detach().cpu().numpy(),
        }

    def state_dict(self):
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "t_step": self.t_step,
            "updates": self.updates,
        }
