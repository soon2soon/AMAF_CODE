from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from amaf.buffers.replay import BootstrappedCropReplayBuffer, CropReplayBuffer
from amaf.envs.crop import state_to_text
from amaf.networks.crop_text import DistilBertBackbone


class SimpleQ(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden=(256, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], action_dim)
        )

    def forward(self, x):
        return self.net(x)


class QuantileQ(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, n_quantiles: int = 25, hidden=(256, 128)):
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_quantiles = int(n_quantiles)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], action_dim * n_quantiles)
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.shape[0], self.action_dim, self.n_quantiles)


class CropEnsembleAgent:
    """Refactored historical Bootstrapped-DQN / REDQ / discrete-TQC baselines.

    These implementations intentionally preserve the notebook behavior. They are
    *legacy baselines*, not claims of canonical REDQ/TQC implementations.
    """

    def __init__(self, kind: str, action_dim: int, cfg: dict[str, Any], device: torch.device):
        self.kind = kind
        self.action_dim = int(action_dim)
        self.cfg = cfg
        self.device = device
        enc = cfg.get("network", {}).get("encoder", {})
        try:
            from transformers import BertTokenizerFast
        except ImportError as exc:
            raise ImportError("Install crop dependencies with `pip install -e '.[crop]'`") from exc
        self.token_size = int(enc.get("token_size", 27))
        model_name = enc.get("model_name", "distilbert-base-uncased")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, use_fast=True)
        # Historical ensemble baselines used frozen DistilBERT CLS representations.
        self.encoder = DistilBertBackbone(
            model_name=model_name,
            token_size=self.token_size,
            representation=enc.get("representation", "cls"),
            trainable=bool(enc.get("trainable", False)),
        ).to(device)
        embed_dim = self.encoder.output_dim

        self.n_heads = int(cfg.get("n_heads", 10))
        self.subset = int(cfg.get("subset", 2))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.update_every = int(cfg.get("update_every", 16))
        self.target_update_every = int(cfg.get("target_update_every", 128))
        self.lr = float(cfg.get("optimizer", {}).get("lr", 1e-5))
        replay_cfg = cfg.get("replay", {})
        self.batch_size = int(replay_cfg.get("batch_size", 512))
        capacity = int(replay_cfg.get("capacity", 100_000))
        self.t_step = 0
        self.updates = 0

        if kind == "bootstrapped_dqn":
            self.q = nn.ModuleList([SimpleQ(embed_dim, action_dim).to(device) for _ in range(self.n_heads)])
            self.target_q = copy.deepcopy(self.q)
            self.optimizers = [
                torch.optim.Adam(
                    q.parameters(),
                    lr=self.lr,
                    betas=tuple(cfg.get("optimizer", {}).get("betas", [0.9, 0.999])),
                    weight_decay=float(cfg.get("optimizer", {}).get("weight_decay", 0.001)),
                )
                for q in self.q
            ]
            self.replay = BootstrappedCropReplayBuffer(capacity, self.n_heads, float(cfg.get("mask_prob", 0.5)))
            self.current_head = 0
        elif kind == "redq":
            self.q = nn.ModuleList([SimpleQ(embed_dim, action_dim).to(device) for _ in range(self.n_heads)])
            self.target_q = copy.deepcopy(self.q)
            self.optimizers = [torch.optim.Adam(q.parameters(), lr=self.lr) for q in self.q]
            self.replay = CropReplayBuffer(capacity)
        elif kind == "tqc":
            self.n_quantiles = int(cfg.get("n_quantiles", 25))
            self.quantiles_to_drop = int(cfg.get("quantiles_to_drop", 5))
            self.q = nn.ModuleList(
                [QuantileQ(embed_dim, action_dim, self.n_quantiles).to(device) for _ in range(self.n_heads)]
            )
            self.target_q = copy.deepcopy(self.q)
            self.optimizers = [torch.optim.Adam(q.parameters(), lr=self.lr) for q in self.q]
            self.replay = CropReplayBuffer(capacity)
        else:
            raise ValueError(f"Unknown legacy crop ensemble: {kind}")

    def _encode(self, states: list[np.ndarray]):
        texts = [state_to_text(s) for s in states]
        tok = self.tokenizer(
            texts,
            max_length=self.token_size,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return self.encoder(tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device))

    def begin_episode(self):
        if self.kind == "bootstrapped_dqn":
            self.current_head = int(np.random.randint(self.n_heads))

    def act(self, state: np.ndarray, epsilon: float):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            z = self._encode([state])
            if self.kind == "tqc":
                values = self.q[0](z).mean(dim=-1)
            else:
                idx = self.current_head if self.kind == "bootstrapped_dqn" else 0
                values = self.q[idx](z)
        return int(values.argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)
        self.t_step += 1
        if self.t_step % self.update_every == 0 and len(self.replay) > self.batch_size:
            metrics = self.learn()
        else:
            metrics = None
        if self.t_step % self.target_update_every == 0:
            for target, online in zip(self.target_q, self.q):
                target.load_state_dict(online.state_dict())
        return metrics

    def learn(self):
        batch = self.replay.sample(self.batch_size)
        if self.kind == "bootstrapped_dqn":
            states = [x[0] for x in batch]
            actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = [x[3] for x in batch]
            dones = torch.tensor([x[4] for x in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            masks = torch.tensor(np.stack([x[5] for x in batch]), dtype=torch.float32, device=self.device)
        else:
            states = [x.state for x in batch]
            actions = torch.tensor([x.action for x in batch], dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = [x.next_state for x in batch]
            dones = torch.tensor([x.done for x in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            masks = None

        z = self._encode(states)
        z_next = self._encode(next_states)
        losses = []

        if self.kind == "bootstrapped_dqn":
            for k in range(self.n_heads):
                head_mask = masks[:, k].unsqueeze(1)
                if head_mask.sum() < 1:
                    continue
                expected = self.q[k](z).gather(1, actions)
                with torch.no_grad():
                    next_action = self.q[k](z_next).argmax(dim=1, keepdim=True)
                    next_q = self.target_q[k](z_next).gather(1, next_action)
                    target = rewards + self.gamma * (1.0 - dones) * next_q
                loss = (((expected - target) ** 2) * head_mask).sum() / (head_mask.sum() + 1e-8)
                self.optimizers[k].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q[k].parameters(), 10.0)
                self.optimizers[k].step()
                losses.append(float(loss.detach().cpu()))

        elif self.kind == "redq":
            with torch.no_grad():
                all_next = torch.stack([q(z_next) for q in self.target_q], dim=0)
                min_q = all_next.min(dim=0).values
                target = rewards + self.gamma * (1.0 - dones) * min_q.max(dim=1, keepdim=True).values
            for i in random.sample(range(self.n_heads), self.subset):
                expected = self.q[i](z).gather(1, actions)
                loss = F.mse_loss(expected, target)
                self.optimizers[i].zero_grad(set_to_none=True)
                loss.backward()
                self.optimizers[i].step()
                losses.append(float(loss.detach().cpu()))

        elif self.kind == "tqc":
            with torch.no_grad():
                q_all = torch.stack([q(z_next) for q in self.target_q], dim=0)  # K,B,A,N
                sorted_q = torch.sort(q_all, dim=-1).values
                kept = sorted_q[..., : self.n_quantiles - self.quantiles_to_drop]
                means = kept.mean(dim=-1)
                conservative = means.min(dim=0).values
                target = rewards + self.gamma * (1.0 - dones) * conservative.max(dim=1, keepdim=True).values
            for i in random.sample(range(self.n_heads), self.subset):
                pred = self.q[i](z)
                gather_idx = actions.unsqueeze(-1).expand(-1, -1, self.n_quantiles)
                expected_quantiles = pred.gather(1, gather_idx).squeeze(1)
                loss = F.mse_loss(expected_quantiles.mean(dim=-1, keepdim=True), target)
                self.optimizers[i].zero_grad(set_to_none=True)
                loss.backward()
                self.optimizers[i].step()
                losses.append(float(loss.detach().cpu()))

        self.updates += 1
        return {"critic_loss": float(np.mean(losses)) if losses else float("nan")}

    def diagnostics(self, state):
        return None

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "q": self.q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "t_step": self.t_step,
            "updates": self.updates,
        }
