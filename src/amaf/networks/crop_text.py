from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .discrete import Gate


@dataclass
class CropAMAFDiagnostics:
    weights: torch.Tensor
    heads: torch.Tensor
    fused_advantage: torch.Tensor
    router_weights: torch.Tensor | None = None
    trust: torch.Tensor | None = None


class DistilBertBackbone(nn.Module):
    """DistilBERT feature extractor supporting the historical flattened-token representation."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        token_size: int = 27,
        representation: str = "flatten",
        trainable: bool = True,
    ):
        super().__init__()
        try:
            from transformers import DistilBertModel
        except ImportError as exc:
            raise ImportError("Install crop dependencies with `pip install -e '.[crop]'`") from exc

        self.bert = DistilBertModel.from_pretrained(model_name)
        self.token_size = int(token_size)
        self.representation = representation
        for p in self.bert.parameters():
            p.requires_grad = bool(trainable)
        self.trainable = bool(trainable)

        hidden = int(self.bert.config.hidden_size)
        if representation == "flatten":
            self.output_dim = hidden * self.token_size
        elif representation == "cls":
            self.output_dim = hidden
        elif representation == "mean":
            self.output_dim = hidden
        else:
            raise ValueError(f"Unknown representation: {representation}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.trainable:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            with torch.no_grad():
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if self.representation == "flatten":
            return out.reshape(out.shape[0], -1)
        if self.representation == "cls":
            return out[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        return (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class CropDQN(nn.Module):
    def __init__(self, backbone: DistilBertBackbone, action_dim: int, fc1: int = 512, fc2: int = 256):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(backbone.output_dim, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.out = nn.Linear(fc2, action_dim)

    def forward(self, input_ids, attention_mask):
        z = self.backbone(input_ids, attention_mask)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        return self.out(z)


class CropDuelingDQN(nn.Module):
    def __init__(self, backbone: DistilBertBackbone, action_dim: int, fc1: int = 512, fc2: int = 256):
        super().__init__()
        self.backbone = backbone
        self.fc1 = nn.Linear(backbone.output_dim, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.value = nn.Linear(fc2, 1)
        self.advantage = nn.Linear(fc2, action_dim)

    def forward(self, input_ids, attention_mask):
        z = self.backbone(input_ids, attention_mask)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class CropAMAFDQN(nn.Module):
    def __init__(
        self,
        backbone: DistilBertBackbone,
        action_dim: int,
        fc1: int = 512,
        fc2: int = 256,
        n_heads: int = 4,
        gate_hidden_dim: int = 64,
        gate_mode: str = "adaptive",
        fixed_weights: Sequence[float] | None = None,
        fusion_mode: str = "direct",
        trust_init: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.n_heads = int(n_heads)
        self.fc1 = nn.Linear(backbone.output_dim, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.value = nn.Linear(fc2, 1)
        self.advantage_heads = nn.ModuleList([nn.Linear(fc2, action_dim) for _ in range(n_heads)])
        self.gate = Gate(
            fc2,
            n_heads,
            hidden_dim=gate_hidden_dim,
            mode=gate_mode,
            fixed_weights=fixed_weights,
        )

        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"direct", "uniform_anchored"}:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

        self.trust_gate = None
        if self.fusion_mode == "uniform_anchored":
            trust_init = float(trust_init)
            if not 0.0 < trust_init < 1.0:
                raise ValueError("trust_init must be in (0, 1)")

            self.trust_gate = nn.Sequential(
                nn.Linear(fc2, gate_hidden_dim),
                nn.ReLU(),
                nn.Linear(gate_hidden_dim, 1),
            )

            # tau(s) starts at trust_init for every state.
            final = self.trust_gate[-1]
            nn.init.zeros_(final.weight)
            nn.init.constant_(
                final.bias,
                math.log(trust_init / (1.0 - trust_init)),
            )

    def _features(self, input_ids, attention_mask):
        z = self.backbone(input_ids, attention_mask)
        z = F.relu(self.fc1(z))
        return F.relu(self.fc2(z))

    def forward(self, input_ids, attention_mask, return_diagnostics: bool = False):
        h = self._features(input_ids, attention_mask)
        value = self.value(h)
        heads = torch.stack([head(h) for head in self.advantage_heads], dim=1)  # [B,H,A]

        router_weights = self.gate(h)
        adaptive = (router_weights.unsqueeze(-1) * heads).sum(dim=1)

        trust = None
        if self.fusion_mode == "uniform_anchored":
            uniform = heads.mean(dim=1)
            assert self.trust_gate is not None
            trust = torch.sigmoid(self.trust_gate(h))  # [B,1]

            # A_fused = A_uniform + tau(s) * (A_adaptive - A_uniform)
            fused = uniform + trust * (adaptive - uniform)

            # Effective simplex weights used by the actual fused output.
            weights = (
                (1.0 - trust) / float(self.n_heads)
                + trust * router_weights
            )
        else:
            fused = adaptive
            weights = router_weights

        q = value + fused - fused.mean(dim=1, keepdim=True)

        if return_diagnostics:
            return q, CropAMAFDiagnostics(
                weights=weights,
                heads=heads,
                fused_advantage=fused,
                router_weights=router_weights,
                trust=trust,
            )
        return q


def build_crop_network(kind: str, action_dim: int, cfg: dict, backbone: DistilBertBackbone | None = None):
    enc_cfg = cfg.get("encoder", {})
    if backbone is None:
        backbone = DistilBertBackbone(
            model_name=enc_cfg.get("model_name", "distilbert-base-uncased"),
            token_size=int(enc_cfg.get("token_size", 27)),
            representation=enc_cfg.get("representation", "flatten"),
            trainable=bool(enc_cfg.get("trainable", True)),
        )

    kind = kind.lower()
    if kind == "dqn":
        return CropDQN(backbone, action_dim, int(cfg.get("fc1", 512)), int(cfg.get("fc2", 256)))
    if kind == "dueling_dqn":
        return CropDuelingDQN(backbone, action_dim, int(cfg.get("fc1", 512)), int(cfg.get("fc2", 256)))
    if kind in {"amaf", "amaf_dqn", "amaf_anchored_dqn"}:
        return CropAMAFDQN(
            backbone,
            action_dim,
            int(cfg.get("fc1", 512)),
            int(cfg.get("fc2", 256)),
            int(cfg.get("n_heads", 4)),
            int(cfg.get("gate_hidden_dim", 64)),
            cfg.get("gate_mode", "adaptive"),
            cfg.get("fixed_weights"),
            cfg.get(
                "fusion_mode",
                "uniform_anchored"
                if kind == "amaf_anchored_dqn"
                else "direct",
            ),
            float(cfg.get("trust_init", 0.1)),
        )
    raise ValueError(f"Unknown crop network kind: {kind}")
