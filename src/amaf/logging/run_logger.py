from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._initialized = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict[str, Any]) -> None:
        cleaned = {name: row.get(name, "") for name in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(cleaned)


EPISODE_FIELDS = [
    "episode",
    "env_steps",
    "train_return",
    "episode_length",
    "epsilon",
    "duration_sec",
    "nitrogen",
    "irrigation",
    "yield",
]

EVAL_FIELDS = ["env_steps", "eval_index", "eval_episode", "return", "episode_length"]

TRAIN_FIELDS = [
    "env_steps",
    "updates",
    "critic_loss",
    "reference_loss",
    "adaptive_loss",
    "actor_loss",
    "actor_base_grad_norm",
    "actor_residual_grad_norm",
    "actor_grad_cosine",
    "actor_conflict",
    "actor_projection_applied",
    "alpha_loss",
    "alpha",
    "q_mean",
    "q_std",
    "td_error_mean",
    "td_error_std",
    "critic_disagreement",
    "grad_norm",
    "reference_grad_norm",
    "adaptive_grad_norm",
]

DIAGNOSTIC_FIELDS = [
    "env_steps",
    "episode",
    "gate_entropy",
    "gate_entropy_norm",
    "effective_heads",
    "gate_max_weight",
    "head_disagreement",
    "dominant_head",
    "adaptive_trust",
    "router_entropy",
    "router_entropy_norm",
    "router_effective_heads",
    "router_max_weight",
    "router_dominant_head",
    "reference_q_abs_mean",
    "correction_abs_mean",
    "correction_ratio",
    "gate_w0",
    "gate_w1",
    "gate_w2",
    "gate_w3",
    "gate_w4",
    "gate_w5",
    "gate_w6",
    "gate_w7",
    "router_w0",
    "router_w1",
    "router_w2",
    "router_w3",
    "router_w4",
    "router_w5",
    "router_w6",
    "router_w7",
    "head0_mean",
    "head1_mean",
    "head2_mean",
    "head3_mean",
    "head4_mean",
    "head5_mean",
    "head6_mean",
    "head7_mean",
]


class RunLoggers:
    def __init__(self, run_dir: str | Path):
        run_dir = Path(run_dir)
        self.episode = CSVLogger(run_dir / "episode_metrics.csv", EPISODE_FIELDS)
        self.eval = CSVLogger(run_dir / "eval_metrics.csv", EVAL_FIELDS)
        self.train = CSVLogger(run_dir / "train_metrics.csv", TRAIN_FIELDS)
        self.diagnostics = CSVLogger(run_dir / "diagnostics.csv", DIAGNOSTIC_FIELDS)
