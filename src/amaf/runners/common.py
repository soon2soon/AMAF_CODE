from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from amaf.config import save_yaml
from amaf.logging.run_logger import RunLoggers
from amaf.utils.io import write_json
from amaf.utils.system import collect_system_info


def choose_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_run(run_dir: str | Path, config: dict[str, Any], repo_root: str | Path | None = None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    save_yaml(config, run_dir / "resolved_config.yaml")
    metadata = collect_system_info(repo_root)
    metadata.update(
        {
            "experiment": config.get("experiment_name"),
            "algorithm": config.get("algorithm"),
            "seed": config.get("seed"),
            "environment": config.get("environment", {}).get("id") or config.get("environment", {}).get("name"),
        }
    )
    write_json(metadata, run_dir / "metadata.json")
    return RunLoggers(run_dir)


def epsilon_by_episode(episode: int, cfg: dict[str, Any]) -> float:
    start = float(cfg.get("start", 1.0))
    end = float(cfg.get("end", 0.01))
    decay = float(cfg.get("decay", 0.995))
    return max(end, start * (decay ** max(episode, 0)))

def epsilon_by_step(env_steps: int, cfg: dict[str, Any]) -> float:
    """Exponential epsilon decay aligned to environment interactions.

    Preserves the configured start/end values while mapping the decay
    over a fixed number of environment steps.
    """
    start = float(cfg.get("start", 1.0))
    end = float(cfg.get("end", 0.01))
    decay_steps = int(cfg.get("decay_steps", 500_000))

    if decay_steps <= 0:
        raise ValueError("epsilon.decay_steps must be positive")
    if start <= 0.0 or end <= 0.0:
        raise ValueError("epsilon start/end must be positive")

    progress = min(max(float(env_steps) / float(decay_steps), 0.0), 1.0)

    return max(
        end,
        start * ((end / start) ** progress),
    )


def diagnostics_to_row(diag: dict | None, env_steps: int, episode: int) -> dict[str, Any] | None:
    if not diag:
        return None
    row: dict[str, Any] = {
        "env_steps": env_steps,
        "episode": episode,
        "gate_entropy": diag.get("gate_entropy", ""),
        "gate_entropy_norm": diag.get("gate_entropy_norm", ""),
        "effective_heads": diag.get("effective_heads", ""),
        "gate_max_weight": diag.get("gate_max_weight", ""),
        "head_disagreement": diag.get("head_disagreement", ""),
        "dominant_head": diag.get("dominant_head", ""),
        "adaptive_trust": diag.get("adaptive_trust", ""),
        "router_entropy": diag.get("router_entropy", ""),
        "router_entropy_norm": diag.get("router_entropy_norm", ""),
        "router_effective_heads": diag.get("router_effective_heads", ""),
        "router_max_weight": diag.get("router_max_weight", ""),
        "router_dominant_head": diag.get("router_dominant_head", ""),
        "reference_q_abs_mean": diag.get("reference_q_abs_mean", ""),
        "correction_abs_mean": diag.get("correction_abs_mean", ""),
        "correction_ratio": diag.get("correction_ratio", ""),
    }
    weights = np.asarray(diag.get("weights", []), dtype=float).reshape(-1)
    router_weights = np.asarray(
        diag.get("router_weights", []), dtype=float
    ).reshape(-1)
    heads = np.asarray(diag.get("heads", []), dtype=float)
    for i in range(min(len(weights), 8)):
        row[f"gate_w{i}"] = float(weights[i])

    for i in range(min(len(router_weights), 8)):
        row[f"router_w{i}"] = float(router_weights[i])
    if heads.ndim >= 1 and heads.size:
        if heads.ndim == 1:
            means = heads
        else:
            means = heads.reshape(heads.shape[0], -1).mean(axis=1)
        for i in range(min(len(means), 8)):
            row[f"head{i}_mean"] = float(means[i])
    return row


def finite_or_blank(value):
    try:
        return value if math.isfinite(float(value)) else ""
    except Exception:
        return ""
