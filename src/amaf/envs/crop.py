from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def dict2array(state: dict[str, Any]) -> np.ndarray:
    """Historical gym-DSSAT dict -> 25D array conversion, preserving dict order."""
    values: list[float] = []
    for key in state.keys():
        if key == "sw":
            values.extend(list(state[key]))
        else:
            values.append(state[key])
    return np.asarray(values, dtype=np.float32)


def state_to_text(state: np.ndarray) -> str:
    """Exact numeric-to-text transformation used by the original DistilBERT notebooks."""
    tokens: list[str] = []
    for i, num in enumerate(state):
        if i == 0:
            v = round(float(num) / 40)
        elif i == 4:
            v = round(float(num) / 100)
        elif i == 7:
            v = round(float(num) / 10)
        elif i == 20:
            v = round(float(num) / 100)
        elif i == 21:
            v = round(float(num) / 6)
        elif i == 23:
            v = round(float(num))
        elif 9 <= i <= 17:
            v = round(float(num) * 1000)
        elif i == 18:
            v = round(float(num) * 100)
        else:
            v = round(float(num))
        tokens.append(str(v))
    return " ".join(tokens) + " "


def decode_action(action_idx: int, state: np.ndarray | None = None) -> dict[str, int]:
    action_idx = int(action_idx)
    action = {
        "anfer": (action_idx % 5) * 40,
        "amir": int(action_idx / 5) * 6,
    }
    if state is not None:
        # Historical constraints from the notebook.
        if state[0] >= 10000:
            action["anfer"] = 0
        if state[21] >= 1600:
            action["amir"] = 0
    return action


def economic_reward(
    state: np.ndarray,
    nitrogen: float,
    irrigation: float,
    done: bool,
    weights: dict[str, float] | None = None,
) -> float:
    weights = weights or {"k1": 0.158, "k2": 0.79, "k3": 1.1, "k4": 0.0}
    reward = -float(weights["k2"]) * float(nitrogen) - float(weights["k3"]) * float(irrigation)
    if done:
        reward += float(weights["k1"]) * float(state[4])
    return float(reward)


def make_crop_env(env_cfg: dict[str, Any], run_dir: str | Path | None = None):
    """Create the legacy gym-DSSAT environment lazily.

    gym-DSSAT historically depends on Gym 0.21 and system-level PDI/DSSAT packages,
    so this import is deliberately isolated from the Gymnasium runners.
    """
    try:
        import gym
        import gym_dssat_pdi  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "gym-DSSAT is not available in this Python environment. Use the crop machine "
            "with the existing PDI/DSSAT installation; see docs/CROP_ENVIRONMENT.md."
        ) from exc

    cfg = dict(env_cfg)
    if run_dir is not None:
        log_path = Path(run_dir) / "dssat-pdi.log"
        cfg["log_saving_path"] = str(log_path)
    return gym.make("gym_dssat_pdi:GymDssatPdi-v0", **cfg)


def reset_crop_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        out = out[0]
    return dict2array(out)


def step_crop_env(env, action: dict[str, int]):
    out = env.step(action)
    if len(out) == 5:
        obs, raw_reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, raw_reward, done, info = out
    return obs, raw_reward, bool(done), info
