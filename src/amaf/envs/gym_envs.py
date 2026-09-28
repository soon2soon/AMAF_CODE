from __future__ import annotations

from typing import Any


def import_gymnasium():
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError("Install Gym dependencies with `pip install -e '.[gym]'`") from exc
    return gym


def make_gym_env(env_id: str, render_mode: str | None = None, **kwargs):
    gym = import_gymnasium()
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)


def reset_env(env: Any, seed: int | None = None):
    if seed is None:
        out = env.reset()
    else:
        out = env.reset(seed=int(seed))
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def step_env(env: Any, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info
    obs, reward, done, info = out
    return obs, float(reward), bool(done), False, info
