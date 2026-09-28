from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from amaf.agents.dqn import DQNAgent
from amaf.agents.protected_dqn import ProtectedReferenceDQNAgent
from amaf.envs.gym_envs import make_gym_env, reset_env, step_env
from amaf.runners.common import (
    choose_device,
    diagnostics_to_row,
    epsilon_by_episode,
    finite_or_blank,
    initialize_run,
    epsilon_by_step,
)
from amaf.utils.checkpoint import save_checkpoint
from amaf.utils.seed import seed_action_space, seed_everything


def evaluate_dqn(agent: DQNAgent, env_id: str, episodes: int, seed_base: int, env_kwargs: dict):
    env = make_gym_env(env_id, **env_kwargs)
    returns = []
    rows = []
    for i in range(episodes):
        state, _ = reset_env(env, seed=seed_base + i)
        done = False
        total = 0.0
        length = 0
        while not done:
            action = agent.act(np.asarray(state, dtype=np.float32), epsilon=0.0)
            state, reward, terminated, truncated, _ = step_env(env, action)
            done = terminated or truncated
            total += reward
            length += 1
        returns.append(total)
        rows.append((i, total, length))
    env.close()
    return float(np.mean(returns)), rows


def run_gym_discrete(config: dict[str, Any], run_dir: str | Path, repo_root=None) -> None:
    seed = int(config["seed"])
    deterministic = bool(config.get("deterministic_torch", False))
    seed_everything(seed, deterministic=deterministic)
    device = choose_device(config.get("device", "auto"))

    env_cfg = config["environment"]
    env_id = env_cfg["id"]
    env_kwargs = dict(env_cfg.get("kwargs", {}))
    env = make_gym_env(env_id, **env_kwargs)
    seed_action_space(env, seed)
    state, _ = reset_env(env, seed=seed)
    state_dim = int(np.asarray(state).shape[0])
    action_dim = int(env.action_space.n)

    if config["algorithm"] == "amaf_protected_dqn":
        agent = ProtectedReferenceDQNAgent(
            state_dim, action_dim, config, device
        )
    else:
        agent = DQNAgent(
            state_dim, action_dim, config, device
        )
    loggers = initialize_run(run_dir, config, repo_root)
    run_dir = Path(run_dir)

    train_cfg = config.get("training", {})
    eps_cfg = config.get("epsilon", {})
    max_episodes = int(train_cfg.get("max_episodes", 1000))
    total_steps_limit = int(train_cfg.get("total_steps", 10**12))
    eval_cfg = config.get("evaluation", {})
    eval_enabled = bool(eval_cfg.get("enabled", True))
    eval_interval = int(eval_cfg.get("interval_steps", 10_000))
    eval_episodes = int(eval_cfg.get("episodes", 10))
    train_log_interval = int(config.get("logging", {}).get("train_interval_updates", 100))
    diagnostic_interval = int(config.get("logging", {}).get("diagnostic_interval_steps", 5000))
    checkpoint_interval = int(config.get("checkpoint_interval_steps", 50_000))

    env_steps = 0
    next_eval = eval_interval
    next_diag = diagnostic_interval
    next_ckpt = checkpoint_interval
    best_eval = -float("inf")
    eval_index = 0

    for episode in range(max_episodes):
        if env_steps >= total_steps_limit:
            break
        if episode > 0:
            state, _ = reset_env(env)
        state = np.asarray(state, dtype=np.float32)

        epsilon_schedule = str(eps_cfg.get("schedule", "episode")).lower()

        epsilon = epsilon_by_episode(episode, eps_cfg)
        episode_return = 0.0
        episode_len = 0
        start = time.time()
        done = False

        while not done and env_steps < total_steps_limit:

            if epsilon_schedule == "step_exponential":
                epsilon = epsilon_by_step(env_steps, eps_cfg)

            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = step_env(env, action)
            done = terminated or truncated
            next_state = np.asarray(next_state, dtype=np.float32)
            metrics = agent.observe(state, action, reward, next_state, done)
            env_steps += 1
            episode_return += reward
            episode_len += 1
            state = next_state

            if metrics and agent.updates % train_log_interval == 0:
                loggers.train.log(
                    {
                        "env_steps": env_steps,
                        "updates": agent.updates,
                        **{k: finite_or_blank(v) for k, v in metrics.items()},
                    }
                )

            if env_steps >= next_diag:
                row = diagnostics_to_row(agent.diagnostics(state), env_steps, episode)
                if row:
                    loggers.diagnostics.log(row)
                next_diag += diagnostic_interval

            if eval_enabled and env_steps >= next_eval:
                mean_eval, eval_rows = evaluate_dqn(
                    agent,
                    env_id,
                    eval_episodes,
                    seed_base=seed + 100_000 + eval_index * 1000,
                    env_kwargs=env_kwargs,
                )
                for eval_ep, ret, length in eval_rows:
                    loggers.eval.log(
                        {
                            "env_steps": env_steps,
                            "eval_index": eval_index,
                            "eval_episode": eval_ep,
                            "return": ret,
                            "episode_length": length,
                        }
                    )
                if mean_eval > best_eval:
                    best_eval = mean_eval
                    save_checkpoint(agent.state_dict(), run_dir / "checkpoints" / "best.pt")
                eval_index += 1
                next_eval += eval_interval

            if env_steps >= next_ckpt:
                save_checkpoint(agent.state_dict(), run_dir / "checkpoints" / "latest.pt")
                next_ckpt += checkpoint_interval

        loggers.episode.log(
            {
                "episode": episode,
                "env_steps": env_steps,
                "train_return": episode_return,
                "episode_length": episode_len,
                "epsilon": epsilon,
                "duration_sec": time.time() - start,
            }
        )

    save_checkpoint(agent.state_dict(), run_dir / "checkpoints" / "final.pt")
    (run_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    env.close()
