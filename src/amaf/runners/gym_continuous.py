from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from amaf.agents.sac import SACAgent
from amaf.agents.td3 import TD3Agent
from amaf.envs.gym_envs import make_gym_env, reset_env, step_env
from amaf.runners.common import (
    choose_device,
    diagnostics_to_row,
    finite_or_blank,
    initialize_run,
)
from amaf.utils.checkpoint import save_checkpoint
from amaf.utils.seed import seed_action_space, seed_everything


def evaluate_continuous(agent, algorithm: str, env_id: str, env_kwargs: dict, episodes: int, seed_base: int):
    env = make_gym_env(env_id, **env_kwargs)
    rows = []
    returns = []
    for ep in range(episodes):
        state, _ = reset_env(env, seed=seed_base + ep)
        done = False
        total = 0.0
        length = 0
        while not done:
            if algorithm == "sac":
                action = agent.select_action(np.asarray(state, dtype=np.float32), deterministic=True)
            else:
                action = agent.select_action(np.asarray(state, dtype=np.float32), noise_std=0.0)
            state, reward, terminated, truncated, _ = step_env(env, action)
            done = terminated or truncated
            total += reward
            length += 1
        returns.append(total)
        rows.append((ep, total, length))
    env.close()
    return float(np.mean(returns)), rows


def run_gym_continuous(config: dict[str, Any], run_dir: str | Path, repo_root=None) -> None:
    seed = int(config["seed"])
    seed_everything(seed, deterministic=bool(config.get("deterministic_torch", False)))
    device = choose_device(config.get("device", "auto"))

    env_cfg = config["environment"]
    env_id = env_cfg["id"]
    env_kwargs = dict(env_cfg.get("kwargs", {}))
    env = make_gym_env(env_id, **env_kwargs)
    seed_action_space(env, seed)
    state, _ = reset_env(env, seed=seed)
    state = np.asarray(state, dtype=np.float32)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    max_action = float(np.asarray(env.action_space.high).max())

    algorithm = config["algorithm"]
    if algorithm == "sac":
        agent = SACAgent(state_dim, action_dim, max_action, config, device)
    else:
        agent = TD3Agent(state_dim, action_dim, max_action, config, device)

    loggers = initialize_run(run_dir, config, repo_root)
    run_dir = Path(run_dir)
    train_cfg = config.get("training", {})
    total_steps_limit = int(train_cfg.get("total_steps", 1_000_000))
    random_steps = int(train_cfg.get("random_steps", 25_000))
    exploration_noise = float(train_cfg.get("exploration_noise", 0.1)) * max_action
    updates_per_step = int(train_cfg.get("updates_per_step", 1))
    treat_truncation_as_done = bool(train_cfg.get("treat_truncation_as_done", False))

    eval_cfg = config.get("evaluation", {})
    eval_interval = int(eval_cfg.get("interval_steps", 10_000))
    eval_episodes = int(eval_cfg.get("episodes", 10))
    eval_enabled = bool(eval_cfg.get("enabled", True))
    log_cfg = config.get("logging", {})
    train_log_interval = int(log_cfg.get("train_interval_updates", 100))
    diagnostic_interval = int(log_cfg.get("diagnostic_interval_steps", 10_000))
    checkpoint_interval = int(config.get("checkpoint_interval_steps", 100_000))

    env_steps = 0
    episode = 0
    episode_return = 0.0
    episode_len = 0
    episode_start = time.time()
    eval_index = 0
    next_eval = eval_interval
    next_diag = diagnostic_interval
    next_ckpt = checkpoint_interval
    best_eval = -float("inf")

    while env_steps < total_steps_limit:
        if env_steps < random_steps:
            action = env.action_space.sample()
        elif algorithm == "sac":
            action = agent.select_action(state, deterministic=False)
        else:
            action = agent.select_action(state, noise_std=exploration_noise)

        next_state, reward, terminated, truncated, _ = step_env(env, action)
        next_state = np.asarray(next_state, dtype=np.float32)
        episode_done = terminated or truncated
        replay_done = episode_done if treat_truncation_as_done else terminated

        diagnostic_state = state.copy()
        agent.add_transition(state, action, next_state, reward, float(replay_done))

        state = next_state
        env_steps += 1
        episode_return += reward
        episode_len += 1

        if env_steps >= random_steps:
            for _ in range(updates_per_step):
                metrics = agent.train_step()
                if metrics:
                    updates = getattr(agent, "total_it", getattr(agent, "updates", 0))
                    if updates % train_log_interval == 0:
                        loggers.train.log(
                            {
                                "env_steps": env_steps,
                                "updates": updates,
                                **{k: finite_or_blank(v) for k, v in metrics.items()},
                            }
                        )

        if env_steps >= next_diag and algorithm != "sac":
            diag = agent.diagnostics(diagnostic_state, action)
            row = diagnostics_to_row(diag, env_steps, episode)
            if row:
                loggers.diagnostics.log(row)
            next_diag += diagnostic_interval

        if eval_enabled and env_steps >= next_eval:
            mean_eval, rows = evaluate_continuous(
                agent,
                algorithm,
                env_id,
                env_kwargs,
                eval_episodes,
                seed_base=seed + 100_000 + eval_index * 1000,
            )
            for eval_ep, ret, length in rows:
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

        if episode_done:
            loggers.episode.log(
                {
                    "episode": episode,
                    "env_steps": env_steps,
                    "train_return": episode_return,
                    "episode_length": episode_len,
                    "duration_sec": time.time() - episode_start,
                }
            )
            episode += 1
            episode_return = 0.0
            episode_len = 0
            episode_start = time.time()
            state, _ = reset_env(env)
            state = np.asarray(state, dtype=np.float32)

    save_checkpoint(agent.state_dict(), run_dir / "checkpoints" / "final.pt")
    (run_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    env.close()
