from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from amaf.agents.crop_dqn import CropDQNAgent
from amaf.agents.crop_ensemble import CropEnsembleAgent
from amaf.envs.crop import (
    decode_action,
    economic_reward,
    make_crop_env,
    reset_crop_env,
    step_crop_env,
)
from amaf.runners.common import diagnostics_to_row, epsilon_by_episode, finite_or_blank, initialize_run, choose_device
from amaf.utils.checkpoint import save_checkpoint
from amaf.utils.seed import seed_everything


def _resolved_env_cfg(config: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = dict(config["environment"].get("kwargs", {}))
    mode = config["environment"].get("seed_mode", "fixed")
    if mode == "run_seed":
        base = int(config["environment"].get("seed_base", 123456))
        cfg["seed"] = base + int(seed)
    elif mode == "fixed":
        cfg.setdefault("seed", int(config["environment"].get("seed_base", 123456)))
    else:
        raise ValueError(f"Unknown crop environment seed_mode: {mode}")
    return cfg


def run_crop(config: dict[str, Any], run_dir: str | Path, repo_root=None) -> None:
    seed = int(config["seed"])
    seed_everything(seed, deterministic=bool(config.get("deterministic_torch", False)))
    device = choose_device(config.get("device", "auto"))
    run_dir = Path(run_dir)
    loggers = initialize_run(run_dir, config, repo_root)

    env_cfg = _resolved_env_cfg(config, seed)
    env = make_crop_env(env_cfg, run_dir=run_dir)
    state = reset_crop_env(env)
    action_dim = int(config.get("action_dim", 25))
    algorithm = config["algorithm"]
    if algorithm in {
        "dqn",
        "dueling_dqn",
        "amaf_dqn",
        "amaf_anchored_dqn",
    }:
        agent = CropDQNAgent(action_dim, config, device)
    elif algorithm in {"bootstrapped_dqn", "redq", "tqc"}:
        agent = CropEnsembleAgent(algorithm, action_dim, config, device)
    else:
        raise ValueError(f"Unsupported crop algorithm: {algorithm}")

    train_cfg = config.get("training", {})
    max_episodes = int(train_cfg.get("max_episodes", 3000))
    max_steps = int(train_cfg.get("max_steps_per_episode", 500))
    eps_cfg = config.get("epsilon", {})
    reward_weights = config.get("reward_weights", {"k1": 0.158, "k2": 0.79, "k3": 1.1, "k4": 0.0})
    log_cfg = config.get("logging", {})
    diag_interval_ep = int(log_cfg.get("diagnostic_interval_episodes", 25))
    train_log_interval = int(log_cfg.get("train_interval_updates", 20))
    checkpoint_interval_ep = int(config.get("checkpoint_interval_episodes", 250))
    best_score = -float("inf")
    global_steps = 0

    for episode in range(max_episodes):
        if episode > 0:
            state = reset_crop_env(env)
        if hasattr(agent, "begin_episode"):
            agent.begin_episode()
        epsilon = epsilon_by_episode(episode, eps_cfg)
        score = 0.0
        nitrogen_total = 0.0
        irrigation_total = 0.0
        yield_value = 0.0
        episode_len = 0
        start = time.time()

        for _ in range(max_steps):
            action_idx = agent.act(state, epsilon)
            action = decode_action(action_idx, state)
            next_obs, _, done, _ = step_crop_env(env, action)

            if done:
                yield_value = float(state[4])
                next_state = state.copy()  # exact historical terminal convention
                reward = economic_reward(
                    state, action["anfer"], action["amir"], True, reward_weights
                )
            else:
                from amaf.envs.crop import dict2array
                next_state = dict2array(next_obs)
                reward = economic_reward(
                    state, action["anfer"], action["amir"], False, reward_weights
                )

            metrics = agent.observe(state, action_idx, reward, next_state, done)
            global_steps += 1
            episode_len += 1
            score += reward

            # Historical notebooks only accumulated resource counters on non-terminal steps.
            if not done:
                nitrogen_total += action["anfer"]
                irrigation_total += action["amir"]
                state = next_state

            updates = int(getattr(agent, "updates", 0))
            if metrics and updates and updates % train_log_interval == 0:
                loggers.train.log(
                    {
                        "env_steps": global_steps,
                        "updates": updates,
                        **{k: finite_or_blank(v) for k, v in metrics.items()},
                    }
                )
            if done:
                break

        if algorithm in {"amaf_dqn", "amaf_anchored_dqn"} and (
            episode + 1
        ) % diag_interval_ep == 0:
            row = diagnostics_to_row(agent.diagnostics(state), global_steps, episode)
            if row:
                loggers.diagnostics.log(row)

        loggers.episode.log(
            {
                "episode": episode,
                "env_steps": global_steps,
                "train_return": score,
                "episode_length": episode_len,
                "epsilon": epsilon,
                "duration_sec": time.time() - start,
                "nitrogen": nitrogen_total,
                "irrigation": irrigation_total,
                "yield": yield_value,
            }
        )

        if score > best_score:
            best_score = score
            save_checkpoint(agent.state_dict(), run_dir / "checkpoints" / "best.pt")
        if (episode + 1) % checkpoint_interval_ep == 0:
            save_checkpoint(agent.state_dict(), run_dir / "checkpoints" / "latest.pt")

    save_checkpoint(agent.state_dict(), run_dir / "checkpoints" / "final.pt")
    (run_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    try:
        env.close()
    except Exception:
        pass
