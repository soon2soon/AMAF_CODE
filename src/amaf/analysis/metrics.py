from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(x[-1] - x[0]) <= 0:
        return float("nan")
    if hasattr(np, "trapezoid"):
        trap = np.trapezoid
    else:
        trap = np.trapz
    return float(trap(y, x) / (x[-1] - x[0]))


def first_threshold(x: np.ndarray, y: np.ndarray, threshold: float | None) -> float:
    if threshold is None:
        return float("nan")
    idx = np.flatnonzero(y >= float(threshold))
    return float(x[idx[0]]) if len(idx) else float("nan")


def run_metrics(run_dir: str | Path, threshold: float | None = None, final_window: int = 100) -> dict[str, Any]:
    run_dir = Path(run_dir)
    episode_path = run_dir / "episode_metrics.csv"
    eval_path = run_dir / "eval_metrics.csv"
    result: dict[str, Any] = {}

    if eval_path.exists() and eval_path.stat().st_size > 0:
        ev = pd.read_csv(eval_path)
        if len(ev):
            curve = ev.groupby(["eval_index", "env_steps"], as_index=False)["return"].mean()
            x = curve["env_steps"].to_numpy(dtype=float)
            y = curve["return"].to_numpy(dtype=float)
            result.update(
                {
                    "metric_source": "evaluation",
                    "final_performance": float(y[-1]),
                    "final_eval_mean": float(y[-1]),
                    "auc": normalized_auc(x, y),
                    "peak": float(np.max(y)),
                    "time_to_threshold": first_threshold(x, y, threshold),
                    "eval_points": len(y),
                }
            )
            final_idx = ev["eval_index"].max()
            final_returns = ev.loc[ev["eval_index"] == final_idx, "return"]
            result["final_policy_std_across_eval_episodes"] = float(final_returns.std(ddof=1)) if len(final_returns) > 1 else 0.0

    if episode_path.exists() and episode_path.stat().st_size > 0:
        ep = pd.read_csv(episode_path)
        if len(ep):
            returns = ep["train_return"].to_numpy(dtype=float)
            steps = ep["env_steps"].to_numpy(dtype=float)
            fw = returns[-min(final_window, len(returns)) :]
            result.setdefault("metric_source", "training")
            result.setdefault("final_performance", float(np.mean(fw)))
            result.setdefault("auc", normalized_auc(steps, returns))
            result.setdefault("peak", float(np.max(returns)))
            result.setdefault("time_to_threshold", first_threshold(steps, returns, threshold))
            result.update(
                {
                    "episodes": len(ep),
                    "total_env_steps": float(steps[-1]),
                    "final_window_mean": float(np.mean(fw)),
                    "final_window_std": float(np.std(fw, ddof=1)) if len(fw) > 1 else 0.0,
                    "peak_train_return": float(np.max(returns)),
                }
            )
            if "yield" in ep and ep["yield"].notna().any():
                peak_idx = ep["train_return"].idxmax()
                result["yield_at_peak"] = float(ep.loc[peak_idx, "yield"])
                result["nitrogen_at_peak"] = float(ep.loc[peak_idx, "nitrogen"])
                result["irrigation_at_peak"] = float(ep.loc[peak_idx, "irrigation"])
                result["yield_final_window"] = float(ep["yield"].tail(final_window).mean())
                result["nitrogen_final_window"] = float(ep["nitrogen"].tail(final_window).mean())
                result["irrigation_final_window"] = float(ep["irrigation"].tail(final_window).mean())
    return result
