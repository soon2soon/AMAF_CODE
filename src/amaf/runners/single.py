from __future__ import annotations

from pathlib import Path
from typing import Any

from amaf.runners.crop import run_crop
from amaf.runners.gym_continuous import run_gym_continuous
from amaf.runners.gym_discrete import run_gym_discrete


def run_single(config: dict[str, Any], run_dir: str | Path, repo_root=None) -> None:
    runner = config.get("runner")
    if runner == "gym_discrete":
        run_gym_discrete(config, run_dir, repo_root=repo_root)
    elif runner == "gym_continuous":
        run_gym_continuous(config, run_dir, repo_root=repo_root)
    elif runner == "crop":
        run_crop(config, run_dir, repo_root=repo_root)
    else:
        raise ValueError(f"Unknown runner: {runner}")
