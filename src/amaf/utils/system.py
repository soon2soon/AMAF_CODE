from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def _git_commit(repo_root: Path | None = None) -> str | None:
    try:
        cwd = str(repo_root) if repo_root else None
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def collect_system_info(repo_root: str | Path | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gymnasium": _version("gymnasium"),
        "gym": _version("gym"),
        "transformers": _version("transformers"),
        "scipy": _version("scipy"),
        "git_commit": _git_commit(Path(repo_root) if repo_root else None),
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    else:
        info["gpu_count"] = 0
        info["gpus"] = []
    return info
