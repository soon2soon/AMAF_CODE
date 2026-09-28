from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument
        return torch.load(Path(path), map_location=map_location)
