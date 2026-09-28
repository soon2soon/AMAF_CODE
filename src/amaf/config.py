from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def save_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(data), f, sort_keys=False, allow_unicode=True)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_path(path: str | Path, anchor: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    anchor = Path(anchor or Path.cwd())
    return (anchor / path).resolve()


def load_config_with_base(path: str | Path) -> dict[str, Any]:
    """Load a run config, optionally merging a `base:` YAML file first."""
    path = Path(path).resolve()
    config = load_yaml(path)
    base_ref = config.pop("base", None)
    if base_ref:
        base_path = resolve_path(base_ref, path.parent)
        base = load_config_with_base(base_path)
        config = deep_merge(base, config)
    config.setdefault("_meta", {})
    config["_meta"]["config_path"] = str(path)
    return config


def apply_seed(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    out = copy.deepcopy(dict(config))
    out["seed"] = int(seed)
    return out


def parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def apply_overrides(config: Mapping[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted overrides such as `training.total_steps=10000`."""
    out = copy.deepcopy(dict(config))
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value: {item}")
        key, raw_value = item.split("=", 1)
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = parse_scalar(raw_value)
    return out
