from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from amaf.config import apply_overrides, apply_seed, deep_merge, load_config_with_base, load_yaml
from amaf.runners.single import run_single


def _resolve_algorithm_config(exp_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    cfg_path = (exp_path.parent / entry["config"]).resolve()
    config = load_config_with_base(cfg_path)
    config = deep_merge(config, entry.get("overrides", {}))
    return config


def plan_experiment(
    experiment_path: str | Path,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
    cli_overrides: list[str] | None = None,
):
    exp_path = Path(experiment_path).resolve()
    exp = load_yaml(exp_path)
    name = exp.get("name", exp_path.stem)
    seeds = [int(s) for s in exp.get("seeds", [])]
    if seed_filter is not None:
        seeds = [s for s in seeds if s == int(seed_filter)]
    jobs = []
    for entry in exp.get("algorithms", []):
        label = entry["name"]
        if algorithm_filter and label != algorithm_filter:
            continue
        base = _resolve_algorithm_config(exp_path, entry)
        for seed in seeds:
            cfg = apply_seed(base, seed)
            cfg["experiment_name"] = name
            cfg["algorithm_label"] = label
            cfg = apply_overrides(cfg, cli_overrides)
            jobs.append((name, label, seed, cfg))
    return exp, jobs


def run_experiment(
    experiment_path: str | Path,
    algorithm_filter: str | None = None,
    seed_filter: int | None = None,
    dry_run: bool = False,
    skip_completed: bool = True,
    output_root: str | Path | None = None,
    cli_overrides: list[str] | None = None,
):
    exp_path = Path(experiment_path).resolve()
    exp, jobs = plan_experiment(exp_path, algorithm_filter, seed_filter, cli_overrides)
    repo_root = exp_path.parents[2] if exp_path.parent.name in {"revision", "paper_original"} else Path.cwd()
    root = Path(output_root or exp.get("output_root", repo_root / "outputs" / "runs"))
    if not root.is_absolute():
        root = (repo_root / root).resolve()

    print(f"Experiment: {exp.get('name', exp_path.stem)}")
    print(f"Jobs: {len(jobs)}")
    failures = []
    for name, label, seed, cfg in jobs:
        run_dir = root / name / label / f"seed_{seed}"
        print(f"\n[{label} | seed={seed}] -> {run_dir}")
        if dry_run:
            continue
        if skip_completed and (run_dir / "COMPLETED").exists():
            print("  skip: COMPLETED")
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        failed_marker = run_dir / "FAILED.txt"
        if failed_marker.exists():
            failed_marker.unlink()
        try:
            run_single(cfg, run_dir, repo_root=repo_root)
        except Exception as exc:
            tb = traceback.format_exc()
            failed_marker.write_text(tb, encoding="utf-8")
            failures.append((label, seed, str(exc)))
            traceback.print_exc()
            print(f"  FAILED: {exc}")
            continue
    return failures
