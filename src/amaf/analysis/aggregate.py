from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from amaf.analysis.metrics import run_metrics
from amaf.analysis.statistics import confidence_interval, paired_tests
from amaf.config import load_yaml


def _repo_root(exp_path: Path) -> Path:
    if exp_path.parent.name in {"revision", "paper_original"}:
        return exp_path.parents[2]
    return Path.cwd()


def analyze_experiment(experiment_path: str | Path, output_root: str | Path | None = None):
    exp_path = Path(experiment_path).resolve()
    exp = load_yaml(exp_path)
    name = exp.get("name", exp_path.stem)
    repo = _repo_root(exp_path)
    runs_root = Path(output_root or exp.get("output_root", repo / "outputs" / "runs"))
    if not runs_root.is_absolute():
        runs_root = (repo / runs_root).resolve()
    experiment_root = runs_root / name
    summary_root = repo / "outputs" / "summaries" / name
    summary_root.mkdir(parents=True, exist_ok=True)

    threshold = exp.get("analysis", {}).get("threshold")
    final_window = int(exp.get("analysis", {}).get("final_window", 100))
    rows = []
    curve_rows = []
    diagnostic_rows = []

    for algo in exp.get("algorithms", []):
        label = algo["name"]
        for seed in exp.get("seeds", []):
            run_dir = experiment_root / label / f"seed_{int(seed)}"
            if not (run_dir / "COMPLETED").exists():
                continue
            m = run_metrics(run_dir, threshold=threshold, final_window=final_window)
            m.update({"algorithm": label, "seed": int(seed), "run_dir": str(run_dir)})
            rows.append(m)

            eval_path = run_dir / "eval_metrics.csv"
            ep_path = run_dir / "episode_metrics.csv"
            if eval_path.exists() and eval_path.stat().st_size:
                ev = pd.read_csv(eval_path)
                curve = ev.groupby(["eval_index", "env_steps"], as_index=False)["return"].mean()
                for _, r in curve.iterrows():
                    curve_rows.append(
                        {"algorithm": label, "seed": int(seed), "x": r["env_steps"], "return": r["return"], "x_kind": "env_steps", "source": "evaluation"}
                    )
            elif ep_path.exists():
                ep = pd.read_csv(ep_path)
                smooth = ep["train_return"].rolling(final_window, min_periods=1).mean()
                for i, r in ep.iterrows():
                    curve_rows.append(
                        {"algorithm": label, "seed": int(seed), "x": i, "return": smooth.iloc[i], "x_kind": "episode", "source": "training_rolling"}
                    )

            diag_path = run_dir / "diagnostics.csv"
            if diag_path.exists() and diag_path.stat().st_size:
                d = pd.read_csv(diag_path)
                if len(d):
                    diagnostic_rows.append(
                        {
                            "algorithm": label,
                            "seed": int(seed),
                            "gate_entropy_mean": d["gate_entropy"].mean(),
                            "gate_entropy_final": d["gate_entropy"].iloc[-1],
                            "dominant_head_final": d["dominant_head"].iloc[-1],
                        }
                    )

    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(summary_root / "seed_level_metrics.csv", index=False)

    summary_rows = []
    if len(seed_df):
        numeric_cols = [c for c in seed_df.select_dtypes(include=[np.number]).columns if c != "seed"]
        for label, group in seed_df.groupby("algorithm"):
            row: dict[str, Any] = {"algorithm": label, "n_seeds": len(group)}
            for c in numeric_cols:
                vals = group[c].dropna().to_numpy(dtype=float)
                if not len(vals):
                    continue
                lo, hi = confidence_interval(vals)
                row[f"{c}_mean"] = float(np.mean(vals))
                row[f"{c}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row[f"{c}_ci95_low"] = lo
                row[f"{c}_ci95_high"] = hi
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_root / "summary.csv", index=False)

    tests = []
    metric = exp.get("analysis", {}).get("comparison_metric", "final_performance")
    for pair in exp.get("analysis", {}).get("comparison_pairs", []):
        a_name, b_name = pair
        if metric not in seed_df.columns:
            continue
        a = seed_df[seed_df.algorithm == a_name].set_index("seed")[metric]
        b = seed_df[seed_df.algorithm == b_name].set_index("seed")[metric]
        common = a.index.intersection(b.index)
        if len(common):
            result = paired_tests(a.loc[common].to_numpy(), b.loc[common].to_numpy())
            result.update({"algorithm_a": a_name, "algorithm_b": b_name, "metric": metric})
            tests.append(result)
    pd.DataFrame(tests).to_csv(summary_root / "statistical_tests.csv", index=False)

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(summary_root / "learning_curves.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(summary_root / "diagnostics_summary.csv", index=False)
    return summary_root
