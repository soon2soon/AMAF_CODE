#!/usr/bin/env python3
"""Prepare validated LunarLander manuscript figure data for F1 through F4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import FigureDataRegistry
from .stats import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE,
    DEFAULT_RNG_SEED,
    summarize_curve_by_seed,
    summarize_paired_frame,
    summarize_sample,
)
from .validation import ROOT


EXPECTED_SEEDS = frozenset({11, 22, 33, 44, 55, 66, 77, 88, 99, 111})
EXPECTED_STEPS = np.arange(10_000, 500_001, 10_000, dtype=int)
EXPECTED_EVAL_EPISODES = 10

F1_METHOD_DIRECTORIES = {
    "Uniform": "amaf_uniform",
    "Naive": "amaf_naive",
    "Protected": "amaf_protected",
    "DQN": "dqn",
    "Dueling": "dueling_dqn",
}

FIGURE_DATA_DIR = ROOT / "analysis" / "figures" / "figure_data"
F1_DATA_PATH = FIGURE_DATA_DIR / "lunar_f1_learning.csv"
F2_DATA_PATH = FIGURE_DATA_DIR / "lunar_f2_failure_rescue.csv"
F2_DELTA_BAR_DATA_PATH = FIGURE_DATA_DIR / "lunar_f2_delta_bars.csv"
F3_DATA_PATH = FIGURE_DATA_DIR / "lunar_f3_protected_vs_uniform.csv"
F4_SUMMARY_DATA_PATH = FIGURE_DATA_DIR / "lunar_f4_method_summary.csv"
F4_POINTS_DATA_PATH = FIGURE_DATA_DIR / "lunar_f4_method_points.csv"
FIGURE_OUTPUT_DIR = ROOT / "paper_figures" / "gnuplot" / "lunar"

F4_METHOD_ORDER = ("DQN", "Dueling", "Uniform", "Naive", "Protected")


def _load_seed_curve(path: Path, seed: int) -> pd.DataFrame:
    """Reduce evaluation episodes to one checkpoint mean per seed."""

    frame = pd.read_csv(path)
    required = {"env_steps", "eval_index", "eval_episode", "return"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {', '.join(sorted(missing))}"
        )
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path} contains missing evaluation values.")

    duplicate = frame.duplicated(
        subset=["env_steps", "eval_index", "eval_episode"],
        keep=False,
    )
    if duplicate.any():
        raise ValueError(f"{path} contains duplicate evaluation episodes.")

    counts = frame.groupby("env_steps", sort=True).size()
    if not (counts == EXPECTED_EVAL_EPISODES).all():
        raise ValueError(
            f"{path} must contain {EXPECTED_EVAL_EPISODES} episodes per checkpoint."
        )
    if not np.array_equal(counts.index.to_numpy(dtype=int), EXPECTED_STEPS):
        raise ValueError(f"{path} does not cover every 10k checkpoint through 500k.")

    curve = (
        frame.groupby("env_steps", as_index=False, sort=True)["return"]
        .mean()
        .rename(columns={"return": "evaluation_return"})
    )
    curve.insert(0, "seed", int(seed))
    return curve


def load_lunar_frozen_curves(
    registry: FigureDataRegistry | None = None,
) -> pd.DataFrame:
    """Load only manuscript-approved Lunar primary evidence for F1."""

    reg = registry or FigureDataRegistry()
    dataset_id = "lunar_primary_frozen"
    if reg.dataset_kind(dataset_id) != "frozen_dir":
        raise ValueError(f"{dataset_id} is not registered as a frozen directory.")
    if reg.dataset_lifecycle(dataset_id) != "frozen":
        raise ValueError(f"{dataset_id} does not have frozen lifecycle.")

    frozen_root = reg.load_path(dataset_id, manuscript=True)
    frames: list[pd.DataFrame] = []
    for method, method_directory in F1_METHOD_DIRECTORIES.items():
        method_root = frozen_root / method_directory
        observed_seeds = {
            int(path.name.removeprefix("seed_"))
            for path in method_root.glob("seed_*")
            if path.is_dir() and path.name.removeprefix("seed_").isdigit()
        }
        if observed_seeds != EXPECTED_SEEDS:
            raise ValueError(f"Unexpected {method} seeds: {sorted(observed_seeds)}")

        for seed in sorted(EXPECTED_SEEDS):
            eval_path = method_root / f"seed_{seed}" / "eval_metrics.csv"
            if not eval_path.is_file():
                raise FileNotFoundError(eval_path)
            curve = _load_seed_curve(eval_path, seed)
            curve.insert(0, "method", method)
            frames.append(curve)

    frozen = pd.concat(frames, ignore_index=True)
    expected_pairs = {
        (method, seed)
        for method in F1_METHOD_DIRECTORIES
        for seed in EXPECTED_SEEDS
    }
    observed_pairs = set(
        frozen[["method", "seed"]].itertuples(index=False, name=None)
    )
    if observed_pairs != expected_pairs:
        raise ValueError("Lunar curves do not contain all matched method-seed pairs.")
    return frozen


def _assert_close(label: str, observed: float, expected: float) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=0.02):
        raise ValueError(f"{label} fingerprint changed: {observed} != {expected}")


def _paired_summary(
    canonical: pd.DataFrame,
    *,
    metric: str,
    comparator: str,
):
    return summarize_paired_frame(
        canonical,
        seed_col="seed",
        method_col="method",
        metric_col=metric,
        focal="Protected",
        comparator=comparator,
        direction="higher",
        confidence=DEFAULT_CONFIDENCE,
        resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        rng_seed=DEFAULT_RNG_SEED,
    )


def _validate_numerical_fingerprints(
    registry: FigureDataRegistry,
    canonical: pd.DataFrame,
) -> None:
    """Validate known group and paired Lunar results before export."""

    expected_means = {
        "DQN": (223.17, 81.13),
        "Dueling": (223.06, 77.21),
        "Uniform": (252.94, 90.52),
        "Naive": (163.33, 59.66),
        "Protected": (247.89, 89.26),
    }
    means = canonical.groupby("method")[["last5", "retention_pct"]].mean()
    for method, (last5, retention) in expected_means.items():
        _assert_close(f"{method} Last-5 mean", means.loc[method, "last5"], last5)
        _assert_close(
            f"{method} retention mean",
            means.loc[method, "retention_pct"],
            retention,
        )

    expected_paired = {
        ("last5", "Naive"): (84.56, 5.82, 6),
        ("last5", "Uniform"): (-5.05, -8.09, 3),
        ("retention_pct", "Naive"): (29.60, 1.76, 5),
        ("retention_pct", "Uniform"): (-1.26, -1.55, 5),
    }
    paired_canonical = registry.load_csv("lunar_paired_deltas", manuscript=True)
    for (metric, comparator), (mean, median, wins) in expected_paired.items():
        _, summary = _paired_summary(
            canonical,
            metric=metric,
            comparator=comparator,
        )
        _assert_close(
            f"Protected-{comparator} {metric} mean",
            summary.mean_delta_focal_minus_comparator,
            mean,
        )
        _assert_close(
            f"Protected-{comparator} {metric} median",
            summary.median_delta_focal_minus_comparator,
            median,
        )
        if summary.wins != wins or summary.n != 10:
            raise ValueError(f"Protected-{comparator} {metric} paired counts changed.")

        rows = paired_canonical.loc[
            (paired_canonical["focal"] == "Protected")
            & (paired_canonical["comparator"] == comparator)
            & (paired_canonical["metric"] == metric)
        ]
        if len(rows) != 1:
            raise ValueError(f"No unique canonical paired row for {metric}/{comparator}.")
        row = rows.iloc[0]
        if int(row["wins"]) != summary.wins or int(row["n"]) != summary.n:
            raise ValueError(f"Canonical paired counts disagree for {metric}/{comparator}.")
        _assert_close(
            f"Canonical paired mean {metric}/{comparator}",
            float(row["mean_delta_focal_minus_comparator"]),
            summary.mean_delta_focal_minus_comparator,
        )


def build_f1_learning(
    registry: FigureDataRegistry,
    frozen: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """Build the wide renderer-ready F1 learning-curve table."""

    summaries = []
    for method in F1_METHOD_DIRECTORIES:
        summary = summarize_curve_by_seed(
            frozen.loc[frozen["method"] == method],
            seed_col="seed",
            x_col="env_steps",
            value_col="evaluation_return",
            confidence=DEFAULT_CONFIDENCE,
            resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            rng_seed=DEFAULT_RNG_SEED,
        )
        if len(summary) != len(EXPECTED_STEPS):
            raise ValueError(f"{method} does not have 50 summarized checkpoints.")
        if set(summary["n_seeds"].astype(int)) != {len(EXPECTED_SEEDS)}:
            raise ValueError(f"{method} does not have 10 seeds at every checkpoint.")

        prefix = method.lower()
        summaries.append(summary[[
            "env_steps",
            "mean",
            "ci95_low",
            "ci95_high",
        ]].rename(columns={
            "mean": f"{prefix}_mean",
            "ci95_low": f"{prefix}_ci_low",
            "ci95_high": f"{prefix}_ci_high",
        }))

    learning = summaries[0]
    for summary in summaries[1:]:
        learning = learning.merge(summary, on="env_steps", validate="one_to_one")
    if not np.array_equal(learning["env_steps"].to_numpy(dtype=int), EXPECTED_STEPS):
        raise ValueError("F1 method checkpoint coverage is not identical.")

    endpoint = frozen.loc[frozen["env_steps"] == 500_000]
    for method in F1_METHOD_DIRECTORIES:
        frozen_values = endpoint.loc[endpoint["method"] == method].set_index("seed")
        canonical_values = canonical.loc[canonical["method"] == method].set_index("seed")
        if not np.allclose(
            frozen_values.loc[sorted(EXPECTED_SEEDS), "evaluation_return"],
            canonical_values.loc[sorted(EXPECTED_SEEDS), "final"],
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"{method} frozen endpoint disagrees with canonical final.")
    return learning


def build_f2_failure_rescue(canonical: pd.DataFrame) -> pd.DataFrame:
    """Build a 10-seed by 6-metric renderer-ready matrix."""

    subset = canonical.loc[
        canonical["method"].isin(["Naive", "Protected"]),
        ["method", "seed", "last5", "retention_pct"],
    ]
    if subset.duplicated(["method", "seed"]).any():
        raise ValueError("Duplicate Lunar method-seed rows detected.")

    last5 = subset.pivot(index="seed", columns="method", values="last5")
    retention = subset.pivot(index="seed", columns="method", values="retention_pct")
    if set(last5.index.astype(int)) != EXPECTED_SEEDS or last5.isna().any().any():
        raise ValueError("F2 Last-5 pairing is incomplete.")
    if set(retention.index.astype(int)) != EXPECTED_SEEDS or retention.isna().any().any():
        raise ValueError("F2 retention pairing is incomplete.")

    metric_specs = (
        ("naive_last5", 1, last5["Naive"], False),
        ("protected_last5", 2, last5["Protected"], False),
        ("last5_delta", 3, last5["Protected"] - last5["Naive"], True),
        ("naive_retention_pct", 4, retention["Naive"], False),
        ("protected_retention_pct", 5, retention["Protected"], False),
        ("retention_delta_pp", 6, retention["Protected"] - retention["Naive"], True),
    )

    rows = []
    ordered_seeds = sorted(EXPECTED_SEEDS)
    seed_y = {seed: len(ordered_seeds) - index for index, seed in enumerate(ordered_seeds)}
    for metric, metric_x, values, show_sign in metric_specs:
        scale = float(np.max(np.abs(values.to_numpy(dtype=float))))
        if scale == 0:
            raise ValueError(f"Cannot color-normalize constant-zero metric {metric}.")
        for seed in ordered_seeds:
            value = float(values.loc[seed])
            rows.append({
                "seed": seed,
                "seed_y": seed_y[seed],
                "metric": metric,
                "metric_x": metric_x,
                "value": value,
                "display": f"{value:+.0f}" if show_sign else f"{value:.0f}",
                "color_score": value / scale,
                "x_low": metric_x - 0.48,
                "x_high": metric_x + 0.48,
                "y_low": seed_y[seed] - 0.46,
                "y_high": seed_y[seed] + 0.46,
            })

    matrix = pd.DataFrame(rows)
    if len(matrix) != 60 or matrix["seed"].nunique() != 10:
        raise ValueError("F2 matrix must contain 60 cells across 10 seeds.")
    return matrix


def build_f2_delta_bars(matrix: pd.DataFrame) -> pd.DataFrame:
    """Build the seed-ordered paired deltas used by the F2 bar alternative."""

    last5 = matrix.loc[
        matrix["metric"] == "last5_delta", ["seed", "value"]
    ].rename(columns={"value": "delta_last5"})
    retention = matrix.loc[
        matrix["metric"] == "retention_delta_pp", ["seed", "value"]
    ].rename(columns={"value": "delta_retention_pp"})
    bars = last5.merge(retention, on="seed", validate="one_to_one")
    bars = bars.set_index("seed").loc[sorted(EXPECTED_SEEDS)].reset_index()

    if len(bars) != 10 or bars["seed"].tolist() != sorted(EXPECTED_SEEDS):
        raise ValueError("F2 delta bars must preserve the canonical 10-seed order.")
    if not np.isfinite(bars[["delta_last5", "delta_retention_pp"]]).all().all():
        raise ValueError("F2 delta bars contain non-finite values.")
    return bars


def build_f3_protected_vs_uniform(canonical: pd.DataFrame) -> pd.DataFrame:
    """Build paired Protected-minus-Uniform values for every canonical seed."""

    subset = canonical.loc[
        canonical["method"].isin(["Uniform", "Protected"]),
        ["method", "seed", "last5", "retention_pct"],
    ]
    if subset.duplicated(["method", "seed"]).any():
        raise ValueError("Duplicate F3 method-seed rows detected.")

    last5 = subset.pivot(index="seed", columns="method", values="last5")
    retention = subset.pivot(index="seed", columns="method", values="retention_pct")
    ordered_seeds = sorted(EXPECTED_SEEDS)
    if set(last5.index.astype(int)) != EXPECTED_SEEDS or last5.isna().any().any():
        raise ValueError("F3 Last-5 pairing is incomplete.")
    if set(retention.index.astype(int)) != EXPECTED_SEEDS or retention.isna().any().any():
        raise ValueError("F3 retention pairing is incomplete.")

    frame = pd.DataFrame({
        "seed": ordered_seeds,
        "uniform_last5": last5.loc[ordered_seeds, "Uniform"].to_numpy(),
        "protected_last5": last5.loc[ordered_seeds, "Protected"].to_numpy(),
        "uniform_retention": retention.loc[ordered_seeds, "Uniform"].to_numpy(),
        "protected_retention": retention.loc[ordered_seeds, "Protected"].to_numpy(),
    })
    frame.insert(
        3,
        "delta_last5",
        frame["protected_last5"] - frame["uniform_last5"],
    )
    frame["delta_retention_pp"] = (
        frame["protected_retention"] - frame["uniform_retention"]
    )

    checks = (
        ("last5", "delta_last5", -5.05, -8.09, 3),
        ("retention_pct", "delta_retention_pp", -1.26, -1.55, 5),
    )
    for metric, column, expected_mean, expected_median, expected_wins in checks:
        paired, summary = _paired_summary(canonical, metric=metric, comparator="Uniform")
        _assert_close(f"F3 {metric} mean delta", frame[column].mean(), expected_mean)
        _assert_close(f"F3 {metric} median delta", frame[column].median(), expected_median)
        if int((frame[column] > 0).sum()) != expected_wins:
            raise ValueError(f"F3 {metric} wins changed.")
        if not np.allclose(
            frame[column],
            paired.set_index("seed").loc[
                ordered_seeds, "delta_focal_minus_comparator"
            ],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"F3 {metric} paired deltas changed.")
    return frame


def build_f4_method_data(
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build raw seed points and seed-bootstrap summaries for F4."""

    required = {"method", "seed", "last5", "retention_pct"}
    missing = required - set(canonical.columns)
    if missing:
        raise ValueError(f"F4 canonical data missing columns: {sorted(missing)}")

    method_y = {
        method: len(F4_METHOD_ORDER) - index
        for index, method in enumerate(F4_METHOD_ORDER)
    }
    seed_offsets = np.linspace(-0.16, 0.16, len(EXPECTED_SEEDS))
    offset_by_seed = dict(zip(sorted(EXPECTED_SEEDS), seed_offsets, strict=True))

    points = canonical.loc[
        canonical["method"].isin(F4_METHOD_ORDER),
        ["method", "seed", "last5", "retention_pct"],
    ].copy()
    points["method"] = pd.Categorical(
        points["method"], categories=F4_METHOD_ORDER, ordered=True
    )
    points = points.sort_values(["method", "seed"]).reset_index(drop=True)
    points.insert(1, "method_y", points["method"].map(method_y).astype(float))
    points.insert(
        3,
        "seed_y",
        points["method_y"] + points["seed"].map(offset_by_seed),
    )
    points["method"] = points["method"].astype("object")

    if len(points) != 50 or set(points["seed"].astype(int)) != EXPECTED_SEEDS:
        raise ValueError("F4 must contain 10 raw points for each of five methods.")
    if set(points.groupby("method", observed=False).size()) != {10}:
        raise ValueError("F4 method sample sizes changed.")

    summary_rows = []
    for method in F4_METHOD_ORDER:
        method_frame = points.loc[points["method"] == method]
        for metric in ("last5", "retention_pct"):
            result = summarize_sample(
                method_frame[metric],
                confidence=DEFAULT_CONFIDENCE,
                resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                rng_seed=DEFAULT_RNG_SEED,
            )
            summary_rows.append({
                "method": method,
                "method_y": method_y[method],
                "metric": metric,
                "n": result.n,
                "mean": result.mean,
                "median": result.median,
                "ci_low": result.ci95_low,
                "ci_high": result.ci95_high,
            })

    summaries = pd.DataFrame(summary_rows)
    if len(summaries) != 10 or set(summaries["n"].astype(int)) != {10}:
        raise ValueError("F4 must contain ten n=10 method-metric summaries.")
    return summaries, points


def build_lunar_main() -> dict[str, pd.DataFrame]:
    """Build renderer-ready Lunar F1 and F2 data."""

    registry = FigureDataRegistry()
    canonical = registry.load_csv("lunar_per_seed", manuscript=True)
    if set(canonical["seed"].astype(int)) != EXPECTED_SEEDS:
        raise ValueError("Canonical Lunar seeds do not match the expected 10 seeds.")

    _validate_numerical_fingerprints(registry, canonical)
    frozen = load_lunar_frozen_curves(registry)
    f1 = build_f1_learning(registry, frozen, canonical)
    f2 = build_f2_failure_rescue(canonical)
    f2_delta_bars = build_f2_delta_bars(f2)
    f3 = build_f3_protected_vs_uniform(canonical)
    f4_summary, f4_points = build_f4_method_data(canonical)

    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    f1.to_csv(F1_DATA_PATH, index=False)
    f2.to_csv(F2_DATA_PATH, index=False)
    f2_delta_bars.to_csv(F2_DELTA_BAR_DATA_PATH, index=False)
    f3.to_csv(F3_DATA_PATH, index=False)
    f4_summary.to_csv(F4_SUMMARY_DATA_PATH, index=False)
    f4_points.to_csv(F4_POINTS_DATA_PATH, index=False)

    print(f"Wrote {F1_DATA_PATH.relative_to(ROOT)} ({len(f1)} checkpoints).")
    print(f"Wrote {F2_DATA_PATH.relative_to(ROOT)} ({len(f2)} cells).")
    print(
        f"Wrote {F2_DELTA_BAR_DATA_PATH.relative_to(ROOT)} "
        f"({len(f2_delta_bars)} seeds)."
    )
    print(f"Wrote {F3_DATA_PATH.relative_to(ROOT)} ({len(f3)} seeds).")
    print(
        f"Wrote {F4_SUMMARY_DATA_PATH.relative_to(ROOT)} "
        f"({len(f4_summary)} summaries)."
    )
    print(
        f"Wrote {F4_POINTS_DATA_PATH.relative_to(ROOT)} "
        f"({len(f4_points)} raw seed points)."
    )
    fingerprint = registry.fingerprint("lunar_primary_frozen")
    print(
        f"Source lunar_primary_frozen: {fingerprint['fingerprint_type']}="
        f"{fingerprint['sha256']}"
    )
    print(
        "Bootstrap: seed unit, "
        f"{DEFAULT_BOOTSTRAP_RESAMPLES} resamples, "
        f"confidence={DEFAULT_CONFIDENCE}, rng_seed={DEFAULT_RNG_SEED}."
    )
    return {
        "f1": f1,
        "f2": f2,
        "f2_delta_bars": f2_delta_bars,
        "f3": f3,
        "f4_summary": f4_summary,
        "f4_points": f4_points,
    }


def main() -> None:
    build_lunar_main()


if __name__ == "__main__":
    main()
