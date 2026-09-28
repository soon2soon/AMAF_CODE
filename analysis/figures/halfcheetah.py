#!/usr/bin/env python3
"""Prepare validated HalfCheetah manuscript figure data."""

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
)
from .validation import ROOT


EXPECTED_SEEDS = frozenset({11, 22, 33, 44, 55, 66, 77, 88, 99, 111})
EXPECTED_STEPS = np.arange(10_000, 1_500_001, 10_000, dtype=int)
EXPECTED_EVAL_EPISODES = 10
ORIGINAL_SEEDS = frozenset({11, 22, 33, 44, 55})
REPLICATION_SEEDS = frozenset({66, 77, 88, 99, 111})

FROZEN_COHORTS = {
    "halfcheetah_primary_frozen": ORIGINAL_SEEDS,
    "halfcheetah_replication_frozen": REPLICATION_SEEDS,
}

METHOD_DIRECTORIES = {
    "TD3": "td3",
    "AMAF": "amaf_selective_bp_v2_td3",
}

FIGURE_DATA_PATH = (
    ROOT
    / "analysis"
    / "figures"
    / "figure_data"
    / "halfcheetah_panel_a_learning.csv"
)

PANEL_B_DATA_PATH = (
    ROOT
    / "analysis"
    / "figures"
    / "figure_data"
    / "halfcheetah_panel_b_transition.csv"
)

PANEL_C_DATA_PATH = (
    ROOT
    / "analysis"
    / "figures"
    / "figure_data"
    / "halfcheetah_panel_c_effect_summary.csv"
)

FIGURE_OUTPUT_DIR = (
    ROOT
    / "paper_figures"
    / "gnuplot"
    / "halfcheetah"
)


def _load_seed_curve(path: Path, seed: int) -> pd.DataFrame:
    """Reduce evaluation episodes to one checkpoint mean per seed."""

    frame = pd.read_csv(path)
    required = {
        "env_steps",
        "eval_index",
        "eval_episode",
        "return",
    }
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

    observed_steps = counts.index.to_numpy(dtype=int)
    if not np.array_equal(observed_steps, EXPECTED_STEPS):
        raise ValueError(
            f"{path} does not cover every 10k checkpoint through 1.5M."
        )

    curve = (
        frame.groupby("env_steps", as_index=False, sort=True)["return"]
        .mean()
        .rename(columns={"return": "evaluation_return"})
    )
    curve.insert(0, "seed", int(seed))
    return curve


def load_halfcheetah_frozen_curves(
    registry: FigureDataRegistry | None = None,
) -> pd.DataFrame:
    """Load only the contract-approved frozen TD3/AMAF evaluations."""

    reg = registry or FigureDataRegistry()
    frames: list[pd.DataFrame] = []

    for dataset_id, expected_cohort_seeds in FROZEN_COHORTS.items():
        if reg.dataset_kind(dataset_id) != "frozen_dir":
            raise ValueError(f"{dataset_id} is not registered as a frozen directory.")
        if reg.dataset_lifecycle(dataset_id) != "frozen":
            raise ValueError(f"{dataset_id} does not have frozen lifecycle.")

        cohort_root = reg.load_path(dataset_id, manuscript=True)
        for method, method_directory in METHOD_DIRECTORIES.items():
            method_root = cohort_root / method_directory
            observed_seed_dirs = {
                int(path.name.removeprefix("seed_"))
                for path in method_root.glob("seed_*")
                if path.is_dir() and path.name.removeprefix("seed_").isdigit()
            }
            if observed_seed_dirs != expected_cohort_seeds:
                raise ValueError(
                    f"Unexpected {dataset_id}/{method} seeds: "
                    f"{sorted(observed_seed_dirs)}"
                )

            for seed in sorted(expected_cohort_seeds):
                eval_path = method_root / f"seed_{seed}" / "eval_metrics.csv"
                if not eval_path.is_file():
                    raise FileNotFoundError(eval_path)
                curve = _load_seed_curve(eval_path, seed)
                curve.insert(0, "method", method)
                frames.append(curve)

    frozen = pd.concat(frames, ignore_index=True)
    observed_pairs = set(
        frozen[["method", "seed"]].itertuples(index=False, name=None)
    )
    expected_pairs = {
        (method, seed)
        for method in METHOD_DIRECTORIES
        for seed in EXPECTED_SEEDS
    }
    if observed_pairs != expected_pairs:
        raise ValueError("Frozen curves do not contain all matched method-seed pairs.")

    return frozen


def _validate_against_canonical(
    registry: FigureDataRegistry,
    learning: pd.DataFrame,
) -> None:
    """Anchor the exported checkpoint means to canonical per-seed results."""

    canonical = registry.load_csv("halfcheetah_per_seed", manuscript=True)
    if set(canonical["seed"].astype(int)) != EXPECTED_SEEDS:
        raise ValueError("Canonical HalfCheetah seeds do not match the frozen evidence.")

    anchors = {
        1_000_000: "at_1m",
        1_500_000: "at_1p5m",
    }
    for env_steps, metric in anchors.items():
        row = learning.loc[learning["env_steps"] == env_steps]
        if len(row) != 1:
            raise ValueError(f"Missing exported anchor checkpoint {env_steps}.")
        for method in METHOD_DIRECTORIES:
            expected = canonical.loc[
                canonical["method"] == method,
                metric,
            ].mean()
            observed = float(row[f"{method.lower()}_mean"].iloc[0])
            if not np.isclose(observed, expected, rtol=0.0, atol=1e-9):
                raise ValueError(
                    f"{method} {metric} differs from the canonical table: "
                    f"{observed} != {expected}"
                )


def _paired_metric(
    canonical: pd.DataFrame,
    metric: str,
):
    """Return aligned seed observations and their paired summary."""

    return summarize_paired_frame(
        canonical,
        seed_col="seed",
        method_col="method",
        metric_col=metric,
        focal="AMAF",
        comparator="TD3",
        direction="higher",
        confidence=DEFAULT_CONFIDENCE,
        resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        rng_seed=DEFAULT_RNG_SEED,
    )


def _assert_summary_fingerprint(
    label: str,
    summary,
    *,
    mean: float,
    median: float | None,
    wins: int,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> None:
    """Fail before export if a known canonical result has drifted."""

    checks = {"mean": (summary.mean_delta_focal_minus_comparator, mean)}
    if median is not None:
        checks["median"] = (
            summary.median_delta_focal_minus_comparator,
            median,
        )
    if ci_low is not None:
        checks["ci_low"] = (summary.ci95_low_focal_minus_comparator, ci_low)
    if ci_high is not None:
        checks["ci_high"] = (summary.ci95_high_focal_minus_comparator, ci_high)

    for field, (observed, expected) in checks.items():
        if not np.isclose(observed, expected, rtol=0.0, atol=0.02):
            raise ValueError(
                f"{label} {field} fingerprint changed: {observed} != {expected}"
            )
    if summary.wins != wins:
        raise ValueError(
            f"{label} wins fingerprint changed: {summary.wins} != {wins}"
        )


def build_transition_table(
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the shared Panel B raw table and paired summaries."""

    paired_1m, summary_1m = _paired_metric(canonical, "at_1m")
    paired_1p5m, terminal = _paired_metric(canonical, "at_1p5m")
    paired_gain, late_gain = _paired_metric(canonical, "gain_1m_to_1p5m")

    transition = paired_1m[["seed", "delta_focal_minus_comparator"]].rename(
        columns={"delta_focal_minus_comparator": "delta_1m"}
    )
    transition = transition.merge(
        paired_1p5m[["seed", "delta_focal_minus_comparator"]].rename(
            columns={"delta_focal_minus_comparator": "delta_1p5m"}
        ),
        on="seed",
        validate="one_to_one",
    )
    transition = transition.merge(
        paired_gain[["seed", "delta_focal_minus_comparator"]].rename(
            columns={"delta_focal_minus_comparator": "late_gain_delta"}
        ),
        on="seed",
        validate="one_to_one",
    )

    if set(transition["seed"].astype(int)) != EXPECTED_SEEDS:
        raise ValueError("Panel B does not contain all 10 matched seeds.")
    if 77 not in set(transition["seed"].astype(int)):
        raise ValueError("Replication seed 77 is missing from Panel B.")

    transition.insert(
        1,
        "cohort",
        transition["seed"].map(
            lambda seed: "original5" if seed in ORIGINAL_SEEDS else "replication5"
        ),
    )
    cohort_counts = transition.groupby("cohort").size().to_dict()
    if cohort_counts != {"original5": 5, "replication5": 5}:
        raise ValueError(f"Unexpected Panel B cohort counts: {cohort_counts}")

    arithmetic_gain = transition["delta_1p5m"] - transition["delta_1m"]
    if not np.allclose(
        transition["late_gain_delta"],
        arithmetic_gain,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Late-stage gain is inconsistent with the endpoint gaps.")

    transition["improved"] = (transition["late_gain_delta"] > 0).astype(int)
    if int(transition["improved"].sum()) != late_gain.wins:
        raise ValueError("Panel B improvement count disagrees with paired wins.")

    _assert_summary_fingerprint(
        "Combined terminal",
        terminal,
        mean=68.72,
        median=528.17,
        wins=8,
    )
    _assert_summary_fingerprint(
        "Combined late-stage gain",
        late_gain,
        mean=375.13,
        median=253.13,
        wins=8,
        ci_low=1.90,
        ci_high=786.48,
    )

    cohort_summaries = {}
    for cohort, seeds, expected in (
        ("original5", ORIGINAL_SEEDS, (682.01, 4, 60.98, 1282.37)),
        ("replication5", REPLICATION_SEEDS, (68.26, 4, -258.98, 336.55)),
    ):
        _, summary = _paired_metric(
            canonical.loc[canonical["seed"].isin(seeds)],
            "gain_1m_to_1p5m",
        )
        _assert_summary_fingerprint(
            f"{cohort} late-stage gain",
            summary,
            mean=expected[0],
            median=None,
            wins=expected[1],
            ci_low=expected[2],
            ci_high=expected[3],
        )
        cohort_summaries[cohort] = summary

    summaries = {
        "at_1m": summary_1m,
        "terminal": terminal,
        "late_gain": late_gain,
        **cohort_summaries,
    }
    return transition, summaries


def build_effect_summary(summaries: dict[str, object]) -> pd.DataFrame:
    """Build the renderer-ready Panel C mean/CI table."""

    rows = []
    for metric, metric_y, summary in (
        ("Late-stage gain", 1, summaries["late_gain"]),
        ("Terminal @1.5M", 2, summaries["terminal"]),
    ):
        rows.append({
            "metric": metric,
            "metric_y": metric_y,
            "n": summary.n,
            "mean": summary.mean_delta_focal_minus_comparator,
            "median": summary.median_delta_focal_minus_comparator,
            "ci_low": summary.ci95_low_focal_minus_comparator,
            "ci_high": summary.ci95_high_focal_minus_comparator,
            "wins": summary.wins,
            "losses": summary.losses,
            "wins_label": f"{summary.wins}/{summary.n} wins",
            "label_x": 2350,
        })

    return pd.DataFrame(rows)


def _validate_paired_canonical(
    paired_canonical: pd.DataFrame,
    summaries: dict[str, object],
) -> None:
    """Cross-check computed effects against the canonical paired table."""

    for metric, summary_key in (
        ("at_1p5m", "terminal"),
        ("gain_1m_to_1p5m", "late_gain"),
    ):
        rows = paired_canonical.loc[
            (paired_canonical["focal"] == "AMAF")
            & (paired_canonical["comparator"] == "TD3")
            & (paired_canonical["metric"] == metric)
        ]
        if len(rows) != 1:
            raise ValueError(f"Canonical paired table has no unique {metric} row.")

        row = rows.iloc[0]
        summary = summaries[summary_key]
        if int(row["n"]) != summary.n or int(row["wins"]) != summary.wins:
            raise ValueError(f"Canonical paired counts disagree for {metric}.")
        if not np.isclose(
            float(row["mean_delta_focal_minus_comparator"]),
            summary.mean_delta_focal_minus_comparator,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"Canonical paired mean disagrees for {metric}.")


def build_halfcheetah_main(
    output_path: Path = FIGURE_DATA_PATH,
) -> pd.DataFrame:
    """Build renderer-ready data for HalfCheetah Panels A, B, and C."""

    registry = FigureDataRegistry()
    frozen = load_halfcheetah_frozen_curves(registry)
    method_summaries: dict[str, pd.DataFrame] = {}

    for method in METHOD_DIRECTORIES:
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
            raise ValueError(f"{method} does not have 150 summarized checkpoints.")
        if set(summary["n_seeds"].astype(int)) != {len(EXPECTED_SEEDS)}:
            raise ValueError(f"{method} does not have 10 seeds at every checkpoint.")

        method_summaries[method] = summary[[
            "env_steps",
            "mean",
            "ci95_low",
            "ci95_high",
        ]].rename(columns={
            "mean": f"{method.lower()}_mean",
            "ci95_low": f"{method.lower()}_ci_low",
            "ci95_high": f"{method.lower()}_ci_high",
        })

    learning = method_summaries["TD3"].merge(
        method_summaries["AMAF"],
        on="env_steps",
        how="inner",
        validate="one_to_one",
    )
    if not np.array_equal(
        learning["env_steps"].to_numpy(dtype=int),
        EXPECTED_STEPS,
    ):
        raise ValueError("TD3 and AMAF checkpoint coverage is not identical.")

    paired = frozen.pivot(
        index=["seed", "env_steps"],
        columns="method",
        values="evaluation_return",
    )
    if set(paired.columns) != set(METHOD_DIRECTORIES):
        raise ValueError("Could not form checkpoint-wise TD3/AMAF pairs.")
    if paired.isna().any().any():
        raise ValueError("Checkpoint-wise TD3/AMAF pairing is incomplete.")

    paired = paired.reset_index()
    paired["delta_amaf_minus_td3"] = paired["AMAF"] - paired["TD3"]
    delta_summary = summarize_curve_by_seed(
        paired,
        seed_col="seed",
        x_col="env_steps",
        value_col="delta_amaf_minus_td3",
        confidence=DEFAULT_CONFIDENCE,
        resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        rng_seed=DEFAULT_RNG_SEED,
    )
    if len(delta_summary) != len(EXPECTED_STEPS):
        raise ValueError("Paired delta does not have 150 summarized checkpoints.")
    if set(delta_summary["n_seeds"].astype(int)) != {len(EXPECTED_SEEDS)}:
        raise ValueError("Paired delta does not have 10 seeds at every checkpoint.")

    delta_summary = delta_summary[[
        "env_steps",
        "mean",
        "ci95_low",
        "ci95_high",
    ]].rename(columns={
        "mean": "delta_mean",
        "ci95_low": "delta_ci_low",
        "ci95_high": "delta_ci_high",
    })
    learning = learning.merge(
        delta_summary,
        on="env_steps",
        how="inner",
        validate="one_to_one",
    )

    arithmetic_delta = learning["amaf_mean"] - learning["td3_mean"]
    if not np.allclose(
        learning["delta_mean"],
        arithmetic_delta,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Paired delta mean is inconsistent with method means.")

    _validate_against_canonical(registry, learning)
    canonical = registry.load_csv("halfcheetah_per_seed", manuscript=True)
    transition, summaries = build_transition_table(canonical)
    paired_canonical = registry.load_csv("halfcheetah_paired", manuscript=True)
    _validate_paired_canonical(paired_canonical, summaries)
    effect_summary = build_effect_summary(summaries)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    learning.to_csv(output_path, index=False)
    transition.to_csv(PANEL_B_DATA_PATH, index=False)
    effect_summary.to_csv(PANEL_C_DATA_PATH, index=False)

    print(f"Wrote {output_path.relative_to(ROOT)} ({len(learning)} checkpoints).")
    print(f"Wrote {PANEL_B_DATA_PATH.relative_to(ROOT)} ({len(transition)} seeds).")
    print(f"Wrote {PANEL_C_DATA_PATH.relative_to(ROOT)} ({len(effect_summary)} effects).")
    for dataset_id in FROZEN_COHORTS:
        fingerprint = registry.fingerprint(dataset_id)
        print(
            f"Source {dataset_id}: {fingerprint['fingerprint_type']}="
            f"{fingerprint['sha256']}"
        )
    print(
        "Bootstrap: seed unit (checkpoint-wise paired deltas preserved), "
        f"{DEFAULT_BOOTSTRAP_RESAMPLES} resamples, "
        f"confidence={DEFAULT_CONFIDENCE}, rng_seed={DEFAULT_RNG_SEED}."
    )
    return learning


def main() -> None:
    build_halfcheetah_main()


if __name__ == "__main__":
    main()
