#!/usr/bin/env python3
"""Export validated, plot-ready data for the manuscript Discussion figures.

Only the frozen Discussion handoff is used to construct plotted values.  The
existing Results handoff is read solely to verify exact agreement where the
same paired quantities appear in both packages.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .validation import ROOT


HANDOFF_DIR = ROOT / "paper_tables" / "discussion_handoff_20260915"
RESULTS_DIR = ROOT / "paper_tables" / "results_handoff_20260915"
OUTPUT_DIR = ROOT / "analysis" / "figures" / "data" / "discussion"
FIGURE_OUTPUT_DIR = ROOT / "paper_figures" / "gnuplot" / "discussion"

EXPECTED_SEEDS = {
    "Lunar": (11, 22, 33, 44, 55, 66, 77, 88, 99, 111),
    "Crop": (11, 22, 33, 44, 55),
    "HalfCheetah": (11, 22, 33, 44, 55, 66, 77, 88, 99, 111),
}

FOCAL_METHODS = {
    "Lunar": "Protected",
    "Crop": "Anchored",
    "HalfCheetah": "AMAF",
}

HEATMAP_VIEWS = (
    ("Lunar", "Protected", "raw_router", "uniform_0.25"),
    ("Lunar", "Protected", "effective_gate", "uniform_0.25"),
    ("Crop", "Anchored", "raw_router", "uniform_0.333333"),
    ("Crop", "Anchored", "effective_gate", "uniform_0.333333"),
    ("HalfCheetah", "AMAF", "effective_gate", "not_recorded"),
)

CONCENTRATION_VIEWS = {
    "Lunar": "raw_router",
    "Crop": "raw_router",
    "HalfCheetah": "effective_gate",
}

ENTROPY_DISTRIBUTION_VIEWS = (
    ("Lunar", "Protected", "raw_router"),
    ("Crop", "Anchored", "raw_router"),
    ("HalfCheetah", "AMAF", "effective_gate"),
)

DISPLAY_PHASES = ("Early", "Middle", "Late")

EXPECTED_HEADS = {
    "Lunar": (0, 1, 2, 3),
    "Crop": (0, 1, 2),
    "HalfCheetah": (0, 1, 2, 3),
}

PHASE_ORDER = {
    "time_early": 1,
    "time_mid": 2,
    "time_late": 3,
}

SWITCH_GROUPS = (
    ("Lunar", "Naive", 1),
    ("Lunar", "Protected", 2),
    ("Crop", "Naive", 4),
    ("Crop", "Anchored", 5),
    ("HalfCheetah", "AMAF", 7),
)


def _read_csv(directory: Path, filename: str, required: set[str]) -> pd.DataFrame:
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    return frame


def _assert_close(label: str, observed, expected, *, atol: float = 1e-9) -> None:
    if not np.allclose(
        np.asarray(observed, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=0.0,
        atol=atol,
        equal_nan=False,
    ):
        raise ValueError(f"{label} differs from its frozen handoff source.")


def _assert_seed_set(label: str, observed, expected) -> None:
    observed_set = {int(seed) for seed in observed}
    expected_set = {int(seed) for seed in expected}
    if observed_set != expected_set:
        raise ValueError(
            f"{label} seed set differs: {sorted(observed_set)} != {sorted(expected_set)}"
        )


def build_d1_head_heatmap(mechanism_long: pd.DataFrame) -> pd.DataFrame:
    """Average recorded raw/effective head weights in normalized time bins.

    The normalization is based on each seed's recorded checkpoint range.  It
    is a temporal coordinate and must not be interpreted as a semantic regime.
    HalfCheetah contributes only its recorded conditional residual-head weights;
    no reference or NULL weight is reconstructed.
    """

    rows: list[pd.DataFrame] = []
    for environment, method, weight_type, reference_note in HEATMAP_VIEWS:
        focal = mechanism_long.loc[
            (mechanism_long["environment"] == environment)
            & (mechanism_long["method"] == method)
            & (mechanism_long["weight_type"] == weight_type)
        ].copy()
        if focal.empty:
            raise ValueError(
                f"No {weight_type} rows for {environment}/{method}."
            )
        _assert_seed_set(
            f"D1 heatmap {environment}/{method}/{weight_type}",
            focal["seed"],
            EXPECTED_SEEDS[environment],
        )
        if set(focal["head_id"].astype(int)) != set(EXPECTED_HEADS[environment]):
            raise ValueError(
                f"Unexpected head IDs for {environment}/{method}/{weight_type}."
            )
        if focal.duplicated(["seed", "checkpoint_or_step", "head_id"]).any():
            raise ValueError(
                f"Duplicate checkpoint/head rows for {environment}/{method}/{weight_type}."
            )
        if focal[["checkpoint_or_step", "head_id", "gate_weight_mean"]].isna().any().any():
            raise ValueError(
                f"Missing heatmap values for {environment}/{method}/{weight_type}."
            )

        minimum = focal.groupby("seed")["checkpoint_or_step"].transform("min")
        maximum = focal.groupby("seed")["checkpoint_or_step"].transform("max")
        if (maximum <= minimum).any():
            raise ValueError(
                f"Degenerate checkpoint range for {environment}/{method}/{weight_type}."
            )
        progress = (focal["checkpoint_or_step"] - minimum) / (maximum - minimum)
        focal["progress_bin"] = np.minimum(
            np.floor(progress.to_numpy(dtype=float) * 10).astype(int),
            9,
        )

        seed_bin = (
            focal.groupby(["seed", "progress_bin", "head_id"], as_index=False)
            .agg(
                seed_mean_gate_weight=("gate_weight_mean", "mean"),
                checkpoint_count=("checkpoint_or_step", "nunique"),
            )
        )
        expected_cells = {
            (seed, progress_bin, head)
            for seed in EXPECTED_SEEDS[environment]
            for progress_bin in range(10)
            for head in EXPECTED_HEADS[environment]
        }
        observed_cells = set(
            seed_bin[["seed", "progress_bin", "head_id"]].itertuples(
                index=False, name=None
            )
        )
        if observed_cells != expected_cells:
            raise ValueError(
                f"Incomplete time-bin/head coverage for "
                f"{environment}/{method}/{weight_type}."
            )

        aggregated = (
            seed_bin.groupby(["progress_bin", "head_id"], as_index=False)
            .agg(
                mean_gate_weight=("seed_mean_gate_weight", "mean"),
                sd_across_seeds=("seed_mean_gate_weight", "std"),
                n_seeds=("seed", "nunique"),
                min_checkpoints_per_seed=("checkpoint_count", "min"),
            )
        )
        aggregated.insert(0, "method", method)
        aggregated.insert(0, "environment", environment)
        aggregated["weight_type"] = weight_type
        aggregated["reference_note"] = reference_note
        aggregated["progress_center"] = (aggregated["progress_bin"] + 0.5) / 10.0
        aggregated["head_label"] = "H" + aggregated["head_id"].astype(int).astype(str)
        aggregated["progress_low"] = aggregated["progress_bin"] / 10.0
        aggregated["progress_high"] = (aggregated["progress_bin"] + 1) / 10.0
        aggregated["head_low"] = aggregated["head_id"] - 0.5
        aggregated["head_high"] = aggregated["head_id"] + 0.5

        sums = aggregated.groupby("progress_bin")["mean_gate_weight"].sum()
        _assert_close(
            f"{environment}/{method}/{weight_type} heatmap simplex",
            sums,
            np.ones(10),
            atol=2e-6,
        )
        if not (aggregated["n_seeds"] == len(EXPECTED_SEEDS[environment])).all():
            raise ValueError(
                f"Incomplete across-seed heatmap mean for "
                f"{environment}/{method}/{weight_type}."
            )
        rows.append(aggregated)

    result = pd.concat(rows, ignore_index=True)
    return result[
        [
            "environment",
            "method",
            "progress_bin",
            "progress_center",
            "head_id",
            "head_label",
            "mean_gate_weight",
            "sd_across_seeds",
            "n_seeds",
            "min_checkpoints_per_seed",
            "progress_low",
            "progress_high",
            "head_low",
            "head_high",
            "weight_type",
            "reference_note",
        ]
    ]


def build_d1_concentration(mechanism_regime: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    environment_offsets = {"Lunar": -0.12, "Crop": 0.0, "HalfCheetah": 0.12}
    for environment, method in FOCAL_METHODS.items():
        weight_type = CONCENTRATION_VIEWS[environment]
        focal = mechanism_regime.loc[
            (mechanism_regime["environment"] == environment)
            & (mechanism_regime["method"] == method)
            & (mechanism_regime["weight_type"] == weight_type)
        ].copy()
        focal = focal.drop_duplicates(
            ["environment", "method", "seed", "regime", "weight_type"]
        )
        _assert_seed_set(
            f"D1 concentration {environment}/{method}",
            focal["seed"],
            EXPECTED_SEEDS[environment],
        )
        expected = {
            (seed, phase)
            for seed in EXPECTED_SEEDS[environment]
            for phase in PHASE_ORDER
        }
        observed = set(focal[["seed", "regime"]].itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(f"Incomplete time-phase rows for {environment}/{method}.")
        columns = ["normalized_entropy", "effective_heads", "mean_max_weight"]
        if focal[columns].isna().any().any():
            raise ValueError(f"Missing concentration statistic for {environment}/{method}.")

        focal["phase_order"] = focal["regime"].map(PHASE_ORDER).astype(int)
        seed_rank = focal["seed"].map(
            {seed: rank for rank, seed in enumerate(EXPECTED_SEEDS[environment])}
        )
        centered_rank = seed_rank - (len(EXPECTED_SEEDS[environment]) - 1) / 2
        focal["x_position"] = (
            focal["phase_order"]
            + environment_offsets[environment]
            + centered_rank * 0.008
        )
        focal["row_type"] = "seed"
        focal["n_seeds"] = len(EXPECTED_SEEDS[environment])

        mean = (
            focal.groupby(["regime", "phase_order"], as_index=False)[columns]
            .mean()
        )
        mean.insert(0, "method", method)
        mean.insert(0, "environment", environment)
        mean["weight_type"] = weight_type
        mean["seed"] = 0
        mean["x_position"] = mean["phase_order"] + environment_offsets[environment]
        mean["row_type"] = "mean"
        mean["n_seeds"] = len(EXPECTED_SEEDS[environment])
        rows.extend([focal, mean])

    result = pd.concat(rows, ignore_index=True)
    result = result.rename(columns={"regime": "time_phase"})
    return result[
        [
            "row_type",
            "environment",
            "method",
            "seed",
            "time_phase",
            "phase_order",
            "x_position",
            "normalized_entropy",
            "effective_heads",
            "mean_max_weight",
            "n_seeds",
            "weight_type",
        ]
    ]


def _valid_checkpoint_phases(size: int) -> np.ndarray:
    """Split valid checkpoint order into Early/Middle/Late thirds."""

    labels = np.empty(size, dtype=object)
    for phase, positions in zip(
        DISPLAY_PHASES,
        np.array_split(np.arange(size), 3),
        strict=True,
    ):
        labels[positions] = phase
    return labels


def build_d1_entropy_observations(mechanism_long: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct normalized entropy at every logged valid checkpoint.

    These are descriptive checkpoint-time observations.  They are not
    independent experimental units and are not used for inferential claims.
    """

    rows: list[dict[str, float | int | str]] = []
    for environment, method, weight_type in ENTROPY_DISTRIBUTION_VIEWS:
        selected = mechanism_long.loc[
            (mechanism_long["environment"] == environment)
            & (mechanism_long["method"] == method)
            & (mechanism_long["weight_type"] == weight_type)
        ].copy()
        _assert_seed_set(
            f"D1 entropy observations {environment}/{method}/{weight_type}",
            selected["seed"],
            EXPECTED_SEEDS[environment],
        )
        expected_heads = set(EXPECTED_HEADS[environment])
        group_columns = [
            "source_dataset_id",
            "seed",
            "checkpoint_or_step",
        ]
        for keys, checkpoint in selected.groupby(group_columns, sort=True):
            source_dataset_id, seed, checkpoint_or_step = keys
            if checkpoint["head_id"].duplicated().any():
                raise ValueError(
                    f"Duplicate entropy head at {environment}/{method}/{seed}/"
                    f"{checkpoint_or_step}."
                )
            if set(checkpoint["head_id"].astype(int)) != expected_heads:
                raise ValueError(
                    f"Incomplete entropy head set at {environment}/{method}/{seed}/"
                    f"{checkpoint_or_step}."
                )
            weights = checkpoint.sort_values("head_id")["gate_weight_mean"].to_numpy(
                dtype=float
            )
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("Entropy reconstruction received invalid weights.")
            if not np.isclose(weights.sum(), 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError("Entropy reconstruction received a non-simplex row.")
            entropy = -np.sum(weights * np.log(np.clip(weights, 1e-12, None)))
            normalized_entropy = entropy / np.log(len(weights))
            rows.append(
                {
                    "environment": environment,
                    "method": method,
                    "weight_type": weight_type,
                    "source_dataset_id": source_dataset_id,
                    "seed": int(seed),
                    "checkpoint_or_step": int(checkpoint_or_step),
                    "normalized_entropy": float(normalized_entropy),
                }
            )

    observations = pd.DataFrame(rows).sort_values(
        ["environment", "method", "weight_type", "seed", "checkpoint_or_step"]
    )
    observations["checkpoint_index"] = (
        observations.groupby(["environment", "method", "weight_type", "seed"])
        .cumcount()
        .add(1)
    )
    observations["phase"] = ""
    observations["phase_order"] = 0
    for _, indexes in observations.groupby(
        ["environment", "method", "weight_type", "seed"], sort=False
    ).groups.items():
        ordered_indexes = list(indexes)
        phases = _valid_checkpoint_phases(len(ordered_indexes))
        observations.loc[ordered_indexes, "phase"] = phases
        observations.loc[ordered_indexes, "phase_order"] = [
            DISPLAY_PHASES.index(phase) + 1 for phase in phases
        ]

    observations["phase_order"] = observations["phase_order"].astype(int)
    # Stable, data-derived horizontal jitter for the raw checkpoint points.
    # The arithmetic is intentionally deterministic: no RNG or subsampling is
    # involved, so repeated exports place every observation identically.
    jitter_bucket = (
        observations["seed"].astype(int) * 37
        + observations["checkpoint_index"].astype(int) * 53
    ) % 101
    observations["jitter_offset"] = (jitter_bucket - 50) / 250.0
    observations["phase_x"] = (
        observations["phase_order"] + observations["jitter_offset"]
    )
    observations["phase_definition"] = "valid_checkpoint_order_thirds"
    return observations[
        [
            "environment",
            "method",
            "weight_type",
            "source_dataset_id",
            "seed",
            "checkpoint_index",
            "checkpoint_or_step",
            "phase",
            "phase_order",
            "normalized_entropy",
            "jitter_offset",
            "phase_x",
            "phase_definition",
        ]
    ].reset_index(drop=True)


def build_d1_entropy_phase_summary(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize the plotted checkpoint observations without inference."""

    summary = (
        observations.groupby(
            ["environment", "method", "weight_type", "phase", "phase_order"],
            sort=False,
        )["normalized_entropy"]
        .agg(
            n="size",
            q05=lambda values: values.quantile(0.05),
            q10=lambda values: values.quantile(0.10),
            q25=lambda values: values.quantile(0.25),
            median="median",
            q75=lambda values: values.quantile(0.75),
            q90=lambda values: values.quantile(0.90),
            q95=lambda values: values.quantile(0.95),
            mean="mean",
        )
        .reset_index()
        .rename(
            columns={
                "environment": "domain",
                "weight_type": "router_type",
            }
        )
    )
    return summary[
        [
            "domain",
            "method",
            "router_type",
            "phase",
            "phase_order",
            "n",
            "q05",
            "q10",
            "q25",
            "median",
            "q75",
            "q90",
            "q95",
            "mean",
        ]
    ]


def _kde_bandwidth(values: np.ndarray) -> float:
    """Return a deterministic bounded Silverman-style bandwidth."""

    if len(values) < 2:
        raise ValueError("KDE requires at least two observations.")
    standard_deviation = float(np.std(values, ddof=1))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    robust_scale = min(standard_deviation, float((q75 - q25) / 1.34))
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        robust_scale = standard_deviation
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        robust_scale = 0.015
    candidate = 0.9 * robust_scale * len(values) ** (-0.2)
    return float(np.clip(candidate, 0.015, 0.08))


def build_d1_entropy_kde(observations: pd.DataFrame) -> pd.DataFrame:
    """Build boundary-reflected KDE grids for rendering only."""

    grid = np.linspace(0.0, 1.0, 201)
    rows: list[pd.DataFrame] = []
    for (environment, method, weight_type, phase, phase_order), group in observations.groupby(
        ["environment", "method", "weight_type", "phase", "phase_order"],
        sort=False,
    ):
        values = group["normalized_entropy"].to_numpy(dtype=float)
        bandwidth = _kde_bandwidth(values)
        reflected = np.concatenate([values, -values, 2.0 - values])
        standardized = (grid[:, None] - reflected[None, :]) / bandwidth
        density = np.exp(-0.5 * standardized**2).sum(axis=1)
        density /= len(values) * bandwidth * np.sqrt(2.0 * np.pi)
        maximum = float(density.max())
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError(f"Degenerate KDE for {environment}/{phase}.")
        rows.append(
            pd.DataFrame(
                {
                    "environment": environment,
                    "method": method,
                    "weight_type": weight_type,
                    "phase": phase,
                    "phase_order": int(phase_order),
                    "entropy_grid": grid,
                    "density": density,
                    "density_scaled": density / maximum,
                    "n_observations": len(values),
                    "mean_entropy": float(np.mean(values)),
                    "median_entropy": float(np.median(values)),
                    "bandwidth": bandwidth,
                    "kde_boundary": "reflection_0_1",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _validate_entropy_against_regime_handoff(
    observations: pd.DataFrame,
    mechanism_regime: pd.DataFrame,
) -> None:
    """Verify Lunar/Crop checkpoint entropy against the frozen phase summaries."""

    phase_to_regime = {
        "Early": "time_early",
        "Middle": "time_mid",
        "Late": "time_late",
    }
    for environment in ("Lunar", "Crop"):
        method = FOCAL_METHODS[environment]
        weight_type = CONCENTRATION_VIEWS[environment]
        observed = (
            observations.loc[observations["environment"] == environment]
            .groupby(["seed", "phase"], as_index=False)["normalized_entropy"]
            .mean()
        )
        observed["regime"] = observed["phase"].map(phase_to_regime)
        expected = mechanism_regime.loc[
            (mechanism_regime["environment"] == environment)
            & (mechanism_regime["method"] == method)
            & (mechanism_regime["weight_type"] == weight_type)
        ].drop_duplicates(["seed", "regime", "weight_type"])
        merged = observed.merge(
            expected[["seed", "regime", "normalized_entropy"]],
            on=["seed", "regime"],
            suffixes=("_observed", "_handoff"),
            validate="one_to_one",
        )
        if len(merged) != len(EXPECTED_SEEDS[environment]) * 3:
            raise ValueError(f"Entropy phase validation lost {environment} rows.")
        _assert_close(
            f"{environment} checkpoint entropy phase means",
            merged["normalized_entropy_observed"],
            merged["normalized_entropy_handoff"],
            atol=1e-9,
        )


def build_d1_switching(mechanism_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for environment, method, category_order in SWITCH_GROUPS:
        group = mechanism_summary.loc[
            (mechanism_summary["environment"] == environment)
            & (mechanism_summary["method"] == method)
        ].copy()
        _assert_seed_set(
            f"D1 switching {environment}/{method}",
            group["seed"],
            EXPECTED_SEEDS[environment],
        )
        if group["seed"].duplicated().any():
            raise ValueError(f"Duplicate switching seed rows for {environment}/{method}.")

        defined = group.loc[group["head_switch_rate"].notna()].copy()
        if defined.empty:
            raise ValueError(f"No defined switch rates for {environment}/{method}.")
        n_expected = len(EXPECTED_SEEDS[environment])
        n_defined = len(defined)
        undefined_seeds = sorted(
            set(EXPECTED_SEEDS[environment]) - set(defined["seed"].astype(int))
        )
        rank = np.arange(n_defined, dtype=float) - (n_defined - 1) / 2
        defined["x_position"] = category_order + rank * 0.025
        defined["row_type"] = "seed"
        defined["category_order"] = category_order
        defined["n_expected_seeds"] = n_expected
        defined["n_defined_seeds"] = n_defined
        defined["undefined_seeds"] = (
            "none" if not undefined_seeds else "+".join(map(str, undefined_seeds))
        )

        mean = pd.DataFrame(
            {
                "row_type": ["mean"],
                "environment": [environment],
                "method": [method],
                "seed": [0],
                "category_order": [category_order],
                "x_position": [float(category_order)],
                "head_switch_rate": [defined["head_switch_rate"].mean()],
                "n_expected_seeds": [n_expected],
                "n_defined_seeds": [n_defined],
                "undefined_seeds": [
                    "none" if not undefined_seeds else "+".join(map(str, undefined_seeds))
                ],
            }
        )
        rows.extend([defined, mean])

    result = pd.concat(rows, ignore_index=True)
    return result[
        [
            "row_type",
            "environment",
            "method",
            "seed",
            "category_order",
            "x_position",
            "head_switch_rate",
            "n_expected_seeds",
            "n_defined_seeds",
            "undefined_seeds",
        ]
    ]


def build_d2_lunar_loo(robustness: pd.DataFrame) -> pd.DataFrame:
    selected = robustness.loc[
        (robustness["environment"] == "Lunar")
        & (robustness["comparison"] == "Protected - Naive")
        & (robustness["metric"] == "last5_return")
    ].copy()
    full = selected.loc[selected["excluded_seed"].astype(str) == "NONE"]
    if len(full) != 1:
        raise ValueError("Lunar LOO table must contain one full-sample row.")
    leave_one_out = selected.loc[selected["excluded_seed"].astype(str) != "NONE"].copy()
    leave_one_out["excluded_seed"] = pd.to_numeric(
        leave_one_out["excluded_seed"], errors="raise"
    ).astype(int)
    _assert_seed_set(
        "D2 Lunar excluded seeds",
        leave_one_out["excluded_seed"],
        EXPECTED_SEEDS["Lunar"],
    )
    if leave_one_out["excluded_seed"].duplicated().any():
        raise ValueError("Duplicate Lunar leave-one-out rows.")
    order = {seed: index + 1 for index, seed in enumerate(EXPECTED_SEEDS["Lunar"])}
    leave_one_out["x_order"] = leave_one_out["excluded_seed"].map(order).astype(int)
    leave_one_out["full_mean_delta"] = float(full["loo_mean_delta"].iloc[0])
    return leave_one_out[
        [
            "excluded_seed",
            "x_order",
            "loo_mean_delta",
            "loo_median_delta",
            "loo_wins",
            "full_mean_delta",
        ]
    ].sort_values("x_order")


def build_d2_crop_tradeoff(crop: pd.DataFrame) -> pd.DataFrame:
    _assert_seed_set("D2 Crop", crop["seed"], EXPECTED_SEEDS["Crop"])
    if crop["seed"].duplicated().any():
        raise ValueError("Duplicate Crop trade-off seed rows.")
    columns = [
        "seed",
        "yield_delta",
        "nitrogen_saving",
        "irrigation_saving",
        "return_delta",
    ]
    if crop[columns].isna().any().any():
        raise ValueError("Crop trade-off contains missing values.")
    return crop[columns].sort_values("seed").reset_index(drop=True)


def build_d2_halfcheetah_boundary(halfcheetah: pd.DataFrame) -> pd.DataFrame:
    seed_rows = halfcheetah.loc[halfcheetah["row_type"] == "seed"].copy()
    seed_rows["seed"] = pd.to_numeric(seed_rows["seed"], errors="raise").astype(int)
    _assert_seed_set(
        "D2 HalfCheetah", seed_rows["seed"], EXPECTED_SEEDS["HalfCheetah"]
    )
    if 77 not in set(seed_rows["seed"].astype(int)):
        raise ValueError("D2 HalfCheetah is missing seed 77.")
    if seed_rows["seed"].duplicated().any():
        raise ValueError("Duplicate HalfCheetah boundary seed rows.")
    if set(seed_rows["cohort"]) != {"original5", "replication5"}:
        raise ValueError("HalfCheetah cohort labels changed.")
    columns = ["cohort", "seed", "terminal_delta_1p5m", "gain_delta"]
    if seed_rows[columns].isna().any().any():
        raise ValueError("HalfCheetah boundary contains missing values.")
    return seed_rows[columns].sort_values(["cohort", "seed"]).reset_index(drop=True)


def _validate_against_results(
    lunar_loo: pd.DataFrame,
    crop_tradeoff: pd.DataFrame,
    halfcheetah_boundary: pd.DataFrame,
) -> None:
    lunar_results = _read_csv(
        RESULTS_DIR,
        "lunar_paired_effects.csv",
        {"comparison", "metric", "mean_delta"},
    )
    lunar_row = lunar_results.loc[
        (lunar_results["comparison"] == "Protected - Naive")
        & (lunar_results["metric"] == "last5_return")
    ]
    if len(lunar_row) != 1:
        raise ValueError("Results handoff Lunar contrast is not unique.")
    _assert_close(
        "Lunar full mean",
        lunar_loo["full_mean_delta"].unique(),
        [float(lunar_row["mean_delta"].iloc[0])],
    )

    crop_results = _read_csv(
        RESULTS_DIR,
        "crop_per_seed.csv",
        {"seed", "method", "final100_return", "yield100", "n100", "irr100"},
    )
    crop_pivot = crop_results.pivot(index="seed", columns="method")
    crop_expected = pd.DataFrame(
        {
            "seed": crop_pivot.index.astype(int),
            "return_delta": (
                crop_pivot["final100_return"]["Anchored"]
                - crop_pivot["final100_return"]["Uniform"]
            ).to_numpy(),
            "yield_delta": (
                crop_pivot["yield100"]["Anchored"]
                - crop_pivot["yield100"]["Uniform"]
            ).to_numpy(),
            "nitrogen_saving": (
                crop_pivot["n100"]["Uniform"] - crop_pivot["n100"]["Anchored"]
            ).to_numpy(),
            "irrigation_saving": (
                crop_pivot["irr100"]["Uniform"] - crop_pivot["irr100"]["Anchored"]
            ).to_numpy(),
        }
    ).sort_values("seed")
    crop_observed = crop_tradeoff.sort_values("seed")
    for column in ("return_delta", "yield_delta", "nitrogen_saving", "irrigation_saving"):
        _assert_close(
            f"Crop {column}", crop_observed[column], crop_expected[column]
        )

    hc_results = _read_csv(
        RESULTS_DIR,
        "halfcheetah_per_seed.csv",
        {"cohort", "seed", "terminal_delta", "gain_delta"},
    )
    merged = halfcheetah_boundary.merge(
        hc_results[["cohort", "seed", "terminal_delta", "gain_delta"]],
        on=["cohort", "seed"],
        validate="one_to_one",
    )
    if len(merged) != len(EXPECTED_SEEDS["HalfCheetah"]):
        raise ValueError("HalfCheetah Results alignment lost a seed.")
    _assert_close(
        "HalfCheetah terminal delta",
        merged["terminal_delta_1p5m"],
        merged["terminal_delta"],
    )
    _assert_close(
        "HalfCheetah gain delta", merged["gain_delta_x"], merged["gain_delta_y"]
    )


def _validate_outputs(frames: dict[str, pd.DataFrame]) -> None:
    expected_rows = {
        "discussion_d1_head_heatmap.tsv": 180,
        "discussion_d1_concentration.tsv": 84,
        "discussion_d1_switching.tsv": 44,
        "discussion_d1_entropy_observations.csv": 3026,
        "routing_entropy_phase_quantiles.csv": 9,
        "discussion_d1_entropy_kde.csv": 1809,
        "discussion_d2_lunar_loo.tsv": 10,
        "discussion_d2_crop_tradeoff.tsv": 5,
        "discussion_d2_halfcheetah_boundary.tsv": 10,
    }
    for filename, frame in frames.items():
        if len(frame) != expected_rows[filename]:
            raise ValueError(
                f"{filename} row count changed: {len(frame)} != {expected_rows[filename]}"
            )
        if frame.isna().any().any():
            raise ValueError(f"{filename} contains NaN values.")

    duplicate_keys = {
        "discussion_d1_head_heatmap.tsv": [
            "environment",
            "weight_type",
            "progress_bin",
            "head_id",
        ],
        "discussion_d1_concentration.tsv": [
            "row_type",
            "environment",
            "seed",
            "time_phase",
        ],
        "discussion_d1_switching.tsv": ["row_type", "environment", "method", "seed"],
        "discussion_d1_entropy_observations.csv": [
            "environment",
            "method",
            "weight_type",
            "seed",
            "checkpoint_or_step",
        ],
        "routing_entropy_phase_quantiles.csv": [
            "domain",
            "method",
            "router_type",
            "phase",
        ],
        "discussion_d1_entropy_kde.csv": [
            "environment",
            "method",
            "weight_type",
            "phase",
            "entropy_grid",
        ],
        "discussion_d2_lunar_loo.tsv": ["excluded_seed"],
        "discussion_d2_crop_tradeoff.tsv": ["seed"],
        "discussion_d2_halfcheetah_boundary.tsv": ["seed"],
    }
    for filename, keys in duplicate_keys.items():
        if frames[filename].duplicated(keys).any():
            raise ValueError(f"{filename} contains duplicate keys: {keys}")

    heatmap = frames["discussion_d1_head_heatmap.tsv"]
    observed_views = set(
        heatmap[["environment", "method", "weight_type", "reference_note"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_views != set(HEATMAP_VIEWS):
        raise ValueError("D1 raw/effective heatmap views changed.")
    if not heatmap["mean_gate_weight"].between(0.0, 1.0).all():
        raise ValueError("D1 head weights fall outside the common 0-to-1 scale.")
    halfcheetah_views = heatmap.loc[
        heatmap["environment"] == "HalfCheetah", "weight_type"
    ].unique()
    if list(halfcheetah_views) != ["effective_gate"]:
        raise ValueError("HalfCheetah reference/NULL routing was synthesized.")

    concentration = frames["discussion_d1_concentration.tsv"]
    expected_concentration_types = {
        (environment, weight_type)
        for environment, weight_type in CONCENTRATION_VIEWS.items()
    }
    observed_concentration_types = set(
        concentration[["environment", "weight_type"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_concentration_types != expected_concentration_types:
        raise ValueError("D1 entropy raw/conditional routing sources changed.")

    entropy_observations = frames["discussion_d1_entropy_observations.csv"]
    entropy_views = set(
        entropy_observations[["environment", "method", "weight_type"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if entropy_views != set(ENTROPY_DISTRIBUTION_VIEWS):
        raise ValueError("D1 entropy distribution views changed.")
    if set(entropy_observations["phase"]) != set(DISPLAY_PHASES):
        raise ValueError("D1 entropy phases changed.")
    if not entropy_observations["normalized_entropy"].between(0.0, 1.0).all():
        raise ValueError("D1 normalized entropy falls outside [0, 1].")
    if not entropy_observations["jitter_offset"].between(-0.2, 0.2).all():
        raise ValueError("D1 deterministic jitter falls outside [-0.2, 0.2].")
    _assert_close(
        "D1 entropy phase x positions",
        entropy_observations["phase_x"],
        entropy_observations["phase_order"]
        + entropy_observations["jitter_offset"],
    )
    phase_counts = entropy_observations.groupby(
        ["environment", "method", "weight_type", "seed", "phase"]
    ).size()
    per_seed_spread = phase_counts.groupby(
        ["environment", "method", "weight_type", "seed"]
    ).agg(lambda values: int(values.max() - values.min()))
    if (per_seed_spread > 1).any():
        raise ValueError("Valid checkpoints were not split into balanced thirds.")

    entropy_summary = frames["routing_entropy_phase_quantiles.csv"]
    if set(entropy_summary["phase"]) != set(DISPLAY_PHASES):
        raise ValueError("D1 entropy summary phases changed.")
    ordered_quantiles = entropy_summary[
        ["q05", "q10", "q25", "median", "q75", "q90", "q95"]
    ].to_numpy(dtype=float)
    if not (np.diff(ordered_quantiles, axis=1) >= 0.0).all():
        raise ValueError("D1 entropy quantiles are not ordered.")
    summary_counts = entropy_observations.groupby(
        ["environment", "method", "weight_type", "phase"], sort=False
    ).size()
    exported_counts = entropy_summary.set_index(
        ["domain", "method", "router_type", "phase"]
    )["n"]
    if not summary_counts.equals(exported_counts.reindex(summary_counts.index)):
        raise ValueError("D1 entropy summary dropped checkpoint observations.")

    entropy_kde = frames["discussion_d1_entropy_kde.csv"]
    kde_groups = entropy_kde.groupby(
        ["environment", "method", "weight_type", "phase"]
    )
    if not (kde_groups.size() == 201).all():
        raise ValueError("Every entropy KDE must contain 201 grid points.")
    if not np.allclose(kde_groups["entropy_grid"].min(), 0.0):
        raise ValueError("Entropy KDE grid does not start at zero.")
    if not np.allclose(kde_groups["entropy_grid"].max(), 1.0):
        raise ValueError("Entropy KDE grid does not end at one.")
    if (entropy_kde["density"] < 0.0).any():
        raise ValueError("Entropy KDE contains negative density.")
    if not entropy_kde["density_scaled"].between(0.0, 1.0).all():
        raise ValueError("Scaled entropy KDE falls outside [0, 1].")

    switching = frames["discussion_d1_switching.tsv"]
    protected_mean = switching.loc[
        (switching["row_type"] == "mean")
        & (switching["environment"] == "Lunar")
        & (switching["method"] == "Protected")
    ]
    if (
        len(protected_mean) != 1
        or int(protected_mean["n_defined_seeds"].iloc[0]) != 9
        or str(protected_mean["undefined_seeds"].iloc[0]) != "55"
    ):
        raise ValueError("The structurally undefined Lunar switching seed changed.")


def export() -> dict[str, pd.DataFrame]:
    mechanism_long = _read_csv(
        HANDOFF_DIR,
        "mechanism_head_long.csv",
        {
            "environment",
            "method",
            "seed",
            "checkpoint_or_step",
            "weight_type",
            "head_id",
            "gate_weight_mean",
        },
    )
    mechanism_regime = _read_csv(
        HANDOFF_DIR,
        "mechanism_head_by_regime.csv",
        {
            "environment",
            "method",
            "seed",
            "regime",
            "weight_type",
            "normalized_entropy",
            "effective_heads",
            "mean_max_weight",
        },
    )
    mechanism_summary = _read_csv(
        HANDOFF_DIR,
        "mechanism_head_summary.csv",
        {"environment", "method", "seed", "head_switch_rate"},
    )
    robustness = _read_csv(
        HANDOFF_DIR,
        "robustness_seed_sensitivity.csv",
        {
            "environment",
            "comparison",
            "metric",
            "excluded_seed",
            "loo_mean_delta",
            "loo_median_delta",
            "loo_wins",
        },
    )
    crop = _read_csv(
        HANDOFF_DIR,
        "crop_tradeoff_discussion.csv",
        {"seed", "return_delta", "yield_delta", "nitrogen_saving", "irrigation_saving"},
    )
    halfcheetah = _read_csv(
        HANDOFF_DIR,
        "halfcheetah_late_stage_discussion.csv",
        {"row_type", "cohort", "seed", "terminal_delta_1p5m", "gain_delta"},
    )

    entropy_observations = build_d1_entropy_observations(mechanism_long)
    entropy_summary = build_d1_entropy_phase_summary(entropy_observations)
    entropy_kde = build_d1_entropy_kde(entropy_observations)
    _validate_entropy_against_regime_handoff(
        entropy_observations,
        mechanism_regime,
    )

    frames = {
        "discussion_d1_head_heatmap.tsv": build_d1_head_heatmap(mechanism_long),
        "discussion_d1_concentration.tsv": build_d1_concentration(mechanism_regime),
        "discussion_d1_switching.tsv": build_d1_switching(mechanism_summary),
        "discussion_d1_entropy_observations.csv": entropy_observations,
        "routing_entropy_phase_quantiles.csv": entropy_summary,
        "discussion_d1_entropy_kde.csv": entropy_kde,
        "discussion_d2_lunar_loo.tsv": build_d2_lunar_loo(robustness),
        "discussion_d2_crop_tradeoff.tsv": build_d2_crop_tradeoff(crop),
        "discussion_d2_halfcheetah_boundary.tsv": build_d2_halfcheetah_boundary(
            halfcheetah
        ),
    }
    _validate_against_results(
        frames["discussion_d2_lunar_loo.tsv"],
        frames["discussion_d2_crop_tradeoff.tsv"],
        frames["discussion_d2_halfcheetah_boundary.tsv"],
    )
    _validate_outputs(frames)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".discussion_figure_data_", dir=OUTPUT_DIR))
    try:
        for filename, frame in frames.items():
            output = temporary / filename
            separator = "," if output.suffix == ".csv" else "\t"
            frame.to_csv(output, sep=separator, index=False, float_format="%.12g")
            serialized = pd.read_csv(output, sep=separator)
            if serialized.shape != frame.shape:
                raise ValueError(f"Serialized shape mismatch for {filename}.")
            if serialized.isna().any().any():
                raise ValueError(f"Serialized {filename} contains NaN values.")
            os.replace(output, OUTPUT_DIR / filename)
    finally:
        temporary.rmdir()

    for filename, frame in frames.items():
        print(f"Wrote {OUTPUT_DIR.relative_to(ROOT) / filename} ({len(frame)} rows).")
    print("Expected seeds, duplicate keys, NaN, seed 77, cohort, and Results alignment: PASS")
    return frames


def main() -> None:
    export()


if __name__ == "__main__":
    main()
