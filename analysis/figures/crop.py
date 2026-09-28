#!/usr/bin/env python3
"""Prepare validated Crop manuscript figure data for production Panels A-C."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import FigureDataRegistry
from .stats import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE,
    DEFAULT_RNG_SEED,
    summarize_paired_frame,
    summarize_sample,
)
from .validation import ROOT


EXPECTED_SEEDS = (11, 22, 33, 44, 55)
EXPECTED_EPISODES = np.arange(3000, dtype=int)
BLOCK_SIZE = 100
METHOD_ORDER = ("DQN", "Dueling", "Uniform", "Naive", "Anchored")
PANEL_A_REFERENCE_METHOD = "Dueling"
PANEL_A_FOCAL_METHODS = ("Uniform", "Anchored")
FROZEN_METHODS = {
    "DQN": ("crop_baseline_primary_frozen", "dqn"),
    "Dueling": ("crop_baseline_primary_frozen", "dueling_dqn"),
    "Uniform": ("crop_baseline_primary_frozen", "amaf_uniform"),
    "Naive": ("crop_baseline_primary_frozen", "amaf_dqn"),
    "Anchored": ("crop_anchored_primary_frozen", None),
}

FIGURE_DATA_DIR = ROOT / "analysis" / "figures" / "figure_data"
PANEL_A_MAIN_PATH = FIGURE_DATA_DIR / "crop_panel_a_main_curve.csv"
PANEL_A_ZOOM_PATH = FIGURE_DATA_DIR / "crop_panel_a_zoom_curve.csv"
PANEL_A_REVIEWER_DELTA_PATH = (
    FIGURE_DATA_DIR / "crop_panel_a_dueling_reference_delta.csv"
)
PER_SEED_LEARNING_PATH = FIGURE_DATA_DIR / "crop_per_seed_learning.csv"
PANEL_B_PATH = FIGURE_DATA_DIR / "crop_panel_b_seed_resource_yield.csv"
PANEL_C_PROFILE_PATH = FIGURE_DATA_DIR / "crop_panel_c_outcome_profile.csv"
PANEL_C_N_PATH = FIGURE_DATA_DIR / "crop_panel_c_nitrogen_tradeoff.csv"
PANEL_C_IRR_PATH = FIGURE_DATA_DIR / "crop_panel_c_irrigation_tradeoff.csv"
PANEL_C_CONNECTION_PATH = FIGURE_DATA_DIR / "crop_panel_c_paired_connections.csv"
FIGURE_OUTPUT_DIR = ROOT / "paper_figures" / "gnuplot" / "crop"


def _assert_close(
    label: str,
    observed: float,
    expected: float,
    *,
    atol: float = 1e-8,
) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{label} anchor changed: {observed} != {expected}")


def _validate_canonical(canonical: pd.DataFrame) -> None:
    required = {
        "method",
        "seed",
        "full",
        "early500",
        "early1000",
        "mid1000",
        "late1000",
        "final500",
        "final100",
        "yield100",
        "N100",
        "irr100",
        "episodes",
        "max_env_steps",
    }
    missing = required - set(canonical.columns)
    if missing:
        raise ValueError(f"Crop canonical table is missing: {sorted(missing)}")
    if canonical[list(required)].isna().any().any():
        raise ValueError("Crop canonical table contains missing figure values.")
    if canonical.duplicated(["method", "seed"]).any():
        raise ValueError("Crop canonical table contains duplicate method-seed rows.")
    if set(canonical["method"].astype(str)) != set(METHOD_ORDER):
        raise ValueError("Crop canonical method set changed.")
    for method in METHOD_ORDER:
        seeds = tuple(
            sorted(canonical.loc[canonical["method"] == method, "seed"].astype(int))
        )
        if seeds != EXPECTED_SEEDS:
            raise ValueError(f"Unexpected {method} seeds: {seeds}")
    if len(canonical) != len(METHOD_ORDER) * len(EXPECTED_SEEDS):
        raise ValueError("Crop canonical table must contain exactly 25 rows.")


def _load_episode_file(path: Path, *, method: str, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "episode",
        "env_steps",
        "train_return",
        "nitrogen",
        "irrigation",
        "yield",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path} contains missing progression values.")
    if frame["episode"].duplicated().any():
        raise ValueError(f"{path} contains duplicate episode indices.")
    episodes = frame["episode"].to_numpy(dtype=int)
    if not np.array_equal(episodes, EXPECTED_EPISODES):
        raise ValueError(f"{path} must contain ordered episodes 0 through 2999.")

    out = frame[[
        "episode",
        "env_steps",
        "train_return",
        "nitrogen",
        "irrigation",
        "yield",
    ]].copy()
    out.insert(0, "seed", int(seed))
    out.insert(0, "method", method)
    return out


def load_crop_frozen_episodes(
    registry: FigureDataRegistry | None = None,
) -> pd.DataFrame:
    """Load all checksum-protected Crop episode progression via the registry."""

    reg = registry or FigureDataRegistry()
    frames: list[pd.DataFrame] = []
    for method, (dataset_id, method_directory) in FROZEN_METHODS.items():
        if reg.dataset_kind(dataset_id) != "frozen_dir":
            raise ValueError(f"{dataset_id} is not registered as a frozen directory.")
        if reg.dataset_lifecycle(dataset_id) != "frozen":
            raise ValueError(f"{dataset_id} does not have frozen lifecycle.")
        root = reg.load_path(dataset_id, manuscript=True)
        method_root = root / method_directory if method_directory else root
        observed_seeds = {
            int(path.name.removeprefix("seed_"))
            for path in method_root.glob("seed_*")
            if path.is_dir() and path.name.removeprefix("seed_").isdigit()
        }
        if observed_seeds != set(EXPECTED_SEEDS):
            raise ValueError(f"Unexpected {method} seed directories: {observed_seeds}")
        for seed in EXPECTED_SEEDS:
            path = method_root / f"seed_{seed}" / "episode_metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(_load_episode_file(path, method=method, seed=seed))

    episodes = pd.concat(frames, ignore_index=True)
    expected_rows = len(METHOD_ORDER) * len(EXPECTED_SEEDS) * len(EXPECTED_EPISODES)
    if len(episodes) != expected_rows:
        raise ValueError(f"Crop frozen progression must contain {expected_rows} rows.")
    return episodes


def _block_seed_progression(episodes: pd.DataFrame) -> pd.DataFrame:
    work = episodes.copy()
    work["block"] = work["episode"] // BLOCK_SIZE + 1
    blocks = (
        work.groupby(["method", "seed", "block"], as_index=False, sort=True)
        .agg(
            train_return=("train_return", "mean"),
            block_rows=("episode", "size"),
        )
    )
    if set(blocks["block_rows"].astype(int)) != {BLOCK_SIZE}:
        raise ValueError("Every Crop seed block must contain exactly 100 episodes.")
    blocks["progress"] = blocks["block"].astype(int) * BLOCK_SIZE
    return blocks


def _validate_progression_anchors(
    episodes: pd.DataFrame,
    blocks: pd.DataFrame,
    canonical: pd.DataFrame,
) -> None:
    canonical_indexed = canonical.set_index(["method", "seed"])
    for method in METHOD_ORDER:
        for seed in EXPECTED_SEEDS:
            key = (method, seed)
            curve = blocks.loc[
                (blocks["method"] == method) & (blocks["seed"] == seed)
            ].set_index("progress")["train_return"]
            raw = episodes.loc[
                (episodes["method"] == method) & (episodes["seed"] == seed)
            ]
            expected = canonical_indexed.loc[key]
            observed = {
                "full": curve.mean(),
                "early500": curve.loc[100:500].mean(),
                "early1000": curve.loc[100:1000].mean(),
                "mid1000": curve.loc[1100:2000].mean(),
                "late1000": curve.loc[2100:3000].mean(),
                "final500": curve.loc[2600:3000].mean(),
                "final100": curve.loc[3000],
            }
            for metric, value in observed.items():
                _assert_close(
                    f"{method}/seed-{seed} {metric}",
                    float(value),
                    float(expected[metric]),
                )
            if int(expected["episodes"]) != len(raw):
                raise ValueError(f"{method}/seed-{seed} episode count drifted.")
            if int(expected["max_env_steps"]) != int(raw["env_steps"].max()):
                raise ValueError(f"{method}/seed-{seed} max_env_steps drifted.")


def build_panel_a(
    episodes: pd.DataFrame,
    canonical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build full and late-stage absolute 100-episode-block curves."""

    blocks = _block_seed_progression(episodes)
    _validate_progression_anchors(episodes, blocks, canonical)
    rows = []
    for method in METHOD_ORDER:
        method_blocks = blocks.loc[blocks["method"] == method]
        for progress in range(BLOCK_SIZE, 3001, BLOCK_SIZE):
            values = method_blocks.loc[
                method_blocks["progress"] == progress, "train_return"
            ]
            result = summarize_sample(
                values,
                confidence=DEFAULT_CONFIDENCE,
                resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                rng_seed=DEFAULT_RNG_SEED,
            )
            rows.append({
                "method": method,
                "progress": progress,
                "n": result.n,
                "mean": result.mean,
                "ci_low": result.ci95_low,
                "ci_high": result.ci95_high,
            })
    main = pd.DataFrame(rows)
    if len(main) != 150 or set(main["n"].astype(int)) != {5}:
        raise ValueError("Panel A must contain 30 n=5 blocks for every method.")

    zoom = main.loc[main["progress"] >= 2100].reset_index(drop=True)
    if len(zoom) != 50 or set(zoom["n"].astype(int)) != {5}:
        raise ValueError("Panel A zoom must contain ten n=5 blocks per method.")
    return main, zoom


def _validate_panel_a_reference(canonical: pd.DataFrame) -> None:
    """Lock the strongest conventional baseline using manuscript windows."""

    conventional = (
        canonical.loc[canonical["method"].isin(["DQN", "Dueling"])]
        .groupby("method")[["late1000", "final500", "final100"]]
        .mean()
    )
    if PANEL_A_REFERENCE_METHOD not in conventional.index:
        raise ValueError("Crop Panel A reference method is unavailable.")
    for metric in ("late1000", "final500", "final100"):
        if not conventional.loc["Dueling", metric] > conventional.loc["DQN", metric]:
            raise ValueError(
                f"Dueling is not the strongest conventional baseline for {metric}."
            )


def build_panel_a_reference_delta(
    episodes: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """Build paired focal-minus-Dueling block margins with seed bootstrap CIs."""

    _validate_panel_a_reference(canonical)
    blocks = _block_seed_progression(episodes)
    _validate_progression_anchors(episodes, blocks, canonical)
    rows = []
    for focal in PANEL_A_FOCAL_METHODS:
        for progress in range(BLOCK_SIZE, 3001, BLOCK_SIZE):
            checkpoint = blocks.loc[blocks["progress"] == progress].pivot(
                index="seed",
                columns="method",
                values="train_return",
            )
            paired_delta = (
                checkpoint[focal] - checkpoint[PANEL_A_REFERENCE_METHOD]
            ).reindex(EXPECTED_SEEDS)
            result = summarize_sample(
                paired_delta,
                confidence=DEFAULT_CONFIDENCE,
                resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                rng_seed=DEFAULT_RNG_SEED,
            )
            rows.append({
                "focal_method": focal,
                "reference_method": PANEL_A_REFERENCE_METHOD,
                "progress": progress,
                "n": result.n,
                "mean_delta": result.mean,
                "ci_low": result.ci95_low,
                "ci_high": result.ci95_high,
            })

    delta = pd.DataFrame(rows)
    if len(delta) != 60 or set(delta["n"].astype(int)) != {5}:
        raise ValueError("Panel A reference delta must contain 30 n=5 blocks per focal method.")
    if delta.duplicated(["focal_method", "progress"]).any():
        raise ValueError("Panel A reference delta contains duplicate checkpoints.")
    if set(delta["reference_method"]) != {PANEL_A_REFERENCE_METHOD}:
        raise ValueError("Panel A reference method drifted.")
    return delta


def build_crop_reviewer_performance() -> dict[str, pd.DataFrame]:
    """Export reviewer trajectories and the aligned final-100 outcome profile."""

    registry = FigureDataRegistry()
    canonical = registry.load_csv("crop_per_seed", manuscript=True)
    _validate_canonical(canonical)
    episodes = load_crop_frozen_episodes(registry)
    panel_a_main, _ = build_panel_a(episodes, canonical)
    panel_c_profile = build_panel_c_outcome_profile(canonical)

    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_a_main.to_csv(PANEL_A_MAIN_PATH, index=False)
    panel_c_profile.to_csv(PANEL_C_PROFILE_PATH, index=False)
    print(f"Wrote {PANEL_A_MAIN_PATH.relative_to(ROOT)} ({len(panel_a_main)} rows).")
    print(f"Wrote {PANEL_C_PROFILE_PATH.relative_to(ROOT)} ({len(panel_c_profile)} rows).")
    return {
        "panel_a_main": panel_a_main,
        "panel_c_outcome_profile": panel_c_profile,
    }


def build_per_seed_learning(
    episodes: pd.DataFrame,
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """Build one validated long-format block-mean table for all Crop seeds."""

    blocks = _block_seed_progression(episodes)
    _validate_progression_anchors(episodes, blocks, canonical)
    per_seed = blocks.rename(columns={
        "progress": "block_end",
        "train_return": "block_mean",
    })[["seed", "block_end", "method", "block_mean"]].copy()
    per_seed["late_stage_flag"] = (per_seed["block_end"] >= 2100).astype(int)
    method_rank = {method: rank for rank, method in enumerate(METHOD_ORDER)}
    per_seed["_method_rank"] = per_seed["method"].map(method_rank)
    per_seed = (
        per_seed.sort_values(["seed", "_method_rank", "block_end"])
        .drop(columns="_method_rank")
        .reset_index(drop=True)
    )

    expected_rows = len(EXPECTED_SEEDS) * len(METHOD_ORDER) * (3000 // BLOCK_SIZE)
    if len(per_seed) != expected_rows:
        raise ValueError(f"Per-seed Crop learning export must contain {expected_rows} rows.")
    if per_seed.duplicated(["seed", "method", "block_end"]).any():
        raise ValueError("Per-seed Crop learning export contains duplicate blocks.")
    if per_seed[["seed", "block_end", "method", "block_mean"]].isna().any().any():
        raise ValueError("Per-seed Crop learning export contains missing values.")

    expected_blocks = tuple(range(BLOCK_SIZE, 3001, BLOCK_SIZE))
    canonical_indexed = canonical.set_index(["method", "seed"])
    for seed in EXPECTED_SEEDS:
        seed_frame = per_seed.loc[per_seed["seed"] == seed]
        if set(seed_frame["method"]) != set(METHOD_ORDER):
            raise ValueError(f"Seed {seed} does not contain all Crop methods.")
        for method in METHOD_ORDER:
            curve = seed_frame.loc[seed_frame["method"] == method].set_index(
                "block_end"
            )["block_mean"]
            if tuple(curve.index.astype(int)) != expected_blocks:
                raise ValueError(f"{method}/seed-{seed} block coverage changed.")
            expected = canonical_indexed.loc[(method, seed)]
            _assert_close(
                f"per-seed export {method}/seed-{seed} final100",
                float(curve.loc[3000]),
                float(expected["final100"]),
            )
            _assert_close(
                f"per-seed export {method}/seed-{seed} late1000",
                float(curve.loc[2100:3000].mean()),
                float(expected["late1000"]),
            )
    return per_seed


def build_crop_per_seed_performance() -> pd.DataFrame:
    """Export reviewer-facing Crop trajectories from approved frozen sources."""

    registry = FigureDataRegistry()
    canonical = registry.load_csv("crop_per_seed", manuscript=True)
    _validate_canonical(canonical)
    episodes = load_crop_frozen_episodes(registry)
    per_seed = build_per_seed_learning(episodes, canonical)
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(PER_SEED_LEARNING_PATH, index=False)
    print(f"Wrote {PER_SEED_LEARNING_PATH.relative_to(ROOT)} ({len(per_seed)} rows).")
    return per_seed


def _paired(canonical: pd.DataFrame, *, metric: str, direction: str):
    return summarize_paired_frame(
        canonical,
        seed_col="seed",
        method_col="method",
        metric_col=metric,
        focal="Anchored",
        comparator="Uniform",
        direction=direction,
        confidence=DEFAULT_CONFIDENCE,
        resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        rng_seed=DEFAULT_RNG_SEED,
    )


def build_panel_b(canonical: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build matched-seed benefits; positive consistently favors Anchored."""

    nitrogen, nitrogen_summary = _paired(canonical, metric="N100", direction="lower")
    irrigation, irrigation_summary = _paired(
        canonical, metric="irr100", direction="lower"
    )
    yield_change, yield_summary = _paired(
        canonical, metric="yield100", direction="higher"
    )
    nitrogen = nitrogen.set_index("seed").sort_index()
    irrigation = irrigation.set_index("seed").sort_index()
    yield_change = yield_change.set_index("seed").sort_index()
    seeds = list(EXPECTED_SEEDS)
    panel = pd.DataFrame({
        "seed": EXPECTED_SEEDS,
        "nitrogen_saving": nitrogen.loc[seeds, "benefit_delta"].to_numpy(),
        "irrigation_saving": irrigation.loc[seeds, "benefit_delta"].to_numpy(),
        "yield_change": yield_change.loc[seeds, "benefit_delta"].to_numpy(),
    })
    panel["sign_n"] = np.sign(panel["nitrogen_saving"]).astype(int)
    panel["sign_irr"] = np.sign(panel["irrigation_saving"]).astype(int)
    panel["sign_yield"] = np.sign(panel["yield_change"]).astype(int)

    _assert_close(
        "nitrogen saving mean", panel["nitrogen_saving"].mean(), 9.52, atol=0.02
    )
    _assert_close(
        "irrigation saving mean",
        panel["irrigation_saving"].mean(),
        9.468,
        atol=0.02,
    )
    _assert_close(
        "yield change mean",
        panel["yield_change"].mean(),
        -495.472674,
        atol=0.02,
    )
    summaries = {
        "nitrogen": nitrogen_summary,
        "irrigation": irrigation_summary,
        "yield": yield_summary,
    }
    expected_wins = {"nitrogen": 5, "irrigation": 5, "yield": 1}
    for metric, result in summaries.items():
        if result.n != 5 or result.wins != expected_wins[metric]:
            raise ValueError(f"Unexpected {metric} wins: {result.wins}/{result.n}")
    return panel, summaries


def _build_tradeoff(canonical: pd.DataFrame, resource_col: str) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        method_frame = canonical.loc[canonical["method"] == method].sort_values("seed")
        for row in method_frame.itertuples(index=False):
            rows.append({
                "method": method,
                "seed": int(row.seed),
                "resource_value": float(getattr(row, resource_col)),
                "yield_value": float(row.yield100),
                "centroid_flag": 0,
            })
        rows.append({
            "method": method,
            "seed": 0,
            "resource_value": float(method_frame[resource_col].mean()),
            "yield_value": float(method_frame["yield100"].mean()),
            "centroid_flag": 1,
        })
    tradeoff = pd.DataFrame(rows)
    if len(tradeoff) != 30:
        raise ValueError("Each trade-off export must contain 25 seeds and 5 centroids.")
    if int((tradeoff["centroid_flag"] == 0).sum()) != 25:
        raise ValueError("Trade-off raw seed count changed.")
    return tradeoff


def build_panel_c_outcome_profile(canonical: pd.DataFrame) -> pd.DataFrame:
    """Build aligned raw-seed and method-mean outcomes in their native units."""

    metric_columns = {
        "yield": "yield100",
        "nitrogen": "N100",
        "irrigation": "irr100",
    }
    row_positions = {
        method: len(METHOD_ORDER) - rank
        for rank, method in enumerate(METHOD_ORDER)
    }
    rows = []
    for metric, column in metric_columns.items():
        method_frames = {
            method: canonical.loc[
                canonical["method"] == method
            ].sort_values("seed")
            for method in METHOD_ORDER
        }

        # Keep each metric's raw observations first and its method means
        # contiguous.  The contiguous mean block lets gnuplot render a simple
        # profile connector without calculating or reconstructing statistics.
        for method in METHOD_ORDER:
            method_frame = method_frames[method]
            for row in method_frame.itertuples(index=False):
                rows.append({
                    "method": method,
                    "row_position": row_positions[method],
                    "metric": metric,
                    "seed": int(row.seed),
                    "value": float(getattr(row, column)),
                    "mean_flag": 0,
                })

        for method in METHOD_ORDER:
            method_frame = method_frames[method]
            rows.append({
                "method": method,
                "row_position": row_positions[method],
                "metric": metric,
                "seed": 0,
                "value": float(method_frame[column].mean()),
                "mean_flag": 1,
            })

    profile = pd.DataFrame(rows)
    if len(profile) != 90:
        raise ValueError("Crop outcome profile must contain 75 seeds and 15 means.")
    raw = profile.loc[profile["mean_flag"] == 0]
    means = profile.loc[profile["mean_flag"] == 1]
    if len(raw) != 75 or len(means) != 15:
        raise ValueError("Crop outcome profile raw/mean row counts changed.")
    if raw.duplicated(["metric", "method", "seed"]).any():
        raise ValueError("Crop outcome profile contains duplicate seed observations.")
    if not (raw.groupby(["metric", "method"]).size() == 5).all():
        raise ValueError("Every Crop outcome row must contain all five seeds.")
    if not (means.groupby(["metric", "method"]).size() == 1).all():
        raise ValueError("Every Crop outcome row must contain one method mean.")
    return profile


def _build_tradeoff_connections(canonical: pd.DataFrame) -> pd.DataFrame:
    """Build matched Uniform-to-Anchored endpoints for optional neutral segments."""

    uniform = (
        canonical.loc[canonical["method"] == "Uniform"]
        .set_index("seed")
        .sort_index()
    )
    anchored = (
        canonical.loc[canonical["method"] == "Anchored"]
        .set_index("seed")
        .sort_index()
    )
    connections = pd.DataFrame({
        "seed": EXPECTED_SEEDS,
        "uniform_nitrogen": uniform.loc[list(EXPECTED_SEEDS), "N100"].to_numpy(),
        "anchored_nitrogen": anchored.loc[list(EXPECTED_SEEDS), "N100"].to_numpy(),
        "uniform_irrigation": uniform.loc[
            list(EXPECTED_SEEDS), "irr100"
        ].to_numpy(),
        "anchored_irrigation": anchored.loc[
            list(EXPECTED_SEEDS), "irr100"
        ].to_numpy(),
        "uniform_yield": uniform.loc[
            list(EXPECTED_SEEDS), "yield100"
        ].to_numpy(),
        "anchored_yield": anchored.loc[
            list(EXPECTED_SEEDS), "yield100"
        ].to_numpy(),
    })
    if len(connections) != len(EXPECTED_SEEDS):
        raise ValueError("Crop trade-off connections must contain all five seeds.")
    if connections["seed"].duplicated().any() or connections.isna().any().any():
        raise ValueError("Crop trade-off connections are incomplete or duplicated.")
    if not (
        connections["anchored_nitrogen"] < connections["uniform_nitrogen"]
    ).all():
        raise ValueError("Anchored nitrogen saving direction changed for a seed.")
    if not (
        connections["anchored_irrigation"] < connections["uniform_irrigation"]
    ).all():
        raise ValueError("Anchored irrigation saving direction changed for a seed.")
    return connections


def _validate_resource_canonical(
    registry: FigureDataRegistry,
    summaries: dict[str, object],
) -> None:
    paired = registry.load_csv("crop_resource_paired", manuscript=True)
    metric_map = {
        "N100": summaries["nitrogen"],
        "irr100": summaries["irrigation"],
        "yield100": summaries["yield"],
    }
    for metric, result in metric_map.items():
        rows = paired.loc[
            (paired["focal"] == "Anchored")
            & (paired["comparator"] == "Uniform")
            & (paired["metric"] == metric)
        ]
        if len(rows) != 1:
            raise ValueError(f"No unique canonical Crop paired row for {metric}.")
        row = rows.iloc[0]
        _assert_close(
            f"canonical {metric} raw paired mean",
            result.mean_delta_focal_minus_comparator,
            float(row["mean_delta_focal_minus_comparator"]),
        )
        if result.n != int(row["n"]) or result.wins != int(row["wins"]):
            raise ValueError(f"Canonical paired counts disagree for {metric}.")


def build_crop_main() -> dict[str, pd.DataFrame]:
    """Build all renderer-ready Crop production data from approved evidence."""

    registry = FigureDataRegistry()
    canonical = registry.load_csv("crop_per_seed", manuscript=True)
    _validate_canonical(canonical)
    episodes = load_crop_frozen_episodes(registry)
    panel_a_main, panel_a_zoom = build_panel_a(episodes, canonical)
    per_seed_learning = build_per_seed_learning(episodes, canonical)
    panel_b, paired_summaries = build_panel_b(canonical)
    _validate_resource_canonical(registry, paired_summaries)
    panel_c_profile = build_panel_c_outcome_profile(canonical)

    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_a_main.to_csv(PANEL_A_MAIN_PATH, index=False)
    panel_a_zoom.to_csv(PANEL_A_ZOOM_PATH, index=False)
    per_seed_learning.to_csv(PER_SEED_LEARNING_PATH, index=False)
    panel_b.to_csv(PANEL_B_PATH, index=False)
    panel_c_profile.to_csv(PANEL_C_PROFILE_PATH, index=False)

    outputs = {
        "panel_a_main": panel_a_main,
        "panel_a_zoom": panel_a_zoom,
        "per_seed_learning": per_seed_learning,
        "panel_b": panel_b,
        "panel_c_outcome_profile": panel_c_profile,
    }
    for name, frame in outputs.items():
        path = {
            "panel_a_main": PANEL_A_MAIN_PATH,
            "panel_a_zoom": PANEL_A_ZOOM_PATH,
            "per_seed_learning": PER_SEED_LEARNING_PATH,
            "panel_b": PANEL_B_PATH,
            "panel_c_outcome_profile": PANEL_C_PROFILE_PATH,
        }[name]
        print(f"Wrote {path.relative_to(ROOT)} ({len(frame)} rows).")
    for dataset_id in (
        "crop_baseline_primary_frozen",
        "crop_anchored_primary_frozen",
        "crop_per_seed",
    ):
        fingerprint = registry.fingerprint(dataset_id)
        print(
            f"Source {dataset_id}: {fingerprint['fingerprint_type']}="
            f"{fingerprint['sha256']}"
        )
    print(
        "Bootstrap: seed unit after within-seed 100-episode blocks, "
        f"{DEFAULT_BOOTSTRAP_RESAMPLES} resamples, "
        f"confidence={DEFAULT_CONFIDENCE}, rng_seed={DEFAULT_RNG_SEED}."
    )
    for metric, result in paired_summaries.items():
        print(
            f"{metric}: mean benefit={result.mean_benefit_delta:.6f}, "
            f"wins={result.wins}/{result.n}."
        )
    return outputs


def main() -> None:
    build_crop_main()


if __name__ == "__main__":
    main()
