#!/usr/bin/env python3
"""Build the manuscript Discussion evidence handoff from frozen sources."""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .crop import EXPECTED_SEEDS as CROP_SEEDS, FROZEN_METHODS as CROP_FROZEN_METHODS
from .halfcheetah import (
    EXPECTED_SEEDS as HC_SEEDS,
    FROZEN_COHORTS,
    METHOD_DIRECTORIES as HC_METHOD_DIRECTORIES,
    ORIGINAL_SEEDS,
    REPLICATION_SEEDS,
    load_halfcheetah_frozen_curves,
)
from .io import FigureDataRegistry
from .lunar import EXPECTED_SEEDS as LUNAR_SEEDS, F1_METHOD_DIRECTORIES
from .results_handoff import OUTPUT_DIR as RESULTS_OUTPUT_DIR
from .results_handoff_sac import (
    OUTPUT_DIR as SAC_RESULTS_OUTPUT_DIR,
    SAC_DATASET_ID,
    _canonical_metric_view,
    _derive_metrics as derive_sac_metrics,
    _load_manifest_canonical,
    _load_sac_curves,
    _validate_canonical as validate_sac_canonical,
)
from .stats import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE,
    DEFAULT_RNG_SEED,
    summarize_paired_frame,
    summarize_sample,
)
from .validation import ROOT


OUTPUT_DIR = ROOT / "paper_tables" / "discussion_handoff_20260915"
ARCHIVE_PATH = ROOT / "paper_tables" / "discussion_handoff_20260915.tar.gz"
PHASES = ("time_early", "time_mid", "time_late")
TIE_TOLERANCE = 1e-8


def _assert_close(label: str, observed, expected, *, atol: float = 1e-9) -> None:
    if not np.allclose(
        np.asarray(observed, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=0.0,
        atol=atol,
        equal_nan=True,
    ):
        raise ValueError(f"{label} differs.")


def _phase_labels(size: int) -> np.ndarray:
    labels = np.empty(size, dtype=object)
    for phase, positions in zip(PHASES, np.array_split(np.arange(size), 3), strict=True):
        labels[positions] = phase
    return labels


def _weight_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [
        column
        for column in frame.columns
        if column.startswith(prefix)
        and column[len(prefix):].isdigit()
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    return sorted(columns, key=lambda column: int(column[len(prefix):]))


def _weight_data(frame: pd.DataFrame, prefix: str):
    columns = _weight_columns(frame, prefix)
    if not columns:
        return None
    weights = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    has_weights = np.all(np.isfinite(weights), axis=1)
    sums = weights.sum(axis=1)
    valid = (
        has_weights
        & np.all(weights >= -TIE_TOLERANCE, axis=1)
        & np.all(weights <= 1.0 + 1e-6, axis=1)
        & np.isclose(sums, 1.0, rtol=1e-4, atol=1e-5)
    )
    valid_weights = np.clip(weights[valid], 0.0, 1.0)
    entropy = -(valid_weights * np.log(np.clip(valid_weights, 1e-12, None))).sum(axis=1)
    normalized_entropy = entropy / np.log(len(columns)) if len(columns) > 1 else np.ones(len(entropy))
    effective_heads = np.exp(entropy)
    max_weight = valid_weights.max(axis=1)
    if len(columns) > 1:
        ordered = np.sort(valid_weights, axis=1)
        tied = (ordered[:, -1] - ordered[:, -2]) <= TIE_TOLERANCE
    else:
        tied = np.zeros(len(valid_weights), dtype=bool)
    dominant = valid_weights.argmax(axis=1)
    dominant[tied] = -1
    return {
        "columns": columns,
        "weights": weights,
        "valid": valid,
        "valid_weights": valid_weights,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "effective_heads": effective_heads,
        "max_weight": max_weight,
        "dominant": dominant,
        "tied": tied,
    }


def _switch_summary(dominant: np.ndarray) -> tuple[float, int]:
    if len(dominant) < 2:
        return np.nan, 0
    eligible = (dominant[:-1] >= 0) & (dominant[1:] >= 0)
    count = int(eligible.sum())
    if count == 0:
        return np.nan, 0
    return float((dominant[:-1][eligible] != dominant[1:][eligible]).mean()), count


def _mechanism_specs(registry: FigureDataRegistry):
    lunar_root = registry.load_path("lunar_primary_frozen", manuscript=True)
    for method in ("Uniform", "Naive", "Protected"):
        yield (
            "Lunar",
            method,
            "lunar_primary_frozen",
            lunar_root / F1_METHOD_DIRECTORIES[method],
            tuple(sorted(LUNAR_SEEDS)),
        )

    for method in ("Uniform", "Naive", "Anchored"):
        dataset_id, method_directory = CROP_FROZEN_METHODS[method]
        root = registry.load_path(dataset_id, manuscript=True)
        yield (
            "Crop",
            method,
            dataset_id,
            root / method_directory if method_directory else root,
            tuple(CROP_SEEDS),
        )

    for dataset_id, seeds in FROZEN_COHORTS.items():
        root = registry.load_path(dataset_id, manuscript=True)
        yield (
            "HalfCheetah",
            "AMAF",
            dataset_id,
            root / HC_METHOD_DIRECTORIES["AMAF"],
            tuple(sorted(seeds)),
        )


def load_mechanism_diagnostics(registry: FigureDataRegistry) -> pd.DataFrame:
    frames = []
    for environment, method, dataset_id, method_root, expected_seeds in _mechanism_specs(registry):
        observed_seeds = {
            int(path.name.removeprefix("seed_"))
            for path in method_root.glob("seed_*")
            if path.is_dir() and path.name.removeprefix("seed_").isdigit()
        }
        if observed_seeds != set(expected_seeds):
            raise ValueError(f"Unexpected mechanism seed set for {environment}/{method}.")
        for seed in expected_seeds:
            path = method_root / f"seed_{seed}" / "diagnostics.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path).sort_values("env_steps").reset_index(drop=True)
            if frame["env_steps"].duplicated().any():
                raise ValueError(f"Duplicate diagnostic checkpoints: {path}")
            frame.insert(0, "source_dataset_id", dataset_id)
            frame.insert(0, "seed", int(seed))
            frame.insert(0, "method", method)
            frame.insert(0, "environment", environment)
            frame.insert(4, "regime", _phase_labels(len(frame)))
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _mechanism_group_summary(group: pd.DataFrame) -> dict[str, object]:
    effective = _weight_data(group, "gate_w")
    if effective is None:
        raise ValueError("Mechanism group has no effective gate weights.")
    valid = effective["valid"]
    dominant = effective["dominant"]
    switch_rate, switch_n = _switch_summary(dominant)
    usage = [float((dominant == head).mean()) for head in range(len(effective["columns"]))]

    trust = pd.to_numeric(group.get("adaptive_trust"), errors="coerce").dropna()
    reference_weight = float((1.0 - trust).mean()) if len(trust) else np.nan
    router = _weight_data(group, "router_w")
    row: dict[str, object] = {
        "environment": group["environment"].iloc[0],
        "method": group["method"].iloc[0],
        "seed": int(group["seed"].iloc[0]),
        "source_dataset_id": group["source_dataset_id"].iloc[0],
        "n_heads": len(effective["columns"]),
        "mean_entropy": float(effective["entropy"].mean()),
        "normalized_entropy": float(effective["normalized_entropy"].mean()),
        "effective_heads": float(effective["effective_heads"].mean()),
        "mean_max_weight": float(effective["max_weight"].mean()),
        "dominant_head_fraction": max(usage),
        "dominant_tie_fraction": float(effective["tied"].mean()),
        "head_switch_rate": switch_rate,
        "n_switch_transitions": switch_n,
        "reference_or_null_weight_mean": reference_weight,
        "n_observations": int(valid.sum()),
        "n_total_observations": int(len(group)),
        "invalid_gate_rows": int((~valid).sum()),
        "invalid_gate_rate": float((~valid).mean()),
    }
    for head in range(4):
        row[f"head{head}_mean_gate_weight"] = (
            float(effective["valid_weights"][:, head].mean())
            if head < len(effective["columns"])
            else np.nan
        )

    if router is None:
        row.update({
            "router_mean_entropy": np.nan,
            "router_normalized_entropy": np.nan,
            "router_effective_heads": np.nan,
            "router_mean_max_weight": np.nan,
            "router_dominant_head_fraction": np.nan,
            "router_dominant_tie_fraction": np.nan,
            "router_switch_rate": np.nan,
            "router_n_observations": 0,
            "router_invalid_rows": 0,
        })
    else:
        router_usage = [
            float((router["dominant"] == head).mean())
            for head in range(len(router["columns"]))
        ]
        router_switch, _ = _switch_summary(router["dominant"])
        row.update({
            "router_mean_entropy": float(router["entropy"].mean()),
            "router_normalized_entropy": float(router["normalized_entropy"].mean()),
            "router_effective_heads": float(router["effective_heads"].mean()),
            "router_mean_max_weight": float(router["max_weight"].mean()),
            "router_dominant_head_fraction": max(router_usage),
            "router_dominant_tie_fraction": float(router["tied"].mean()),
            "router_switch_rate": router_switch,
            "router_n_observations": int(router["valid"].sum()),
            "router_invalid_rows": int((~router["valid"]).sum()),
        })
    return row


def build_mechanism_tables(
    diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    regime_rows = []
    long_rows = []
    keys = ["environment", "method", "seed", "source_dataset_id"]
    for _, group in diagnostics.groupby(keys, sort=False):
        group = group.sort_values("env_steps").reset_index(drop=True)
        summary_rows.append(_mechanism_group_summary(group))
        for regime, phase_group in group.groupby("regime", sort=False):
            for weight_type, prefix in (("effective_gate", "gate_w"), ("raw_router", "router_w")):
                data = _weight_data(phase_group, prefix)
                if data is None:
                    continue
                valid_group = phase_group.loc[data["valid"]].reset_index(drop=True)
                dominant = data["dominant"]
                for head in range(len(data["columns"])):
                    head_output = pd.to_numeric(
                        valid_group.get(f"head{head}_mean"), errors="coerce"
                    )
                    regime_rows.append({
                        "environment": group["environment"].iloc[0],
                        "method": group["method"].iloc[0],
                        "seed": int(group["seed"].iloc[0]),
                        "regime": regime,
                        "weight_type": weight_type,
                        "head_id": head,
                        "mean_gate_weight": float(data["valid_weights"][:, head].mean()),
                        "std_gate_weight": float(data["valid_weights"][:, head].std(ddof=1)),
                        "dominant_fraction": float((dominant == head).mean()),
                        "dominant_tie_fraction": float(data["tied"].mean()),
                        "entropy": float(data["entropy"].mean()),
                        "normalized_entropy": float(data["normalized_entropy"].mean()),
                        "effective_heads": float(data["effective_heads"].mean()),
                        "mean_max_weight": float(data["max_weight"].mean()),
                        "mean_head_output": float(head_output.mean()),
                        "std_head_output": float(head_output.std(ddof=1)),
                        "n_observations": int(data["valid"].sum()),
                        "n_total_observations": int(len(phase_group)),
                        "invalid_gate_rows": int((~data["valid"]).sum()),
                    })
                for position, diagnostic_row in valid_group.iterrows():
                    original_position = np.flatnonzero(data["valid"])[position]
                    weights = data["valid_weights"][position]
                    for head, weight in enumerate(weights):
                        long_rows.append({
                            "environment": group["environment"].iloc[0],
                            "method": group["method"].iloc[0],
                            "seed": int(group["seed"].iloc[0]),
                            "source_dataset_id": group["source_dataset_id"].iloc[0],
                            "checkpoint_or_step": int(diagnostic_row["env_steps"]),
                            "episode": int(diagnostic_row["episode"]),
                            "regime": regime,
                            "weight_type": weight_type,
                            "head_id": head,
                            "gate_weight_mean": float(weight),
                            "gate_weight_std": 0.0,
                            "head_output_mean": float(diagnostic_row[f"head{head}_mean"]),
                            "n_observations": 1,
                            "source_row_position": int(original_position),
                        })
    summary = pd.DataFrame(summary_rows).sort_values(["environment", "method", "seed"])
    by_regime = pd.DataFrame(regime_rows).sort_values(
        ["environment", "method", "seed", "regime", "weight_type", "head_id"]
    )
    long = pd.DataFrame(long_rows).sort_values(
        ["environment", "method", "seed", "checkpoint_or_step", "weight_type", "head_id"]
    )
    return summary.reset_index(drop=True), by_regime.reset_index(drop=True), long.reset_index(drop=True)


def _paired_effect(
    frame: pd.DataFrame,
    *,
    metric: str,
    focal: str,
    comparator: str,
    direction: str = "higher",
):
    return summarize_paired_frame(
        frame,
        seed_col="seed",
        method_col="method",
        metric_col=metric,
        focal=focal,
        comparator=comparator,
        direction=direction,
        confidence=DEFAULT_CONFIDENCE,
        resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        rng_seed=DEFAULT_RNG_SEED,
    )


def _effect_row(
    environment: str,
    comparison: str,
    metric: str,
    result,
    interpretation_direction: str,
    *,
    benefit: bool = False,
) -> dict[str, object]:
    return {
        "environment": environment,
        "comparison": comparison,
        "metric": metric,
        "n": result.n,
        "mean_delta": result.mean_benefit_delta if benefit else result.mean_delta_focal_minus_comparator,
        "median_delta": result.median_benefit_delta if benefit else result.median_delta_focal_minus_comparator,
        "wins": result.wins,
        "losses": result.losses,
        "ties": result.ties,
        "ci_low": result.ci95_low_benefit if benefit else result.ci95_low_focal_minus_comparator,
        "ci_high": result.ci95_high_benefit if benefit else result.ci95_high_focal_minus_comparator,
        "interpretation_direction": interpretation_direction,
    }


def _append_effect(
    rows: list[dict[str, object]],
    vectors: list[dict[str, object]],
    frame: pd.DataFrame,
    *,
    environment: str,
    focal: str,
    comparator: str,
    source_metric: str,
    output_metric: str,
    direction: str,
    interpretation: str,
) -> None:
    paired, result = _paired_effect(
        frame,
        metric=source_metric,
        focal=focal,
        comparator=comparator,
        direction=direction,
    )
    benefit = direction == "lower"
    comparison = f"{focal} - {comparator}"
    rows.append(
        _effect_row(
            environment,
            comparison,
            output_metric,
            result,
            interpretation,
            benefit=benefit,
        )
    )
    vector_column = "benefit_delta" if benefit else "delta_focal_minus_comparator"
    vectors.append({
        "environment": environment,
        "comparison": comparison,
        "metric": output_metric,
        "values": paired.set_index("seed")[vector_column],
    })


def build_ablation_and_vectors(
    lunar: pd.DataFrame,
    crop: pd.DataFrame,
    halfcheetah: pd.DataFrame,
    sac_source: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    vectors: list[dict[str, object]] = []
    for focal, comparator in (
        ("Naive", "Uniform"),
        ("Protected", "Naive"),
        ("Protected", "Uniform"),
    ):
        for source_metric, output_metric, interpretation in (
            ("last5", "last5_return", f"positive = {focal} higher Last-5"),
            ("retention_pct", "retention_pct", f"positive = {focal} higher retention (pp)"),
        ):
            _append_effect(
                rows,
                vectors,
                lunar,
                environment="Lunar",
                focal=focal,
                comparator=comparator,
                source_metric=source_metric,
                output_metric=output_metric,
                direction="higher",
                interpretation=interpretation,
            )

    for focal, comparator in (
        ("Anchored", "Uniform"),
        ("Naive", "Uniform"),
        ("Anchored", "Naive"),
    ):
        for source_metric, output_metric, direction, interpretation in (
            ("final100", "final100_return", "higher", f"positive = {focal} higher final-100 return"),
            ("yield100", "yield100", "higher", f"positive = {focal} higher final-100 yield"),
            ("N100", "nitrogen_saving", "lower", f"positive = {focal} uses less nitrogen"),
            ("irr100", "irrigation_saving", "lower", f"positive = {focal} uses less irrigation"),
        ):
            _append_effect(
                rows,
                vectors,
                crop,
                environment="Crop",
                focal=focal,
                comparator=comparator,
                source_metric=source_metric,
                output_metric=output_metric,
                direction=direction,
                interpretation=interpretation,
            )

    for source_metric, output_metric in (
        ("at_1p5m", "terminal_delta"),
        ("gain_1m_to_1p5m", "gain_delta"),
        ("last5", "last5_delta"),
        ("last10", "last10_delta"),
        ("auc_1p5m", "auc_delta"),
    ):
        _append_effect(
            rows,
            vectors,
            halfcheetah,
            environment="HalfCheetah",
            focal="AMAF",
            comparator="TD3",
            source_metric=source_metric,
            output_metric=output_metric,
            direction="higher",
            interpretation="positive = AMAF higher than TD3",
        )

    for source_metric, output_metric in (
        ("at_1p5m", "terminal_delta"),
        ("gain_1m_to_1p5m", "gain_delta"),
        ("last5", "last5_delta"),
        ("last10", "last10_delta"),
        ("auc_norm", "auc_delta"),
    ):
        _append_effect(
            rows,
            vectors,
            sac_source,
            environment="SAC context",
            focal="AMAF",
            comparator="SAC",
            source_metric=source_metric,
            output_metric=output_metric,
            direction="higher",
            interpretation="positive = AMAF higher than SAC",
        )
    ablation_rows = [row for row in rows if row["environment"] != "SAC context"]
    return pd.DataFrame(ablation_rows), vectors


def build_robustness(vectors: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in vectors:
        include = (
            (
                spec["environment"] == "Lunar"
                and spec["comparison"] in {"Protected - Naive", "Protected - Uniform"}
            )
            or (
                spec["environment"] == "Crop"
                and spec["comparison"] == "Anchored - Uniform"
            )
            or spec["environment"] == "HalfCheetah"
            or (
                spec["environment"] == "SAC context"
                and spec["metric"] in {"terminal_delta", "gain_delta", "auc_delta"}
            )
        )
        if not include:
            continue
        values: pd.Series = spec["values"].sort_index()
        exclusions: list[str | int] = ["NONE", *values.index.astype(int).tolist()]
        for excluded in exclusions:
            subset = values if excluded == "NONE" else values.drop(int(excluded))
            mean = float(subset.mean())
            rows.append({
                "environment": spec["environment"],
                "comparison": spec["comparison"],
                "metric": spec["metric"],
                "excluded_seed": excluded,
                "loo_mean_delta": mean,
                "loo_median_delta": float(subset.median()),
                "loo_wins": int((subset > 0).sum()),
                "sign_of_mean": "positive" if mean > 0 else "negative" if mean < 0 else "zero",
            })
    return pd.DataFrame(rows)


def build_crop_tradeoff(crop: pd.DataFrame) -> pd.DataFrame:
    indexed = crop.set_index(["method", "seed"])
    rows = []
    for seed in CROP_SEEDS:
        anchored = indexed.loc[("Anchored", seed)]
        uniform = indexed.loc[("Uniform", seed)]
        return_delta = float(anchored["final100"] - uniform["final100"])
        yield_delta = float(anchored["yield100"] - uniform["yield100"])
        nitrogen_saving = float(uniform["N100"] - anchored["N100"])
        irrigation_saving = float(uniform["irr100"] - anchored["irr100"])
        rows.append({
            "seed": seed,
            "return_delta": return_delta,
            "yield_delta": yield_delta,
            "nitrogen_saving": nitrogen_saving,
            "irrigation_saving": irrigation_saving,
            "resource_saved_both": int(nitrogen_saving > 0 and irrigation_saving > 0),
            "yield_improved": int(yield_delta > 0),
            "return_improved": int(return_delta > 0),
        })
    return pd.DataFrame(rows)


def _hc_seed_deltas(halfcheetah: pd.DataFrame) -> pd.DataFrame:
    wide = halfcheetah.pivot(index="seed", columns="method")
    rows = []
    for seed in sorted(HC_SEEDS):
        rows.append({
            "cohort": "original5" if seed in ORIGINAL_SEEDS else "replication5",
            "seed": seed,
            "delta_1m": float(wide["at_1m"].loc[seed, "AMAF"] - wide["at_1m"].loc[seed, "TD3"]),
            "terminal_delta_1p5m": float(wide["at_1p5m"].loc[seed, "AMAF"] - wide["at_1p5m"].loc[seed, "TD3"]),
            "gain_delta": float(wide["gain_1m_to_1p5m"].loc[seed, "AMAF"] - wide["gain_1m_to_1p5m"].loc[seed, "TD3"]),
            "last5_delta": float(wide["last5"].loc[seed, "AMAF"] - wide["last5"].loc[seed, "TD3"]),
            "last10_delta": float(wide["last10"].loc[seed, "AMAF"] - wide["last10"].loc[seed, "TD3"]),
            "auc_delta": float(wide["auc_1p5m"].loc[seed, "AMAF"] - wide["auc_1p5m"].loc[seed, "TD3"]),
        })
    return pd.DataFrame(rows)


def build_halfcheetah_late_stage(halfcheetah: pd.DataFrame) -> pd.DataFrame:
    seeds = _hc_seed_deltas(halfcheetah)
    rows = []
    metrics = ("delta_1m", "terminal_delta_1p5m", "gain_delta", "last5_delta", "last10_delta", "auc_delta")
    for row in seeds.itertuples(index=False):
        record = {
            "row_type": "seed",
            "cohort": row.cohort,
            "seed": str(row.seed),
            "n": 1,
            **{metric: float(getattr(row, metric)) for metric in metrics},
            "terminal_median": float(row.terminal_delta_1p5m),
            "gain_median": float(row.gain_delta),
            "terminal_ci_low": float(row.terminal_delta_1p5m),
            "terminal_ci_high": float(row.terminal_delta_1p5m),
            "gain_ci_low": float(row.gain_delta),
            "gain_ci_high": float(row.gain_delta),
            "terminal_wins": int(row.terminal_delta_1p5m > 0),
            "gain_wins": int(row.gain_delta > 0),
        }
        rows.append(record)

    for cohort, selected in (
        ("original5", seeds.loc[seeds["cohort"] == "original5"]),
        ("replication5", seeds.loc[seeds["cohort"] == "replication5"]),
        ("combined10", seeds),
    ):
        terminal = summarize_sample(selected["terminal_delta_1p5m"])
        gain = summarize_sample(selected["gain_delta"])
        rows.append({
            "row_type": "cohort_summary",
            "cohort": cohort,
            "seed": "ALL",
            "n": len(selected),
            **{metric: float(selected[metric].mean()) for metric in metrics},
            "terminal_median": terminal.median,
            "gain_median": gain.median,
            "terminal_ci_low": terminal.ci95_low,
            "terminal_ci_high": terminal.ci95_high,
            "gain_ci_low": gain.ci95_low,
            "gain_ci_high": gain.ci95_high,
            "terminal_wins": int((selected["terminal_delta_1p5m"] > 0).sum()),
            "gain_wins": int((selected["gain_delta"] > 0).sum()),
        })
    return pd.DataFrame(rows)


def load_sac_source(
    registry: FigureDataRegistry,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    sac_root = registry.load_path(SAC_DATASET_ID, manuscript=True)
    canonical, _ = _load_manifest_canonical(sac_root)
    sac_curves = _load_sac_curves(registry)
    td3_amaf_curves = load_halfcheetah_frozen_curves(registry)
    td3_amaf_curves = td3_amaf_curves.loc[
        td3_amaf_curves["seed"].astype(int).isin(ORIGINAL_SEEDS)
    ]
    source = derive_sac_metrics(pd.concat([td3_amaf_curves, sac_curves], ignore_index=True))
    registered = registry.load_csv("halfcheetah_per_seed", manuscript=True)
    validate_sac_canonical(source, registered, canonical)
    return source, canonical


def build_sac_context(
    source: pd.DataFrame,
    canonical: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    aggregate = canonical[
        "halfcheetah_primary5_td3_amaf_sac_aggregate.csv"
    ].set_index(["method", "metric"])
    metric_map = (
        ("at_1m", "return_1m"),
        ("at_1p5m", "return_1p5m"),
        ("gain_1m_to_1p5m", "gain"),
        ("last5", "last5"),
        ("last10", "last10"),
        ("auc_norm", "auc"),
    )
    for method in ("TD3", "AMAF", "SAC"):
        for source_metric, output_metric in metric_map:
            result = aggregate.loc[(method, source_metric)]
            rows.append({
                "record_type": "method_summary",
                "method": method,
                "comparison": "not_applicable",
                "metric": output_metric,
                "n": int(result["n"]),
                "mean": float(result["mean"]),
                "median": float(result["median"]),
                "ci_low": float(result["ci_low"]),
                "ci_high": float(result["ci_high"]),
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "direction": "descriptive absolute method metric",
            })
    for source_metric, output_metric in (
        ("at_1p5m", "terminal_delta"),
        ("gain_1m_to_1p5m", "gain_delta"),
        ("last5", "last5_delta"),
        ("last10", "last10_delta"),
        ("auc_norm", "auc_delta"),
    ):
        _, result = _paired_effect(
            source,
            metric=source_metric,
            focal="AMAF",
            comparator="SAC",
        )
        rows.append({
            "record_type": "paired_effect",
            "method": "not_applicable",
            "comparison": "AMAF - SAC",
            "metric": output_metric,
            "n": result.n,
            "mean": result.mean_delta_focal_minus_comparator,
            "median": result.median_delta_focal_minus_comparator,
            "ci_low": result.ci95_low_focal_minus_comparator,
            "ci_high": result.ci95_high_focal_minus_comparator,
            "wins": result.wins,
            "losses": result.losses,
            "ties": result.ties,
            "direction": "positive = AMAF higher than SAC",
        })
    return pd.DataFrame(rows)


def _validate_results_alignment(
    lunar: pd.DataFrame,
    crop: pd.DataFrame,
    hc_late: pd.DataFrame,
    sac_context: pd.DataFrame,
) -> None:
    lunar_handoff = pd.read_csv(RESULTS_OUTPUT_DIR / "lunar_per_seed.csv")
    lunar_source = lunar[["seed", "method", "last5", "retention_pct"]].rename(
        columns={"last5": "last5_return"}
    )
    left = lunar_source.sort_values(["method", "seed"]).reset_index(drop=True)
    right = lunar_handoff.sort_values(["method", "seed"]).reset_index(drop=True)
    _assert_close("Lunar Results handoff", left[["last5_return", "retention_pct"]], right[["last5_return", "retention_pct"]])

    crop_handoff = pd.read_csv(RESULTS_OUTPUT_DIR / "crop_per_seed.csv")
    crop_source = crop[["seed", "method", "late1000", "final100", "yield100", "N100", "irr100"]].rename(
        columns={"late1000": "late1000_return", "final100": "final100_return", "N100": "n100"}
    )
    left = crop_source.sort_values(["method", "seed"]).reset_index(drop=True)
    right = crop_handoff.sort_values(["method", "seed"]).reset_index(drop=True)
    columns = ["late1000_return", "final100_return", "yield100", "n100", "irr100"]
    _assert_close("Crop Results handoff", left[columns], right[columns])

    hc_handoff = pd.read_csv(RESULTS_OUTPUT_DIR / "halfcheetah_per_seed.csv").set_index("seed")
    hc_seed = hc_late.loc[hc_late["row_type"] == "seed"].copy()
    hc_seed["seed"] = hc_seed["seed"].astype(int)
    hc_seed = hc_seed.set_index("seed")
    for discussion_col, result_col in (
        ("delta_1m", "delta_1m"),
        ("terminal_delta_1p5m", "terminal_delta"),
        ("gain_delta", "gain_delta"),
        ("last5_delta", "last5_delta"),
        ("last10_delta", "last10_delta"),
        ("auc_delta", "auc_delta"),
    ):
        _assert_close(f"HalfCheetah Results {discussion_col}", hc_seed[discussion_col], hc_handoff[result_col])

    sac_summary = pd.read_csv(SAC_RESULTS_OUTPUT_DIR / "halfcheetah_sac_primary5_summary.csv")
    method_context = sac_context.loc[sac_context["record_type"] == "method_summary"]
    for row in sac_summary.itertuples(index=False):
        for result_col, metric in (
            ("return_1m_mean", "return_1m"),
            ("return_1p5m_mean", "return_1p5m"),
            ("gain_mean", "gain"),
            ("last5_mean", "last5"),
            ("last10_mean", "last10"),
            ("auc_mean", "auc"),
        ):
            observed = method_context.loc[
                (method_context["method"] == row.method) & (method_context["metric"] == metric), "mean"
            ].iloc[0]
            _assert_close(f"SAC Results {row.method}/{metric}", observed, getattr(row, result_col))
    sac_paired = pd.read_csv(SAC_RESULTS_OUTPUT_DIR / "halfcheetah_amaf_vs_sac_paired.csv")
    paired_context = sac_context.loc[sac_context["record_type"] == "paired_effect"].set_index("metric")
    for row in sac_paired.itertuples(index=False):
        observed = paired_context.loc[row.metric]
        for discussion_col, result_col in (
            ("mean", "mean_delta"),
            ("median", "median_delta"),
            ("ci_low", "ci_low"),
            ("ci_high", "ci_high"),
        ):
            _assert_close(f"SAC paired {row.metric}/{discussion_col}", observed[discussion_col], getattr(row, result_col))


def _source_lines(registry: FigureDataRegistry) -> str:
    dataset_ids = (
        "lunar_primary_frozen",
        "lunar_per_seed",
        "crop_baseline_primary_frozen",
        "crop_anchored_primary_frozen",
        "crop_per_seed",
        "halfcheetah_primary_frozen",
        "halfcheetah_replication_frozen",
        "halfcheetah_per_seed",
        SAC_DATASET_ID,
    )
    lines = []
    for dataset_id in dataset_ids:
        fingerprint = registry.fingerprint(dataset_id)
        lines.append(
            f"- `{dataset_id}`: `{fingerprint['path']}`; lifecycle `{fingerprint['lifecycle']}`; "
            f"{fingerprint['fingerprint_type']} `{fingerprint['sha256']}`"
        )
    return "\n".join(lines)


def _readme(registry: FigureDataRegistry, frames: dict[str, pd.DataFrame]) -> str:
    mechanism = frames["mechanism_head_summary.csv"]
    invalid = int(mechanism["invalid_gate_rows"].sum())
    valid = int(mechanism["n_observations"].sum())
    return f"""# AMAF Discussion evidence handoff (2026-09-15)

## 생성 파일과 Discussion 질문

- `mechanism_head_summary.csv`: head 사용의 전역 집중도·entropy·switching; 질문 1–3
- `mechanism_head_by_regime.csv`: state regime 대신 time-phase별 head/router 요약; 질문 1–3
- `mechanism_head_long.csv`: checkpoint × head long-format 진단; 질문 1–3
- `ablation_discussion_summary.csv`: 보호/anchoring 및 기준법 contrast; 질문 3
- `robustness_seed_sensitivity.csv`: 전체 및 leave-one-seed-out 민감도; 질문 7
- `crop_tradeoff_discussion.csv`: paired resource/performance/yield trade-off; 질문 4
- `halfcheetah_late_stage_discussion.csv`: terminal과 1M→1.5M gain 분리; 질문 5, 7
- `sac_discussion_context.csv`: primary-five TD3/AMAF/SAC와 AMAF−SAC; 질문 6
- `DISCUSSION_CLAIM_MAP.md`: claim-to-evidence와 limitation mapping

## Source registry

{_source_lines(registry)}

모든 source는 `figure_contract.yaml` registry ID로 해결했다. `paper_data`, canonical table,
MANIFEST, SHA256SUMS는 수정하지 않았다. Seed population은 Lunar 10개(11–111의 지정 seed),
Crop 5개(11–55), HalfCheetah combined10, SAC primary5이며 seed 77을 포함해 제외한 seed는 없다.

## Mechanism data availability and extraction

Frozen `diagnostics.csv`에는 diagnostic checkpoint의 `env_steps`, `episode`, effective
`gate_w*`, entropy/max-weight/effective-head/dominant-head와 `head*_mean`이 있다. Lunar
Protected 및 Crop Anchored에는 raw `router_w*`와 `adaptive_trust`도 있다. Effective gate와
raw router는 섞지 않고 `weight_type`으로 분리했다.

Probability-simplex 조건(유한·비음수·각 weight ≤1·합≈1)을 통과한 행만 gate distribution
통계와 long table에 사용했다. 전체 유효 행은 {valid:,}개이며 제외된 invalid conditional
gate 행은 {invalid:,}개다. Entropy는 각 행에서 직접 재계산했고 normalized entropy는
`H/log(n_heads)`, effective heads는 각 행의 `exp(H)`를 평균했다. Equal-maximum tie는
임의 dominant head로 지정하지 않았다. `dominant_head_fraction`은 유효 관측 전체 중 가장
자주 유일 dominant였던 head의 비율이며 tie 비율을 별도로 제공한다. Switching은 시간상
인접하고 양쪽 모두 unique-dominant인 관측만 분모로 사용한다.

Protected/Anchored의 `reference_or_null_weight_mean`은 기록된 `1-adaptive_trust`, 즉
uniform/reference 쪽 계수다. HalfCheetah selective AMAF의 explicit NULL weight는 frozen
historical logger에 기록되지 않아 재구성하지 않았다. `reference_or_null_weight_mean`,
router 관련 열, head 수가 적은 method의 여분 head 열, dominant tie만 존재해 switching이
정의되지 않는 경우의 NaN은 모두 구조적으로 정당한 missing이다.

## Regime definition and missing mechanism evidence

State feature, action, altitude/velocity, crop growth/resource 상태, 의미론적 regime label은
frozen diagnostics에 기록되지 않았다. 따라서 `time_early`, `time_mid`, `time_late`는 각
seed의 기록된 diagnostic checkpoint를 시간순으로 3등분한 대체 구간이며 state-conditioned
regime가 아니다. 각 diagnostic row도 full state distribution이 아니라 logging 시점의 한
state 표본이다.

**MISSING / NOT RECORDED:** state-conditioned gate distribution, semantic regime별 head
usage, HalfCheetah full H+1 gate/NULL weight, state/action과 head output의 완전한 alignment는
현재 frozen logs에서 재구성할 수 없다. 따라서 head specialization/collapse에 대한 직접
state-conditioned empirical evidence는 현재 frozen logs에서 재구성할 수 없으며, Results
성능 수치만으로 mechanism을 추론하지 않는다. 이 package는 checkpoint-time selectivity와
switching을 기술통계로 제공할 뿐이다.

## Metric definitions and bootstrap

- Lunar: Last-5 evaluation return, retention=`100×Last-5/Best-5`.
- Crop: final-100 train return/yield/nitrogen/irrigation. Saving은 comparator−focal usage.
- HalfCheetah: exact 1M/1.5M evaluation return, gain=1.5M−1M, Last-5/Last-10,
  10k–1.5M time-normalized trapezoidal AUC.
- SAC context는 같은 primary-five seed와 동일 checkpoint/metric 정의를 사용한다.
- CI는 seed-level percentile bootstrap mean CI, {DEFAULT_BOOTSTRAP_RESAMPLES:,} resamples,
  95% confidence, RNG seed {DEFAULT_RNG_SEED}. Leave-one-out은 지정 seed를 한 번씩 제외하되
  전체 row(`excluded_seed=NONE`)도 포함한다.

## Limitations and counterevidence / boundary cases

- Lunar: Protected는 Uniform을 일관되게 능가하지 않으며 Protected−Naive도 일부 seed에서
  음수다. 낮은 entropy만으로 좋은 specialization이라 해석하지 않는다.
- Crop: Anchored는 Uniform보다 질소·관개를 5/5 seed에서 절감하지만 final-100 return과
  yield는 각각 1/5 seed에서만 개선됐다. Resource saving은 performance/yield cost와 함께
  나타날 수 있다.
- HalfCheetah: combined terminal CI가 넓고 0을 포함한다. seed 77을 포함한 모든 seed를
  유지했으며 Last-5/Last-10/AUC combined mean은 양수가 아니다.
- SAC: SAC는 primary-five 1M return과 AUC가 AMAF보다 높다. AMAF terminal wins는 2/5로,
  terminal mean 우세가 seed-wise dominance를 뜻하지 않는다. Sample-efficiency superiority
  claim은 지원되지 않는다.
- SAC frozen `SHA256SUMS`에 선언된 model checkpoint weight 15개가 tree에 없어 full archive
  checksum은 PASS가 아니다. Numerical input인 MANIFEST/COMPLETED/eval CSV는 검증됐다.

기존 Results handoff와 공통 metric은 export 전에 수치 단위로 일치 검증했다.
"""


def _claim_map() -> str:
    return """# Discussion claim-to-evidence map

## Claim 1

Potential claim: Protection mitigates severe adaptive failures in Lunar.

Supported by:
- `ablation_discussion_summary.csv`: Protected−Naive Last-5/retention
- `robustness_seed_sensitivity.csv`: 해당 contrast의 전체 및 leave-one-seed-out 결과
- `mechanism_head_summary.csv`, `mechanism_head_by_regime.csv`: checkpoint-time gate/router 기술통계

Metric/direction: 양의 Protected−Naive Last-5 또는 retention(pp)는 Protected 우세.

Limits:
- Protected는 Uniform보다 일관되게 높지 않다.
- 일부 Protected−Naive seed delta는 음수다.
- state/regime가 기록되지 않아 time-varying routing을 conditional specialization의 직접 증거로 쓸 수 없다.

## Claim 2

Potential claim: Anchoring moves Crop control toward a lower-resource operating point.

Supported by:
- `crop_tradeoff_discussion.csv`: seed별 nitrogen/irrigation saving
- `ablation_discussion_summary.csv`: Anchored−Uniform 및 Anchored−Naive contrast
- `robustness_seed_sensitivity.csv`: resource-saving LOO 부호

Metric/direction: nitrogen/irrigation saving은 comparator usage−Anchored usage; 양수는 Anchored 절감.

Limits:
- `crop_tradeoff_discussion.csv`의 yield/return delta가 음수인 seed가 다수다.
- 임의 efficiency composite score를 만들지 않았으며 resource saving과 outcome cost를 함께 보고해야 한다.

## Claim 3

Potential claim: AMAF shows late-stage relative improvement in HalfCheetah.

Supported by:
- `halfcheetah_late_stage_discussion.csv`: original5/replication5/combined10 gain delta
- `ablation_discussion_summary.csv`: combined AMAF−TD3 terminal/gain/late-window/AUC
- `robustness_seed_sensitivity.csv`: 모든 seed와 LOO 결과

Metric/direction: gain delta=`(AMAF 1.5M−1M)−(TD3 1.5M−1M)`; 양수는 AMAF의 더 큰 late gain.

Limits:
- Terminal superiority와 gain improvement는 동일 주장이 아니다.
- Terminal CI는 0을 포함하고 Last-5/Last-10/AUC combined mean은 양수가 아니다.
- 불리한 seed 77도 포함했다.

## Claim 4

Potential claim: AMAF late-stage gain is visible relative to SAC, but this does not establish sample-efficiency superiority.

Supported by:
- `sac_discussion_context.csv`: TD3/AMAF/SAC absolute metrics와 AMAF−SAC paired gain
- `robustness_seed_sensitivity.csv`: AMAF−SAC terminal/gain/AUC LOO

Metric/direction: paired delta는 AMAF−SAC; gain delta가 양수면 AMAF의 1M→1.5M 증가가 더 큼.

Limits:
- SAC의 1M return과 normalized AUC가 더 높다.
- AMAF terminal wins는 2/5이고 terminal CI는 0을 포함한다.
- 따라서 terminal mean 또는 late gain만으로 전반적 sample-efficiency 우위를 주장하지 않는다.

## Mechanism boundary

Potential claim: checkpoint-time routing becomes selective or switches heads.

Supported by:
- `mechanism_head_summary.csv`: entropy, max weight, unique-dominant concentration, switching
- `mechanism_head_by_regime.csv`: early/mid/late time-phase contrasts
- `mechanism_head_long.csv`: checkpoint-level effective gate/raw router/head-output 기록

Limits:
- 이 증거는 semantic state/regime-conditioned specialization이 아니다.
- 낮은 entropy와 높은 dominant fraction은 conditional specialization뿐 아니라 global collapse와도 양립한다.
- state features가 없으므로 specialization과 collapse의 결정적 구분은 현재 frozen data로 불가능하다.
"""


def _validate_frames(frames: dict[str, pd.DataFrame]) -> None:
    expected_rows = {
        "mechanism_head_summary.csv": 55,
        "mechanism_head_by_regime.csv": 780,
        "ablation_discussion_summary.csv": 23,
        "robustness_seed_sensitivity.csv": 141,
        "crop_tradeoff_discussion.csv": 5,
        "halfcheetah_late_stage_discussion.csv": 13,
        "sac_discussion_context.csv": 23,
    }
    for filename, count in expected_rows.items():
        if len(frames[filename]) != count:
            raise ValueError(f"{filename} has {len(frames[filename])} rows, expected {count}.")
    if len(frames["mechanism_head_long.csv"]) == 0:
        raise ValueError("mechanism_head_long.csv is empty.")

    if frames["mechanism_head_summary.csv"].duplicated(["environment", "method", "seed"]).any():
        raise ValueError("Mechanism summary contains duplicate keys.")
    if frames["mechanism_head_by_regime.csv"].duplicated(
        ["environment", "method", "seed", "regime", "weight_type", "head_id"]
    ).any():
        raise ValueError("Mechanism regime table contains duplicate keys.")
    if frames["mechanism_head_long.csv"].duplicated(
        ["environment", "method", "seed", "checkpoint_or_step", "weight_type", "head_id"]
    ).any():
        raise ValueError("Mechanism long table contains duplicate keys.")
    if frames["ablation_discussion_summary.csv"].duplicated(
        ["environment", "comparison", "metric"]
    ).any():
        raise ValueError("Ablation table contains duplicate keys.")
    if frames["robustness_seed_sensitivity.csv"].duplicated(
        ["environment", "comparison", "metric", "excluded_seed"]
    ).any():
        raise ValueError("Robustness table contains duplicate keys.")

    allowed_nan = {
        "mechanism_head_summary.csv": {
            "head3_mean_gate_weight",
            "head_switch_rate",
            "reference_or_null_weight_mean",
            "router_mean_entropy",
            "router_normalized_entropy",
            "router_effective_heads",
            "router_mean_max_weight",
            "router_dominant_head_fraction",
            "router_dominant_tie_fraction",
            "router_switch_rate",
        }
    }
    for filename, frame in frames.items():
        nan_columns = set(frame.columns[frame.isna().any()])
        if nan_columns - allowed_nan.get(filename, set()):
            raise ValueError(f"{filename} has unjustified NaN columns: {sorted(nan_columns)}")

    crop = frames["crop_tradeoff_discussion.csv"]
    if crop["seed"].tolist() != list(CROP_SEEDS):
        raise ValueError("Crop tradeoff seed set changed.")
    hc = frames["halfcheetah_late_stage_discussion.csv"]
    hc_seed = hc.loc[hc["row_type"] == "seed", "seed"].astype(int)
    if set(hc_seed) != set(HC_SEEDS) or 77 not in set(hc_seed):
        raise ValueError("HalfCheetah seed set is incomplete.")
    if set(frames["mechanism_head_by_regime.csv"]["regime"]) != set(PHASES):
        raise ValueError("Mechanism time-phase labels changed.")


def export() -> dict[str, pd.DataFrame]:
    if OUTPUT_DIR.exists() or ARCHIVE_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing Discussion handoff or archive.")
    registry = FigureDataRegistry()
    diagnostics = load_mechanism_diagnostics(registry)
    mechanism_summary, mechanism_regime, mechanism_long = build_mechanism_tables(diagnostics)

    lunar = registry.load_csv("lunar_per_seed", manuscript=True)
    crop = registry.load_csv("crop_per_seed", manuscript=True)
    halfcheetah = registry.load_csv("halfcheetah_per_seed", manuscript=True)
    sac_source, sac_canonical = load_sac_source(registry)

    ablation, vectors = build_ablation_and_vectors(lunar, crop, halfcheetah, sac_source)
    robustness = build_robustness(vectors)
    crop_tradeoff = build_crop_tradeoff(crop)
    hc_late = build_halfcheetah_late_stage(halfcheetah)
    sac_context = build_sac_context(sac_source, sac_canonical)
    _validate_results_alignment(lunar, crop, hc_late, sac_context)

    frames = {
        "mechanism_head_summary.csv": mechanism_summary,
        "mechanism_head_by_regime.csv": mechanism_regime,
        "mechanism_head_long.csv": mechanism_long,
        "ablation_discussion_summary.csv": ablation,
        "robustness_seed_sensitivity.csv": robustness,
        "crop_tradeoff_discussion.csv": crop_tradeoff,
        "halfcheetah_late_stage_discussion.csv": hc_late,
        "sac_discussion_context.csv": sac_context,
    }
    _validate_frames(frames)

    temporary = Path(tempfile.mkdtemp(prefix=".discussion_handoff_20260915_", dir=OUTPUT_DIR.parent))
    try:
        for filename, frame in frames.items():
            frame.to_csv(temporary / filename, index=False)
        (temporary / "DISCUSSION_DATA_README.md").write_text(
            _readme(registry, frames), encoding="utf-8"
        )
        (temporary / "DISCUSSION_CLAIM_MAP.md").write_text(_claim_map(), encoding="utf-8")
        os.rename(temporary, OUTPUT_DIR)
    except Exception:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
        raise

    archive_temporary = Path(f"{ARCHIVE_PATH}.tmp")
    try:
        with tarfile.open(archive_temporary, "w:gz") as archive:
            archive.add(OUTPUT_DIR, arcname=OUTPUT_DIR.name)
        os.rename(archive_temporary, ARCHIVE_PATH)
    except Exception:
        if archive_temporary.exists():
            archive_temporary.unlink()
        raise

    for filename, frame in frames.items():
        print(f"Wrote {OUTPUT_DIR.relative_to(ROOT) / filename} ({len(frame)} rows).")
    print(f"Wrote {OUTPUT_DIR.relative_to(ROOT) / 'DISCUSSION_DATA_README.md'}.")
    print(f"Wrote {OUTPUT_DIR.relative_to(ROOT) / 'DISCUSSION_CLAIM_MAP.md'}.")
    print(f"Wrote {ARCHIVE_PATH.relative_to(ROOT)}.")
    print("Discussion handoff and Results alignment: PASS")
    return frames


def main() -> None:
    export()


if __name__ == "__main__":
    main()
