#!/usr/bin/env python3
"""Freeze manuscript Results tables from the approved figure evidence."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .crop import (
    EXPECTED_SEEDS as CROP_SEEDS,
    METHOD_ORDER as CROP_METHODS,
    PANEL_A_MAIN_PATH as CROP_PANEL_A_PATH,
    PANEL_C_PROFILE_PATH as CROP_PANEL_C_PATH,
    _validate_canonical as validate_crop_canonical,
    build_panel_a as build_crop_panel_a,
    build_panel_c_outcome_profile,
    load_crop_frozen_episodes,
)
from .halfcheetah import (
    EXPECTED_SEEDS as HALFCHEETAH_SEEDS,
    FIGURE_DATA_PATH as HALFCHEETAH_PANEL_A_PATH,
    METHOD_DIRECTORIES as HALFCHEETAH_METHODS,
    ORIGINAL_SEEDS,
    PANEL_B_DATA_PATH as HALFCHEETAH_PANEL_B_PATH,
    PANEL_C_DATA_PATH as HALFCHEETAH_PANEL_C_PATH,
    REPLICATION_SEEDS,
    build_effect_summary,
    build_transition_table,
    load_halfcheetah_frozen_curves,
)
from .io import FigureDataRegistry
from .lunar import (
    EXPECTED_SEEDS as LUNAR_SEEDS,
    F1_DATA_PATH as LUNAR_F1_PATH,
    F2_DELTA_BAR_DATA_PATH as LUNAR_F2_PATH,
    F4_METHOD_ORDER as LUNAR_METHODS,
    F4_POINTS_DATA_PATH as LUNAR_F4_POINTS_PATH,
    F4_SUMMARY_DATA_PATH as LUNAR_F4_SUMMARY_PATH,
    build_f1_learning,
    build_f2_delta_bars,
    build_f2_failure_rescue,
    build_f4_method_data,
    load_lunar_frozen_curves,
)
from .stats import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE,
    DEFAULT_RNG_SEED,
    summarize_curve_by_seed,
    summarize_paired_frame,
    summarize_sample,
)
from .validation import ROOT


OUTPUT_DIR = ROOT / "paper_tables" / "results_handoff_20260915"


def _assert_close(label: str, observed, expected, *, atol: float = 1e-9) -> None:
    observed_array = np.asarray(observed, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if not np.allclose(
        observed_array,
        expected_array,
        rtol=0.0,
        atol=atol,
        equal_nan=False,
    ):
        difference = float(np.max(np.abs(observed_array - expected_array)))
        raise ValueError(f"{label} differs (maximum absolute difference={difference}).")


def _assert_frame_equal(label: str, observed: pd.DataFrame, expected: pd.DataFrame) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_categorical=False,
            check_names=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-9,
        )
    except AssertionError as error:
        raise ValueError(f"{label} figure-data consistency check failed: {error}") from error


def _sample_columns(values, prefix: str) -> dict[str, float | int]:
    summary = summarize_sample(
        values,
        confidence=DEFAULT_CONFIDENCE,
        resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        rng_seed=DEFAULT_RNG_SEED,
    )
    return {
        f"{prefix}_mean": summary.mean,
        f"{prefix}_median": summary.median,
        f"{prefix}_ci_low": summary.ci95_low,
        f"{prefix}_ci_high": summary.ci95_high,
    }


def _paired(
    canonical: pd.DataFrame,
    *,
    metric: str,
    focal: str,
    comparator: str,
    direction: str,
):
    return summarize_paired_frame(
        canonical,
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


def _paired_row(
    comparison: str,
    metric: str,
    summary,
    *,
    benefit: bool = False,
) -> dict[str, float | int | str]:
    if benefit:
        mean_delta = summary.mean_benefit_delta
        median_delta = summary.median_benefit_delta
        ci_low = summary.ci95_low_benefit
        ci_high = summary.ci95_high_benefit
    else:
        mean_delta = summary.mean_delta_focal_minus_comparator
        median_delta = summary.median_delta_focal_minus_comparator
        ci_low = summary.ci95_low_focal_minus_comparator
        ci_high = summary.ci95_high_focal_minus_comparator
    return {
        "comparison": comparison,
        "metric": metric,
        "n": summary.n,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": summary.wins,
        "losses": summary.losses,
        "ties": summary.ties,
    }


def _validate_method_seed_table(
    frame: pd.DataFrame,
    *,
    label: str,
    methods,
    seeds,
) -> None:
    expected = {(method, int(seed)) for method in methods for seed in seeds}
    observed = set(frame[["method", "seed"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(f"{label} has an unexpected method/seed population.")
    if frame.duplicated(["method", "seed"]).any():
        raise ValueError(f"{label} contains duplicate method/seed keys.")
    if frame.isna().any().any():
        raise ValueError(f"{label} contains NaN values.")


def build_lunar(registry: FigureDataRegistry) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = registry.load_csv("lunar_per_seed", manuscript=True)
    canonical = canonical.copy()
    canonical["seed"] = canonical["seed"].astype(int)
    _validate_method_seed_table(
        canonical[["method", "seed", "last5", "retention_pct"]],
        label="Lunar canonical table",
        methods=LUNAR_METHODS,
        seeds=sorted(LUNAR_SEEDS),
    )

    frozen = load_lunar_frozen_curves(registry)
    derived_rows = []
    for (method, seed), curve in frozen.groupby(["method", "seed"], sort=False):
        values = curve.sort_values("env_steps")["evaluation_return"]
        last5 = float(values.tail(5).mean())
        best5 = float(values.nlargest(5).mean())
        derived_rows.append({
            "method": method,
            "seed": int(seed),
            "final": float(values.iloc[-1]),
            "last5": last5,
            "retention_pct": 100.0 * last5 / best5,
        })
    derived = pd.DataFrame(derived_rows).sort_values(["method", "seed"])
    anchor = canonical[["method", "seed", "final", "last5", "retention_pct"]].sort_values(
        ["method", "seed"]
    )
    _assert_frame_equal("Lunar frozen/canonical metrics", derived, anchor)

    summary_rows = []
    for method in LUNAR_METHODS:
        method_frame = canonical.loc[canonical["method"] == method]
        row: dict[str, float | int | str] = {"method": method, "n": len(method_frame)}
        row.update(_sample_columns(method_frame["last5"], "last5"))
        retention = _sample_columns(method_frame["retention_pct"], "retention")
        row.update({f"{key}_pct": value for key, value in retention.items()})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)[[
        "method",
        "n",
        "last5_mean",
        "last5_median",
        "last5_ci_low",
        "last5_ci_high",
        "retention_mean_pct",
        "retention_median_pct",
        "retention_ci_low_pct",
        "retention_ci_high_pct",
    ]]

    paired_rows = []
    for comparator in ("Naive", "Uniform"):
        comparison = f"Protected - {comparator}"
        for source_metric, output_metric in (
            ("last5", "last5_return"),
            ("retention_pct", "retention_pct"),
        ):
            _, result = _paired(
                canonical,
                metric=source_metric,
                focal="Protected",
                comparator=comparator,
                direction="higher",
            )
            paired_rows.append(_paired_row(comparison, output_metric, result))
    paired = pd.DataFrame(paired_rows)

    per_seed = (
        canonical[["seed", "method", "last5", "retention_pct"]]
        .rename(columns={"last5": "last5_return"})
        .copy()
    )
    per_seed["method"] = pd.Categorical(per_seed["method"], LUNAR_METHODS, ordered=True)
    per_seed = per_seed.sort_values(["method", "seed"]).reset_index(drop=True)
    per_seed["method"] = per_seed["method"].astype("object")

    # Current Lunar renderer inputs must exactly reproduce from these sources.
    current_f1 = pd.read_csv(LUNAR_F1_PATH)
    computed_f1 = build_f1_learning(registry, frozen, canonical)
    _assert_frame_equal("Lunar F1", current_f1, computed_f1)
    computed_f2 = build_f2_delta_bars(build_f2_failure_rescue(canonical))
    _assert_frame_equal("Lunar F2", pd.read_csv(LUNAR_F2_PATH), computed_f2)
    computed_f4_summary, computed_f4_points = build_f4_method_data(canonical)
    _assert_frame_equal(
        "Lunar method summary",
        pd.read_csv(LUNAR_F4_SUMMARY_PATH),
        computed_f4_summary,
    )
    _assert_frame_equal(
        "Lunar seed points",
        pd.read_csv(LUNAR_F4_POINTS_PATH),
        computed_f4_points,
    )

    return summary, paired, per_seed


def build_crop(registry: FigureDataRegistry) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = registry.load_csv("crop_per_seed", manuscript=True).copy()
    canonical["seed"] = canonical["seed"].astype(int)
    validate_crop_canonical(canonical)
    episodes = load_crop_frozen_episodes(registry)

    # The figure exporter validates return windows; validate final-100 outcomes here.
    outcome = (
        episodes.sort_values("episode")
        .groupby(["method", "seed"], as_index=False, sort=True)
        .tail(100)
        .groupby(["method", "seed"], as_index=False, sort=True)
        .agg(yield100=("yield", "mean"), N100=("nitrogen", "mean"), irr100=("irrigation", "mean"))
        .sort_values(["method", "seed"])
    )
    anchor = canonical[["method", "seed", "yield100", "N100", "irr100"]].sort_values(
        ["method", "seed"]
    )
    _assert_frame_equal("Crop frozen/canonical final-100 outcomes", outcome, anchor)

    summary_rows = []
    metric_prefixes = (
        ("late1000", "late1000_return"),
        ("final100", "final100_return"),
        ("yield100", "yield100"),
        ("N100", "n100"),
        ("irr100", "irr100"),
    )
    for method in CROP_METHODS:
        method_frame = canonical.loc[canonical["method"] == method]
        row: dict[str, float | int | str] = {"method": method, "n": len(method_frame)}
        for source_metric, prefix in metric_prefixes:
            row.update(_sample_columns(method_frame[source_metric], prefix))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    paired_rows = []
    for source_metric, output_metric, direction in (
        ("final100", "final100_return", "higher"),
        ("yield100", "yield100", "higher"),
        ("N100", "nitrogen_saving", "lower"),
        ("irr100", "irrigation_saving", "lower"),
    ):
        _, result = _paired(
            canonical,
            metric=source_metric,
            focal="Anchored",
            comparator="Uniform",
            direction=direction,
        )
        paired_rows.append(
            _paired_row(
                "Anchored vs Uniform",
                output_metric,
                result,
                benefit=direction == "lower",
            )
        )
    paired = pd.DataFrame(paired_rows)

    per_seed = canonical[[
        "seed",
        "method",
        "late1000",
        "final100",
        "yield100",
        "N100",
        "irr100",
    ]].rename(columns={
        "late1000": "late1000_return",
        "final100": "final100_return",
        "N100": "n100",
    })
    per_seed = per_seed.copy()
    per_seed["method"] = pd.Categorical(per_seed["method"], CROP_METHODS, ordered=True)
    per_seed = per_seed.sort_values(["method", "seed"]).reset_index(drop=True)
    per_seed["method"] = per_seed["method"].astype("object")

    computed_panel_a, _ = build_crop_panel_a(episodes, canonical)
    _assert_frame_equal("Crop Figure A", pd.read_csv(CROP_PANEL_A_PATH), computed_panel_a)
    computed_panel_c = build_panel_c_outcome_profile(canonical)
    _assert_frame_equal("Crop Figure C", pd.read_csv(CROP_PANEL_C_PATH), computed_panel_c)

    return summary, paired, per_seed


def _normalized_auc(curve: pd.DataFrame) -> float:
    ordered = curve.sort_values("env_steps")
    x = ordered["env_steps"].to_numpy(dtype=float)
    y = ordered["evaluation_return"].to_numpy(dtype=float)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def _build_halfcheetah_learning(frozen: pd.DataFrame) -> pd.DataFrame:
    summaries = {}
    for method in HALFCHEETAH_METHODS:
        frame = summarize_curve_by_seed(
            frozen.loc[frozen["method"] == method],
            seed_col="seed",
            x_col="env_steps",
            value_col="evaluation_return",
            confidence=DEFAULT_CONFIDENCE,
            resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            rng_seed=DEFAULT_RNG_SEED,
        )
        summaries[method] = frame[["env_steps", "mean", "ci95_low", "ci95_high"]].rename(
            columns={
                "mean": f"{method.lower()}_mean",
                "ci95_low": f"{method.lower()}_ci_low",
                "ci95_high": f"{method.lower()}_ci_high",
            }
        )
    learning = summaries["TD3"].merge(summaries["AMAF"], on="env_steps", validate="one_to_one")
    checkpoint_pairs = frozen.pivot(
        index=["seed", "env_steps"], columns="method", values="evaluation_return"
    ).reset_index()
    checkpoint_pairs["delta"] = checkpoint_pairs["AMAF"] - checkpoint_pairs["TD3"]
    delta = summarize_curve_by_seed(
        checkpoint_pairs,
        seed_col="seed",
        x_col="env_steps",
        value_col="delta",
        confidence=DEFAULT_CONFIDENCE,
        resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        rng_seed=DEFAULT_RNG_SEED,
    )[["env_steps", "mean", "ci95_low", "ci95_high"]].rename(columns={
        "mean": "delta_mean",
        "ci95_low": "delta_ci_low",
        "ci95_high": "delta_ci_high",
    })
    return learning.merge(delta, on="env_steps", validate="one_to_one")


def build_halfcheetah(registry: FigureDataRegistry) -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = registry.load_csv("halfcheetah_per_seed", manuscript=True).copy()
    canonical["seed"] = canonical["seed"].astype(int)
    frozen = load_halfcheetah_frozen_curves(registry)

    derived_rows = []
    for (method, seed), curve in frozen.groupby(["method", "seed"], sort=False):
        ordered = curve.sort_values("env_steps")
        values = ordered.set_index("env_steps")["evaluation_return"]
        at_1m = float(values.loc[1_000_000])
        at_1p5m = float(values.loc[1_500_000])
        derived_rows.append({
            "method": method,
            "seed": int(seed),
            "at_1m": at_1m,
            "at_1p5m": at_1p5m,
            "gain_1m_to_1p5m": at_1p5m - at_1m,
            "last5": float(values.tail(5).mean()),
            "last10": float(values.tail(10).mean()),
            "auc_1p5m": _normalized_auc(ordered),
        })
    derived = pd.DataFrame(derived_rows).sort_values(["method", "seed"])
    anchor = canonical[[
        "method",
        "seed",
        "at_1m",
        "at_1p5m",
        "gain_1m_to_1p5m",
        "last5",
        "last10",
        "auc_1p5m",
    ]].sort_values(["method", "seed"])
    _assert_frame_equal("HalfCheetah frozen/canonical metrics", derived, anchor)

    wide = canonical.pivot(index="seed", columns="method")
    per_seed = pd.DataFrame({"seed": sorted(HALFCHEETAH_SEEDS)})
    per_seed.insert(
        0,
        "cohort",
        per_seed["seed"].map(
            lambda seed: "original5" if seed in ORIGINAL_SEEDS else "replication5"
        ),
    )
    for output_name, metric, method in (
        ("td3_1m", "at_1m", "TD3"),
        ("amaf_1m", "at_1m", "AMAF"),
        ("td3_1p5m", "at_1p5m", "TD3"),
        ("amaf_1p5m", "at_1p5m", "AMAF"),
        ("td3_gain", "gain_1m_to_1p5m", "TD3"),
        ("amaf_gain", "gain_1m_to_1p5m", "AMAF"),
    ):
        per_seed[output_name] = wide[metric][method].reindex(per_seed["seed"]).to_numpy()
    per_seed.insert(4, "delta_1m", per_seed["amaf_1m"] - per_seed["td3_1m"])
    per_seed.insert(
        7,
        "terminal_delta",
        per_seed["amaf_1p5m"] - per_seed["td3_1p5m"],
    )
    per_seed["gain_delta"] = per_seed["amaf_gain"] - per_seed["td3_gain"]
    for metric in ("last5", "last10", "auc_1p5m"):
        output_name = "auc_delta" if metric == "auc_1p5m" else f"{metric}_delta"
        per_seed[output_name] = (
            wide[metric]["AMAF"].reindex(per_seed["seed"]).to_numpy()
            - wide[metric]["TD3"].reindex(per_seed["seed"]).to_numpy()
        )
    per_seed = per_seed[[
        "cohort",
        "seed",
        "td3_1m",
        "amaf_1m",
        "delta_1m",
        "td3_1p5m",
        "amaf_1p5m",
        "terminal_delta",
        "td3_gain",
        "amaf_gain",
        "gain_delta",
        "last5_delta",
        "last10_delta",
        "auc_delta",
    ]]

    summary_rows = []
    cohort_specs = (
        ("original5", sorted(ORIGINAL_SEEDS)),
        ("replication5", sorted(REPLICATION_SEEDS)),
        ("combined10", sorted(HALFCHEETAH_SEEDS)),
    )
    for cohort, seeds in cohort_specs:
        subset = canonical.loc[canonical["seed"].isin(seeds)]
        row: dict[str, float | int | str] = {"cohort": cohort, "n": len(seeds)}
        for output_name, metric, method in (
            ("td3_1m_mean", "at_1m", "TD3"),
            ("amaf_1m_mean", "at_1m", "AMAF"),
            ("td3_1p5m_mean", "at_1p5m", "TD3"),
            ("amaf_1p5m_mean", "at_1p5m", "AMAF"),
            ("td3_gain_mean", "gain_1m_to_1p5m", "TD3"),
            ("amaf_gain_mean", "gain_1m_to_1p5m", "AMAF"),
        ):
            row[output_name] = float(subset.loc[subset["method"] == method, metric].mean())
        for prefix, metric in (
            ("terminal", "at_1p5m"),
            ("gain", "gain_1m_to_1p5m"),
        ):
            _, effect = _paired(
                subset,
                metric=metric,
                focal="AMAF",
                comparator="TD3",
                direction="higher",
            )
            row.update({
                f"{prefix}_delta_mean": effect.mean_delta_focal_minus_comparator,
                f"{prefix}_delta_median": effect.median_delta_focal_minus_comparator,
                f"{prefix}_delta_ci_low": effect.ci95_low_focal_minus_comparator,
                f"{prefix}_delta_ci_high": effect.ci95_high_focal_minus_comparator,
                f"{prefix}_wins": effect.wins,
            })
        for output_name, metric in (
            ("last5_delta_mean", "last5"),
            ("last10_delta_mean", "last10"),
            ("auc_delta_mean", "auc_1p5m"),
        ):
            _, effect = _paired(
                subset,
                metric=metric,
                focal="AMAF",
                comparator="TD3",
                direction="higher",
            )
            row[output_name] = effect.mean_delta_focal_minus_comparator
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    computed_panel_a = _build_halfcheetah_learning(frozen)
    _assert_frame_equal(
        "HalfCheetah Panel A",
        pd.read_csv(HALFCHEETAH_PANEL_A_PATH),
        computed_panel_a,
    )
    transition, transition_summaries = build_transition_table(canonical)
    _assert_frame_equal(
        "HalfCheetah Panel B",
        pd.read_csv(HALFCHEETAH_PANEL_B_PATH),
        transition,
    )
    effect_summary = build_effect_summary(transition_summaries)
    _assert_frame_equal(
        "HalfCheetah Panel C",
        pd.read_csv(HALFCHEETAH_PANEL_C_PATH),
        effect_summary,
    )
    panel_b = transition.set_index("seed")
    handoff = per_seed.set_index("seed")
    _assert_close("HalfCheetah Panel B 1M points", panel_b["delta_1m"], handoff["delta_1m"])
    _assert_close(
        "HalfCheetah Panel B terminal points",
        panel_b["delta_1p5m"],
        handoff["terminal_delta"],
    )
    _assert_close(
        "HalfCheetah Panel B gain points",
        panel_b["late_gain_delta"],
        handoff["gain_delta"],
    )

    return summary, per_seed


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
    )
    lines = []
    for dataset_id in dataset_ids:
        fingerprint = registry.fingerprint(dataset_id)
        lines.append(
            f"- `{dataset_id}`: `{fingerprint['path']}`; "
            f"{fingerprint['fingerprint_type']}=`{fingerprint['sha256']}`"
        )
    return "\n".join(lines)


def build_readme(registry: FigureDataRegistry) -> str:
    return f"""# AMAF Results numerical table handoff (2026-09-15)

이 디렉터리는 현재 manuscript main figure와 같은 source, metric, seed population,
aggregation으로 계산한 Results/Appendix용 수치 동결본이다. 통계적 experimental unit은
모두 seed이며 evaluation episode나 Crop episode를 독립 표본으로 취급하지 않았다.

## Figure 대응

- Lunar F1: `lunar_result_summary.csv`, `lunar_per_seed.csv`
- Lunar F2: `lunar_paired_effects.csv`, `lunar_per_seed.csv`
- Crop Fig. A: `crop_result_summary.csv`, `crop_per_seed.csv`
- Crop Fig. C: `crop_result_summary.csv`, `crop_per_seed.csv`
- HalfCheetah Panel A: `halfcheetah_result_summary.csv`, `halfcheetah_per_seed.csv`
- HalfCheetah Panel B: `halfcheetah_per_seed.csv`
- HalfCheetah Panel C: `halfcheetah_result_summary.csv`, `halfcheetah_per_seed.csv`

## Metric 정의와 단위

- Lunar `last5_return`: 마지막 5개 evaluation checkpoint mean return의 평균
  (`lunar.last5`, evaluation return). `retention_pct`: `100 × last5 / best5`, 여기서
  `best5`는 가장 높은 5개 checkpoint mean return의 평균 (`lunar.retention_pct`, %).
- Crop `late1000_return`: Episode 2001–3000 train return 평균
  (`crop.late1000`, train return). `final100_return`: 마지막 100 episode train return 평균
  (`crop.final100`, train return). `yield100`, `n100`, `irr100`: 각각 마지막 100 episode의
  crop yield, nitrogen usage, irrigation usage 평균이며 단위는 contract의 native
  `yield`, `nitrogen`, `irrigation`이다. Figure A 곡선은 seed 내부에서 Episode 1–3000을
  100-episode 비중첩 block으로 먼저 평균한 뒤 seed 간 집계한다.
- HalfCheetah `*_1m`, `*_1p5m`: 정확히 1,000,000/1,500,000 step의 evaluation return.
  `*_gain`: 1.5M 값에서 1.0M 값을 뺀 evaluation-return change. `last5_delta`,
  `last10_delta`: AMAF와 TD3 각각의 마지막 5/10 checkpoint 평균 차이.
  `auc_delta`: 10k–1.5M checkpoint mean curve의 시간 정규화 trapezoidal AUC 차이.

모든 summary CI는 seed-level mean에 대한 percentile bootstrap 95% CI이다. resample 수는
{DEFAULT_BOOTSTRAP_RESAMPLES:,}, RNG seed는 {DEFAULT_RNG_SEED}, confidence는
{DEFAULT_CONFIDENCE:.2f}이다. `median`은 관측 seed 값의 기술통계 중앙값이며 CI는 mean의
CI이다.

## Delta 방향

- Lunar: `Protected - Naive`, `Protected - Uniform`; 양수는 Protected 우세.
- Crop performance/yield: `Anchored - Uniform`; 양수는 Anchored 우세.
- Crop `nitrogen_saving`: `Uniform n100 - Anchored n100`; 양수는 Anchored의 질소 절감.
- Crop `irrigation_saving`: `Uniform irr100 - Anchored irr100`; 양수는 Anchored의 관개 절감.
- HalfCheetah endpoint/last/AUC delta: `AMAF - TD3`.
- HalfCheetah `gain_delta`: `(AMAF 1.5M - AMAF 1.0M) - (TD3 1.5M - TD3 1.0M)`.

`wins`, `losses`, `ties`는 위 benefit 방향의 seed별 부호로 계산했다.

## Seed population

- Lunar: 11, 22, 33, 44, 55, 66, 77, 88, 99, 111; 5 methods, 각 n=10.
- Crop: 11, 22, 33, 44, 55; 5 methods, 각 n=5.
- HalfCheetah original5: 11, 22, 33, 44, 55; replication5: 66, 77, 88, 99, 111;
  combined10은 두 cohort 전체이며 seed 77을 포함한다.

## Source dataset IDs와 fingerprint

{_source_lines(registry)}

source path는 코드에 직접 넣지 않고 `analysis/figures/figure_contract.yaml`의 registry ID로
해결했다. canonical/frozen source는 수정하지 않았다.

## Figure–table consistency

PASS. Export 전에 다음을 Python에서 재계산하고 현재 renderer input CSV와 수치 단위로
대조했다.

- Lunar: frozen evaluation으로 canonical final/Last-5/retention을 재현했고, F1 전체 curve,
  F2 Protected–Naive seed delta, method summary/raw points가 일치했다.
- Crop: 두 frozen episode source로 canonical performance windows와 final-100 outcome을
  재현했고, Figure A 전체 100-episode block curve 및 Figure C raw/mean markers가 일치했다.
- HalfCheetah: 두 frozen cohort로 endpoint/gain/Last-5/Last-10/AUC를 재현했고, Panel A 전체
  curve, Panel B seed points, Panel C combined mean/CI/raw effect가 일치했다.

Exporter와 figure exporter는 동일 registry source, metric definition, seed population,
aggregation, bootstrap helper (`analysis.figures.stats`)를 사용한다.
"""


def _validate_outputs(frames: dict[str, pd.DataFrame]) -> None:
    expected_rows = {
        "lunar_result_summary.csv": 5,
        "lunar_paired_effects.csv": 4,
        "lunar_per_seed.csv": 50,
        "crop_result_summary.csv": 5,
        "crop_paired_effects.csv": 4,
        "crop_per_seed.csv": 25,
        "halfcheetah_result_summary.csv": 3,
        "halfcheetah_per_seed.csv": 10,
    }
    for filename, frame in frames.items():
        if len(frame) != expected_rows[filename]:
            raise ValueError(f"{filename} has {len(frame)} rows, expected {expected_rows[filename]}.")
        if frame.isna().any().any():
            raise ValueError(f"{filename} contains NaN values.")

    _validate_method_seed_table(
        frames["lunar_per_seed.csv"],
        label="lunar_per_seed.csv",
        methods=LUNAR_METHODS,
        seeds=sorted(LUNAR_SEEDS),
    )
    _validate_method_seed_table(
        frames["crop_per_seed.csv"],
        label="crop_per_seed.csv",
        methods=CROP_METHODS,
        seeds=CROP_SEEDS,
    )
    halfcheetah = frames["halfcheetah_per_seed.csv"]
    if halfcheetah["seed"].duplicated().any():
        raise ValueError("halfcheetah_per_seed.csv contains duplicate seeds.")
    if set(halfcheetah["seed"].astype(int)) != set(HALFCHEETAH_SEEDS):
        raise ValueError("halfcheetah_per_seed.csv is missing an expected seed.")
    if 77 not in set(halfcheetah["seed"].astype(int)):
        raise ValueError("halfcheetah_per_seed.csv is missing seed 77.")
    if frames["lunar_paired_effects.csv"].duplicated(["comparison", "metric"]).any():
        raise ValueError("lunar_paired_effects.csv contains duplicate keys.")
    if frames["crop_paired_effects.csv"].duplicated(["comparison", "metric"]).any():
        raise ValueError("crop_paired_effects.csv contains duplicate keys.")
    if frames["halfcheetah_result_summary.csv"]["cohort"].duplicated().any():
        raise ValueError("halfcheetah_result_summary.csv contains duplicate cohorts.")


def export() -> dict[str, pd.DataFrame]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing handoff directory: {OUTPUT_DIR}")

    registry = FigureDataRegistry()
    lunar_summary, lunar_paired, lunar_per_seed = build_lunar(registry)
    crop_summary, crop_paired, crop_per_seed = build_crop(registry)
    halfcheetah_summary, halfcheetah_per_seed = build_halfcheetah(registry)

    frames = {
        "lunar_result_summary.csv": lunar_summary,
        "lunar_paired_effects.csv": lunar_paired,
        "lunar_per_seed.csv": lunar_per_seed,
        "crop_result_summary.csv": crop_summary,
        "crop_paired_effects.csv": crop_paired,
        "crop_per_seed.csv": crop_per_seed,
        "halfcheetah_result_summary.csv": halfcheetah_summary,
        "halfcheetah_per_seed.csv": halfcheetah_per_seed,
    }
    _validate_outputs(frames)

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".results_handoff_20260915_", dir=OUTPUT_DIR.parent))
    try:
        for filename, frame in frames.items():
            frame.to_csv(temporary / filename, index=False)
        (temporary / "RESULT_TABLES_README.md").write_text(
            build_readme(registry), encoding="utf-8"
        )

        # Validate the serialized tables, not only the in-memory frames.
        for filename, expected in frames.items():
            _assert_frame_equal(filename, pd.read_csv(temporary / filename), expected)
        if len(list(temporary.iterdir())) != 9:
            raise ValueError("The handoff package must contain eight CSVs and one README.")
        os.rename(temporary, OUTPUT_DIR)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    for filename, frame in frames.items():
        print(f"Wrote {OUTPUT_DIR.relative_to(ROOT) / filename} ({len(frame)} rows).")
    print(f"Wrote {OUTPUT_DIR.relative_to(ROOT) / 'RESULT_TABLES_README.md'}.")
    print("Figure-table consistency: PASS")
    print("Row-count, duplicate-key, seed-population, and NaN checks: PASS")
    return frames


def main() -> None:
    export()


if __name__ == "__main__":
    main()
