#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ======================================================================
# Global statistical constants
# ======================================================================

DEFAULT_BOOTSTRAP_RESAMPLES = 100_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_RNG_SEED = 20260907


# ======================================================================
# Validation helpers
# ======================================================================

def _clean_numeric(
    values: Iterable[float],
    *,
    allow_nan: bool = False,
) -> np.ndarray:

    x = np.asarray(
        list(values),
        dtype=float,
    ).reshape(-1)

    if x.size == 0:
        raise ValueError(
            "No observations supplied."
        )

    if allow_nan:

        x = x[
            np.isfinite(x)
        ]

        if x.size == 0:
            raise ValueError(
                "No finite observations supplied."
            )

    elif not np.isfinite(x).all():

        bad = int(
            (~np.isfinite(x)).sum()
        )

        raise ValueError(
            (
                f"Found {bad} non-finite "
                "observations."
            )
        )

    return x


def _bootstrap_quantiles(
    confidence: float,
) -> tuple[float, float]:

    if not (
        0.0
        < confidence
        < 1.0
    ):
        raise ValueError(
            (
                "confidence must be "
                "strictly between 0 and 1."
            )
        )

    alpha = (
        1.0
        - confidence
    )

    return (
        alpha / 2.0,
        1.0 - alpha / 2.0,
    )


# ======================================================================
# Bootstrap primitives
# ======================================================================

def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> tuple[float, float]:

    x = _clean_numeric(
        values
    )

    if resamples <= 0:
        raise ValueError(
            "resamples must be positive."
        )

    if len(x) == 1:
        value = float(
            x[0]
        )

        return (
            value,
            value,
        )

    rng = np.random.default_rng(
        rng_seed
    )

    idx = rng.integers(
        low=0,
        high=len(x),
        size=(
            resamples,
            len(x),
        ),
    )

    means = (
        x[idx]
        .mean(
            axis=1
        )
    )

    qlo, qhi = (
        _bootstrap_quantiles(
            confidence
        )
    )

    lo, hi = np.quantile(
        means,
        [
            qlo,
            qhi,
        ],
    )

    return (
        float(lo),
        float(hi),
    )


def bootstrap_median_ci(
    values: Iterable[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> tuple[float, float]:

    x = _clean_numeric(
        values
    )

    if resamples <= 0:
        raise ValueError(
            "resamples must be positive."
        )

    if len(x) == 1:

        value = float(
            x[0]
        )

        return (
            value,
            value,
        )

    rng = np.random.default_rng(
        rng_seed
    )

    idx = rng.integers(
        low=0,
        high=len(x),
        size=(
            resamples,
            len(x),
        ),
    )

    medians = np.median(
        x[idx],
        axis=1,
    )

    qlo, qhi = (
        _bootstrap_quantiles(
            confidence
        )
    )

    lo, hi = np.quantile(
        medians,
        [
            qlo,
            qhi,
        ],
    )

    return (
        float(lo),
        float(hi),
    )


def paired_bootstrap_mean_ci(
    focal: Iterable[float],
    comparator: Iterable[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> tuple[float, float]:
    """
    Paired bootstrap for focal - comparator.

    Pairing is preserved because bootstrap resampling is performed
    over the paired delta vector.
    """

    a = _clean_numeric(
        focal
    )

    b = _clean_numeric(
        comparator
    )

    if a.shape != b.shape:
        raise ValueError(
            (
                "Paired inputs must have "
                "identical shape."
            )
        )

    delta = (
        a
        - b
    )

    return bootstrap_mean_ci(
        delta,
        confidence=confidence,
        resamples=resamples,
        rng_seed=rng_seed,
    )


# ======================================================================
# Summary objects
# ======================================================================

@dataclass(frozen=True)
class SampleSummary:

    n: int

    mean: float
    median: float
    sd: float

    ci95_low: float
    ci95_high: float

    minimum: float
    maximum: float


@dataclass(frozen=True)
class PairedSummary:

    n: int

    focal: str
    comparator: str

    direction: str

    mean_focal: float
    mean_comparator: float

    mean_delta_focal_minus_comparator: float
    median_delta_focal_minus_comparator: float
    sd_delta: float

    mean_benefit_delta: float
    median_benefit_delta: float

    ci95_low_focal_minus_comparator: float
    ci95_high_focal_minus_comparator: float

    ci95_low_benefit: float
    ci95_high_benefit: float

    wins: int
    losses: int
    ties: int


# ======================================================================
# Sample summaries
# ======================================================================

def summarize_sample(
    values: Iterable[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> SampleSummary:

    x = _clean_numeric(
        values
    )

    lo, hi = bootstrap_mean_ci(
        x,
        confidence=confidence,
        resamples=resamples,
        rng_seed=rng_seed,
    )

    if len(x) > 1:

        sd = float(
            np.std(
                x,
                ddof=1,
            )
        )

    else:
        sd = 0.0

    return SampleSummary(
        n=int(
            len(x)
        ),
        mean=float(
            np.mean(x)
        ),
        median=float(
            np.median(x)
        ),
        sd=sd,
        ci95_low=lo,
        ci95_high=hi,
        minimum=float(
            np.min(x)
        ),
        maximum=float(
            np.max(x)
        ),
    )


# ======================================================================
# Paired summaries
# ======================================================================

def paired_delta(
    focal: Iterable[float],
    comparator: Iterable[float],
) -> np.ndarray:

    a = _clean_numeric(
        focal
    )

    b = _clean_numeric(
        comparator
    )

    if a.shape != b.shape:
        raise ValueError(
            (
                "Paired arrays must "
                "have identical shape."
            )
        )

    return (
        a
        - b
    )


def benefit_delta(
    focal: Iterable[float],
    comparator: Iterable[float],
    *,
    direction: str,
) -> np.ndarray:
    """
    Return a delta whose positive direction always means
    'better for focal'.

    higher:
        focal - comparator

    lower:
        comparator - focal
    """

    delta = paired_delta(
        focal,
        comparator,
    )

    if direction == "higher":
        return delta

    if direction == "lower":
        return -delta

    raise ValueError(
        (
            "benefit_delta requires "
            "direction='higher' or 'lower'."
        )
    )


def summarize_paired(
    focal_values: Iterable[float],
    comparator_values: Iterable[float],
    *,
    focal_name: str,
    comparator_name: str,
    direction: str = "higher",
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> PairedSummary:

    focal = _clean_numeric(
        focal_values
    )

    comparator = _clean_numeric(
        comparator_values
    )

    if focal.shape != comparator.shape:
        raise ValueError(
            (
                "Paired arrays must "
                "have identical shape."
            )
        )

    raw_delta = (
        focal
        - comparator
    )

    if direction == "higher":

        benefit = (
            raw_delta
        )

    elif direction == "lower":

        benefit = (
            -raw_delta
        )

    else:
        raise ValueError(
            (
                "Paired performance summary "
                "requires direction='higher' "
                "or direction='lower'."
            )
        )

    raw_lo, raw_hi = (
        bootstrap_mean_ci(
            raw_delta,
            confidence=confidence,
            resamples=resamples,
            rng_seed=rng_seed,
        )
    )

    benefit_lo, benefit_hi = (
        bootstrap_mean_ci(
            benefit,
            confidence=confidence,
            resamples=resamples,
            rng_seed=rng_seed,
        )
    )

    if len(raw_delta) > 1:

        sd_delta = float(
            np.std(
                raw_delta,
                ddof=1,
            )
        )

    else:
        sd_delta = 0.0

    wins = int(
        np.sum(
            benefit > 0
        )
    )

    losses = int(
        np.sum(
            benefit < 0
        )
    )

    ties = int(
        np.sum(
            benefit == 0
        )
    )

    return PairedSummary(
        n=int(
            len(raw_delta)
        ),
        focal=focal_name,
        comparator=comparator_name,
        direction=direction,

        mean_focal=float(
            np.mean(focal)
        ),

        mean_comparator=float(
            np.mean(comparator)
        ),

        mean_delta_focal_minus_comparator=float(
            np.mean(
                raw_delta
            )
        ),

        median_delta_focal_minus_comparator=float(
            np.median(
                raw_delta
            )
        ),

        sd_delta=sd_delta,

        mean_benefit_delta=float(
            np.mean(
                benefit
            )
        ),

        median_benefit_delta=float(
            np.median(
                benefit
            )
        ),

        ci95_low_focal_minus_comparator=raw_lo,
        ci95_high_focal_minus_comparator=raw_hi,

        ci95_low_benefit=benefit_lo,
        ci95_high_benefit=benefit_hi,

        wins=wins,
        losses=losses,
        ties=ties,
    )


# ======================================================================
# DataFrame paired helpers
# ======================================================================

def paired_values_from_frame(
    df: pd.DataFrame,
    *,
    seed_col: str,
    method_col: str,
    metric_col: str,
    focal: str,
    comparator: str,
) -> pd.DataFrame:
    """
    Convert a long-form method/seed table into a paired seed table.

    Output:
        seed
        focal
        comparator
        delta_focal_minus_comparator

    Duplicate method-seed rows are rejected.
    """

    required = {
        seed_col,
        method_col,
        metric_col,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            (
                "Missing columns: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )
        )

    x = df[
        df[
            method_col
        ].isin(
            [
                focal,
                comparator,
            ]
        )
    ][
        [
            seed_col,
            method_col,
            metric_col,
        ]
    ].copy()

    duplicates = x.duplicated(
        subset=[
            seed_col,
            method_col,
        ],
        keep=False,
    )

    if duplicates.any():

        bad = x.loc[
            duplicates,
            [
                seed_col,
                method_col,
            ],
        ]

        raise ValueError(
            (
                "Duplicate seed-method rows "
                "detected:\n"
                + bad.to_string(
                    index=False
                )
            )
        )

    pivot = x.pivot(
        index=seed_col,
        columns=method_col,
        values=metric_col,
    )

    required_methods = {
        focal,
        comparator,
    }

    if not required_methods.issubset(
        set(
            pivot.columns
        )
    ):
        raise ValueError(
            (
                "Could not form complete paired "
                f"comparison for {focal} and "
                f"{comparator}."
            )
        )

    paired = pivot[
        [
            focal,
            comparator,
        ]
    ].dropna().copy()

    paired = (
        paired
        .sort_index()
        .reset_index()
    )

    paired = paired.rename(
        columns={
            focal: "focal",
            comparator: "comparator",
        }
    )

    paired[
        "delta_focal_minus_comparator"
    ] = (
        paired["focal"]
        - paired["comparator"]
    )

    return paired


def summarize_paired_frame(
    df: pd.DataFrame,
    *,
    seed_col: str,
    method_col: str,
    metric_col: str,
    focal: str,
    comparator: str,
    direction: str = "higher",
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> tuple[pd.DataFrame, PairedSummary]:

    paired = paired_values_from_frame(
        df,
        seed_col=seed_col,
        method_col=method_col,
        metric_col=metric_col,
        focal=focal,
        comparator=comparator,
    )

    summary = summarize_paired(
        paired["focal"],
        paired["comparator"],
        focal_name=focal,
        comparator_name=comparator,
        direction=direction,
        confidence=confidence,
        resamples=resamples,
        rng_seed=rng_seed,
    )

    if direction == "higher":

        paired[
            "benefit_delta"
        ] = (
            paired[
                "delta_focal_minus_comparator"
            ]
        )

    elif direction == "lower":

        paired[
            "benefit_delta"
        ] = (
            -paired[
                "delta_focal_minus_comparator"
            ]
        )

    else:
        raise ValueError(
            (
                "direction must be "
                "'higher' or 'lower'."
            )
        )

    return (
        paired,
        summary,
    )


# ======================================================================
# Learning-curve statistics
# ======================================================================

def summarize_curve_by_seed(
    df: pd.DataFrame,
    *,
    seed_col: str,
    x_col: str,
    value_col: str,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    rng_seed: int = DEFAULT_RNG_SEED,
    require_unique_seed_x: bool = True,
) -> pd.DataFrame:
    """
    Aggregate a seed-level learning curve.

    IMPORTANT:
        The input must already contain one value per seed/x position.

    Examples:
        Lunar:
            evaluation episodes
            -> checkpoint mean within seed
            -> this function

        Crop:
            episode returns
            -> rolling-100 within seed
            -> selected episode positions
            -> this function

    This function intentionally does NOT treat evaluation episodes
    or crop episodes as independent statistical replicates.
    """

    required = {
        seed_col,
        x_col,
        value_col,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise KeyError(
            (
                "Missing curve columns: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )
        )

    work = df[
        [
            seed_col,
            x_col,
            value_col,
        ]
    ].copy()

    work = work.dropna(
        subset=[
            seed_col,
            x_col,
            value_col,
        ]
    )

    if require_unique_seed_x:

        duplicate = work.duplicated(
            subset=[
                seed_col,
                x_col,
            ],
            keep=False,
        )

        if duplicate.any():

            bad = work.loc[
                duplicate,
                [
                    seed_col,
                    x_col,
                ],
            ].head(
                20
            )

            raise ValueError(
                (
                    "Learning-curve input must "
                    "contain one observation per "
                    "seed/x position. Duplicates:\n"
                    + bad.to_string(
                        index=False
                    )
                )
            )

    rows = []

    for x_value, group in (
        work
        .groupby(
            x_col,
            sort=True,
        )
    ):

        values = group[
            value_col
        ].to_numpy(
            dtype=float
        )

        summary = summarize_sample(
            values,
            confidence=confidence,
            resamples=resamples,
            rng_seed=(
                rng_seed
                + len(rows)
            ),
        )

        rows.append({
            x_col: x_value,
            "mean": summary.mean,
            "median": summary.median,
            "sd": summary.sd,
            "ci95_low": summary.ci95_low,
            "ci95_high": summary.ci95_high,
            "n_seeds": summary.n,
        })

    return pd.DataFrame(
        rows
    )


# ======================================================================
# Utility conversion
# ======================================================================

def summary_to_dict(
    summary,
) -> dict:

    if hasattr(
        summary,
        "__dataclass_fields__",
    ):

        return {
            key: getattr(
                summary,
                key,
            )
            for key in (
                summary
                .__dataclass_fields__
                .keys()
            )
        }

    raise TypeError(
        (
            "summary_to_dict expects "
            "a dataclass summary."
        )
    )
