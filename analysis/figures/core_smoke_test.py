#!/usr/bin/env python3

from __future__ import annotations

import numpy as np

from .io import (
    FigureDataRegistry,
    ManuscriptSafetyError,
)

from .stats import (
    summarize_paired_frame,
)


def main():

    print(
        "=== FIGURE CORE SMOKE TEST ==="
    )

    reg = FigureDataRegistry()

    # ------------------------------------------------------------------
    # Canonical dataset loading
    # ------------------------------------------------------------------

    lunar = reg.load_csv(
        "lunar_per_seed",
        manuscript=True,
    )

    crop = reg.load_csv(
        "crop_per_seed",
        manuscript=True,
    )

    hc = reg.load_csv(
        "halfcheetah_per_seed",
        manuscript=True,
    )

    print(
        "Lunar rows:",
        len(lunar),
    )

    print(
        "Crop rows:",
        len(crop),
    )

    print(
        "HalfCheetah rows:",
        len(hc),
    )

    assert len(lunar) == 50
    assert len(crop) == 25
    assert len(hc) == 20

    print(
        "CANONICAL LOAD: PASS"
    )

    # ------------------------------------------------------------------
    # Manuscript safety: planned Lunar dataset
    # ------------------------------------------------------------------

    try:
        reg.load_path(
            "lunar_anchored_v31",
            manuscript=True,
        )

    except ManuscriptSafetyError:
        print(
            "PLANNED DATASET BLOCK: PASS"
        )

    else:
        raise AssertionError(
            (
                "Planned Lunar dataset was "
                "incorrectly allowed for manuscript."
            )
        )

    # ------------------------------------------------------------------
    # Manuscript safety: frozen SAC primary-5
    # ------------------------------------------------------------------

    # ------------------------------------------------------------
    # Frozen SAC primary-5 must be manuscript-safe
    # ------------------------------------------------------------

    sac_path = reg.load_path(
        "halfcheetah_sac_primary5",
        manuscript=True,
    )

    if not sac_path.exists():
        raise AssertionError(
            f"Frozen SAC dataset path does not exist: {sac_path}"
        )

    print("FROZEN SAC DATASET LOAD: PASS")

    # ------------------------------------------------------------------
    # Lunar core result sanity
    # ------------------------------------------------------------------

    print()
    print("Lunar late-stage / retention:")

    lunar_methods = (
        lunar.groupby("method")
        .agg(
            last5_mean=("last5", "mean"),
            retention_mean=("retention_pct", "mean"),
        )
    )

    for method in [
        "DQN",
        "Dueling",
        "Uniform",
        "Naive",
        "Protected",
    ]:
        row = lunar_methods.loc[method]

        print(
            f"  {method:10s} "
            f"Last-5={row['last5_mean']:8.2f}  "
            f"Retention={row['retention_mean']:6.2f}%"
        )

    # Protected vs Naive: retention-preservation effect
    lunar_ret_pairs, lunar_ret = (
        summarize_paired_frame(
            lunar,
            seed_col="seed",
            method_col="method",
            metric_col="retention_pct",
            focal="Protected",
            comparator="Naive",
            direction="higher",
        )
    )

    print()
    print("Lunar Protected vs Naive retention:")

    print(
        "  mean Δ  =",
        f"{lunar_ret.mean_delta_focal_minus_comparator:+.2f} pp",
    )

    print(
        "  median Δ=",
        f"{lunar_ret.median_delta_focal_minus_comparator:+.2f} pp",
    )

    print(
        "  wins    =",
        f"{lunar_ret.wins}/{lunar_ret.n}",
    )

    assert lunar_ret.n == 10

    assert (
        lunar_ret.mean_delta_focal_minus_comparator
        > 0
    )

    # Protected vs Uniform: late-stage performance preservation
    lunar_last5_pairs, lunar_last5 = (
        summarize_paired_frame(
            lunar,
            seed_col="seed",
            method_col="method",
            metric_col="last5",
            focal="Protected",
            comparator="Uniform",
            direction="higher",
        )
    )

    print()
    print("Lunar Protected vs Uniform Last-5:")

    print(
        "  Uniform mean   =",
        f"{lunar_last5.mean_comparator:.2f}",
    )

    print(
        "  Protected mean =",
        f"{lunar_last5.mean_focal:.2f}",
    )

    print(
        "  mean Δ         =",
        f"{lunar_last5.mean_delta_focal_minus_comparator:+.2f}",
    )

    print(
        "  wins           =",
        f"{lunar_last5.wins}/{lunar_last5.n}",
    )

    assert lunar_last5.n == 10

    print(
        "LUNAR CORE RESULT CHECK: PASS"
    )

    # ------------------------------------------------------------------
    # HalfCheetah paired terminal result
    # ------------------------------------------------------------------

    paired_terminal, terminal = (
        summarize_paired_frame(
            hc,
            seed_col="seed",
            method_col="method",
            metric_col="at_1p5m",
            focal="AMAF",
            comparator="TD3",
            direction="higher",
        )
    )

    print()
    print(
        "HalfCheetah @1.5M:"
    )

    print(
        "  n       =",
        terminal.n,
    )

    print(
        "  mean Δ  =",
        f"{terminal.mean_delta_focal_minus_comparator:+.2f}",
    )

    print(
        "  median Δ=",
        f"{terminal.median_delta_focal_minus_comparator:+.2f}",
    )

    print(
        "  wins    =",
        f"{terminal.wins}/{terminal.n}",
    )

    assert terminal.n == 10

    assert terminal.wins == 8

    assert np.isclose(
        terminal.mean_delta_focal_minus_comparator,
        68.72,
        atol=0.02,
    )

    # ------------------------------------------------------------------
    # HalfCheetah late-stage relative gain
    # ------------------------------------------------------------------

    paired_gain, gain = (
        summarize_paired_frame(
            hc,
            seed_col="seed",
            method_col="method",
            metric_col="gain_1m_to_1p5m",
            focal="AMAF",
            comparator="TD3",
            direction="higher",
        )
    )

    print()
    print(
        "HalfCheetah 1M→1.5M gain:"
    )

    print(
        "  mean Δ  =",
        f"{gain.mean_delta_focal_minus_comparator:+.2f}",
    )

    print(
        "  median Δ=",
        f"{gain.median_delta_focal_minus_comparator:+.2f}",
    )

    print(
        "  wins    =",
        f"{gain.wins}/{gain.n}",
    )

    assert gain.n == 10

    assert gain.wins == 8

    assert np.isclose(
        gain.mean_delta_focal_minus_comparator,
        375.13,
        atol=0.02,
    )

    # ------------------------------------------------------------------
    # Crop resource direction sanity
    # ------------------------------------------------------------------

    crop_n_pairs, crop_n = (
        summarize_paired_frame(
            crop,
            seed_col="seed",
            method_col="method",
            metric_col="N100",
            focal="Anchored",
            comparator="Uniform",
            direction="lower",
        )
    )

    print()
    print(
        "Crop nitrogen Anchored vs Uniform:"
    )

    print(
        "  raw Δ Anchored-Uniform =",
        f"{crop_n.mean_delta_focal_minus_comparator:+.2f}",
    )

    print(
        "  benefit / saving Δ     =",
        f"{crop_n.mean_benefit_delta:+.2f}",
    )

    print(
        "  wins                   =",
        f"{crop_n.wins}/{crop_n.n}",
    )

    # Current canonical evidence should show lower N
    # in all five matched seeds.
    assert crop_n.n == 5
    assert crop_n.wins == 5

    # Positive benefit delta must mean lower nitrogen
    # usage by Anchored.
    assert (
        crop_n.mean_benefit_delta
        > 0
    )

    print()
    print(
        "STATISTICAL DIRECTION CHECK: PASS"
    )

    # ------------------------------------------------------------------
    # Fingerprint smoke
    # ------------------------------------------------------------------

    fp = reg.fingerprint(
        "halfcheetah_per_seed"
    )

    assert fp[
        "exists"
    ]

    assert fp[
        "sha256"
    ]

    print(
        "SOURCE FINGERPRINT: PASS"
    )

    print()
    print(
        "========================================"
    )

    print(
        "FIGURE CORE SMOKE TEST: PASS"
    )


if __name__ == "__main__":
    main()
