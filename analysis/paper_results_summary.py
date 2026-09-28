#!/usr/bin/env python3
"""
AMAF_UNIFIED canonical paper-result audit and summary.

Reads ONLY frozen data under paper_data/, verifies SHA256 manifests and expected
matched-seed completion/horizons, then reproduces the descriptive metrics used
for the paper.

Default frozen bundles
----------------------
paper_data/lunar/primary_20260828/
paper_data/crop/baseline_primary_20260828/
paper_data/crop/anchored_primary_20260830/
paper_data/halfcheetah/primary_1p5m_20260830/

Usage
-----
From AMAF_UNIFIED repository root:

    python analysis/paper_results_summary.py --no-write

To also write derived CSV/JSON summaries:

    python analysis/paper_results_summary.py

Raw frozen data are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# ======================================================================================================================
# Canonical experiment specification
# ======================================================================================================================

LUNAR_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
CROP_SEEDS = [11, 22, 33, 44, 55]
CHEETAH_SEEDS = [11, 22, 33, 44, 55]

LUNAR_TARGET_STEPS = 500_000
CROP_TARGET_EPISODES = 3_000
CHEETAH_TARGET_STEPS = 1_500_000

EXPECTED_MANIFEST_COUNTS = {
    "Lunar primary": 333,
    "Crop baseline": 113,
    "Crop anchored": 30,
    "HalfCheetah 1.5M": 65,
}

LUNAR_METHODS = {
    "DQN": "dqn",
    "Dueling": "dueling_dqn",
    "Uniform": "amaf_uniform",
    "Naive": "amaf_naive",
    "Protected": "amaf_protected",
}

CROP_BASE_METHODS = {
    "DQN": "dqn",
    "Dueling": "dueling_dqn",
    "Uniform": "amaf_uniform",
    "Naive": "amaf_dqn",
}

CHEETAH_METHODS = {
    "TD3": "td3",
    "AMAF": "amaf_selective_bp_v2_td3",
}


class AuditError(RuntimeError):
    pass


# ======================================================================================================================
# Generic helpers
# ======================================================================================================================

def fmt(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{float(x):.{digits}f}"


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return math.nan, math.nan
    if len(x) == 1:
        return float(x[0]), math.nan
    return float(x.mean()), float(x.std(ddof=1))


def find_seed_dir(method_root: Path, seed: int) -> Path:
    for name in (f"seed_{seed}", f"seed{seed}", str(seed)):
        p = method_root / name
        if p.is_dir():
            return p
    raise AuditError(f"Missing seed directory: root={method_root}, seed={seed}")


def require_file(root: Path, filename: str) -> Path:
    p = root / filename
    if p.is_file():
        return p
    hits = sorted(root.rglob(filename))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise AuditError(f"Ambiguous {filename} below {root}: {hits}")
    raise AuditError(f"Missing {filename}: {root}")


def require_completed(seed_dir: Path) -> None:
    if not (seed_dir / "COMPLETED").is_file():
        raise AuditError(f"Missing COMPLETED marker: {seed_dir}")


def infer_step_col(df: pd.DataFrame) -> str:
    for c in ("env_steps", "step", "steps"):
        if c in df.columns:
            return c
    raise AuditError(f"No step column. columns={list(df.columns)}")


def infer_return_col(df: pd.DataFrame) -> str:
    for c in ("return", "eval_return", "episode_return", "eval_episode_return", "reward"):
        if c in df.columns:
            return c
    candidates = [
        c for c in df.columns
        if ("return" in c.lower() or "reward" in c.lower())
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise AuditError(f"No unique return column. columns={list(df.columns)}")


def checkpoint_means(csv_path: Path) -> pd.Series:
    """
    One mean value per evaluation checkpoint.

    This intentionally groups by env_steps first, so duplicate episode-level
    evaluation rows never become pseudo-independent observations.
    """
    d = pd.read_csv(csv_path)
    step_col = infer_step_col(d)
    return_col = infer_return_col(d)
    g = (
        d.groupby(step_col, sort=True)[return_col]
        .mean()
        .astype(float)
        .sort_index()
    )
    g.index = g.index.astype(int)
    return g


def normalized_auc(g: pd.Series) -> float:
    if len(g) < 2:
        return math.nan
    x = g.index.to_numpy(dtype=float)
    y = g.to_numpy(dtype=float)
    denom = x[-1] - x[0]
    if denom <= 0:
        return math.nan
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(y, x)
    else:
        area = np.trapz(y, x)
    return float(area / denom)


def threshold_episode(r: pd.Series, threshold: float, window: int = 100) -> float:
    q = r.rolling(window).mean().to_numpy(dtype=float)
    hit = np.flatnonzero(q >= threshold)
    return math.nan if len(hit) == 0 else float(hit[0] + 1)


def verify_manifest(root: Path) -> dict:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise AuditError(f"Missing SHA256SUMS: {root}")

    checked = 0
    failures = []

    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            expected, rel = raw.split(maxsplit=1)
        except ValueError as exc:
            raise AuditError(f"Malformed SHA256 line: {raw!r}") from exc

        rel = rel.lstrip("*").strip()
        p = root / rel
        if not p.is_file():
            failures.append(f"MISSING {rel}")
            continue

        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        actual = h.hexdigest()
        checked += 1

        if actual != expected:
            failures.append(
                f"HASH_MISMATCH {rel}: expected={expected}, actual={actual}"
            )

    if failures:
        raise AuditError(
            f"SHA256 audit failed for {root}:\n  " + "\n  ".join(failures)
        )

    return {
        "root": str(root),
        "manifest_entries": checked,
        "sha256_ok": True,
    }


def aggregate_table(
    per_seed: pd.DataFrame,
    experiment: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for method in per_seed["method"].drop_duplicates():
        x = per_seed[per_seed.method == method]
        for metric in metrics:
            vals = pd.to_numeric(x[metric], errors="coerce").to_numpy(dtype=float)
            m, sd = mean_sd(vals)
            rows.append({
                "experiment": experiment,
                "method": method,
                "metric": metric,
                "mean": m,
                "sd": sd,
                "n": int(np.isfinite(vals).sum()),
            })
    return pd.DataFrame(rows)


def paired_table(
    per_seed: pd.DataFrame,
    experiment: str,
    focal: str,
    comparators: Sequence[str],
    metrics: Sequence[str],
    lower_is_better: set[str] | None = None,
) -> pd.DataFrame:
    lower_is_better = lower_is_better or set()
    rows = []
    f = per_seed[per_seed.method == focal].set_index("seed")

    for comp in comparators:
        b = per_seed[per_seed.method == comp].set_index("seed")
        seeds = sorted(set(f.index).intersection(b.index))

        for metric in metrics:
            deltas = np.array(
                [float(f.loc[s, metric]) - float(b.loc[s, metric]) for s in seeds],
                dtype=float,
            )
            finite = np.isfinite(deltas)
            valid = deltas[finite]
            used_seeds = [s for s, ok in zip(seeds, finite) if ok]
            m, sd = mean_sd(valid)
            wins = int(np.sum(valid < 0)) if metric in lower_is_better else int(np.sum(valid > 0))

            rows.append({
                "experiment": experiment,
                "focal": focal,
                "comparator": comp,
                "metric": metric,
                "direction": "lower" if metric in lower_is_better else "higher",
                "mean_delta_focal_minus_comparator": m,
                "sd_delta": sd,
                "wins": wins,
                "n": int(len(valid)),
                "seed_deltas": json.dumps(
                    {str(s): float(v) for s, v in zip(used_seeds, valid)},
                    ensure_ascii=False,
                ),
            })

    return pd.DataFrame(rows)


# ======================================================================================================================
# LunarLander
# ======================================================================================================================

LUNAR_METRICS = ["final", "last5", "auc", "best5", "retention_pct"]


def analyze_lunar(root: Path):
    rows = []

    for method, dirname in LUNAR_METHODS.items():
        method_root = root / dirname
        if not method_root.is_dir():
            raise AuditError(f"Missing Lunar method directory: {method_root}")

        for seed in LUNAR_SEEDS:
            seed_dir = find_seed_dir(method_root, seed)
            g = checkpoint_means(require_file(seed_dir, "eval_metrics.csv"))
            x = g[g.index <= LUNAR_TARGET_STEPS]

            if not len(x) or int(x.index.max()) != LUNAR_TARGET_STEPS:
                raise AuditError(
                    f"Lunar incomplete/wrong horizon: {method} seed={seed}, "
                    f"max_step={int(g.index.max()) if len(g) else None}"
                )

            final = float(x.iloc[-1])
            last5 = float(x.tail(5).mean())
            best5 = float(x.nlargest(min(5, len(x))).mean())

            rows.append({
                "experiment": "lunar",
                "method": method,
                "seed": seed,
                "final": final,
                "last5": last5,
                "auc": normalized_auc(x),
                "best5": best5,
                "retention_pct": 100.0 * last5 / best5 if best5 else math.nan,
                "final_gt_200": int(final > 200.0),
                "checkpoints": int(len(x)),
                "max_step": int(x.index.max()),
            })

    per_seed = pd.DataFrame(rows)
    agg = aggregate_table(per_seed, "lunar", LUNAR_METRICS)

    success = []
    for method in per_seed.method.drop_duplicates():
        x = per_seed[per_seed.method == method]
        success.append({
            "experiment": "lunar",
            "method": method,
            "metric": "final_gt_200_count",
            "mean": float(x.final_gt_200.sum()),
            "sd": math.nan,
            "n": int(len(x)),
        })
    agg = pd.concat([agg, pd.DataFrame(success)], ignore_index=True)

    paired = paired_table(
        per_seed,
        "lunar",
        "Protected",
        ["Uniform", "Naive", "Dueling", "DQN"],
        LUNAR_METRICS,
    )

    return per_seed, agg, paired


# ======================================================================================================================
# Crop
# ======================================================================================================================

CROP_METRICS = [
    "full",
    "early500",
    "early1000",
    "mid1000",
    "late1000",
    "final500",
    "final250",
    "final100",
    "peak100",
    "peak250",
    "yield100",
    "N100",
    "irr100",
    "T800",
    "T900",
    "T1000",
]


def crop_seed_metrics(seed_dir: Path) -> dict:
    require_completed(seed_dir)
    d = pd.read_csv(require_file(seed_dir, "episode_metrics.csv"))

    if len(d) != CROP_TARGET_EPISODES:
        raise AuditError(
            f"Crop expected exactly {CROP_TARGET_EPISODES} episodes: "
            f"{seed_dir}, rows={len(d)}"
        )
    if "train_return" not in d.columns:
        raise AuditError(f"Crop missing train_return: {seed_dir}")

    r = pd.to_numeric(d.train_return, errors="raise").astype(float)
    q100 = r.rolling(100).mean()
    q250 = r.rolling(250).mean()

    def tail_mean(col: str, n: int = 100) -> float:
        if col not in d.columns:
            return math.nan
        return float(pd.to_numeric(d[col], errors="coerce").tail(n).mean())

    return {
        "full": float(r.mean()),
        "early500": float(r.iloc[:500].mean()),
        "early1000": float(r.iloc[:1000].mean()),
        "mid1000": float(r.iloc[1000:2000].mean()),
        "late1000": float(r.iloc[2000:3000].mean()),
        "final500": float(r.tail(500).mean()),
        "final250": float(r.tail(250).mean()),
        "final100": float(r.tail(100).mean()),
        "peak100": float(q100.max()),
        "peak100_episode": int(q100.idxmax() + 1),
        "peak250": float(q250.max()),
        "peak250_episode": int(q250.idxmax() + 1),
        "yield100": tail_mean("yield"),
        "N100": tail_mean("nitrogen"),
        "irr100": tail_mean("irrigation"),
        "T800": threshold_episode(r, 800.0),
        "T900": threshold_episode(r, 900.0),
        "T1000": threshold_episode(r, 1000.0),
        "episodes": int(len(d)),
        "max_env_steps": int(d.env_steps.max()) if "env_steps" in d.columns else None,
    }


def analyze_crop(baseline_root: Path, anchored_root: Path):
    rows = []

    # Historical baseline methods have method directories.
    for method, dirname in CROP_BASE_METHODS.items():
        method_root = baseline_root / dirname
        if not method_root.is_dir():
            raise AuditError(f"Missing Crop baseline method directory: {method_root}")

        for seed in CROP_SEEDS:
            rows.append({
                "experiment": "crop",
                "method": method,
                "seed": seed,
                **crop_seed_metrics(find_seed_dir(method_root, seed)),
            })

    # Frozen Anchored bundle intentionally stores seed_* directly below its root.
    for seed in CROP_SEEDS:
        rows.append({
            "experiment": "crop",
            "method": "Anchored",
            "seed": seed,
            **crop_seed_metrics(find_seed_dir(anchored_root, seed)),
        })

    per_seed = pd.DataFrame(rows)
    agg = aggregate_table(per_seed, "crop", CROP_METRICS)

    perf_paired = paired_table(
        per_seed,
        "crop",
        "Anchored",
        ["Uniform", "Dueling", "DQN", "Naive"],
        [
            "full", "late1000", "final500", "final250", "final100",
            "peak100", "peak250",
        ],
    )

    resource_paired = paired_table(
        per_seed,
        "crop",
        "Anchored",
        ["Uniform", "Dueling", "DQN", "Naive"],
        ["yield100", "N100", "irr100"],
        lower_is_better={"N100", "irr100"},
    )

    threshold_rows = []
    for method in per_seed.method.drop_duplicates():
        x = per_seed[per_seed.method == method]
        for metric in ("T800", "T900", "T1000"):
            vals = pd.to_numeric(x[metric], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(vals)
            threshold_rows.append({
                "experiment": "crop",
                "method": method,
                "threshold": int(metric[1:]),
                "mean_episode_reached": float(np.nanmean(vals)) if finite.any() else math.nan,
                "sd_episode_reached": float(np.nanstd(vals, ddof=1)) if finite.sum() > 1 else math.nan,
                "reached": int(finite.sum()),
                "n": int(len(vals)),
                "seed_values": json.dumps(
                    {
                        str(seed): (None if not np.isfinite(v) else float(v))
                        for seed, v in zip(x.seed, vals)
                    },
                    ensure_ascii=False,
                ),
            })

    return (
        per_seed,
        agg,
        perf_paired,
        resource_paired,
        pd.DataFrame(threshold_rows),
    )


def crop_trust_summary(anchored_root: Path) -> pd.DataFrame:
    rows = []

    for seed in CROP_SEEDS:
        seed_dir = find_seed_dir(anchored_root, seed)
        p = seed_dir / "diagnostics.csv"
        if not p.is_file():
            continue

        d = pd.read_csv(p)
        if "adaptive_trust" not in d.columns:
            continue

        trust = pd.to_numeric(d.adaptive_trust, errors="coerce")
        d = d.loc[trust.notna()].copy()
        d["adaptive_trust"] = trust.loc[trust.notna()].to_numpy()
        if not len(d):
            continue

        phase_indices = np.array_split(np.arange(len(d)), 3)
        for phase, idx in zip(("early", "mid", "late"), phase_indices):
            chunk = d.iloc[idx]
            v = chunk.adaptive_trust.to_numpy(dtype=float)
            rows.append({
                "experiment": "crop",
                "method": "Anchored",
                "seed": seed,
                "phase": phase,
                "trust_mean": float(v.mean()),
                "trust_median": float(np.median(v)),
                "trust_max": float(v.max()),
                "rows": int(len(v)),
            })

    return pd.DataFrame(rows)


# ======================================================================================================================
# HalfCheetah
# ======================================================================================================================

CHEETAH_METRICS = [
    "at_1m",
    "at_1p5m",
    "gain_1m_to_1p5m",
    "late500k",
    "last5",
    "last10",
    "best",
    "auc_1p5m",
]


def cheetah_seed_metrics(seed_dir: Path) -> dict:
    require_completed(seed_dir)
    g = checkpoint_means(require_file(seed_dir, "eval_metrics.csv"))
    x = g[g.index <= CHEETAH_TARGET_STEPS]

    if not len(x) or int(x.index.max()) != CHEETAH_TARGET_STEPS:
        raise AuditError(
            f"HalfCheetah wrong/incomplete horizon: {seed_dir}, "
            f"max_step={int(g.index.max()) if len(g) else None}"
        )

    for step in (1_000_000, 1_500_000):
        if step not in x.index:
            raise AuditError(f"HalfCheetah missing exact checkpoint {step}: {seed_dir}")

    at_1m = float(x.loc[1_000_000])
    at_1p5m = float(x.loc[1_500_000])

    return {
        "at_1m": at_1m,
        "at_1p5m": at_1p5m,
        "gain_1m_to_1p5m": at_1p5m - at_1m,
        "late500k": float(x[x.index >= 1_000_000].mean()),
        "last5": float(x.tail(5).mean()),
        "last10": float(x.tail(10).mean()),
        "best": float(x.max()),
        "best_step": int(x.idxmax()),
        "auc_1p5m": normalized_auc(x),
        "checkpoints": int(len(x)),
        "max_step": int(x.index.max()),
    }


def analyze_cheetah(root: Path):
    rows = []

    for method, dirname in CHEETAH_METHODS.items():
        method_root = root / dirname
        if not method_root.is_dir():
            raise AuditError(f"Missing HalfCheetah method directory: {method_root}")

        for seed in CHEETAH_SEEDS:
            rows.append({
                "experiment": "halfcheetah",
                "method": method,
                "seed": seed,
                **cheetah_seed_metrics(find_seed_dir(method_root, seed)),
            })

    per_seed = pd.DataFrame(rows)
    agg = aggregate_table(per_seed, "halfcheetah", CHEETAH_METRICS)

    paired = paired_table(
        per_seed,
        "halfcheetah",
        "AMAF",
        ["TD3"],
        CHEETAH_METRICS,
    )

    a = per_seed[per_seed.method == "AMAF"].set_index("seed")
    t = per_seed[per_seed.method == "TD3"].set_index("seed")
    gap_rows = []

    for seed in CHEETAH_SEEDS:
        g1 = float(a.loc[seed, "at_1m"] - t.loc[seed, "at_1m"])
        g15 = float(a.loc[seed, "at_1p5m"] - t.loc[seed, "at_1p5m"])
        gap_rows.append({
            "experiment": "halfcheetah",
            "seed": seed,
            "gap_1m_amaf_minus_td3": g1,
            "gap_1p5m_amaf_minus_td3": g15,
            "gap_change": g15 - g1,
            "amaf_win_1m": int(g1 > 0),
            "amaf_win_1p5m": int(g15 > 0),
        })

    return per_seed, agg, paired, pd.DataFrame(gap_rows)


def cheetah_training_diagnostics(root: Path) -> pd.DataFrame:
    """
    Descriptive base-safe actor diagnostics if corresponding columns exist.

    No NULL weight is reconstructed from diagnostics.csv here because the current
    generic logger stores conditional residual-head weights, not the H+1 full
    selective gate.
    """
    method_root = root / CHEETAH_METHODS["AMAF"]
    rows = []

    for seed in CHEETAH_SEEDS:
        seed_dir = find_seed_dir(method_root, seed)
        p = seed_dir / "train_metrics.csv"
        if not p.is_file():
            continue

        d = pd.read_csv(p)
        rec = {
            "experiment": "halfcheetah",
            "method": "AMAF",
            "seed": seed,
            "train_rows": int(len(d)),
        }

        for out_name, candidates in {
            "actor_conflict_rate": ["actor_conflict"],
            "actor_projection_rate": [
                "actor_projection_applied",
                "projection_applied",
            ],
            "actor_grad_cosine_mean": [
                "actor_grad_cosine",
                "actor_gradient_cosine",
                "actor_cosine",
                "grad_cosine",
            ],
        }.items():
            for c in candidates:
                if c in d.columns:
                    v = pd.to_numeric(d[c], errors="coerce").dropna()
                    if len(v):
                        rec[out_name] = float(v.mean())
                    break

        rows.append(rec)

    return pd.DataFrame(rows)


def cheetah_gate_summary(root: Path) -> pd.DataFrame:
    """
    Audit and summarize the logged CONDITIONAL residual-head gate distribution.

    IMPORTANT
    ---------
    gate_w0..gate_w3 are conditional residual-family weights produced by

        residual_weights / residual_trust.clamp_min(1e-12)

    in the current implementation. When residual_trust becomes extremely small
    (near explicit NULL/base fallback), that clamped normalization can cease to
    represent a probability simplex numerically. In such rows, logged
    gate_entropy/gate_max_weight may become mathematically invalid
    (e.g. entropy < 0 or max weight > 1).

    Therefore this analysis:
      * NEVER interprets invalid rows as a normalized conditional distribution;
      * validates gate_w values directly;
      * summarizes entropy/effective-head/max-weight only on valid simplex rows;
      * reports the number/rate of degenerate rows separately.

    The frozen performance metrics are independent of this diagnostic filtering.
    """
    method_root = root / CHEETAH_METHODS["AMAF"]
    rows = []

    for seed in CHEETAH_SEEDS:
        p = find_seed_dir(method_root, seed) / "diagnostics.csv"
        if not p.is_file():
            continue

        d = pd.read_csv(p)
        weight_cols = [
            c for c in d.columns
            if c.startswith("gate_w")
            and pd.to_numeric(d[c], errors="coerce").notna().any()
        ]
        if not weight_cols:
            continue

        numeric = d[weight_cols].apply(pd.to_numeric, errors="coerce")
        has_weights = numeric.notna().any(axis=1)
        valid_positions = np.flatnonzero(has_weights.to_numpy())
        if len(valid_positions) == 0:
            continue

        # Keep chronological thirds over diagnostic checkpoints.
        phase_positions = np.array_split(valid_positions, 3)

        for phase, positions in zip(("early", "mid", "late"), phase_positions):
            if len(positions) == 0:
                continue

            w = numeric.iloc[positions].to_numpy(dtype=float)

            finite = np.all(np.isfinite(w), axis=1)
            nonnegative = np.all(w >= -1e-8, axis=1)
            bounded = np.all(w <= 1.0 + 1e-6, axis=1)
            sums = np.nansum(w, axis=1)
            simplex_sum = np.isclose(sums, 1.0, rtol=1e-4, atol=1e-5)
            valid_simplex = finite & nonnegative & bounded & simplex_sum

            n_total = int(len(w))
            n_valid = int(valid_simplex.sum())
            n_degenerate = int(n_total - n_valid)

            rec = {
                "experiment": "halfcheetah",
                "method": "AMAF",
                "seed": seed,
                "phase": phase,
                "rows": n_total,
                "valid_simplex_rows": n_valid,
                "degenerate_conditional_rows": n_degenerate,
                "degenerate_conditional_rate": (
                    float(n_degenerate / n_total) if n_total else math.nan
                ),
                "gate_weight_sum_mean_all_rows": float(np.nanmean(sums)),
                "gate_weight_sum_min_all_rows": float(np.nanmin(sums)),
                "gate_weight_sum_max_all_rows": float(np.nanmax(sums)),
            }

            # Only probability-simplex rows are eligible for selectivity metrics.
            if n_valid:
                valid_idx = np.asarray(positions)[valid_simplex]
                chunk = d.iloc[valid_idx]

                ent = (
                    pd.to_numeric(chunk["gate_entropy"], errors="coerce")
                    if "gate_entropy" in chunk.columns
                    else pd.Series(dtype=float)
                )
                eff = (
                    pd.to_numeric(chunk["effective_heads"], errors="coerce")
                    if "effective_heads" in chunk.columns
                    else pd.Series(dtype=float)
                )
                maxw = (
                    pd.to_numeric(chunk["gate_max_weight"], errors="coerce")
                    if "gate_max_weight" in chunk.columns
                    else pd.Series(dtype=float)
                )

                # Additional sanity constraints before reporting each logged metric.
                ent = ent[(ent >= -1e-8) & np.isfinite(ent)]
                eff = eff[(eff >= 1.0 - 1e-6) & (eff <= len(weight_cols) + 1e-6) & np.isfinite(eff)]
                maxw = maxw[(maxw >= -1e-8) & (maxw <= 1.0 + 1e-6) & np.isfinite(maxw)]

                rec.update({
                    "conditional_gate_entropy_mean_valid": (
                        float(ent.mean()) if len(ent) else math.nan
                    ),
                    "conditional_effective_heads_mean_valid": (
                        float(eff.mean()) if len(eff) else math.nan
                    ),
                    "conditional_gate_max_weight_mean_valid": (
                        float(maxw.mean()) if len(maxw) else math.nan
                    ),
                })
            else:
                rec.update({
                    "conditional_gate_entropy_mean_valid": math.nan,
                    "conditional_effective_heads_mean_valid": math.nan,
                    "conditional_gate_max_weight_mean_valid": math.nan,
                })

            rows.append(rec)

    return pd.DataFrame(rows)


# ======================================================================================================================
# Regression/headline checks
# ======================================================================================================================

def assert_close(label: str, actual: float, expected: float, tol: float = 0.15):
    if not np.isfinite(actual) or abs(actual - expected) > tol:
        raise AuditError(
            f"Headline regression failed: {label}: "
            f"actual={actual:.6f}, expected≈{expected:.6f}, tol={tol}"
        )
    print(f"[REGRESSION OK] {label:48s} {actual:.2f}")


def run_headline_regressions(
    lunar_seed: pd.DataFrame,
    crop_seed: pd.DataFrame,
    crop_thresholds: pd.DataFrame,
    cheetah_seed: pd.DataFrame,
    cheetah_gap: pd.DataFrame,
):
    print("\n" + "=" * 120)
    print("HEADLINE REGRESSION CHECKS")
    print("=" * 120)

    # Lunar frozen headline values previously audited.
    def lunar_mean(method, metric):
        return float(lunar_seed[lunar_seed.method == method][metric].mean())

    assert_close("Lunar Protected final mean", lunar_mean("Protected", "final"), 240.33)
    assert_close("Lunar Uniform final mean", lunar_mean("Uniform", "final"), 248.78)
    assert_close("Lunar Naive final mean", lunar_mean("Naive", "final"), 196.44)
    assert_close("Lunar Protected last5 mean", lunar_mean("Protected", "last5"), 247.89)
    assert_close("Lunar Protected AUC mean", lunar_mean("Protected", "auc"), 116.96, tol=0.25)
    protected_success = int(
        lunar_seed[lunar_seed.method == "Protected"].final_gt_200.sum()
    )
    if protected_success != 8:
        raise AuditError(
            f"Lunar Protected final>200 regression failed: {protected_success}/10 != 8/10"
        )
    print("[REGRESSION OK] Lunar Protected final>200                 8/10")

    # Crop 5-seed headline values.
    c = crop_seed[crop_seed.method == "Anchored"]
    assert_close("Crop Anchored full mean", float(c.full.mean()), -1214.33)
    assert_close("Crop Anchored late1000 mean", float(c.late1000.mean()), 882.82)
    assert_close("Crop Anchored final100 mean", float(c.final100.mean()), 996.74)
    assert_close("Crop Anchored peak250 mean", float(c.peak250.mean()), 1026.14)
    assert_close("Crop Anchored irrigation100 mean", float(c.irr100.mean()), 213.68)

    for threshold, expected_mean, expected_reached in [
        (900, 2455.4, 5),
        (1000, 2711.4, 5),
    ]:
        row = crop_thresholds[
            (crop_thresholds.method == "Anchored")
            & (crop_thresholds.threshold == threshold)
        ].iloc[0]
        assert_close(
            f"Crop Anchored T{threshold} mean episode",
            float(row.mean_episode_reached),
            expected_mean,
            tol=0.2,
        )
        if int(row.reached) != expected_reached:
            raise AuditError(
                f"Crop Anchored T{threshold} reached regression failed: "
                f"{int(row.reached)}/5 != {expected_reached}/5"
            )
        print(
            f"[REGRESSION OK] Crop Anchored T{threshold} reached"
            f"{'':20s}{int(row.reached)}/5"
        )

    # HalfCheetah 5-seed headline values.
    td3 = cheetah_seed[cheetah_seed.method == "TD3"]
    amaf = cheetah_seed[cheetah_seed.method == "AMAF"]

    assert_close("HalfCheetah TD3 1.5M mean", float(td3.at_1p5m.mean()), 11711.4)
    assert_close("HalfCheetah AMAF 1.5M mean", float(amaf.at_1p5m.mean()), 12046.9)
    assert_close("HalfCheetah TD3 1M→1.5M gain", float(td3.gain_1m_to_1p5m.mean()), 1089.0)
    assert_close("HalfCheetah AMAF 1M→1.5M gain", float(amaf.gain_1m_to_1p5m.mean()), 1771.0)

    gap1 = float(cheetah_gap.gap_1m_amaf_minus_td3.mean())
    gap15 = float(cheetah_gap.gap_1p5m_amaf_minus_td3.mean())
    change = float(cheetah_gap.gap_change.mean())
    assert_close("HalfCheetah mean gap @1M", gap1, -346.5)
    assert_close("HalfCheetah mean gap @1.5M", gap15, 335.5)
    assert_close("HalfCheetah mean gap reversal", change, 682.0)

    wins = int(cheetah_gap.amaf_win_1p5m.sum())
    if wins != 4:
        raise AuditError(f"HalfCheetah 1.5M wins regression failed: {wins}/5 != 4/5")
    print("[REGRESSION OK] HalfCheetah AMAF wins @1.5M               4/5")


# ======================================================================================================================
# Console rendering
# ======================================================================================================================

def print_aggregate(title: str, df: pd.DataFrame):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    for metric in df.metric.drop_duplicates():
        print(f"\n[{metric}]")
        x = df[df.metric == metric]
        for _, r in x.iterrows():
            if metric.endswith("_count"):
                print(f"{r.method:10s} {int(r['mean'])}/{int(r.n)}")
            else:
                print(
                    f"{r.method:10s} "
                    f"{fmt(r['mean']):>10s} ± {fmt(r.sd):>8s} "
                    f"(n={int(r.n)})"
                )


def print_paired(title: str, df: pd.DataFrame):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    for comp in df.comparator.drop_duplicates():
        print(f"\nvs {comp}")
        for _, r in df[df.comparator == comp].iterrows():
            print(
                f"{r.metric:20s} "
                f"Δ={fmt(r.mean_delta_focal_minus_comparator):>9s} "
                f"wins={int(r.wins)}/{int(r.n)} "
                f"preferred={r.direction}"
            )


def print_crop_thresholds(df: pd.DataFrame):
    print("\n" + "=" * 120)
    print("CROP SAMPLE EFFICIENCY — FIRST ROLLING-100 THRESHOLD")
    print("=" * 120)

    for threshold in df.threshold.drop_duplicates():
        print(f"\n[rolling100 >= {int(threshold)}] lower episode = better")
        for _, r in df[df.threshold == threshold].iterrows():
            print(
                f"{r.method:10s} "
                f"mean_episode={fmt(r.mean_episode_reached, 1):>8s} "
                f"reached={int(r.reached)}/{int(r.n)}"
            )


def print_cheetah_gap(df: pd.DataFrame):
    print("\n" + "=" * 120)
    print("HALFCHEETAH GAP REVERSAL — AMAF MINUS TD3")
    print("=" * 120)

    for _, r in df.iterrows():
        print(
            f"seed={int(r.seed):3d} "
            f"1M={r.gap_1m_amaf_minus_td3:+9.1f} "
            f"1.5M={r.gap_1p5m_amaf_minus_td3:+9.1f} "
            f"change={r.gap_change:+9.1f}"
        )

    print(
        "\nMEAN "
        f"1M={df.gap_1m_amaf_minus_td3.mean():+.1f}  "
        f"1.5M={df.gap_1p5m_amaf_minus_td3.mean():+.1f}  "
        f"change={df.gap_change.mean():+.1f}  "
        f"wins@1.5M={int(df.amaf_win_1p5m.sum())}/{len(df)}"
    )


# ======================================================================================================================
# Inputs / outputs
# ======================================================================================================================

@dataclass(frozen=True)
class Inputs:
    lunar: Path
    crop_baseline: Path
    crop_anchored: Path
    halfcheetah: Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Audit frozen AMAF_UNIFIED paper data and reproduce canonical metrics."
    )
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--lunar", type=Path, default=Path("paper_data/lunar/primary_20260828"))
    p.add_argument(
        "--crop-baseline",
        type=Path,
        default=Path("paper_data/crop/baseline_primary_20260828"),
    )
    p.add_argument(
        "--crop-anchored",
        type=Path,
        default=Path("paper_data/crop/anchored_primary_20260830"),
    )
    p.add_argument(
        "--halfcheetah",
        type=Path,
        default=Path("paper_data/halfcheetah/primary_1p5m_20260830"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs/paper_results_summary"),
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Audit and print only; do not write derived CSV/JSON outputs.",
    )
    return p.parse_args()


def resolve_paths(args):
    if args.repo_root is not None:
        repo = args.repo_root.resolve()
    else:
        here = Path(__file__).resolve()
        repo = here.parents[1] if here.parent.name == "analysis" else Path.cwd().resolve()

    def abs_path(p: Path) -> Path:
        return p.resolve() if p.is_absolute() else (repo / p).resolve()

    return (
        repo,
        Inputs(
            lunar=abs_path(args.lunar),
            crop_baseline=abs_path(args.crop_baseline),
            crop_anchored=abs_path(args.crop_anchored),
            halfcheetah=abs_path(args.halfcheetah),
        ),
        abs_path(args.out_dir),
    )


def main() -> int:
    args = parse_args()
    repo, inputs, out_dir = resolve_paths(args)

    print("=" * 120)
    print("AMAF_UNIFIED — CANONICAL PAPER RESULT AUDIT")
    print("=" * 120)
    print(f"repo_root: {repo}")

    audits = []
    for label, root in [
        ("Lunar primary", inputs.lunar),
        ("Crop baseline", inputs.crop_baseline),
        ("Crop anchored", inputs.crop_anchored),
        ("HalfCheetah 1.5M", inputs.halfcheetah),
    ]:
        if not root.is_dir():
            raise AuditError(f"Missing frozen bundle: {label}: {root}")
        result = verify_manifest(root)
        expected_entries = EXPECTED_MANIFEST_COUNTS[label]
        if int(result["manifest_entries"]) != expected_entries:
            raise AuditError(
                f"Manifest-count regression failed for {label}: "
                f"{result['manifest_entries']} != {expected_entries}"
            )
        checkpoints = list(root.rglob("checkpoints"))
        if checkpoints:
            raise AuditError(
                f"Frozen analysis bundle unexpectedly contains checkpoint directories: {checkpoints}"
            )

        # Crop raw DSSAT runtime logs are intentionally excluded from the paper-data
        # freeze, matching the existing Crop baseline policy and repository *.log ignore.
        if label == "Crop anchored":
            runtime_logs = list(root.rglob("*.log"))
            if runtime_logs:
                raise AuditError(
                    f"Crop anchored frozen bundle unexpectedly contains runtime logs: {runtime_logs}"
                )

        result["label"] = label
        result["expected_manifest_entries"] = expected_entries
        audits.append(result)
        print(
            f"[HASH OK] {label:18s} "
            f"entries={result['manifest_entries']:3d}/{expected_entries:3d}  {root}"
        )

    lunar_seed, lunar_agg, lunar_pair = analyze_lunar(inputs.lunar)

    (
        crop_seed,
        crop_agg,
        crop_perf_pair,
        crop_resource_pair,
        crop_threshold,
    ) = analyze_crop(inputs.crop_baseline, inputs.crop_anchored)
    crop_trust = crop_trust_summary(inputs.crop_anchored)

    cheetah_seed, cheetah_agg, cheetah_pair, cheetah_gap = analyze_cheetah(
        inputs.halfcheetah
    )
    cheetah_train_diag = cheetah_training_diagnostics(inputs.halfcheetah)
    cheetah_gate = cheetah_gate_summary(inputs.halfcheetah)

    print_aggregate("LUNAR — 10-SEED CANONICAL BOARD", lunar_agg)
    print_paired("LUNAR — PROTECTED PAIRED DELTAS", lunar_pair)

    print_aggregate("CROP — 5-SEED CANONICAL BOARD", crop_agg)
    print_paired("CROP — ANCHORED PERFORMANCE DELTAS", crop_perf_pair)
    print_paired("CROP — ANCHORED RESOURCE/OUTCOME DELTAS", crop_resource_pair)
    print_crop_thresholds(crop_threshold)

    print_aggregate("HALFCHEETAH — 5-SEED CANONICAL BOARD", cheetah_agg)
    print_paired("HALFCHEETAH — AMAF PAIRED DELTAS", cheetah_pair)
    print_cheetah_gap(cheetah_gap)

    if len(crop_trust):
        print("\n" + "=" * 120)
        print("CROP ANCHORED — TRUST DYNAMICS (DESCRIPTIVE)")
        print("=" * 120)
        print(crop_trust.to_string(index=False))

    if len(cheetah_train_diag):
        print("\n" + "=" * 120)
        print("HALFCHEETAH AMAF — BASE-SAFE TRAINING DIAGNOSTICS (DESCRIPTIVE)")
        print("=" * 120)
        print(cheetah_train_diag.to_string(index=False))

    if len(cheetah_gate):
        print("\n" + "=" * 120)
        print("HALFCHEETAH AMAF — CONDITIONAL RESIDUAL-GATE SELECTIVITY (DESCRIPTIVE)")
        print("=" * 120)
        print(cheetah_gate.to_string(index=False))

    run_headline_regressions(
        lunar_seed,
        crop_seed,
        crop_threshold,
        cheetah_seed,
        cheetah_gap,
    )

    if len(cheetah_gate):
        for col in (
            "conditional_gate_entropy_mean_valid",
            "conditional_effective_heads_mean_valid",
            "conditional_gate_max_weight_mean_valid",
        ):
            vals = pd.to_numeric(cheetah_gate[col], errors="coerce").dropna()
            if col == "conditional_gate_entropy_mean_valid" and (vals < -1e-8).any():
                raise AuditError("Invalid negative conditional gate entropy survived filtering")
            if col == "conditional_gate_max_weight_mean_valid" and (vals > 1.0 + 1e-6).any():
                raise AuditError("Invalid conditional gate max weight > 1 survived filtering")
        print(
            "[DIAGNOSTIC AUDIT OK] HalfCheetah conditional gate: "
            "invalid near-degenerate rows separated from valid simplex summaries"
        )

    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)

        tables = {
            "lunar_per_seed.csv": lunar_seed,
            "lunar_aggregate.csv": lunar_agg,
            "lunar_paired_deltas.csv": lunar_pair,
            "crop_per_seed.csv": crop_seed,
            "crop_aggregate.csv": crop_agg,
            "crop_performance_paired_deltas.csv": crop_perf_pair,
            "crop_resource_paired_deltas.csv": crop_resource_pair,
            "crop_thresholds.csv": crop_threshold,
            "crop_anchored_trust.csv": crop_trust,
            "halfcheetah_per_seed.csv": cheetah_seed,
            "halfcheetah_aggregate.csv": cheetah_agg,
            "halfcheetah_paired_deltas.csv": cheetah_pair,
            "halfcheetah_gap_reversal.csv": cheetah_gap,
            "halfcheetah_training_diagnostics.csv": cheetah_train_diag,
            "halfcheetah_conditional_gate.csv": cheetah_gate,
        }
        for name, df in tables.items():
            df.to_csv(out_dir / name, index=False)

        summary = {
            "audit": audits,
            "headline": {
                "crop_anchored_full_mean": float(
                    crop_seed[crop_seed.method == "Anchored"].full.mean()
                ),
                "crop_anchored_late1000_mean": float(
                    crop_seed[crop_seed.method == "Anchored"].late1000.mean()
                ),
                "crop_anchored_final100_mean": float(
                    crop_seed[crop_seed.method == "Anchored"].final100.mean()
                ),
                "crop_anchored_T1000_reached": int(
                    crop_seed[
                        (crop_seed.method == "Anchored")
                        & crop_seed.T1000.notna()
                    ].shape[0]
                ),
                "halfcheetah_td3_1p5m_mean": float(
                    cheetah_seed[cheetah_seed.method == "TD3"].at_1p5m.mean()
                ),
                "halfcheetah_amaf_1p5m_mean": float(
                    cheetah_seed[cheetah_seed.method == "AMAF"].at_1p5m.mean()
                ),
                "halfcheetah_amaf_wins_1p5m": int(
                    cheetah_gap.amaf_win_1p5m.sum()
                ),
                "halfcheetah_gap_change_mean": float(
                    cheetah_gap.gap_change.mean()
                ),
            },
        }

        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nDerived summaries written to: {out_dir}")

    print("\n" + "=" * 120)
    print("AUDIT PASS — frozen bytes, expected seeds/horizons, and headline regressions all validated.")
    print("=" * 120)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print("\nAUDIT FAILED", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
