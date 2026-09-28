#!/usr/bin/env python3
"""Freeze and verify primary-five HalfCheetah TD3/AMAF/SAC Results tables."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .halfcheetah import (
    EXPECTED_STEPS,
    ORIGINAL_SEEDS,
    _load_seed_curve,
    load_halfcheetah_frozen_curves,
)
from .io import FigureDataRegistry
from .stats import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE,
    DEFAULT_RNG_SEED,
    summarize_paired_frame,
)
from .validation import ROOT


SAC_DATASET_ID = "halfcheetah_sac_primary5"
TD3_AMAF_CANONICAL_ID = "halfcheetah_per_seed"
EXPECTED_SEEDS = tuple(sorted(ORIGINAL_SEEDS))
METHODS = ("TD3", "AMAF", "SAC")
OUTPUT_DIR = ROOT / "paper_tables" / "results_handoff_sac_20260915"

CANONICAL_BASENAMES = {
    "halfcheetah_sac_per_seed.csv",
    "halfcheetah_sac_aggregate.csv",
    "halfcheetah_primary5_td3_amaf_sac_per_seed.csv",
    "halfcheetah_primary5_td3_amaf_sac_aggregate.csv",
    "halfcheetah_primary5_td3_amaf_sac_pairwise.csv",
}

METRICS = (
    ("at_1m", "return_1m"),
    ("at_1p5m", "return_1p5m"),
    ("gain_1m_to_1p5m", "gain"),
    ("last5", "last5"),
    ("last10", "last10"),
    ("auc_norm", "auc"),
)

PAIRED_METRICS = (
    ("at_1p5m", "terminal_delta"),
    ("gain_1m_to_1p5m", "gain_delta"),
    ("last5", "last5_delta"),
    ("last10", "last10_delta"),
    ("auc_norm", "auc_delta"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frozen_analysis_checksums(frozen_root: Path) -> list[str]:
    checksum_path = frozen_root / "SHA256SUMS"
    declared: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative_path = line.split(maxsplit=1)
        declared[relative_path.lstrip("*")] = digest

    required = {"MANIFEST.json"}
    for seed in EXPECTED_SEEDS:
        required.add(f"seed_{seed}/COMPLETED")
        required.add(f"seed_{seed}/eval_metrics.csv")
    for relative_path in sorted(required):
        if relative_path not in declared:
            raise ValueError(f"Analysis input is absent from SHA256SUMS: {relative_path}")
        path = frozen_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen analysis input: {path}")
        if _sha256(path) != declared[relative_path]:
            raise ValueError(f"Frozen analysis input checksum mismatch: {relative_path}")

    return sorted(
        relative_path
        for relative_path in declared
        if not (frozen_root / relative_path).is_file()
    )


def _assert_frame_equal(label: str, observed: pd.DataFrame, expected: pd.DataFrame) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_names=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-9,
        )
    except AssertionError as error:
        raise ValueError(f"{label} mismatch: {error}") from error


def _assert_close(label: str, observed: float, expected: float, *, atol: float) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=atol):
        raise ValueError(f"{label} mismatch: observed={observed}, expected={expected}")


def _load_manifest_canonical(
    frozen_root: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    manifest_path = frozen_root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing frozen manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("lifecycle") != "frozen" or not manifest.get("manuscript_allowed"):
        raise ValueError("SAC MANIFEST is not manuscript-safe frozen evidence.")
    if tuple(manifest.get("seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("SAC MANIFEST seed population changed.")
    if int(manifest.get("target_env_steps", 0)) != 1_500_000:
        raise ValueError("SAC MANIFEST target is not 1.5M steps.")

    declared = manifest.get("canonical_tables", {})
    if {Path(path).name for path in declared} != CANONICAL_BASENAMES:
        raise ValueError("SAC MANIFEST canonical table set changed.")

    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    root_resolved = ROOT.resolve()
    for relative_path, expected_hash in declared.items():
        path = (ROOT / relative_path).resolve()
        if root_resolved not in path.parents:
            raise ValueError(f"Canonical path escapes repository root: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise ValueError(f"Canonical SHA-256 mismatch: {relative_path}")
        frames[path.name] = pd.read_csv(path)
        hashes[relative_path] = observed_hash
    return frames, hashes


def _derive_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, seed), curve in curves.groupby(["method", "seed"], sort=False):
        ordered = curve.sort_values("env_steps")
        if not np.array_equal(ordered["env_steps"].to_numpy(dtype=int), EXPECTED_STEPS):
            raise ValueError(f"{method}/seed-{seed} checkpoint coverage changed.")
        checkpoint = ordered.set_index("env_steps")["evaluation_return"]
        x = checkpoint.index.to_numpy(dtype=float)
        y = checkpoint.to_numpy(dtype=float)
        at_1m = float(checkpoint.loc[1_000_000])
        at_1p5m = float(checkpoint.loc[1_500_000])
        rows.append({
            "method": str(method),
            "seed": int(seed),
            "at_1m": at_1m,
            "at_1p5m": at_1p5m,
            "gain_1m_to_1p5m": at_1p5m - at_1m,
            "last5": float(checkpoint.tail(5).mean()),
            "last10": float(checkpoint.tail(10).mean()),
            "auc_norm": float(np.trapezoid(y, x) / (x[-1] - x[0])),
        })
    frame = pd.DataFrame(rows)
    method_rank = {method: rank for rank, method in enumerate(METHODS)}
    frame["_method_rank"] = frame["method"].map(method_rank)
    return frame.sort_values(["_method_rank", "seed"]).drop(columns="_method_rank").reset_index(drop=True)


def _load_sac_curves(registry: FigureDataRegistry) -> pd.DataFrame:
    if registry.dataset_kind(SAC_DATASET_ID) != "frozen_dir":
        raise ValueError(f"{SAC_DATASET_ID} is not a frozen directory dataset.")
    if registry.dataset_lifecycle(SAC_DATASET_ID) != "frozen":
        raise ValueError(f"{SAC_DATASET_ID} lifecycle is not frozen.")
    registry.assert_manuscript_allowed(SAC_DATASET_ID)
    frozen_root = registry.load_path(SAC_DATASET_ID, manuscript=True)
    observed_seeds = {
        int(path.name.removeprefix("seed_"))
        for path in frozen_root.glob("seed_*")
        if path.is_dir() and path.name.removeprefix("seed_").isdigit()
    }
    if observed_seeds != set(EXPECTED_SEEDS):
        raise ValueError(f"Unexpected SAC seed directories: {sorted(observed_seeds)}")

    frames = []
    for seed in EXPECTED_SEEDS:
        seed_root = frozen_root / f"seed_{seed}"
        if not (seed_root / "COMPLETED").is_file():
            raise ValueError(f"SAC seed {seed} is not marked COMPLETED.")
        curve = _load_seed_curve(seed_root / "eval_metrics.csv", seed)
        curve.insert(0, "method", "SAC")
        frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def _canonical_metric_view(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "auc_1p5m" in work.columns and "auc_norm" not in work.columns:
        work = work.rename(columns={"auc_1p5m": "auc_norm"})
    columns = ["method", "seed", "at_1m", "at_1p5m", "gain_1m_to_1p5m", "last5", "last10", "auc_norm"]
    return work[columns].sort_values(["method", "seed"]).reset_index(drop=True)


def _validate_canonical(
    source: pd.DataFrame,
    registered: pd.DataFrame,
    canonical: dict[str, pd.DataFrame],
) -> None:
    registered_primary = registered.loc[
        registered["seed"].astype(int).isin(EXPECTED_SEEDS)
        & registered["method"].isin(["TD3", "AMAF"])
    ]
    _assert_frame_equal(
        "Frozen TD3/AMAF versus registered canonical",
        _canonical_metric_view(source.loc[source["method"].isin(["TD3", "AMAF"])]),
        _canonical_metric_view(registered_primary),
    )
    _assert_frame_equal(
        "Frozen SAC versus standalone canonical",
        _canonical_metric_view(source.loc[source["method"] == "SAC"]),
        _canonical_metric_view(canonical["halfcheetah_sac_per_seed.csv"]),
    )
    _assert_frame_equal(
        "Frozen three-method source versus combined canonical",
        _canonical_metric_view(source),
        _canonical_metric_view(
            canonical["halfcheetah_primary5_td3_amaf_sac_per_seed.csv"]
        ),
    )


def _build_summary(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        method_frame = source.loc[source["method"] == method]
        row: dict[str, float | int | str] = {"method": method, "n": len(method_frame)}
        for source_metric, output_metric in METRICS:
            row[f"{output_metric}_mean"] = float(method_frame[source_metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_paired(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_metric, output_metric in PAIRED_METRICS:
        _, summary = summarize_paired_frame(
            source,
            seed_col="seed",
            method_col="method",
            metric_col=source_metric,
            focal="AMAF",
            comparator="SAC",
            direction="higher",
            confidence=DEFAULT_CONFIDENCE,
            resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
            rng_seed=DEFAULT_RNG_SEED,
        )
        rows.append({
            "comparison": "AMAF - SAC",
            "metric": output_metric,
            "n": summary.n,
            "mean_delta": summary.mean_delta_focal_minus_comparator,
            "median_delta": summary.median_delta_focal_minus_comparator,
            "ci_low": summary.ci95_low_focal_minus_comparator,
            "ci_high": summary.ci95_high_focal_minus_comparator,
            "wins": summary.wins,
            "losses": summary.losses,
            "ties": summary.ties,
        })
    return pd.DataFrame(rows)


def _build_per_seed(source: pd.DataFrame) -> pd.DataFrame:
    indexed = source.set_index(["seed", "method"])
    rows = []
    for seed in EXPECTED_SEEDS:
        row: dict[str, float | int] = {"seed": seed}
        for metric, output_metric in METRICS:
            for method in METHODS:
                row[f"{method.lower()}_{output_metric.removeprefix('return_')}"] = float(
                    indexed.loc[(seed, method), metric]
                )
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame[[
        "seed",
        "td3_1m",
        "amaf_1m",
        "sac_1m",
        "td3_1p5m",
        "amaf_1p5m",
        "sac_1p5m",
        "td3_gain",
        "amaf_gain",
        "sac_gain",
        "td3_last5",
        "amaf_last5",
        "sac_last5",
        "td3_last10",
        "amaf_last10",
        "sac_last10",
        "td3_auc",
        "amaf_auc",
        "sac_auc",
    ]]


def _validate_aggregate_canonical(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    canonical: dict[str, pd.DataFrame],
) -> None:
    summary_index = summary.set_index("method")
    for filename in (
        "halfcheetah_sac_aggregate.csv",
        "halfcheetah_primary5_td3_amaf_sac_aggregate.csv",
    ):
        aggregate = canonical[filename]
        for row in aggregate.itertuples(index=False):
            output_metric = dict(METRICS)[row.metric]
            _assert_close(
                f"{filename} {row.method}/{row.metric} mean",
                float(row.mean),
                float(summary_index.loc[row.method, f"{output_metric}_mean"]),
                atol=1e-9,
            )

    canonical_paired = canonical[
        "halfcheetah_primary5_td3_amaf_sac_pairwise.csv"
    ]
    canonical_paired = canonical_paired.loc[canonical_paired["comparison"] == "AMAF-SAC"]
    paired_index = paired.set_index("metric")
    for source_metric, output_metric in PAIRED_METRICS:
        row = canonical_paired.loc[canonical_paired["metric"] == source_metric]
        if len(row) != 1:
            raise ValueError(f"No unique AMAF-SAC canonical row for {source_metric}.")
        expected = row.iloc[0]
        observed = paired_index.loc[output_metric]
        for canonical_col, output_col in (
            ("mean_delta", "mean_delta"),
            ("median_delta", "median_delta"),
            ("ci_low", "ci_low"),
            ("ci_high", "ci_high"),
        ):
            _assert_close(
                f"Canonical AMAF-SAC {source_metric} {canonical_col}",
                float(expected[canonical_col]),
                float(observed[output_col]),
                atol=1e-9,
            )
        for column in ("n", "wins", "losses"):
            if int(expected[column]) != int(observed[column]):
                raise ValueError(f"Canonical AMAF-SAC {source_metric} {column} mismatch.")


def _validate_sanity_fingerprints(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    method_expected = {
        "TD3": (10622.36, 11711.40, 1089.04, 11676.75, 11641.36, 9232.05),
        "AMAF": (10275.82, 12046.86, 1771.05, 11926.21, 11799.37, 9108.73),
        "SAC": (10859.52, 11780.38, 920.87, 11862.65, 11822.04, 9536.05),
    }
    method_columns = (
        "return_1m_mean",
        "return_1p5m_mean",
        "gain_mean",
        "last5_mean",
        "last10_mean",
        "auc_mean",
    )
    indexed_summary = summary.set_index("method")
    for method, expected_values in method_expected.items():
        for column, expected in zip(method_columns, expected_values, strict=True):
            _assert_close(
                f"Sanity fingerprint {method}/{column}",
                float(indexed_summary.loc[method, column]),
                expected,
                atol=0.02,
            )

    paired_expected = {
        "terminal_delta": (266.48, -234.64, 2),
        "gain_delta": (850.18, 814.71, 5),
        "last5_delta": (63.56, None, 2),
        "last10_delta": (-22.67, None, 2),
        "auc_delta": (-427.32, None, 2),
    }
    indexed_paired = paired.set_index("metric")
    for metric, (expected_mean, expected_median, expected_wins) in paired_expected.items():
        row = indexed_paired.loc[metric]
        _assert_close(
            f"Sanity fingerprint {metric}/mean",
            float(row["mean_delta"]),
            expected_mean,
            atol=0.02,
        )
        if expected_median is not None:
            _assert_close(
                f"Sanity fingerprint {metric}/median",
                float(row["median_delta"]),
                expected_median,
                atol=0.02,
            )
        if int(row["wins"]) != expected_wins:
            raise ValueError(f"Sanity fingerprint {metric}/wins mismatch.")


def _validate_outputs(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    per_seed: pd.DataFrame,
) -> None:
    if len(summary) != 3 or summary["method"].tolist() != list(METHODS):
        raise ValueError("Summary must contain TD3, AMAF, SAC in order.")
    if len(paired) != 5 or paired.duplicated(["comparison", "metric"]).any():
        raise ValueError("Paired table must contain five unique AMAF-SAC metrics.")
    if len(per_seed) != 5 or per_seed["seed"].tolist() != list(EXPECTED_SEEDS):
        raise ValueError("Per-seed table must contain all five primary seeds.")
    if per_seed["seed"].duplicated().any():
        raise ValueError("Per-seed table contains duplicate seeds.")
    for label, frame in (("summary", summary), ("paired", paired), ("per_seed", per_seed)):
        if frame.isna().any().any():
            raise ValueError(f"{label} contains NaN values.")
    if not (paired["n"] == 5).all():
        raise ValueError("Every paired metric must have n=5.")
    if not ((paired["wins"] + paired["losses"] + paired["ties"]) == 5).all():
        raise ValueError("Paired win/loss/tie counts do not sum to five.")


def _readme(
    registry: FigureDataRegistry,
    canonical_hashes: dict[str, str],
    missing_archive_files: list[str],
    summary: pd.DataFrame,
    paired: pd.DataFrame,
) -> str:
    sac_fingerprint = registry.fingerprint(SAC_DATASET_ID)
    td3_amaf_fingerprint = registry.fingerprint(TD3_AMAF_CANONICAL_ID)
    canonical_lines = "\n".join(
        f"- `{path}`: `{digest}`" for path, digest in canonical_hashes.items()
    )
    summary_lines = "\n".join(
        f"- {row.method}: 1M {row.return_1m_mean:.2f}; 1.5M {row.return_1p5m_mean:.2f}; "
        f"gain {row.gain_mean:.2f}; Last-5 {row.last5_mean:.2f}; "
        f"Last-10 {row.last10_mean:.2f}; AUC {row.auc_mean:.2f}"
        for row in summary.itertuples(index=False)
    )
    paired_lines = "\n".join(
        f"- {row.metric}: mean {row.mean_delta:+.2f}; median {row.median_delta:+.2f}; "
        f"95% CI [{row.ci_low:+.2f}, {row.ci_high:+.2f}]; "
        f"wins/losses/ties {row.wins}/{row.losses}/{row.ties}"
        for row in paired.itertuples(index=False)
    )
    return f"""# HalfCheetah SAC primary-five Results check

## Scope

Training과 re-run 없이 frozen/canonical evidence만 사용한 manuscript-safe handoff이다.
Experimental unit은 seed이며 seed 집합은 11, 22, 33, 44, 55이다. 세 방법 모두 10k
간격 evaluation checkpoint를 1.5M까지 동일하게 사용한다.

## Sources

- SAC registry ID: `{SAC_DATASET_ID}`
- SAC path: `{sac_fingerprint['path']}`
- lifecycle/manuscript allowed: `frozen` / `true`
- SAC {sac_fingerprint['fingerprint_type']}: `{sac_fingerprint['sha256']}`
- TD3/AMAF canonical registry ID: `{TD3_AMAF_CANONICAL_ID}`
- TD3/AMAF canonical fingerprint: `{td3_amaf_fingerprint['sha256']}`

SAC source path는 `figure_contract.yaml` registry로 해결했다. MANIFEST에 선언된 canonical
파일과 검증된 SHA-256은 다음과 같다.

{canonical_lines}

## Metrics and bootstrap

- 1M/1.5M: 정확한 checkpoint의 evaluation-episode mean return
- gain: 1.5M return − 1M return
- Last-5/Last-10: 마지막 5/10 checkpoint mean return의 평균
- normalized AUC: 10k–1.5M checkpoint mean curve의 시간 정규화 trapezoidal AUC
- paired delta: 모두 AMAF − SAC. gain delta는 두 방법 gain의 차이
- bootstrap: seed-level percentile mean CI, {DEFAULT_BOOTSTRAP_RESAMPLES:,} resamples,
  95% confidence, RNG seed {DEFAULT_RNG_SEED}

## Primary-five method means

{summary_lines}

## AMAF − SAC paired effects

{paired_lines}

## Verification

- Frozen checkpoint data에서 세 방법의 모든 필수 metric을 재계산했다.
- Registered TD3/AMAF canonical, standalone SAC canonical, combined three-method canonical,
  aggregate canonical, AMAF-SAC paired canonical과 수치가 일치했다.
- 요청에 제공된 method mean 및 AMAF-SAC mean/median/wins sanity fingerprint와 모두 일치했다.
- 출력 행 수, seed 집합, duplicate key, NaN, wins/losses/ties 합계를 검증했다.
- 수치 계산에 사용한 `MANIFEST.json`, 5개 `COMPLETED`, 5개 `eval_metrics.csv`는
  `SHA256SUMS`와 모두 일치했다.

전체 `sha256sum -c SHA256SUMS`는 목록에 선언됐지만 frozen tree에 없는 model-weight
파일 {len(missing_archive_files)}개 때문에 PASS가 아니다. 누락 항목은 각 seed의
`checkpoints/best.pt`, `checkpoints/final.pt`, `checkpoints/latest.pt`이며 numerical table
계산에는 사용되지 않았다. 따라서 Results 수치는 frozen evaluation input으로 재현 및
검증되지만, 이 디렉터리를 checkpoint weight까지 완전한 archive로 기술하면 안 된다.

## Results-safe reading

SAC는 primary-five 평균에서 1M return과 normalized AUC가 AMAF보다 높았다. AMAF는
1M→1.5M gain이 SAC보다 컸으며 paired gain delta는 5/5 seed에서 양수이고 95% CI도
0보다 높았다. AMAF terminal mean은 SAC보다 높았지만 terminal wins는 2/5이고 CI가
0을 포함하므로 seed-wise dominance로 기술하면 안 된다.
"""


def export() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing handoff directory: {OUTPUT_DIR}")

    registry = FigureDataRegistry()
    sac_root = registry.load_path(SAC_DATASET_ID, manuscript=True)
    missing_archive_files = _validate_frozen_analysis_checksums(sac_root)
    canonical, canonical_hashes = _load_manifest_canonical(sac_root)

    sac_curves = _load_sac_curves(registry)
    td3_amaf_curves = load_halfcheetah_frozen_curves(registry)
    td3_amaf_curves = td3_amaf_curves.loc[
        td3_amaf_curves["seed"].astype(int).isin(EXPECTED_SEEDS)
    ]
    source = _derive_metrics(pd.concat([td3_amaf_curves, sac_curves], ignore_index=True))
    if len(source) != 15 or set(source["method"]) != set(METHODS):
        raise ValueError("Source must contain three methods by five matched seeds.")
    if source.duplicated(["method", "seed"]).any() or source.isna().any().any():
        raise ValueError("Source metrics are duplicated or incomplete.")

    registered = registry.load_csv(TD3_AMAF_CANONICAL_ID, manuscript=True)
    _validate_canonical(source, registered, canonical)
    summary = _build_summary(source)
    paired = _build_paired(source)
    per_seed = _build_per_seed(source)
    _validate_aggregate_canonical(summary, paired, canonical)
    _validate_sanity_fingerprints(summary, paired)
    _validate_outputs(summary, paired, per_seed)

    temporary = Path(tempfile.mkdtemp(prefix=".results_handoff_sac_20260915_", dir=OUTPUT_DIR.parent))
    try:
        outputs = {
            "halfcheetah_sac_primary5_summary.csv": summary,
            "halfcheetah_amaf_vs_sac_paired.csv": paired,
            "halfcheetah_sac_primary5_per_seed.csv": per_seed,
        }
        for filename, frame in outputs.items():
            frame.to_csv(temporary / filename, index=False)
            _assert_frame_equal(filename, pd.read_csv(temporary / filename), frame)
        (temporary / "SAC_RESULTS_CHECK.md").write_text(
            _readme(
                registry,
                canonical_hashes,
                missing_archive_files,
                summary,
                paired,
            ),
            encoding="utf-8",
        )
        if len(list(temporary.iterdir())) != 4:
            raise ValueError("SAC handoff must contain three CSVs and one Markdown report.")
        os.rename(temporary, OUTPUT_DIR)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"SAC source: {registry.fingerprint(SAC_DATASET_ID)}")
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print("Frozen analysis-input/canonical/fingerprint/sanity/output checks: PASS")
    print(f"Full frozen archive caveat: {len(missing_archive_files)} listed weight files are absent.")
    return summary, paired, per_seed


def main() -> None:
    export()


if __name__ == "__main__":
    main()
