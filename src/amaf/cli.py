from __future__ import annotations

import argparse

from amaf.runners.experiment import run_experiment


def build_parser():
    p = argparse.ArgumentParser(description="AMAF experiment runner")
    p.add_argument("--experiment", required=True, help="Experiment YAML path")
    p.add_argument("--algorithm", help="Run only this algorithm label")
    p.add_argument("--seed", type=int, help="Run only this seed")
    p.add_argument("--output-root", help="Override outputs/runs root")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-skip-completed", action="store_true")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config key, e.g. --set training.total_steps=10000",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    failures = run_experiment(
        args.experiment,
        algorithm_filter=args.algorithm,
        seed_filter=args.seed,
        dry_run=args.dry_run,
        skip_completed=not args.no_skip_completed,
        output_root=args.output_root,
        cli_overrides=args.set,
    )
    if failures:
        print("\nFailures:")
        for algorithm, seed, reason in failures:
            print(f"- {algorithm} seed={seed}: {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
