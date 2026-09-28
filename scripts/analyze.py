#!/usr/bin/env python
import argparse
from amaf.analysis.aggregate import analyze_experiment

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--output-root")
    args = p.parse_args()
    out = analyze_experiment(args.experiment, args.output_root)
    print(out)
