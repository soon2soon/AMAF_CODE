#!/usr/bin/env python
import argparse
from pathlib import Path
from amaf.analysis.aggregate import analyze_experiment
from amaf.config import load_yaml
from amaf.tables.make_tables import make_tables
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--experiment",required=True); p.add_argument("--output-root"); a=p.parse_args()
    exp=Path(a.experiment).resolve(); cfg=load_yaml(exp); name=cfg.get("name",exp.stem); summary=analyze_experiment(exp,a.output_root)
    repo=exp.parents[2] if exp.parent.name in {"revision","paper_original"} else Path.cwd()
    for x in make_tables(summary,repo/"outputs"/"tables"/name): print(x)
