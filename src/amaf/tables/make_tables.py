from __future__ import annotations
from pathlib import Path
import pandas as pd


def make_tables(summary_root: str | Path, output_dir: str | Path):
    summary_root, output_dir=Path(summary_root),Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for name in ["summary","statistical_tests"]:
        src=summary_root/f"{name}.csv"
        if not src.exists() or not src.stat().st_size: continue
        df=pd.read_csv(src); dst=output_dir/f"{name}.csv"; df.to_csv(dst,index=False); outputs.append(dst)
        tex=output_dir/f"{name}.tex"
        tex.write_text(df.to_latex(index=False,float_format=lambda x:f"{x:.4g}"),encoding="utf-8"); outputs.append(tex)
    return outputs
