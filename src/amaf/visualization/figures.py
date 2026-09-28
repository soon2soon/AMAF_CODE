from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read(summary_root: Path, name: str) -> pd.DataFrame:
    p = summary_root / name
    return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()


def learning_curve(summary_root: str | Path, output_dir: str | Path) -> Path | None:
    summary_root, output_dir = Path(summary_root), Path(output_dir)
    df = _read(summary_root, "learning_curves.csv")
    if df.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for algo, g in df.groupby("algorithm"):
        # Curves may not share exact x across seeds; aggregate on recorded x values.
        agg = g.groupby("x")["return"].agg(["mean", "std", "count"]).reset_index()
        x = agg["x"].to_numpy(float); mean = agg["mean"].to_numpy(float)
        std = agg["std"].fillna(0).to_numpy(float); n = agg["count"].to_numpy(float)
        ci = 1.96 * std / np.sqrt(np.maximum(n, 1))
        line, = ax.plot(x, mean, label=algo)
        ax.fill_between(x, mean-ci, mean+ci, alpha=0.16, color=line.get_color())
    ax.set_xlabel("Environment steps / episode index")
    ax.set_ylabel("Return")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = output_dir / "learning_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(output_dir / "learning_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def final_performance(summary_root: str | Path, output_dir: str | Path) -> Path | None:
    summary_root, output_dir = Path(summary_root), Path(output_dir)
    df = _read(summary_root, "seed_level_metrics.csv")
    if df.empty or "final_performance" not in df:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    algos = list(dict.fromkeys(df["algorithm"].tolist()))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.15*len(algos)), 4.2))
    for i, algo in enumerate(algos):
        vals = df.loc[df.algorithm == algo, "final_performance"].dropna().to_numpy(float)
        if not len(vals): continue
        jitter = np.linspace(-0.06, 0.06, len(vals)) if len(vals)>1 else np.array([0.0])
        ax.scatter(np.full(len(vals), i)+jitter, vals, s=24, alpha=0.75)
        mean = np.mean(vals); se = np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else 0
        ax.errorbar(i, mean, yerr=1.96*se, fmt="o", capsize=4)
    ax.set_xticks(range(len(algos)), algos, rotation=25, ha="right")
    ax.set_ylabel("Final policy performance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out=output_dir/"final_performance.pdf"
    fig.savefig(out,bbox_inches="tight"); fig.savefig(output_dir/"final_performance.png",dpi=300,bbox_inches="tight")
    plt.close(fig); return out


def gating_entropy(summary_root: str | Path, output_dir: str | Path) -> Path | None:
    # Uses raw run directories referenced from seed_level_metrics to preserve temporal dynamics.
    summary_root, output_dir = Path(summary_root), Path(output_dir)
    seed = _read(summary_root, "seed_level_metrics.csv")
    if seed.empty: return None
    rows=[]
    for _, r in seed.iterrows():
        p=Path(r["run_dir"])/"diagnostics.csv"
        if p.exists() and p.stat().st_size:
            d=pd.read_csv(p)
            if "gate_entropy" in d:
                for _, z in d.iterrows(): rows.append({"algorithm":r["algorithm"],"seed":r["seed"],"x":z["env_steps"],"entropy":z["gate_entropy"]})
    df=pd.DataFrame(rows)
    if df.empty: return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.6,4.2))
    for algo,g in df.groupby("algorithm"):
        agg=g.groupby("x")["entropy"].agg(["mean","std","count"]).reset_index()
        x=agg.x.to_numpy(float); m=agg["mean"].to_numpy(float); sd=agg["std"].fillna(0).to_numpy(float); n=agg["count"].to_numpy(float)
        ci=1.96*sd/np.sqrt(np.maximum(n,1)); line,=ax.plot(x,m,label=algo); ax.fill_between(x,m-ci,m+ci,alpha=.16,color=line.get_color())
    ax.set_xlabel("Environment steps"); ax.set_ylabel("Gate entropy"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
    out=output_dir/"gating_entropy.pdf"; fig.savefig(out,bbox_inches="tight"); fig.savefig(output_dir/"gating_entropy.png",dpi=300,bbox_inches="tight"); plt.close(fig); return out
