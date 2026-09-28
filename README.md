# AMAF research code

Official research-code release for **Adaptive Multi-View Advantage Fusion
(AMAF)**. The repository contains the implementations, experiment
configurations, tests, paper-analysis source, frozen derived tables, and
plot-ready data used for the LunarLander, Crop, and HalfCheetah studies.

## What is included

- DQN, Dueling DQN, TD3, SAC, and AMAF variants under `src/amaf/`.
- Final matched-seed protocols and ablations under `experiments/revision/`.
- Reusable YAML configurations under `configs/`.
- Canonical derived tables and manuscript handoff tables under `paper_tables/`.
- Plot-ready CSV/TSV data and Gnuplot sources under `analysis/figures/`.
- Unit tests for agents, networks, configuration resolution, metrics, and crop
  utilities.

Large raw trajectories, model checkpoints, legacy notebooks, machine-specific
environment captures, and generated PDF/PNG files are intentionally not
included. See [Public release scope](docs/PUBLIC_RELEASE.md) for the exact
boundary and [Reproducibility](docs/REPRODUCIBILITY.md) for what can be rebuilt
from this repository alone.

## Installation

Python 3.10 or newer is required by the released type annotations and analysis
tools.

```bash
git clone https://github.com/soon2soon/AMAF_CODE.git
cd AMAF_CODE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gym,dev]"
pytest -q
```

For a CPU-only environment, install the appropriate PyTorch build first if the
default pip package is unsuitable for your platform.

## Quick protocol check

Resolve the final LunarLander matrix without starting training:

```bash
amaf-run \
  --experiment experiments/revision/lunarlander_fixed500k.yaml \
  --dry-run
```

Run a short LunarLander smoke experiment:

```bash
amaf-run \
  --experiment experiments/revision/lunarlander_fixed500k.yaml \
  --algorithm dqn \
  --seed 11 \
  --set training.total_steps=2000 \
  --set training.max_episodes=5 \
  --set evaluation.enabled=false
```

Experiment outputs are written to
`outputs/runs/<experiment>/<algorithm>/seed_<seed>/`. A completed run contains
the resolved configuration, system metadata, metric CSV files, checkpoints,
and a `COMPLETED` marker.

## Final experiment matrices

The paper protocols are intentionally split so that every run has an explicit
method, seed set, and budget.

```bash
# LunarLander baselines/AMAF variants and protected AMAF
amaf-run --experiment experiments/revision/lunarlander_fixed500k.yaml
amaf-run --experiment experiments/revision/lunarlander_protected500k_v32.yaml

# Crop baselines/AMAF variants and anchored AMAF
amaf-run --experiment experiments/revision/crop_multiseed.yaml
amaf-run --experiment experiments/revision/crop_anchored_dev.yaml

# HalfCheetah primary five, replication five, and SAC baseline
amaf-run --experiment experiments/revision/halfcheetah_td3_vs_v2_1p5m_fresh.yaml
amaf-run --experiment experiments/revision/halfcheetah_td3_vs_v2_1p5m_extra5.yaml
amaf-run --experiment experiments/revision/halfcheetah_sac_1p5m_fresh.yaml
```

These are full research runs and can require substantial compute. Start with a
dry run or the bounded examples in [examples/README.md](examples/README.md).

## Rebuild the paper figures

The repository includes the frozen plot-ready inputs, so the paper figures can
be rendered without the omitted raw trajectories. On Debian/Ubuntu:

```bash
sudo apt-get install gnuplot-nox fonts-liberation
bash scripts/render_paper_figures.sh
```

PDF and PNG outputs are written below `paper_figures/` and are ignored by Git.
The Discussion plot tables can also be regenerated from the included handoff:

```bash
python -m analysis.figures.export_discussion_figures
bash scripts/render_paper_figures.sh discussion
```

## Analyze new runs

```bash
python scripts/analyze.py \
  --experiment experiments/revision/halfcheetah_td3_vs_v2_1p5m_fresh.yaml
python scripts/make_figures.py \
  --experiment experiments/revision/halfcheetah_td3_vs_v2_1p5m_fresh.yaml
python scripts/make_tables.py \
  --experiment experiments/revision/halfcheetah_td3_vs_v2_1p5m_fresh.yaml
```

## Crop environment

The reported crop experiments used an external gym-DSSAT/PDI installation that
cannot be redistributed here. The runner and configurations are included for
inspection and use with an existing installation. Read
[docs/CROP_ENVIRONMENT.md](docs/CROP_ENVIRONMENT.md) before attempting a crop
run.

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff). Publication DOI
and venue fields should be added when available. The code is released under the
[MIT License](LICENSE). Result tables remain subject to the citation request in
`CITATION.cff`.
