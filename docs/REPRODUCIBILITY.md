# Reproducibility guide

The release separates four reproducibility levels so that code, derived data,
and unavailable external systems are not conflated.

## 1. Code-level verification

Install the development dependencies and run:

```bash
pytest -q
python -m compileall -q src scripts analysis
amaf-run --experiment experiments/revision/lunarlander_fixed500k.yaml --dry-run
```

The tests cover network construction, agent update steps, AMAF gating
properties, configuration inheritance, seed handling, analysis metrics, and
crop utility transformations.

## 2. Figure-level verification

All manuscript Gnuplot inputs are committed under:

- `analysis/figures/figure_data/`
- `analysis/figures/data/discussion/`

Render every figure with:

```bash
bash scripts/render_paper_figures.sh
```

The Discussion plot-ready tables can be reconstructed from the included
frozen handoff and checked against Results quantities:

```bash
python -m analysis.figures.export_discussion_figures
git diff --exit-code -- analysis/figures/data/discussion
```

## 3. Experiment reruns

Every experiment is keyed by `(experiment, algorithm, seed)`. The resolved
configuration and system metadata are stored with each run. Algorithms in a
comparison use explicit matched seeds and budgets; evaluation is separated
from exploration/training.

The final seed sets are:

- LunarLander: 11, 22, 33, 44, 55, 66, 77, 88, 99, 111.
- Crop: 11, 22, 33, 44, 55.
- HalfCheetah primary: 11, 22, 33, 44, 55.
- HalfCheetah replication: 66, 77, 88, 99, 111.
- SAC baseline: 11, 22, 33, 44, 55.

`COMPLETED` marks a finished seed. Checkpoints preserve model and optimizer
state but not full replay-buffer/environment state, so they are recovery
artifacts rather than bit-exact mid-run resumptions.

## 4. Raw-result audit

The full raw trajectory archive is not stored in this code repository. The
raw-to-canonical builders and their source registry are included for
transparency, but builders that require `paper_data/` will report missing data
until that archive is supplied separately. Canonical and handoff tables in
this release are frozen derived artifacts and are covered by
`DATA_SHA256SUMS`.

## Determinism limits

Seeds are applied to Python, NumPy, PyTorch, and supported environments.
Hardware, simulator, driver, and library differences can still affect
floating-point training trajectories. Report package versions and hardware
metadata with rerun results rather than assuming bit-identical learning.
