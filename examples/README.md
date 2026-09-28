# Execution examples

Run commands from the repository root after installing the relevant optional
dependencies.

## Configuration-only check

```bash
amaf-run --experiment experiments/revision/halfcheetah_td3_vs_v2_1p5m_fresh.yaml --dry-run
```

## Bounded LunarLander run

```bash
amaf-run \
  --experiment experiments/revision/lunarlander_fixed500k.yaml \
  --algorithm dqn \
  --seed 11 \
  --set training.total_steps=2000 \
  --set training.max_episodes=5 \
  --set evaluation.enabled=false
```

## Bounded HalfCheetah run

```bash
amaf-run \
  --experiment experiments/revision/halfcheetah_td3_vs_v2_1p5m_fresh.yaml \
  --algorithm td3 \
  --seed 11 \
  --set training.total_steps=5000 \
  --set training.random_steps=1000 \
  --set evaluation.interval_steps=2500 \
  --set evaluation.episodes=2
```

## Analyze a completed matrix

```bash
python scripts/analyze.py --experiment experiments/revision/lunarlander_fixed500k.yaml
python scripts/make_figures.py --experiment experiments/revision/lunarlander_fixed500k.yaml
python scripts/make_tables.py --experiment experiments/revision/lunarlander_fixed500k.yaml
```

## Render paper figures

```bash
bash scripts/render_paper_figures.sh
```

Render a single family:

```bash
bash scripts/render_paper_figures.sh lunar
bash scripts/render_paper_figures.sh crop
bash scripts/render_paper_figures.sh halfcheetah
bash scripts/render_paper_figures.sh discussion
```
