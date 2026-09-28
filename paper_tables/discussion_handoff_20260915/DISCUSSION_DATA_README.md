# Discussion data

These tables support the mechanism, robustness, and boundary analyses in the
Discussion. They complement the performance tables in
`results_handoff_20260915/`.

## Files

| File | Description |
| --- | --- |
| `mechanism_head_long.csv` | Checkpoint-by-head routing and head-output records |
| `mechanism_head_by_regime.csv` | Early, middle, and late checkpoint summaries |
| `mechanism_head_summary.csv` | Per-seed entropy, concentration, and switching summaries |
| `ablation_discussion_summary.csv` | Protection, anchoring, and reference-method contrasts |
| `robustness_seed_sensitivity.csv` | Full-sample and leave-one-seed-out summaries |
| `crop_tradeoff_discussion.csv` | Paired resource, return, and yield changes |
| `halfcheetah_late_stage_discussion.csv` | Terminal and 1M-to-1.5M gain effects |
| `sac_discussion_context.csv` | TD3, AMAF, and SAC primary-five context |
| `DISCUSSION_CLAIM_MAP.md` | Supported interpretations and their limits |

## Routing summaries

Routing statistics use rows whose recorded weights are finite, non-negative,
no greater than one, and sum to one within tolerance. Entropy is recomputed
from each valid row. Normalized entropy is `H / log(K)`, and effective head
count is `exp(H)`. Equal maxima are recorded as ties instead of being assigned
to an arbitrary head. A switch is counted only when two adjacent checkpoints
both have a unique dominant head.

Lunar Protected and Crop Anchored runs record both the raw router and the
effective gate. HalfCheetah records the conditional residual-head gate but not
an explicit null/reference weight. Missing values that follow from an
unrecorded quantity or an undefined switch denominator are retained as
missing, not imputed.

The labels `time_early`, `time_mid`, and `time_late` divide each seed's ordered
checkpoints into thirds. They are training-time bins, not semantic state
regimes. The logs do not contain the state-aligned information needed to
distinguish specialization from global head collapse.

## Statistical conventions

- LunarLander uses Last-5 evaluation return and retention.
- Crop uses final-100 return, yield, nitrogen, and irrigation.
- HalfCheetah uses exact 1M/1.5M returns, late gain, Last-5/Last-10, and
  time-normalized AUC.
- SAC comparisons use the same five seeds and checkpoint definitions as the
  primary HalfCheetah cohort.

Mean confidence intervals use a seed-level 95% percentile bootstrap with
100,000 resamples and RNG seed 20260907. Leave-one-seed-out rows remove one
listed seed at a time; `excluded_seed=NONE` denotes the full sample.

## Interpretation limits

- Protected does not consistently outperform Uniform on LunarLander.
- Crop resource savings may coincide with lower return or yield.
- The combined HalfCheetah terminal interval includes zero, and its late-window
  and AUC mean effects are not uniformly positive.
- SAC has higher primary-five 1M return and AUC than AMAF; a larger late gain
  alone is not evidence of better sample efficiency.
- Routing entropy and head switching are descriptive checkpoint diagnostics,
  not direct evidence of state-conditioned specialization.
