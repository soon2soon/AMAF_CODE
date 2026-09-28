# Discussion claim map

This note identifies the tables that support each interpretation and records
the corresponding boundary on the claim.

## LunarLander protection

**Supported statement.** Protected AMAF reduces the severe late-stage failures
seen in Naive AMAF on average.

**Evidence.** `ablation_discussion_summary.csv` reports Protected-minus-Naive
Last-5 and retention effects. `robustness_seed_sensitivity.csv` reports the
full-sample and leave-one-seed-out results.

**Boundary.** Protected does not consistently exceed Uniform, and some
Protected-minus-Naive seed effects are negative. Routing summaries do not show
state-conditioned specialization.

## Crop anchoring

**Supported statement.** Anchoring moves the learned policy toward lower
nitrogen and irrigation use relative to Uniform AMAF.

**Evidence.** `crop_tradeoff_discussion.csv` contains paired resource, return,
and yield changes. `ablation_discussion_summary.csv` and
`robustness_seed_sensitivity.csv` provide aggregate and leave-one-seed-out
contrasts.

**Boundary.** Resource savings must be reported with the associated return and
yield changes. No composite efficiency score is defined.

## HalfCheetah late-stage gain

**Supported statement.** AMAF shows a larger 1M-to-1.5M improvement than TD3 in
the combined ten-seed analysis.

**Evidence.** `halfcheetah_late_stage_discussion.csv` separates terminal return
from late gain. `robustness_seed_sensitivity.csv` includes all seeds, including
seed 77, and the leave-one-seed-out results.

**Boundary.** A larger late gain is not the same as terminal superiority. The
combined terminal interval includes zero, and the combined Last-5, Last-10,
and AUC mean effects are not positive.

## SAC comparison

**Supported statement.** AMAF has a larger primary-five 1M-to-1.5M gain than
SAC.

**Evidence.** `sac_discussion_context.csv` reports absolute method metrics and
paired AMAF-minus-SAC effects.

**Boundary.** SAC has higher 1M return and normalized AUC. AMAF wins the
terminal comparison on two of five seeds, and the terminal interval includes
zero. The result does not support a general sample-efficiency claim.

## Routing behavior

**Supported statement.** The recorded routing distributions vary in
concentration and dominant-head switching over training.

**Evidence.** `mechanism_head_summary.csv`, `mechanism_head_by_regime.csv`, and
`mechanism_head_long.csv` contain the per-seed and checkpoint-level summaries.

**Boundary.** Low entropy is compatible with both selective routing and global
collapse. Because aligned state features and semantic regime labels were not
recorded, these tables cannot establish state-conditioned specialization.
