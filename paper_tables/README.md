# Paper tables

This directory contains the derived data released with the paper. The tables
are organized by use rather than by the script that produced them.

| Directory | Contents |
| --- | --- |
| `canonical/` | Per-seed results, aggregate statistics, paired comparisons, and diagnostics |
| `results_handoff_20260915/` | Values used in the main Results figures and tables |
| `results_handoff_sac_20260915/` | Primary-five TD3, AMAF, and SAC comparison |
| `discussion_handoff_20260915/` | Mechanism summaries, robustness checks, and claim boundaries |

The seed is the experimental unit. Evaluation episodes, training episodes, and
logged checkpoints are not treated as independent replicates. Confidence
intervals for seed-level means use a 95% percentile bootstrap with 100,000
resamples and RNG seed 20260907.

The public repository contains derived tables, not the full raw trajectory
archive. `DATA_SHA256SUMS` records the checksum of every released CSV and TSV.
The analysis code is available under `analysis/`.
