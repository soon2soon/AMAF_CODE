# Frozen derived paper tables

This directory contains the derived numerical artifacts selected for the
public code release.

- `canonical/`: per-seed, aggregate, paired, and diagnostic summary tables.
- `results_handoff_20260915/`: Results/Appendix tables aligned with the main
  figures.
- `results_handoff_sac_20260915/`: matched primary-five SAC comparison.
- `discussion_handoff_20260915/`: mechanism, robustness, trade-off, and claim
  boundary tables.

The statistical experimental unit is the seed. Evaluation episodes and crop
episodes are not treated as independent replicates. Metric definitions,
bootstrap parameters, comparison directions, and limitations are documented
inside the handoff directories.

The raw frozen trajectories used to derive these files are deliberately not
duplicated in this code repository. The derivation source remains under
`analysis/`, and every released CSV/TSV is covered by `DATA_SHA256SUMS`.
