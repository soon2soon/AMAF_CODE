# HalfCheetah SAC comparison

This directory compares TD3, AMAF, and SAC on the same five HalfCheetah seeds
(11, 22, 33, 44, and 55). Each method was evaluated every 10k environment
steps through 1.5M steps.

## Files

| File | Description |
| --- | --- |
| `halfcheetah_sac_primary5_per_seed.csv` | Matched per-seed metrics for all three methods |
| `halfcheetah_sac_primary5_summary.csv` | Method means |
| `halfcheetah_amaf_vs_sac_paired.csv` | Paired AMAF-minus-SAC effects and bootstrap intervals |

## Metrics

- `return_1m` and `return_1p5m`: evaluation return at the exact checkpoint.
- `gain`: return at 1.5M minus return at 1M.
- `last5` and `last10`: mean return over the final 5 or 10 checkpoints.
- `auc`: time-normalized trapezoidal AUC from 10k to 1.5M steps.
- Paired effects: AMAF minus SAC; `gain_delta` is the difference between the
  two methods' gains.

Confidence intervals are 95% percentile-bootstrap intervals for the paired
seed-level mean, using 100,000 resamples and RNG seed 20260907.

## Summary

| Method | 1M | 1.5M | Gain | Last-5 | Last-10 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TD3 | 10622.36 | 11711.40 | 1089.04 | 11676.75 | 11641.36 | 9232.05 |
| AMAF | 10275.82 | 12046.86 | 1771.05 | 11926.21 | 11799.37 | 9108.73 |
| SAC | 10859.52 | 11780.38 | 920.87 | 11862.65 | 11822.04 | 9536.05 |

AMAF had a larger 1M-to-1.5M gain than SAC on all five seeds; the mean paired
gain difference was 850.18 with a 95% interval of [482.40, 1252.38]. This does
not establish overall sample-efficiency superiority: SAC had the higher 1M
mean and higher normalized AUC. The AMAF-minus-SAC terminal interval also
included zero, with AMAF winning on two of five seeds.

The released files are derived numerical tables. Model checkpoints and raw
per-run logs are not part of this public code repository and are not required
to inspect or render the reported comparison.
