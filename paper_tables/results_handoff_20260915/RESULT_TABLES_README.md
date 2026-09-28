# Main result tables

These files contain the numerical values used in the LunarLander, Crop, and
HalfCheetah Results sections. All summaries use seed-level observations.

## Files

| File | Description |
| --- | --- |
| `lunar_per_seed.csv` | LunarLander Last-5 return and retention by method and seed |
| `lunar_result_summary.csv` | LunarLander method summaries and bootstrap intervals |
| `lunar_paired_effects.csv` | Protected-minus-comparator paired effects |
| `crop_per_seed.csv` | Crop return, yield, nitrogen, and irrigation values by method and seed |
| `crop_result_summary.csv` | Crop method summaries and bootstrap intervals |
| `crop_paired_effects.csv` | Anchored-versus-Uniform paired effects |
| `halfcheetah_per_seed.csv` | TD3 and AMAF endpoints, late gain, late windows, and AUC by seed |
| `halfcheetah_result_summary.csv` | Original-five, replication-five, and combined-ten summaries |

## Metrics

**LunarLander.** `last5_return` is the mean of the final five evaluation
checkpoint means. `retention_pct` is `100 * Last-5 / Best-5`, where Best-5 is
the mean of the five highest checkpoint means.

**Crop.** `late1000_return` covers episodes 2001--3000. The `final100_*`
columns summarize the final 100 training episodes. `yield100`, `n100`, and
`irr100` use the native simulator units. Learning curves first average each
seed in non-overlapping 100-episode blocks and then aggregate across seeds.

**HalfCheetah.** The 1M and 1.5M columns are deterministic evaluation returns
at those exact checkpoints. `gain` is the 1M-to-1.5M change. Last-5 and Last-10
average the corresponding final checkpoint means. AUC is the time-normalized
trapezoidal area from 10k to 1.5M steps.

## Contrasts

- LunarLander: `Protected - Naive` and `Protected - Uniform`.
- Crop return and yield: `Anchored - Uniform`.
- Crop resource saving: `Uniform usage - Anchored usage`.
- HalfCheetah endpoint, late-window, and AUC effects: `AMAF - TD3`.
- HalfCheetah gain effect: `(AMAF 1.5M - AMAF 1M) - (TD3 1.5M - TD3 1M)`.

Positive values favor the focal method under these definitions. Wins, losses,
and ties are counted from the paired seed-level signs.

## Seeds and uncertainty

- LunarLander: 10 seeds (11, 22, 33, 44, 55, 66, 77, 88, 99, 111).
- Crop: 5 seeds (11, 22, 33, 44, 55).
- HalfCheetah: original five (11--55), replication five (66--111), and their
  combined ten-seed cohort.

Intervals are 95% percentile-bootstrap intervals for the seed-level mean,
using 100,000 resamples and RNG seed 20260907. Medians are descriptive sample
medians; their adjacent confidence limits apply to the mean, not the median.
