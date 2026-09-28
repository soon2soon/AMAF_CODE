# Discussion claim-to-evidence map

## Claim 1

Potential claim: Protection mitigates severe adaptive failures in Lunar.

Supported by:
- `ablation_discussion_summary.csv`: Protected−Naive Last-5/retention
- `robustness_seed_sensitivity.csv`: 해당 contrast의 전체 및 leave-one-seed-out 결과
- `mechanism_head_summary.csv`, `mechanism_head_by_regime.csv`: checkpoint-time gate/router 기술통계

Metric/direction: 양의 Protected−Naive Last-5 또는 retention(pp)는 Protected 우세.

Limits:
- Protected는 Uniform보다 일관되게 높지 않다.
- 일부 Protected−Naive seed delta는 음수다.
- state/regime가 기록되지 않아 time-varying routing을 conditional specialization의 직접 증거로 쓸 수 없다.

## Claim 2

Potential claim: Anchoring moves Crop control toward a lower-resource operating point.

Supported by:
- `crop_tradeoff_discussion.csv`: seed별 nitrogen/irrigation saving
- `ablation_discussion_summary.csv`: Anchored−Uniform 및 Anchored−Naive contrast
- `robustness_seed_sensitivity.csv`: resource-saving LOO 부호

Metric/direction: nitrogen/irrigation saving은 comparator usage−Anchored usage; 양수는 Anchored 절감.

Limits:
- `crop_tradeoff_discussion.csv`의 yield/return delta가 음수인 seed가 다수다.
- 임의 efficiency composite score를 만들지 않았으며 resource saving과 outcome cost를 함께 보고해야 한다.

## Claim 3

Potential claim: AMAF shows late-stage relative improvement in HalfCheetah.

Supported by:
- `halfcheetah_late_stage_discussion.csv`: original5/replication5/combined10 gain delta
- `ablation_discussion_summary.csv`: combined AMAF−TD3 terminal/gain/late-window/AUC
- `robustness_seed_sensitivity.csv`: 모든 seed와 LOO 결과

Metric/direction: gain delta=`(AMAF 1.5M−1M)−(TD3 1.5M−1M)`; 양수는 AMAF의 더 큰 late gain.

Limits:
- Terminal superiority와 gain improvement는 동일 주장이 아니다.
- Terminal CI는 0을 포함하고 Last-5/Last-10/AUC combined mean은 양수가 아니다.
- 불리한 seed 77도 포함했다.

## Claim 4

Potential claim: AMAF late-stage gain is visible relative to SAC, but this does not establish sample-efficiency superiority.

Supported by:
- `sac_discussion_context.csv`: TD3/AMAF/SAC absolute metrics와 AMAF−SAC paired gain
- `robustness_seed_sensitivity.csv`: AMAF−SAC terminal/gain/AUC LOO

Metric/direction: paired delta는 AMAF−SAC; gain delta가 양수면 AMAF의 1M→1.5M 증가가 더 큼.

Limits:
- SAC의 1M return과 normalized AUC가 더 높다.
- AMAF terminal wins는 2/5이고 terminal CI는 0을 포함한다.
- 따라서 terminal mean 또는 late gain만으로 전반적 sample-efficiency 우위를 주장하지 않는다.

## Mechanism boundary

Potential claim: checkpoint-time routing becomes selective or switches heads.

Supported by:
- `mechanism_head_summary.csv`: entropy, max weight, unique-dominant concentration, switching
- `mechanism_head_by_regime.csv`: early/mid/late time-phase contrasts
- `mechanism_head_long.csv`: checkpoint-level effective gate/raw router/head-output 기록

Limits:
- 이 증거는 semantic state/regime-conditioned specialization이 아니다.
- 낮은 entropy와 높은 dominant fraction은 conditional specialization뿐 아니라 global collapse와도 양립한다.
- state features가 없으므로 specialization과 collapse의 결정적 구분은 현재 frozen data로 불가능하다.
