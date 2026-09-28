# AMAF Discussion evidence handoff (2026-09-15)

## 생성 파일과 Discussion 질문

- `mechanism_head_summary.csv`: head 사용의 전역 집중도·entropy·switching; 질문 1–3
- `mechanism_head_by_regime.csv`: state regime 대신 time-phase별 head/router 요약; 질문 1–3
- `mechanism_head_long.csv`: checkpoint × head long-format 진단; 질문 1–3
- `ablation_discussion_summary.csv`: 보호/anchoring 및 기준법 contrast; 질문 3
- `robustness_seed_sensitivity.csv`: 전체 및 leave-one-seed-out 민감도; 질문 7
- `crop_tradeoff_discussion.csv`: paired resource/performance/yield trade-off; 질문 4
- `halfcheetah_late_stage_discussion.csv`: terminal과 1M→1.5M gain 분리; 질문 5, 7
- `sac_discussion_context.csv`: primary-five TD3/AMAF/SAC와 AMAF−SAC; 질문 6
- `DISCUSSION_CLAIM_MAP.md`: claim-to-evidence와 limitation mapping

## Source registry

- `lunar_primary_frozen`: `paper_data/lunar/primary_20260828`; lifecycle `frozen`; SHA256SUMS_sha256 `bb9cd9bb6e5bca65894d706a892b8b326b16a98ebbbc4d02db088d2815e22901`
- `lunar_per_seed`: `paper_tables/canonical/lunar_per_seed.csv`; lifecycle `canonical`; file_sha256 `166b5e8ec4972c7655f9b4dde2c9b4333286a39cf2c3ff609445a541c84b8adc`
- `crop_baseline_primary_frozen`: `paper_data/crop/baseline_primary_20260828`; lifecycle `frozen`; SHA256SUMS_sha256 `169fb5a11e039b6981419448c12e87faa9b58e3d598eaa69001aea6b66a83d3f`
- `crop_anchored_primary_frozen`: `paper_data/crop/anchored_primary_20260830`; lifecycle `frozen`; SHA256SUMS_sha256 `b345a69f28ab8330c33a8dd4145f7ef6876339ef38042680c31cf20f98f6ce6b`
- `crop_per_seed`: `paper_tables/canonical/crop_per_seed.csv`; lifecycle `canonical`; file_sha256 `07123c5a06e401ff8622725fa47c9ae0b8c69106e667f391a51f76dffc7b8576`
- `halfcheetah_primary_frozen`: `paper_data/halfcheetah/primary_1p5m_20260830`; lifecycle `frozen`; SHA256SUMS_sha256 `81df478ff8e9ebef068f792f97bf7ab77391edad1cd86df99d2d0d3c6d297214`
- `halfcheetah_replication_frozen`: `paper_data/halfcheetah/replication_extra5_1p5m_20260907`; lifecycle `frozen`; SHA256SUMS_sha256 `89dd7366173dbb23b5830611a3b80986ed4755cc48df56a4b3ca72f4f0c63651`
- `halfcheetah_per_seed`: `paper_tables/canonical/halfcheetah_per_seed.csv`; lifecycle `canonical`; file_sha256 `7856c4cd2a32bd28e8a067ddd8025fe031799b1a464a305e6d4a6ea6cc554082`
- `halfcheetah_sac_primary5`: `paper_data/halfcheetah/sac_primary5_1p5m_20260910`; lifecycle `frozen`; SHA256SUMS_sha256 `0c5795b36a614822c17bb79a26e2db41318c5ba1d41cc8ca7f428fc3848f8f7c`

모든 source는 `figure_contract.yaml` registry ID로 해결했다. `paper_data`, canonical table,
MANIFEST, SHA256SUMS는 수정하지 않았다. Seed population은 Lunar 10개(11–111의 지정 seed),
Crop 5개(11–55), HalfCheetah combined10, SAC primary5이며 seed 77을 포함해 제외한 seed는 없다.

## Mechanism data availability and extraction

Frozen `diagnostics.csv`에는 diagnostic checkpoint의 `env_steps`, `episode`, effective
`gate_w*`, entropy/max-weight/effective-head/dominant-head와 `head*_mean`이 있다. Lunar
Protected 및 Crop Anchored에는 raw `router_w*`와 `adaptive_trust`도 있다. Effective gate와
raw router는 섞지 않고 `weight_type`으로 분리했다.

Probability-simplex 조건(유한·비음수·각 weight ≤1·합≈1)을 통과한 행만 gate distribution
통계와 long table에 사용했다. 전체 유효 행은 6,226개이며 제외된 invalid conditional
gate 행은 74개다. Entropy는 각 행에서 직접 재계산했고 normalized entropy는
`H/log(n_heads)`, effective heads는 각 행의 `exp(H)`를 평균했다. Equal-maximum tie는
임의 dominant head로 지정하지 않았다. `dominant_head_fraction`은 유효 관측 전체 중 가장
자주 유일 dominant였던 head의 비율이며 tie 비율을 별도로 제공한다. Switching은 시간상
인접하고 양쪽 모두 unique-dominant인 관측만 분모로 사용한다.

Protected/Anchored의 `reference_or_null_weight_mean`은 기록된 `1-adaptive_trust`, 즉
uniform/reference 쪽 계수다. HalfCheetah selective AMAF의 explicit NULL weight는 frozen
historical logger에 기록되지 않아 재구성하지 않았다. `reference_or_null_weight_mean`,
router 관련 열, head 수가 적은 method의 여분 head 열, dominant tie만 존재해 switching이
정의되지 않는 경우의 NaN은 모두 구조적으로 정당한 missing이다.

## Regime definition and missing mechanism evidence

State feature, action, altitude/velocity, crop growth/resource 상태, 의미론적 regime label은
frozen diagnostics에 기록되지 않았다. 따라서 `time_early`, `time_mid`, `time_late`는 각
seed의 기록된 diagnostic checkpoint를 시간순으로 3등분한 대체 구간이며 state-conditioned
regime가 아니다. 각 diagnostic row도 full state distribution이 아니라 logging 시점의 한
state 표본이다.

**MISSING / NOT RECORDED:** state-conditioned gate distribution, semantic regime별 head
usage, HalfCheetah full H+1 gate/NULL weight, state/action과 head output의 완전한 alignment는
현재 frozen logs에서 재구성할 수 없다. 따라서 head specialization/collapse에 대한 직접
state-conditioned empirical evidence는 현재 frozen logs에서 재구성할 수 없으며, Results
성능 수치만으로 mechanism을 추론하지 않는다. 이 package는 checkpoint-time selectivity와
switching을 기술통계로 제공할 뿐이다.

## Metric definitions and bootstrap

- Lunar: Last-5 evaluation return, retention=`100×Last-5/Best-5`.
- Crop: final-100 train return/yield/nitrogen/irrigation. Saving은 comparator−focal usage.
- HalfCheetah: exact 1M/1.5M evaluation return, gain=1.5M−1M, Last-5/Last-10,
  10k–1.5M time-normalized trapezoidal AUC.
- SAC context는 같은 primary-five seed와 동일 checkpoint/metric 정의를 사용한다.
- CI는 seed-level percentile bootstrap mean CI, 100,000 resamples,
  95% confidence, RNG seed 20260907. Leave-one-out은 지정 seed를 한 번씩 제외하되
  전체 row(`excluded_seed=NONE`)도 포함한다.

## Limitations and counterevidence / boundary cases

- Lunar: Protected는 Uniform을 일관되게 능가하지 않으며 Protected−Naive도 일부 seed에서
  음수다. 낮은 entropy만으로 좋은 specialization이라 해석하지 않는다.
- Crop: Anchored는 Uniform보다 질소·관개를 5/5 seed에서 절감하지만 final-100 return과
  yield는 각각 1/5 seed에서만 개선됐다. Resource saving은 performance/yield cost와 함께
  나타날 수 있다.
- HalfCheetah: combined terminal CI가 넓고 0을 포함한다. seed 77을 포함한 모든 seed를
  유지했으며 Last-5/Last-10/AUC combined mean은 양수가 아니다.
- SAC: SAC는 primary-five 1M return과 AUC가 AMAF보다 높다. AMAF terminal wins는 2/5로,
  terminal mean 우세가 seed-wise dominance를 뜻하지 않는다. Sample-efficiency superiority
  claim은 지원되지 않는다.
- SAC frozen `SHA256SUMS`에 선언된 model checkpoint weight 15개가 tree에 없어 full archive
  checksum은 PASS가 아니다. Numerical input인 MANIFEST/COMPLETED/eval CSV는 검증됐다.

기존 Results handoff와 공통 metric은 export 전에 수치 단위로 일치 검증했다.
