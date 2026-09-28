# Discussion figure data

이 디렉터리는 `paper_tables/discussion_handoff_20260915/`에서 생성한 plot-ready
TSV/CSV를 포함한다. 생성 명령은 다음과 같다.

```bash
python -m analysis.figures.export_discussion_figures
```

## D1 routing mechanism

- `discussion_d1_head_heatmap.tsv`: Lunar Protected, Crop Anchored,
  HalfCheetah AMAF의 recorded head weight를 seed별 checkpoint 범위에서 normalized
  progress 10개 bin으로 집계한 뒤 seed 평균한 값이다. Lunar/Crop은 `raw_router`와
  `effective_gate`를 모두 보존하며, `reference_note`는 각각 Uniform `0.25`,
  `0.333333`을 기록한다. HalfCheetah는 conditional residual-head `effective_gate`만
  포함하고 reference는 `not_recorded`다.
- `discussion_d1_concentration.tsv`: handoff의 checkpoint-time early/middle/late
  thirds별 normalized entropy, effective-head count, maximum gate weight. Lunar/Crop은
  `raw_router`, HalfCheetah는 conditional `effective_gate`를 사용하며 `seed`와 `mean`
  row를 모두 포함한다.
- `discussion_d1_entropy_observations.csv`: 유효 checkpoint를 seed 내부 시간 순서로
  정렬한 뒤 Early/Middle/Late thirds로 재분할하여 계산한 normalized entropy
  `H/log(K)`. Lunar Protected와 Crop Anchored는 `raw_router`, HalfCheetah AMAF는
  기록된 conditional residual-head `effective_gate`를 사용한다. 각 행은 하나의
  logged checkpoint-time observation이며 `seed`, `checkpoint_index`, 원래 step을
  함께 보존한다. `jitter_offset`과 `phase_x`는 seed와 checkpoint index만으로 계산한
  결정적 수평 위치이며 entropy 값 자체를 바꾸지 않는다.
- `routing_entropy_phase_quantiles.csv`: panel (b)에 표시하는 동일 checkpoint
  관측값의 phase별 `n`, q05, q10, q25, median, q75, q90, q95, mean. Figure는
  5--95%, 10--90%, 25--75% nested horizontal interval과 median marker만 표시하며
  KDE, 보간값, raw point cloud를 사용하지 않는다.
- `discussion_d1_entropy_kde.csv`: 위 checkpoint 관측값의 경계 보정 Gaussian KDE를
  고정된 0--1 grid에 평가했던 이전 rendering table. 재현성 기록을 위해 유지하지만
  현재 production figure의 입력으로는 사용하지 않는다.
- `discussion_d1_switching.tsv`: 인접한 두 checkpoint가 모두 unique-dominant일 때만
  정의한 effective/conditional head-switch rate. Lunar Protected seed 55는 eligible
  transition이 없어 제외 사유와 `9/10` 정의 가능 seed 수를 명시했으며 0으로 대체하지
  않았다. Uniform은 모든 최대값이 tie라 switch rate가 정의되지 않으므로 figure에
  포함하지 않았다.

이 자료의 progress bin과 time phase는 semantic state/regime가 아니다. HalfCheetah
weight는 기록된 conditional residual-head 분포이며, 기록되지 않은 NULL/reference
weight를 복원하지 않았다. 낮은 entropy는 더 집중된 routing을 뜻하지만 semantic
specialization의 증거로 해석하지 않는다. seed-level 통계적 근거는 Results/Appendix의
canonical paired analysis를 따른다. Panel (b)의 quantile은 logged checkpoint-time
diagnostic observation을 기술적으로 요약하며 independent experimental replicate에 대한
추론으로 취급하지 않는다. 낮은 entropy와 낮은 switching은 지속적인 집중 선호와, 낮은
entropy와 nontrivial switching은 dominant head가 바뀌는 집중 routing과 양립하지만 어느
경우도 semantic specialization을 증명하지 않는다.

## D2 claim boundaries

- `discussion_d2_lunar_loo.tsv`: Protected−Naive Last-5 effect의 10개
  leave-one-seed-out mean과 full-sample mean.
- `discussion_d2_crop_tradeoff.tsv`: 다섯 paired seed의 Yield100 delta,
  nitrogen saving, irrigation saving, return delta. composite score는 없다.
- `discussion_d2_halfcheetah_boundary.tsv`: combined-10의 terminal delta와
  1M→1.5M gain delta. original5/replication5를 유지하며 seed 77을 포함한다.

Exporter는 expected seed population, duplicate keys, unexpected NaN, seed 77,
HalfCheetah cohort, Discussion handoff provenance를 검사한다. Results handoff와 겹치는
Lunar full mean, Crop paired delta, HalfCheetah seed-level delta도 수치 단위로 exact-match
검증한다.
