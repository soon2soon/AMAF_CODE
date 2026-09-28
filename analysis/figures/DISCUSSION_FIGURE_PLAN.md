# Discussion figure production plan

## D1 Routing / Head Mechanism

### Manuscript role and recommendation

Main Discussion figure candidate. Results의 return comparison을 반복하지 않고,
기록된 diagnostic checkpoint에서 routing distribution, concentration, temporal switching이
어떻게 나타나는지를 기술한다.

### Source data

- `paper_tables/discussion_handoff_20260915/mechanism_head_long.csv`
- `paper_tables/discussion_handoff_20260915/mechanism_head_by_regime.csv`
- `paper_tables/discussion_handoff_20260915/mechanism_head_summary.csv`
- Plot-ready export: `analysis/figures/data/discussion/discussion_d1_*.tsv` 및
  `analysis/figures/data/discussion/discussion_d1_entropy_*.csv`

### Panels and metric definitions

- **(a1) Lunar Protected raw/effective:** `mechanism_head_long.csv`의 `raw_router`와
  `effective_gate` checkpoint sequence를 별도 heatmap으로 수직 배치한다. Lunar는 4개
  head이므로 Uniform reference는 head당 `0.25`다.
- **(a2) Crop Anchored raw/effective:** 같은 source의 `raw_router`와 `effective_gate`를
  별도 heatmap으로 배치한다. Crop은 3개 head이므로 Uniform reference는 head당
  `1/3 ≈ 0.333`이다.
- **(a3) HalfCheetah conditional residual routing:** historical logger가 기록한 4개
  residual-head conditional weight만 표시한다. explicit NULL/reference weight는 기록되지
  않았으며 합성하거나 재구성하지 않는다.
- **Heatmap aggregation:** 각 seed의 기록된 첫/마지막 checkpoint를 0–1로 정규화하고
  10개 time bin 안에서 먼저 seed별 평균한 뒤 seed 평균한다. 모든 raw/effective map은
  동일한 blue–white–red display scale `0–0.5`를 사용한다. 원본 absolute head weight는
  TSV에 그대로 보존하고, figure rendering에서만 `0.5`를 넘는 값을 `0.5` color endpoint로
  포화시킨다(`0=blue`, `0.25=white`, `0.5=red`).
- **(b) Routing entropy across checkpoint phases:** seed별 유효 checkpoint sequence를 시간 순서의
  Early/Middle/Late thirds로 나누고 각 checkpoint의 normalized entropy `H/log(K)`를
  계산한다. Lunar Protected와 Crop Anchored는 `raw_router`, HalfCheetah AMAF는 기록된
  conditional residual-head entropy를 사용한다. 9개 horizontal row에서 얇은 구간은
  q05--q95, 중간 구간은 q10--q90, 굵은 구간은 q25--q75, 점은 median을 나타낸다. 높은
  entropy는 더 diffuse하고 낮은 entropy는 더 concentrated한 routing을 뜻한다. 이
  summary는 logged checkpoint-time diagnostic의 기술통계이며 independent experimental
  replicate 또는 seed-level inferential statistic으로 해석하지 않는다.
- **(c) Temporal switching:** 시간상 인접하고 양쪽 모두 unique-dominant인 checkpoint
  pair 중 dominant head가 바뀐 비율. 작은 점은 seed, 큰 marker는 정의 가능한 seed의
  mean이다. 이 panel은 Lunar/Crop의 effective gate와 HalfCheetah conditional residual
  weights에 대한 switching이다. Lunar Protected seed 55는 eligible pair가 없어 mean의
  분모가 9개 seed이며, undefined 값을 0으로 대체하지 않는다. Uniform은 모든 최대값이
  tie라 switching이 정의되지 않아 제외한다.

### What the figure supports

- Lunar raw router가 선택적으로 나타날 수 있으면서 effective gate는 head당 0.25에
  가까이 유지되는 checkpoint-time 양상.
- Crop raw router가 강하게 집중될 수 있으면서 effective gate는 head당 약 0.333에
  가까이 유지되는 checkpoint-time 양상.
- HalfCheetah residual-head routing의 시간적 변화.
- Lunar Protected와 HalfCheetah AMAF에서 raw/conditional routing entropy 분포가
  training 후반에 낮은 값 쪽으로 이동하는 양상, 그리고 Crop Anchored가 처음부터 낮은
  entropy에 집중된 양상.
- Method/seed별 dominant-head switching 기술통계 차이.
- Across-seed head-identity 평균과 seed 내부 concentration은 서로 다른 요약이라는 점.

### What the figure does **not** support

- State-conditioned 또는 semantic-regime-conditioned head specialization.
- 상태별로 구분되는 expert 역할.
- 낮은 entropy가 specialization인지 global collapse인지에 대한 결정적 판별.
- Protection/anchoring이 collapse를 방지한다는 인과 주장.
- Entropy와 return 사이의 인과 관계.
- HalfCheetah의 기록되지 않은 NULL/reference head behavior.

### Caption draft

**Routing diagnostics for the reference-preserving variants.** For Lunar Protected and Crop
Anchored, a selective raw routing signal can coexist with effective fusion that remains close
to the corresponding Uniform reference (0.25 and 1/3 per head, respectively). HalfCheetah
shows temporal variation in conditional residual-head routing, but its logger does not contain
an explicit NULL/reference weight or the state-aligned information needed to establish semantic
state-conditioned specialization. (b) Distributional summaries of normalized routing entropy
across early, middle, and late checkpoint thirds. Nested horizontal intervals show the
5--95%, 10--90%, and interquartile ranges, and the marker denotes the median. Lower entropy
indicates more concentrated routing. Lunar and Crop use the raw protected/anchored router,
whereas HalfCheetah uses the available conditional residual-head routing. The summaries
describe logged checkpoint-time diagnostics and are not treated as independent experimental
replicates. The switching panel summarizes effective/conditional dominant-head changes over
training.

## D2 Claim Boundaries / Robustness

### Manuscript role and recommendation

Main Discussion figure candidate. 평균 효과만 반복하지 않고 seed 삭제 민감도,
resource–yield trade-off, terminal 수준과 late-stage gain의 분리를 한 화면에서 보여준다.

### Source data

- `paper_tables/discussion_handoff_20260915/robustness_seed_sensitivity.csv`
- `paper_tables/discussion_handoff_20260915/crop_tradeoff_discussion.csv`
- `paper_tables/discussion_handoff_20260915/halfcheetah_late_stage_discussion.csv`
- Plot-ready export: `analysis/figures/data/discussion/discussion_d2_*.tsv`

### Panels and metric definitions

- **(a) Lunar LOO:** Protected−Naive Last-5 paired effect에서 지정 seed를 하나씩 제외한
  mean delta. 점은 excluded seed별 LOO mean이고 점선은 full-sample mean이다.
- **(b1–b2) Crop trade-off:** x는 `Yield100 Anchored−Uniform`, y는 각각
  `Nitrogen Uniform−Anchored`, `Irrigation Uniform−Anchored`. 각 점은 동일 seed의 paired
  observation이며 x=0, y=0 reference를 함께 표시한다. normalized/composite efficiency
  score는 만들지 않는다.
- **(c) HalfCheetah boundary:** x는 terminal delta @1.5M, y는
  `(AMAF 1.5M−1M)−(TD3 1.5M−1M)`. 각 점은 seed이며 original5와 replication5 marker를
  구분하고 seed 77을 명시한다.

### Interpretation boundaries

- Lunar의 positive Protected−Naive LOO mean은 한 seed만으로 생성된 평균은 아니지만,
  Protected가 Uniform보다 항상 우월하다는 뜻은 아니다.
- Crop의 양의 resource saving은 yield 또는 final-return dominance가 아니며, 다수 seed의
  음의 yield delta를 함께 읽어야 한다.
- HalfCheetah의 late-stage gain과 terminal superiority는 별개다. combined-10 terminal
  CI가 0을 포함하고 seed 77이 불리하므로 terminal dominance를 주장하지 않는다.
- 이 figure를 method 전반의 robustness 또는 global superiority 근거로 확대하지 않는다.

### Caption draft

**Seed-level boundaries on the principal Discussion claims.** Leave-one-seed-out Lunar
effects, paired Crop resource–yield outcomes, and HalfCheetah terminal-versus-gain deltas show
which directions are stable and where trade-offs or seed sensitivity limit stronger claims.

## D3 SAC Context

### Decision

**Appendix candidate; current production set에서는 omit.** SAC primary-five의 terminal,
gain, AUC contrast는 Results handoff에서 이미 직접 보고되고, 별도의 세-panel Discussion
figure는 D2가 제공하는 claim-boundary insight를 실질적으로 확장하지 않는다. Main text에는
D1과 D2 두 figure를 우선한다.

Reviewer가 optimizer context의 seed-level 시각화를 요구하면 primary-five만 사용한
appendix figure로 terminal delta, gain delta, AUC delta를 나란히 제시할 수 있다. 그 경우에도
SAC의 1M return/AUC 강점과 AMAF terminal wins 2/5를 함께 명시하고 global
sample-efficiency superiority를 주장하지 않는다.

### Caption draft if promoted to appendix

**Primary-five SAC context.** AMAF shows a consistently larger 1M-to-1.5M gain, whereas SAC
retains stronger early and trajectory-level evidence; terminal differences are not seed-wise
dominant.

## Production commands and outputs

```bash
python -m analysis.figures.export_discussion_figures
gnuplot analysis/figures/gnuplot/discussion/discussion_d1_mechanism.gp
gnuplot analysis/figures/gnuplot/discussion/discussion_d2_claim_boundaries.gp
```

Authoritative PDFs:

- `paper_figures/gnuplot/discussion/discussion_routing_mechanism.pdf`
- `paper_figures/gnuplot/discussion/discussion_claim_boundaries.pdf`

Same-basename PNG files are previews only. Existing repository policy leaves generated
PDF/PNG outputs untracked. Manuscript LaTeX composition and prose are outside this change.
