# AMAF Results numerical table handoff (2026-09-15)

이 디렉터리는 현재 manuscript main figure와 같은 source, metric, seed population,
aggregation으로 계산한 Results/Appendix용 수치 동결본이다. 통계적 experimental unit은
모두 seed이며 evaluation episode나 Crop episode를 독립 표본으로 취급하지 않았다.

## Figure 대응

- Lunar F1: `lunar_result_summary.csv`, `lunar_per_seed.csv`
- Lunar F2: `lunar_paired_effects.csv`, `lunar_per_seed.csv`
- Crop Fig. A: `crop_result_summary.csv`, `crop_per_seed.csv`
- Crop Fig. C: `crop_result_summary.csv`, `crop_per_seed.csv`
- HalfCheetah Panel A: `halfcheetah_result_summary.csv`, `halfcheetah_per_seed.csv`
- HalfCheetah Panel B: `halfcheetah_per_seed.csv`
- HalfCheetah Panel C: `halfcheetah_result_summary.csv`, `halfcheetah_per_seed.csv`

## Metric 정의와 단위

- Lunar `last5_return`: 마지막 5개 evaluation checkpoint mean return의 평균
  (`lunar.last5`, evaluation return). `retention_pct`: `100 × last5 / best5`, 여기서
  `best5`는 가장 높은 5개 checkpoint mean return의 평균 (`lunar.retention_pct`, %).
- Crop `late1000_return`: Episode 2001–3000 train return 평균
  (`crop.late1000`, train return). `final100_return`: 마지막 100 episode train return 평균
  (`crop.final100`, train return). `yield100`, `n100`, `irr100`: 각각 마지막 100 episode의
  crop yield, nitrogen usage, irrigation usage 평균이며 단위는 contract의 native
  `yield`, `nitrogen`, `irrigation`이다. Figure A 곡선은 seed 내부에서 Episode 1–3000을
  100-episode 비중첩 block으로 먼저 평균한 뒤 seed 간 집계한다.
- HalfCheetah `*_1m`, `*_1p5m`: 정확히 1,000,000/1,500,000 step의 evaluation return.
  `*_gain`: 1.5M 값에서 1.0M 값을 뺀 evaluation-return change. `last5_delta`,
  `last10_delta`: AMAF와 TD3 각각의 마지막 5/10 checkpoint 평균 차이.
  `auc_delta`: 10k–1.5M checkpoint mean curve의 시간 정규화 trapezoidal AUC 차이.

모든 summary CI는 seed-level mean에 대한 percentile bootstrap 95% CI이다. resample 수는
100,000, RNG seed는 20260907, confidence는
0.95이다. `median`은 관측 seed 값의 기술통계 중앙값이며 CI는 mean의
CI이다.

## Delta 방향

- Lunar: `Protected - Naive`, `Protected - Uniform`; 양수는 Protected 우세.
- Crop performance/yield: `Anchored - Uniform`; 양수는 Anchored 우세.
- Crop `nitrogen_saving`: `Uniform n100 - Anchored n100`; 양수는 Anchored의 질소 절감.
- Crop `irrigation_saving`: `Uniform irr100 - Anchored irr100`; 양수는 Anchored의 관개 절감.
- HalfCheetah endpoint/last/AUC delta: `AMAF - TD3`.
- HalfCheetah `gain_delta`: `(AMAF 1.5M - AMAF 1.0M) - (TD3 1.5M - TD3 1.0M)`.

`wins`, `losses`, `ties`는 위 benefit 방향의 seed별 부호로 계산했다.

## Seed population

- Lunar: 11, 22, 33, 44, 55, 66, 77, 88, 99, 111; 5 methods, 각 n=10.
- Crop: 11, 22, 33, 44, 55; 5 methods, 각 n=5.
- HalfCheetah original5: 11, 22, 33, 44, 55; replication5: 66, 77, 88, 99, 111;
  combined10은 두 cohort 전체이며 seed 77을 포함한다.

## Source dataset IDs와 fingerprint

- `lunar_primary_frozen`: `paper_data/lunar/primary_20260828`; SHA256SUMS_sha256=`bb9cd9bb6e5bca65894d706a892b8b326b16a98ebbbc4d02db088d2815e22901`
- `lunar_per_seed`: `paper_tables/canonical/lunar_per_seed.csv`; file_sha256=`166b5e8ec4972c7655f9b4dde2c9b4333286a39cf2c3ff609445a541c84b8adc`
- `crop_baseline_primary_frozen`: `paper_data/crop/baseline_primary_20260828`; SHA256SUMS_sha256=`169fb5a11e039b6981419448c12e87faa9b58e3d598eaa69001aea6b66a83d3f`
- `crop_anchored_primary_frozen`: `paper_data/crop/anchored_primary_20260830`; SHA256SUMS_sha256=`b345a69f28ab8330c33a8dd4145f7ef6876339ef38042680c31cf20f98f6ce6b`
- `crop_per_seed`: `paper_tables/canonical/crop_per_seed.csv`; file_sha256=`07123c5a06e401ff8622725fa47c9ae0b8c69106e667f391a51f76dffc7b8576`
- `halfcheetah_primary_frozen`: `paper_data/halfcheetah/primary_1p5m_20260830`; SHA256SUMS_sha256=`81df478ff8e9ebef068f792f97bf7ab77391edad1cd86df99d2d0d3c6d297214`
- `halfcheetah_replication_frozen`: `paper_data/halfcheetah/replication_extra5_1p5m_20260907`; SHA256SUMS_sha256=`89dd7366173dbb23b5830611a3b80986ed4755cc48df56a4b3ca72f4f0c63651`
- `halfcheetah_per_seed`: `paper_tables/canonical/halfcheetah_per_seed.csv`; file_sha256=`7856c4cd2a32bd28e8a067ddd8025fe031799b1a464a305e6d4a6ea6cc554082`

source path는 코드에 직접 넣지 않고 `analysis/figures/figure_contract.yaml`의 registry ID로
해결했다. canonical/frozen source는 수정하지 않았다.

## Figure–table consistency

PASS. Export 전에 다음을 Python에서 재계산하고 현재 renderer input CSV와 수치 단위로
대조했다.

- Lunar: frozen evaluation으로 canonical final/Last-5/retention을 재현했고, F1 전체 curve,
  F2 Protected–Naive seed delta, method summary/raw points가 일치했다.
- Crop: 두 frozen episode source로 canonical performance windows와 final-100 outcome을
  재현했고, Figure A 전체 100-episode block curve 및 Figure C raw/mean markers가 일치했다.
- HalfCheetah: 두 frozen cohort로 endpoint/gain/Last-5/Last-10/AUC를 재현했고, Panel A 전체
  curve, Panel B seed points, Panel C combined mean/CI/raw effect가 일치했다.

Exporter와 figure exporter는 동일 registry source, metric definition, seed population,
aggregation, bootstrap helper (`analysis.figures.stats`)를 사용한다.
